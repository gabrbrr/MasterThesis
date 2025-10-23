""" 
JaxUED script (with minor logging modifications) for running PLR, ACCEL and DR on Minigrid Maze.
"""

import json
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
import chex
from enum import IntEnum
import hydra
from omegaconf import OmegaConf

from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv

from sfl.envs.ltl_env import Level, make_level_generator, LTLEnv, LTLEnvRenderer, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper


from sfl.util.jaxued.jaxued_utils import l1_value_loss
from sfl.train.train_utils import save_params

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, obs, env_state = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        pi, value = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, next_obs, env_state)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, last_obs, last_env_state), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_obs,
            init_env_state,
        ),
        None,
        length=max_episode_length,
    )

    _, last_value = train_state.apply_fn(train_state.params,last_obs)
    return (rng, train_state, last_obs, last_env_state, last_value), traj

def evaluate(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        pi, _ = train_state.apply_fn(train_state.params,obs)
        action = pi.sample(seed=rng_action)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

def update_actor_critic(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    batch_size = n_steps * num_envs
    minibatch_size = batch_size // n_minibatch
    batch_flat = (obs, actions, log_probs, values, targets, advantages)
    batch_flat = jax.tree_map(
        lambda x: x.reshape((batch_size, *x.shape[2:])),
        batch_flat
    )
    obs_flat, actions_flat, log_probs_flat, values_flat, targets_flat, advantages_flat = batch_flat
        
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                pi, values_pred = train_state.apply_fn(params, obs)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, batch_size)
        batch_shuffled = jax.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            (obs_flat, actions_flat, log_probs_flat, values_flat, targets_flat, advantages_flat)
        )
        minibatches = jax.tree_map(
            lambda x: x.reshape((n_minibatch, minibatch_size, *x.shape[1:])),
            batch_shuffled,
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)



def segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    """Computes the sum of elements within segments of an array."""
    # Note: jax.ops.segment_sum is deprecated. Using jax.lax.segment_sum instead.
    # We pad segment_ids to avoid jax.lax.segment_sum's check.
    # This assumes segment_ids are contiguous from 0 to num_segments - 1.
    return jax.lax.segment_sum(data, segment_ids, num_segments=num_segments)

class RelationalUpdate(nn.Module):
    """
    A Flax module to compute messages based on relation type.
    It applies a different linear transformation for each relation.
    """
    features: int
    num_relations: int

    @nn.compact
    def __call__(self, nodes: jnp.ndarray, senders: jnp.ndarray, edge_types: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            nodes: The node features array of shape `[num_nodes, in_features]`.
            senders: The sender node indices for each edge of shape `[num_edges]`.
            edge_types: The integer type for each edge of shape `[num_edges]`.

        Returns:
            An array of computed messages of shape `[num_edges, out_features]`.
        """
        in_features = nodes.shape[-1]
        
        # Create a stack of weight matrices, one for each relation type.
        kernels = self.param(
            'kernels',
            nn.initializers.lecun_normal(),
            (self.num_relations, in_features, self.features)
        )
        
        # Get the features of the sender nodes for each edge.
        sender_features = nodes[senders]  # Shape: [num_edges, in_features]
        
        # Select the appropriate kernel for each edge based on its type.
        edge_kernels = kernels[edge_types] # Shape: [num_edges, in_features, out_features]
        
        # Compute messages: messages[e] = W_type(e) * h_sender(e)
        # einsum is efficient for this batched matrix-vector product.
        messages = jnp.einsum('eif,ei->ef', edge_kernels, sender_features) # Shape: [num_edges, out_features]
        
        return messages

def CustomRelationalGraphConvolution(
    update_node_module: nn.Module,
    symmetric_normalization: bool = True
) -> Callable[[dict], dict]:
    """
    Returns a function that applies a Relational Graph Convolution layer.
    This function wraps the message computation and performs aggregation.
    """
    def _ApplyRGCN(graph: dict) -> dict:
        nodes, senders, receivers, edge_types = (
            graph["nodes"], graph["senders"], graph["receivers"], graph["edge_types"]
        )

        # Compute messages using the provided relation-specific update module.
        messages = update_node_module(nodes, senders, edge_types)
        
        total_num_nodes = nodes.shape[0]

        # Aggregate messages at receiver nodes.
        if symmetric_normalization:
            ones = jnp.ones_like(senders, dtype=jnp.float32)
            # Ensure degrees are calculated correctly even for isolated nodes
            sender_degree = segment_sum(ones, senders, total_num_nodes).clip(1.0)
            receiver_degree = segment_sum(ones, receivers, total_num_nodes).clip(1.0)

            norm_senders = jax.lax.rsqrt(sender_degree)
            norm_receivers = jax.lax.rsqrt(receiver_degree)
            
            messages = messages * norm_senders[senders, None]
            aggregated_nodes = segment_sum(messages, receivers, total_num_nodes)
            aggregated_nodes = aggregated_nodes * norm_receivers[:, None]
        else:
            aggregated_nodes = segment_sum(messages, receivers, total_num_nodes)

        return {**graph, "nodes": aggregated_nodes}
        
    return _ApplyRGCN


# --- GNN Module (Provided) ---

class RGCNRootShared_no_jraph(nn.Module):
    """An RGCN with shared weights and root-based readout."""
    hidden_dim: int
    num_layers: int
    output_dim: int
    num_edge_types: int

    @nn.compact
    def __call__(self, graph: dict) -> jnp.ndarray:
        # Separate the 'is_root' flag from the node features.
        h_features = graph["nodes"][:, :-1]
        is_root_nodes = graph["nodes"][:, -1:]

        # Initial linear projection.
        h_0 = nn.Dense(features=self.hidden_dim, name='input_dense')(h_features)
        h = h_0
        
        # Define the single, shared convolutional layer module.
        # Its input will have size 2 * hidden_dim due to the skip connection.
        shared_update_module = RelationalUpdate(
            features=self.hidden_dim, 
            num_relations=self.num_edge_types,
            name='shared_rgcn_update'
        )
        rgcn_layer = CustomRelationalGraphConvolution(update_node_module=shared_update_module)

        # Prepare graph structure (excluding nodes) for convolution loops
        conv_graph = {key: val for key, val in graph.items() if key != 'nodes'}
        
        for _ in range(self.num_layers):
            h_cat = jnp.concatenate([h, h_0], axis=-1)
            current_layer_graph = {**conv_graph, "nodes": h_cat}
            
            graph_after_rgcn = rgcn_layer(current_layer_graph)
            # Use tanh activation as in the DGL example.
            h = nn.tanh(graph_after_rgcn["nodes"])

        # Graph Readout: Select and sum root node embeddings.
        num_graphs = graph["n_node"].shape[0]
        
        # This logic handles batching (num_graphs > 1) and single instances (num_graphs=1)
        if num_graphs > 0:
            # Create segment_ids for segment_sum
            # This assumes nodes are packed contiguously per graph
            num_total_nodes = h.shape[0]
            # Handle potential padding if n_node doesn't sum to total_nodes
            if "n_node" in graph:
                 segment_ids = jnp.repeat(jnp.arange(num_graphs), repeats=graph["n_node"])
                 # If nodes are padded, we need to ensure segment_ids match h.shape[0]
                 padding_size = num_total_nodes - segment_ids.shape[0]
                 if padding_size > 0:
                     # Pad with an unused segment ID (num_graphs), which will be ignored
                     padding_ids = jnp.full(padding_size, num_graphs, dtype=jnp.int32)
                     segment_ids = jnp.concatenate([segment_ids, padding_ids])
                     # We need num_segments to be num_graphs + 1 to account for padding
                     graph_embeddings = segment_sum(h * is_root_nodes, segment_ids, num_segments=num_graphs + 1)
                     # Slice off the padding segment
                     graph_embeddings = graph_embeddings[:num_graphs]
                 else:
                     graph_embeddings = segment_sum(h * is_root_nodes, segment_ids, num_segments=num_graphs)
            else:
                 # Fallback if n_node is not provided (assumes equal-sized graphs)
                 num_nodes_per_graph = num_total_nodes // num_graphs
                 segment_ids = jnp.repeat(jnp.arange(num_graphs), repeats=num_nodes_per_graph)
                 graph_embeddings = segment_sum(h * is_root_nodes, segment_ids, num_segments=num_graphs)
        else: 
            graph_embeddings = jnp.zeros((0, h.shape[-1]))

        output = nn.Dense(features=self.output_dim, name='output_dense')(graph_embeddings)
        return jnp.squeeze(output, axis=0) # Squeeze just in case batch size was 1


# --- EnvModel (Provided) ---

class EnvModel(nn.Module):
    """
    A CNN to process image-like observations, matching the PyTorch LetterEnvModel.
    It uses three convolutional layers with (2, 2) kernels and no pooling.
    """
    @nn.compact
    def __call__(self, obs_image: jnp.ndarray) -> jnp.ndarray:
        # Note: JAX Conv expects (N, H, W, C)
        x = nn.Conv(features=16, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_image)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        # Flatten the output of the convolutional layers
        return x.reshape((x.shape[0], -1))



class ActorCritic(nn.Module):
    """
    JAX-native Actor-Critic model that combines visual and textual (GNN) embeddings.
    Accepts an Observation dataclass and returns a distrax.Distribution.
    """
    text_embedding_size: int = 32
    output_dim: int = 4
    value_ensemble_size: int = 1


    def setup(self):
        """Initializes the sub-modules of the actor-critic model."""
        self.env_model = EnvModel()
        VmappedGNN = nn.vmap(
            RGCNRootShared_no_jraph,
            in_axes=0,             # Map over the first axis of the input PyTree (the Graph dict)
            out_axes=0,            # Stack outputs along the first axis
            variable_axes={'params': None}, # Do not map/split the model parameters
            split_rngs={'params': False}    # Do not split RNGs for parameter initialization
        )
        
        self.gnn = VmappedGNN(
            output_dim=self.text_embedding_size,
            hidden_dim=32,    
            num_layers=2,     
            num_edge_types=4
        )

        # Define actor network layers directly to output logits.
        actor_layers = []
        hiddens = [64, 64, 64]
        for hidden_size in hiddens:
            actor_layers.append(nn.Dense(features=hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)))
            actor_layers.append(nn.relu)

        
        actor_layers.append(nn.Dense(features=self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)))
        
        self.actor_net = nn.Sequential(actor_layers)

        # Critic network remains the same
        self.critic_net = nn.Sequential([
            nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(features=1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        ])

    def __call__(self, obs: Observation, carry=None, reset=None) -> Tuple[jnp.ndarray, distrax.Distribution, Any]:
        """
        Forward pass for the Actor-Critic model.

        Args:
            obs: An Observation dataclass instance.

        Returns:
            A tuple containing:
            - v (jnp.ndarray): The state-value estimate (critic).
            - distribution (distrax.Distribution): The policy distribution (actor).
            - carry (Any): The recurrent hidden state (None for this model).
        """
        
        # --- MODIFICATION: Use dataclass attributes ---
        # Process image features
        embedding_img = self.env_model(obs.image)

        # Create graph dictionary for the GNN from the Observation dataclass
        graph_dict = {
            "nodes": obs.nodes,
            "senders": obs.senders,
            "receivers": obs.receivers,
            "edge_types": obs.edge_types,
            "n_node": obs.n_node,
        }
        # Process text features with GNN
        embedding_gnn = self.gnn(graph_dict)
        # --- End Modification ---

        # Combine embeddings
        embedding = jnp.concatenate((embedding_img, embedding_gnn), axis=1)

        # --- Actor Pass ---
        # Get unnormalized logits from the actor network
        logits = self.actor_net(embedding)
        
        # --- MODIFICATION: Create distribution ---
        distribution = distrax.Categorical(logits=logits)
        # --- End Modification ---

        # --- Critic Pass ---
        # Get the state-value estimate from the critic network
        v = self.critic_net(embedding)

        
        return distribution, v

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }

def compute_score(config, dones, values, max_returns, advantages):
    if config['SCORE_FUNCTION'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['SCORE_FUNCTION'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['SCORE_FUNCTION'] == "l1vl":
        return l1_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['SCORE_FUNCTION']}")

@hydra.main(version_base=None, config_path="config", config_name="letter-plr")
def main(config):
    config = OmegaConf.to_container(config)
    
    d = {}
    for k, v in config.items():
        if isinstance(v, dict):
            d = d | v
        else:
            d[k] = v
    config = d
    config["TOTAL_TIMESTEPS"] = int(config["TOTAL_TIMESTEPS"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    tags = []
    if not config["EXPLORATORY_GRAD_UPDATES"]:
        tags.append("robust")
    if config["USE_ACCEL"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    run = wandb.init(
        group=config.get('GROUP_NAME'),
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
    )
 
    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["NUM_ENVS"] * config["NUM_STEPS"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["EVAL_LEVELS"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["EVAL_LEVELS"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        
        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        for s in ['dr', 'replay', 'mutation']:
            if train_state_info['info'][f'num_{s}_updates'] > 0:
                log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # animations
        for i, level_name in enumerate(config["EVAL_LEVELS"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    # Setup the environment

    env = LTLEnv(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=False, use_agent_centric_view=True, timeout=100, num_unique_letters=len(set("aabbccddeeffgghhiijjkkll")), intrinsic= 0.0)
    eval_env = env
    sample_random_level = make_level_generator(grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=False )
    env_renderer = LTLEnvRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(2)

    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["PLR_PARAMS"]["capacity"],
        replay_prob=config["PLR_PARAMS"]["replay_prob"],
        staleness_coeff=config["PLR_PARAMS"]["staleness_coeff"],
        minimum_fill_ratio=config["PLR_PARAMS"]["minimum_fill_ratio"],
        prioritization=config["PLR_PARAMS"]["prioritization"],
        prioritization_params=config["PLR_PARAMS"]["prioritization_params"],
        duplicate_check=config["PLR_PARAMS"]['duplicate_check'],
    )
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config["LR"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_map(
            lambda x: jnp.repeat(x[None, ...], config["NUM_ENVS"], axis=0),
            obs,
        )
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, obs) 
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_map(lambda x: jnp.array([x]).repeat(config["NUM_ENVS"], axis=0), pholder_level)
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
            This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
                The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            
            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["NUM_ENVS"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["NUM_ENVS"],
                config["NUM_STEPS"],
            )
            advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["NUM_ENVS"],
                config["NUM_STEPS"],
                config["NUM_MINIBATCHES"],
                config["UPDATE_EPOCHS"],
                config["CLIP_EPS"],
                config["ENT_COEF"],
                config["VF_COEF"],
                update_grad=config["EXPLORATORY_GRAD_UPDATES"],
            )
            
            metrics = {
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "mean_num_nodes": new_levels.num_nodes.sum() / config["NUM_ENVS"],
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics
        
        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This samples levels from the level buffer, and updates the policy on them.
            """
            sampler = train_state.sampler
            
            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["NUM_ENVS"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["NUM_ENVS"],
                config["NUM_STEPS"],
            )
            advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
            max_returns = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], compute_max_returns(dones, rewards))
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})
            
            # Update the policy using trajectories collected from replay levels
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["NUM_ENVS"],
                config["NUM_STEPS"],
                config["NUM_MINIBATCHES"],
                config["UPDATE_EPOCHS"],
                config["CLIP_EPS"],
                config["ENT_COEF"],
                config["VF_COEF"],
                update_grad=True,
            )
                            
            metrics = {
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "mean_num_nodes": levels.num_nodes.sum() / config["NUM_ENVS"],
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, config["NUM_ENVS"]), parent_levels, config["NUM_EDITS"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["NUM_ENVS"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["NUM_ENVS"],
                config["NUM_STEPS"],
            )
            advantages, targets = compute_gae(config["GAMMA"], config["GAE_LAMBDA"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["NUM_ENVS"],
                config["NUM_STEPS"],
                config["NUM_MINIBATCHES"],
                config["UPDATE_EPOCHS"],
                config["CLIP_EPS"],
                config["ENT_COEF"],
                config["VF_COEF"],
                update_grad=config["EXPLORATORY_GRAD_UPDATES"],
            )
            
            metrics = {
                "losses": jax.tree_map(lambda x: x.mean(), losses),
                "mean_num_nodes": child_levels.num_nodes.sum() / config["NUM_ENVS"],
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics
    
        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)
        
        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["USE_ACCEL"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        return jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state
        )
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["EVAL_LEVELS"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["EVAL_LEVELS"])
        num_levels = len(config["EVAL_LEVELS"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate(
            rng,
            eval_env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["EVAL_FREQ"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["EVAL_NUM_ATTEMPTS"]), train_state)
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
        metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
        metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)
        
        highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
        
        metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
        metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["EVAL_NUM_ATTEMPTS"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['EVAL_LEVELS'])
        return states, cum_rewards, episode_lengths

    # Set up the train states
    rng = jax.random.PRNGKey(config["SEED"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["NUM_UPDATES"] // config["EVAL_FREQ"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["CHECKPOINT_SAVE_INTERVAL"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    if config["SAVE_PATH"] is not None:
        params = runner_state[1].params
        
        save_dir = os.path.join(config["SAVE_PATH"], wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()

    return runner_state[1]

if __name__=="__main__":
    main()
