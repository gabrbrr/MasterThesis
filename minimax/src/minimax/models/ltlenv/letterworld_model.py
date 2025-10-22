import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, constant
from typing import Tuple, Any, Sequence, Dict
from functools import partial
from minimax.models.registration import register
from minimax.envs.ltl_env.utils import *
from typing import Callable, List, NamedTuple, Optional

def segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    """Computes the sum of elements within segments of an array."""
    return jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)

# ==============================================================================
# 3. Custom GraphConvolution Layer (no jraph)
# ==============================================================================
def CustomGraphConvolution(
    update_node_fn: Callable[[jnp.ndarray], jnp.ndarray],
    add_self_edges: bool = False,
    symmetric_normalization: bool = True
) -> Callable[[dict], dict]:
    """Returns a function that applies a Graph Convolution layer without jraph."""
    def _ApplyGCN(graph: dict) -> dict:
        nodes, senders, receivers, _= graph["nodes"], graph["senders"], graph["receivers"], graph["n_node"]
        total_num_nodes = nodes.shape[0]

        nodes = update_node_fn(nodes)

        if add_self_edges:
            conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)))
            conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)))
        else:
            conv_senders = senders
            conv_receivers = receivers

        if symmetric_normalization:
            ones = jnp.ones_like(conv_senders, dtype=jnp.float32)
            sender_degree = segment_sum(ones, conv_senders, total_num_nodes)
            receiver_degree = segment_sum(ones, conv_receivers, total_num_nodes)

            norm_senders = jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))
            norm_receivers = jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))

            messages = nodes[conv_senders] * norm_senders[conv_senders, None]
            aggregated_nodes = segment_sum(messages, conv_receivers, total_num_nodes)
            aggregated_nodes = aggregated_nodes * norm_receivers[:, None]
        else:
            messages = nodes[conv_senders]
            aggregated_nodes = segment_sum(messages, conv_receivers, total_num_nodes)

        return {**graph, "nodes": aggregated_nodes}

    return _ApplyGCN

def CustomGraphConvolution_PostUpdate(
    update_node_fn: Callable[[jnp.ndarray], jnp.ndarray],
    add_self_edges: bool = False
) -> Callable[[dict], dict]:
    """
    Returns a function that applies a GCN layer without jraph.

    This version applies the update_node_fn *after* node aggregation to mimic
    the behavior of jraph.GraphConvolution with a segment_sum aggregator.
    """
    def _ApplyGCN(graph: dict) -> dict:
        nodes, senders, receivers, _ = graph
        total_num_nodes = nodes.shape[0]

        # Determine senders and receivers, optionally adding self-edges
        if add_self_edges:
            conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)))
            conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)))
        else:
            conv_senders = senders
            conv_receivers = receivers

        # Aggregate messages from neighbors (using their current features)
        messages = nodes[conv_senders]
        aggregated_nodes = segment_sum(messages, conv_receivers, total_num_nodes)

        # Apply the update function to the aggregated representations
        updated_nodes = update_node_fn(aggregated_nodes)

        return {**graph, "nodes": updated_nodes}

    return _ApplyGCN

class GCNRoot_no_jraph(nn.Module):
    """
    A GCN model with root node readout, implemented without jraph.

    The network architecture has skip connections, and readout is performed
    by summing the features of nodes marked as roots.
    """
    hidden_dims: List[int]
    output_dim: int

    @nn.compact
    def __call__(self, graph: dict, is_root_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the GCNRoot_no_jraph model.

        Args:
            graph: A Graph tuple containing the graph data.
            is_root_mask: A boolean JAX array indicating which nodes are roots.

        Returns:
            A JAX array representing the graph-level embeddings from root nodes.
        """
        # Store initial node features for skip connections
        h_0 = graph["nodes"]
        h = h_0

        # Sequentially apply GCN layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:
                features_for_conv = jnp.concatenate([h, h_0], axis=-1)
            else:
                features_for_conv = h

          
            graph_for_conv={**graph, "nodes": features_for_conv}
            gcn_layer = CustomGraphConvolution_PostUpdate(
                update_node_fn=nn.Sequential([
                    nn.Dense(features=hidden_dim, name=f'gcn_dense_{i}'),
                    nn.relu
                ]),
                add_self_edges=True
            )
            # The output of the layer is the new node features
            h = gcn_layer(graph_for_conv)["nodes"]

        # --- Readout from root nodes ---
        # Mask the final node features to keep only root nodes
        masked_nodes = h * is_root_mask[:, jnp.newaxis]

        # Create segment IDs to map each node to its graph in the batch
        num_graphs = graph["n_node"].shape[0]
        graph_indices = jnp.repeat(jnp.arange(num_graphs), graph["n_node"], axis=0)

        # Sum the masked node features for each graph
        hg = segment_sum(masked_nodes, graph_indices, num_segments=num_graphs)

        # Final dense layer for prediction
        out = nn.Dense(features=self.output_dim, name="g_embed")(hg)

        return jnp.squeeze(out, axis=-1) if self.output_dim == 1 else out


class GCNRootShared_no_jraph(nn.Module):
    """
    A GCN with a shared layer and root node readout, implemented without jraph.

    This model repeatedly applies the same GCN layer, using skip connections
    from the initial projected features. Readout is performed on the root node(s).
    """
    hidden_dim: int
    num_layers: int
    output_dim: int

    @nn.compact
    def __call__(self, graph: dict, is_root_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the GCNRootShared_no_jraph model.

        Args:
            graph: A Graph tuple containing graph data.
            is_root_mask: A boolean JAX array indicating which nodes are roots.

        Returns:
            A JAX array representing the graph-level embeddings from root nodes.
        """
        node_features = graph["nodes"]

        # Initial linear projection.
        h_0 = nn.Dense(features=self.hidden_dim, name="linear_in")(node_features)
        h = h_0

        # Define a single GCN layer to be shared across all steps.
        # In Flax, defining it once and calling it in a loop shares the weights.
        shared_gcn_layer = CustomGraphConvolution_PostUpdate(
            update_node_fn=nn.Sequential([
                nn.Dense(features=self.hidden_dim, name="shared_gcn_dense"),
                nn.relu
            ]),
            add_self_edges=True
        )

        # Apply the shared GCN layer multiple times.
        for _ in range(self.num_layers):
            h_cat = jnp.concatenate([h, h_0], axis=-1)
            graph_for_conv={**graph, "nodes": h_cat}
            h = shared_gcn_layer(graph_for_conv)["nodes"]

        # --- Readout from root nodes ---
        # Mask the final node features to keep only root nodes
        masked_nodes = h * is_root_mask[:, jnp.newaxis]

        # Create segment IDs to map each node to its graph in the batch
        num_graphs = graph["n_node"].shape[0]
        graph_indices = jnp.repeat(jnp.arange(num_graphs), graph["n_node"], axis=0)

        # Sum the masked node features for each graph
        hg = segment_sum(masked_nodes, graph_indices, num_segments=num_graphs)

        # Final dense layer for prediction.
        out = nn.Dense(features=self.output_dim, name="g_embed")(hg)

        return jnp.squeeze(out, axis=-1) if self.output_dim == 1 else out

# ==============================================================================
# 4. GCN Model Definition (no jraph)
# ==============================================================================
class GCN_no_jraph(nn.Module):
    """A GCN with skip connections, implemented without jraph."""
    hidden_dims: List[int]
    output_dim: int

    @nn.compact
    def __call__(self, graph: dict) -> jnp.ndarray:
        h_initial = graph["nodes"]
        h = h_initial

        for i, num_hidden in enumerate(self.hidden_dims):
            if i > 0:
                h = jnp.concatenate([h, h_initial], axis=-1)

            current_layer_graph = {**graph, "nodes": h}
            gcn_layer = CustomGraphConvolution(
                update_node_fn=nn.Dense(features=num_hidden, name=f'gcn_dense_layer_{i}'),
                add_self_edges=False,
                symmetric_normalization=True
            )
            graph_after_gcn = gcn_layer(current_layer_graph)
            h = nn.relu(graph_after_gcn["nodes"])

        final_graph = {**graph, "nodes": h}

        # --- Readout Logic ---
        # When processed by vmap, `num_graphs` will be 1.
        num_graphs = graph["n_node"].shape[0]
        num_total_nodes = graph["nodes"].shape[0]
        
        # This logic correctly handles both vmapped (single graph) and batched inputs.
        # However, for vmap, the result of the aggregation will have a leading dim of 1.
        if num_graphs > 0:
            num_nodes_per_graph = num_total_nodes // num_graphs
            segment_ids = jnp.repeat(jnp.arange(num_graphs), repeats=num_nodes_per_graph)
            graph_sum = segment_sum(final_graph["nodes"], segment_ids, num_segments=num_graphs)
            true_node_counts = jnp.maximum(final_graph["n_node"][:, None], 1.0)
            graph_embeddings = graph_sum / true_node_counts
        else: # Handle case with 0 graphs
            graph_embeddings = jnp.zeros((0, h.shape[-1]))


        output = nn.Dense(features=self.output_dim, name='output_dense')(graph_embeddings)

        # ============================ FIX ============================
        # Squeeze the graph dimension. When vmap processes one graph,
        # the output shape is (1, output_dim). We remove the '1' to make
        # it a simple feature vector of shape (output_dim,).
        # vmap will then correctly stack these vectors into a (batch_size, output_dim) array.
        return jnp.squeeze(output, axis=0)

# JAX-native environment model (CNN for image features)
class EnvModel(nn.Module):
    """
    A CNN to process image-like observations, matching the PyTorch LetterEnvModel.
    It uses three convolutional layers with (2, 2) kernels and no pooling.
    """
    @nn.compact
    def __call__(self, obs_image: jnp.ndarray) -> jnp.ndarray:
        # Note: JAX Conv expects (N, H, W, C), while the PyTorch model description
        # implies a channel-first format. The input data should be appropriately
        # transposed before being passed to this model if needed.
        x = nn.Conv(features=16, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs_image)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(2, 2), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        # Flatten the output of the convolutional layers
        return x.reshape((x.shape[0], -1))



class ACModel(nn.Module):
    """
    JAX-native Actor-Critic model that combines visual and textual (GNN) embeddings.
    """
    text_embedding_size: int = 32
    output_dim: int = 4
    value_ensemble_size: int = 1


    def setup(self):
        """Initializes the sub-modules of the actor-critic model."""
        self.env_model = EnvModel()
        VmappedGNN = nn.vmap(
            GCN_no_jraph,
            in_axes=0,               # Map over the first axis of the input PyTree (the Graph tuple)
            out_axes=0,              # Stack outputs along the first axis
            variable_axes={'params': None}, # Do not map/split the model parameters
            split_rngs={'params': False}    # Do not split RNGs for parameter initialization
        )
        self.gnn = VmappedGNN(output_dim=self.text_embedding_size, hidden_dims=[32, 32])

        # Define actor network layers directly to output logits.
        # This architecture mirrors the one in the original PolicyNetwork.
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

    def __call__(self, obs: Dict, carry=None, reset=None) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
        """
        Forward pass for the Actor-Critic model.

        Args:
            obs: A dictionary containing 'image' and 'text' observations.

        Returns:
            A tuple containing:
            - v (jnp.ndarray): The state-value estimate (critic).
            - logits (jnp.ndarray): The unnormalized policy output (actor).
            - carry (Any): The recurrent hidden state (None for this model).
        """
        print(obs["image"].shape, obs["nodes"].shape, obs["senders"].shape, obs["receivers"].shape, obs["n_node"].shape)
        # Process image features
        embedding_img = self.env_model(obs['image'])

        # Process text features with GNN
        embedding_gnn = self.gnn(obs)

        # Combine embeddings
        embedding = jnp.concatenate((embedding_img, embedding_gnn), axis=1)

        # --- Actor Pass ---
        # Get unnormalized logits from the actor network
        logits = self.actor_net(embedding)

        # --- Critic Pass ---
        # Get the state-value estimate from the critic network
        v = self.critic_net(embedding)

        # The model is not recurrent, so the carry state is None
        carry = None
        print(v.shape, logits.shape)
        return v, logits, carry

    @property
    def is_recurrent(self):
        return False

# Register models
if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
	env_group_id='LTLEnv', model_id='letter_env_default', 
	entry_point=module_path + ':ACModel')