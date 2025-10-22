from dataclasses import dataclass
import jax
import jax.numpy as jnp
from dataclasses import dataclass, replace
from flax.struct import PyTreeNode
from minimax.envs import environment, spaces
from minimax.envs.ltl_env.letter_env import EnvParams, EnvState, LetterEnv
from minimax.envs.ltl_env import progress
import minimax.envs.ltl_env.ast
from minimax.envs.ltl_env.utils import * 
from functools import partial
import chex
from minimax.envs.registration import register
from minimax.envs.ltl_env.sampler import JaxUntilTaskSampler
from collections import OrderedDict
from minimax.envs.ltl_env.ast import JaxASTBuilder
class LTLEnvState(PyTreeNode):
    env_state: any  # Underlying env state
    ltl_goal: jnp.ndarray   # Current LTL formula
    ltl_original: jnp.ndarray  # Original LTL formula
    key: jnp.ndarray  # PRNG key
    num_nodes: jnp.ndarray
    root_idx: jnp.ndarray

class LTLEnv(environment.Environment):
    """
    Functional wrapper adding LTL goals to a Gymnax environment.
    Adds LTL formula to observations, progresses it, and modifies rewards.
    """
    def __init__(self,  grid_size=5, letters="abcdefghijkl", use_fixed_map=False, use_agent_centric_view=False, timeout=100, num_unique_letters=len(set(encode_letters("abcdefghijkl"))), ltl_sampler: str = None, intrinsic: float = 0.0):
        super().__init__()
        self.params=EnvParams(grid_size=5, 
                               letters=encode_letters(letters),
                               use_fixed_map=False,
                               use_agent_centric_view=False,
                               timeout=100,
                               num_unique_letters=len(set(encode_letters(letters))))
        self.env = LetterEnv(self.params)
        self.propositions = self.env.get_propositions()
        self.sampler = JaxUntilTaskSampler(self.propositions)
        self.intrinsic = intrinsic
        self.ast_builder=JaxASTBuilder(LTL_BASE_VOCAB,MAX_NODES)
    @partial(jax.jit, static_argnames=("self"))
    def reset_env(self, key: jnp.ndarray) -> tuple[dict, LTLEnvState]:
        """Reset env, sample LTL goal, return dict obs and state."""
        key, subkey, sample_key= jax.random.split(key,3)
        obs, env_state = self.env.reset_env(subkey)

        final_array, num_nodes, root_idx = self.sample_ltl_goal(sample_key)
        ltl_state = LTLEnvState(
            env_state=env_state,
            ltl_goal=final_array,
            ltl_original=final_array,
            key=key,
            num_nodes=num_nodes,
            root_idx=root_idx
        )
        graph=self.ast_builder(final_array, num_nodes)
        ltl_obs = OrderedDict({'image': obs, **graph})
        return ltl_obs, ltl_state

    @partial(jax.jit, static_argnames=("self"))
    def step_env(self, key: jnp.ndarray, state: LTLEnvState, action: int) -> tuple[dict, float, bool, dict, LTLEnvState]:
        """Step env, progress LTL, return (obs, reward, done, info, new_state)."""
        key, subkey = jax.random.split(key)
        obs, new_env_state, reward, done, info = self.env.step_env(subkey, state.env_state, action)

        # Progress LTL
        truth_assignment = jax.lax.stop_gradient(self.get_events(new_env_state))

        ltl_goal, root_index, num_nodes= progress.progress_and_clean_jax(state.ltl_goal, truth_assignment,0,state.num_nodes)

        new_state = LTLEnvState(
            env_state=new_env_state,
            ltl_goal=ltl_goal,
            ltl_original=state.ltl_original,
            key=key,
            num_nodes=num_nodes,
            root_idx=root_index
        )
        graph=self.ast_builder(ltl_goal, num_nodes)

        # Compute LTL reward and done
        ltl_reward = jax.lax.cond(
            ltl_goal[0][0] == LTL_BASE_VOCAB['True'],
            lambda: 1.0,
            lambda: jax.lax.cond(
                ltl_goal[0][0] == LTL_BASE_VOCAB['False'],
                lambda: -1.0,
                lambda: self.intrinsic
            )
        )
        is_true = (ltl_goal[0][0] == LTL_BASE_VOCAB['True'])
        is_false = (ltl_goal[0][0] == LTL_BASE_VOCAB['False'])
        ltl_done = jnp.logical_or(is_true, is_false)
        ltl_obs = OrderedDict({'image' : obs, **graph})

        return ltl_obs,new_state, reward + ltl_reward,jnp.logical_or(done,ltl_done), info


   
    def sample_ltl_goal(self,key) -> any:
        """Sample LTL formula, adjust timeout for SequenceSampler."""
        final_array, num_nodes, root_idx = self.sampler.sample(key)
        return final_array, num_nodes, root_idx

    def get_events(self, env_state: EnvState) -> chex.Array:
        """Get current propositions from the underlying env using its state."""
        return self.env.get_events(env_state)

    
    def observation_space(self) -> spaces.Dict:
        """Observation space of the LTL-wrapped environment."""
        return spaces.Dict({
        'image': self.env.observation_space(),
        'nodes': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_NODES+1, VOCAB_SIZE+1), dtype=np.float32),
        'senders' : spaces.Box(low=0, high=MAX_NODES - 1, shape=(MAX_NODES*3,), dtype=np.int32),
        'receivers' :  spaces.Box(low=0, high=MAX_NODES - 1, shape=(MAX_NODES*3,), dtype=np.int32),
        'n_node' : spaces.Box(low=0, high=MAX_NODES, shape=(1,), dtype=np.int32),
    
        
    })

    
    def action_space(self) -> spaces.Discrete:
        """Action space of the LTL-wrapped environment."""
        return self.env.action_space()

    
    def state_space(self) -> spaces.Dict:
        """State space of the LTL-wrapped environment."""
        ltl_formula_space = spaces.Box(
            low=0,
            high=VOCAB_SIZE - 1,
            shape=(3, MAX_NODES),
            dtype=jnp.int32
        )
        return spaces.Dict({
            "env_state": self.env.state_space(),
            "ltl_goal": ltl_formula_space,
            "ltl_original": ltl_formula_space,
            "num_nodes": spaces.Discrete(MAX_NODES + 1),
            "root_idx": spaces.Discrete(MAX_NODES)
        })
    def max_episode_steps(self) -> int:
        """Max episode steps of the LTL-wrapped environment."""
        return self.params.timeout  
    def get_env_metrics(self, state: EnvState) -> dict:
        return dict(num_nodes_left=state.num_nodes)
   

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(env_id='LTLEnv', entry_point=module_path + ':LTLEnv')

