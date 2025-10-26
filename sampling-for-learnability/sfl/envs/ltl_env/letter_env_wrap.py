from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax import struct  # Import for struct.dataclass
import chex
from functools import partial
from collections import OrderedDict
from typing import Tuple

from jaxued.environments import UnderspecifiedEnv 
from sfl.envs.ltl_env.ast import JaxASTBuilder
from sfl.envs.ltl_env.sampler import JaxUntilTaskSampler, JaxEventuallySampler
from sfl.envs.ltl_env.utils import *
import sfl.envs.ltl_env.progress
from sfl.envs.ltl_env.letter_env import LetterEnv, LetterEnvState, encode_letters
from typing import Callable
from gymnax.environments import spaces

@struct.dataclass
class Observation:
    image: chex.Array
    nodes: chex.Array
    senders: chex.Array
    receivers: chex.Array   
    n_node: chex.Array
    edge_types: chex.Array

@struct.dataclass
class Level:
    """
    Defines a static instance of an LTL task, including the 
    map layout and the LTL formula, as required by UED.
    """
    letter_map: chex.Array           # The letter grid [grid_size, grid_size, num_unique_letters]
    agent_pos: chex.Array     # Initial agent position (e.g., jnp.array([0, 0]))
    ltl_formula: chex.Array   # The LTL AST array (from an external sampler)
    ltl_num_nodes: int
    ltl_root_idx: int

@struct.dataclass
class EnvParams:
    """UED-style episode parameters."""
    max_steps_in_episode: int = 100

class EnvState(PyTreeNode):
    env_state: LetterEnvState   # Underlying LetterEnv state
    ltl_goal: jnp.ndarray       # Current LTL formula
    ltl_original: jnp.ndarray   # Original LTL formula
    key: jnp.ndarray            # PRNG key
    num_nodes: jnp.ndarray
    root_idx: jnp.ndarray
    terminal: bool              # <-- ADDED: True if LTL goal is True/False

class LTLEnv(UnderspecifiedEnv): # <-- CHANGED: Inherit from UED base
    """
    Functional wrapper adding LTL goals to a Gymnax environment,
    adapted for Unsupervised Environment Design (UED).
    
    This environment expects to be reset with an `Level` object.
    """
    def __init__(self, grid_size=7, letters="aabbccddeeffgghhiijjkkll", use_fixed_map=False, use_agent_centric_view=False, timeout=100, num_unique_letters=len(set(encode_letters("abcdefghijkl"))), intrinsic: float = 0.0):
        super().__init__()
        
        # Store the configuration for the underlying LetterEnv
        self.config_params = dict(
            grid_size=grid_size, 
            letters=encode_letters(letters),
            use_fixed_map=use_fixed_map,
            use_agent_centric_view=use_agent_centric_view,
            timeout=timeout, # This is now only used by LetterEnv if needed, not for termination
            num_unique_letters=len(set(encode_letters(letters)))
        )
        
        self.env = LetterEnv(self.config_params)
        self.propositions = self.env.get_propositions()
        self.intrinsic = intrinsic
        self.ast_builder = JaxASTBuilder(LTL_BASE_VOCAB, MAX_NODES)
        
        
    @property
    def default_params(self) -> EnvParams:
        """Return default UED-style episode parameters."""
        return EnvParams()

    def init_state_from_level(
        self, 
        key: chex.PRNGKey, 
        level: Level,
    ) -> EnvState:
        """Helper to create the initial state from a level definition."""
        
        # 1. Create the base LetterEnv state from the level
        base_env_state = LetterEnvState(
            agent=level.agent_pos,
            map=level.letter_map,
            time=0,
            num_episodes=0,
            key=key
        )
        
        # 2. Create the full EnvState
        ltl_state = EnvState(
            env_state=base_env_state,
            ltl_goal=level.ltl_formula,
            ltl_original=level.ltl_formula,
            key=key,
            num_nodes=level.ltl_num_nodes,
            root_idx=level.ltl_root_idx,
            terminal=False # Always start non-terminal
        )
        return ltl_state

    # --- UED API Implementation ---

    def reset_env_to_level(
        self, 
        rng: chex.PRNGKey, 
        level: Level, 
        params: EnvParams
    ) -> Tuple[dict, EnvState]:
        """Resets the environment to a specific provided level."""
        state = self.init_state_from_level(rng, level)
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnames=("self"))
    def step_env(
        self, 
        key: jnp.ndarray, 
        state: EnvState, 
        action: int,
        params: EnvParams  
    ) -> tuple[dict, EnvState, float, bool, dict]:
        """Step env, progress LTL, return (obs, new_state, reward, done, info)."""
        
        key, subkey = jax.random.split(key)
        
        # 1. Step the underlying LetterEnv
        #    Note: LetterEnv.step_env MUST be modified to not return done=True on timeout
        obs, new_env_state, reward, base_done, info = self.env.step_env(
            subkey, state.env_state, action
        )

        # 2. Progress LTL
        truth_assignment = jax.lax.stop_gradient(self.get_events(new_env_state))
        ltl_goal, root_index, num_nodes = progress.progress_and_clean_jax(
            state.ltl_goal, truth_assignment, 0, state.num_nodes
        )

        # 3. Compute LTL reward and termination
        is_true = (ltl_goal[0][0] == LTL_BASE_VOCAB['True'])
        is_false = (ltl_goal[0][0] == LTL_BASE_VOCAB['False'])
        ltl_terminal = jnp.logical_or(is_true, is_false) # This is the new terminal flag

        ltl_reward = jax.lax.cond(
            is_true,
            lambda: 1.0,
            lambda: jax.lax.cond(
                is_false,
                lambda: -1.0,
                lambda: self.intrinsic
            )
        )
        
        # 4. Create new state
        new_state = EnvState(
            env_state=new_env_state,
            ltl_goal=ltl_goal,
            ltl_original=state.ltl_original,
            key=key,
            num_nodes=num_nodes,
            root_idx=root_index,
            terminal=ltl_terminal 
        )
        
        # 5. Get obs and check for termination using UED method
        ltl_obs = self.get_obs(new_state)
        done = self.is_terminal(new_state, params) 
        
        total_reward = reward + ltl_reward 

        return ltl_obs, new_state, total_reward, done, info

    def get_obs(self, state: EnvState) -> dict:
        """Gets the observation from the current state."""
        # Get base observation from LetterEnv
        base_obs = self.env._get_observation(state.env_state)
        
        # Get LTL graph observation
        graph = self.ast_builder(state.ltl_goal, state.num_nodes)
        
        # Combine them
        ltl_obs = Observation(image=base_obs, senders=graph['senders'], receivers=graph['receivers'], n_node=graph['n_node'], nodes=graph['nodes'], edge_types=graph['edge_types'])
        return ltl_obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check for timeout
        done_steps = state.env_state.time >= params.max_steps_in_episode
        
        # Check for LTL success/failure (which we set in state.terminal)
        return jnp.logical_or(done_steps, state.terminal)

    
    def get_events(self, env_state: LetterEnvState) -> chex.Array:
        """Get current propositions from the underlying env using its state."""
        return self.env.get_events(env_state)
    

    

    def action_space(self) -> spaces.Discrete:
        """Action space of the LTL-wrapped environment."""
        return self.env.action_space()
   
    def get_env_metrics(self, state: EnvState) -> dict:
        return dict(num_nodes_left=state.num_nodes)
   
def make_level_generator(
    grid_size=5, 
    letters="abcdefghijkl", 
    use_fixed_map=False, 
):
    """
    Creates a level generator function for the LTLEnv.

    This factory initializes the map sampler (LetterEnv) and the LTL sampler
    and returns a JIT-compiled function to sample levels.
    """

    # 1. Create the config for the base environment
    encoded_letters = encode_letters(letters)
    num_unique_letters = len(set(encoded_letters))

    config_params = dict(
        grid_size=grid_size,
        letters=encoded_letters,
        use_fixed_map=use_fixed_map,
        use_agent_centric_view=True, 
        timeout=100,                  # Not relevant for level generation
        num_unique_letters=num_unique_letters
    )

    # 2. Instantiate the map sampler (the base LetterEnv)
    # We use this to call its reset_env method, which samples a valid map
    map_sampler_env = LetterEnv(config_params)

    # 3. Instantiate the LTL sampler
    propositions = map_sampler_env.get_propositions()
    ltl_sampler = JaxUntilTaskSampler(propositions)

    @jax.jit
    def sample(key: chex.PRNGKey) -> Level:
        """
        JIT-compiled function to sample a single LTLLevel.
        """
        key_map, key_ltl = jax.random.split(key)

        # a. Sample a map
        # We call reset_env on our base env to get a valid map
        _obs, letter_env_state = map_sampler_env.reset_env(key_map)
        sampled_map = letter_env_state.map

        # b. Sample an LTL formula
        ltl_formula, num_nodes, root_idx = ltl_sampler.sample(key_ltl)

        agent_pos = jnp.array([0, 0], dtype=jnp.int32)

        level = Level(
            letter_map=sampled_map,
            agent_pos=agent_pos,
            ltl_formula=ltl_formula,
            ltl_num_nodes=num_nodes,
            ltl_root_idx=root_idx
        )
        return level

    # Return the pure sample function
    return sample


def make_level_mutator_minimax(max_num_edits: int) -> Callable[[chex.PRNGKey, Level, int], Level]:
    """
    Creates a mutator function that permutes propositions and letter maps.

    The returned mutator applies a cyclical shift to all propositions
    (e.g., 'a' -> 'b', ..., 'l' -> 'a') and the corresponding letter
    representations in the grid.

    The number of shifts (edits) is sampled randomly from [1, max_num_edits]
    using the provided PRNG key.
    """
    
    # Pre-calculate constants for the inner function closure
    _PROP_OFFSET = PROP_OFFSET
    _NUM_PROPS = NUM_PROPS
    
    @jax.jit
    def _mutator(key: chex.PRNGKey, level: Level, solve_steps: int) -> Level:
        """
        Applies a random number of +1 proposition/letter permutations.
        
        Args:
            key: JAX PRNG key to sample the number of edits.
            level: The level to mutate.
            solve_steps: Ignored. Included for a consistent UED mutator API.
        """
        
        # 1. Sample the number of shifts (edits) to apply.
        # We sample n from [1, max_num_edits].
        shift_amount = jax.random.randint(
            key, 
            shape=(), 
            minval=1, 
            maxval=max_num_edits + 1
        )

        # --- 2. Mutate LTL Formula ---
        
        # Get the token IDs (first column of the formula array)
        tokens = level.ltl_formula[:, 0]
        
        # Create a mask to identify which tokens are propositions
        is_prop = (tokens >= _PROP_OFFSET) & (tokens < _PROP_OFFSET + _NUM_PROPS)
        
        # Convert proposition token IDs to zero-based indices (0 for 'a', 1 for 'b', ...)
        # This is the 'neighbour' value from the prompt
        prop_indices = tokens - _PROP_OFFSET
        
        # Apply the circular shift: (neighbour + n) % num_props
        new_prop_indices = (prop_indices + shift_amount) % _NUM_PROPS
        
        # Convert back to token IDs
        new_tokens = new_prop_indices + _PROP_OFFSET
        
        # Only apply the mutation to tokens that were propositions
        mutated_tokens = jnp.where(is_prop, new_tokens, tokens)
        
        # Update the formula array. .at[...].set() is the JAX way to update.
        mutated_formula = level.ltl_formula.at[:, 0].set(mutated_tokens)

        # --- 3. Mutate Letter Map ---
        
        # The letter_map has shape [grid, grid, num_unique_letters].
        # num_unique_letters should be equal to _NUM_PROPS.
        # We apply the same circular shift to the last axis.
        mutated_letter_map = jnp.roll(
            level.letter_map, 
            shift=shift_amount, 
            axis=-1 # The last axis is the one-hot letter dimension
        )
        
        # --- 4. Return new Level ---
        # We use .replace() as Level is a flax struct (PyTreeNode)
        return level.replace(
            letter_map=mutated_letter_map,
            ltl_formula=mutated_formula
        )

    return _mutator