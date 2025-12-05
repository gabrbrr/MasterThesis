from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax import struct  # Import for struct.dataclass
import chex
from functools import partial
from collections import OrderedDict
from typing import Callable, Any, NamedTuple, List, Tuple

from jaxued.environments import UnderspecifiedEnv 
from sfl.envs.ltl_env.eventually_sampler import  JaxEventuallyTaskSampler
from sfl.envs.ltl_env.until_sampler import JaxUntilTaskSampler, ConjunctionState
from sfl.envs.ltl_env.letter_env import LetterEnv, LetterEnvState
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
    ltl_formula: Any 
    num_conjuncts: int
    avg_levels: int

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 75

class EnvState(PyTreeNode):
    env_state: LetterEnvState   # Underlying LetterEnv state
    ltl_goal: Any    # Current LTL formula
    ltl_original: Any  # Original LTL formula
    key: jnp.ndarray            # PRNG key
    terminal: bool              

class LTLEnv(UnderspecifiedEnv): 
    def __init__(self, 
                env,
                sampler):
        super().__init__()
        
        self.env = env 
        self.grid_size = env.grid_size
        self.use_fixed_map = env.use_fixed_map
        self.use_agent_centric_view = env.use_agent_centric_view
        self.intrinsic = 0.0 # what to return if neither satisfied or falsified
        self.sampler=sampler
        self.letters=env.letters
        
        
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def init_state_from_level(
        self, 
        key: chex.PRNGKey, 
        level: Level,
    ) -> EnvState:
        """Helper to create the initial state from a level definition."""
        
        base_env_state = LetterEnvState(
            agent=level.agent_pos,
            map=level.letter_map,
            num_episodes=0,
            time=0,
            key=key
        )
        
        ltl_state = EnvState(
            env_state=base_env_state,
            ltl_goal=level.ltl_formula,
            ltl_original=level.ltl_formula,
            key=key,
            terminal=False # Always start non-terminal
        )
        return ltl_state


    def reset_to_level(
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
        
  
        obs, new_env_state, reward, base_done, info = self.env.step_env(
            subkey, state.env_state, action
        )

        truth_assignment = jax.lax.stop_gradient(self.get_events(new_env_state))
        ltl_goal, is_true, is_false= self.sampler.progress(state.ltl_goal, truth_assignment)

        ltl_terminal = jnp.logical_or(is_true, is_false) 

        ltl_reward = jax.lax.cond(
            is_true,
            lambda: 1.0,
            lambda: jax.lax.cond(
                is_false,
                lambda: -1.0,
                lambda: self.intrinsic
            )
        )
        
        new_state = EnvState(
            env_state=new_env_state,
            ltl_goal=ltl_goal,
            ltl_original=state.ltl_original,
            key=key,
            terminal=ltl_terminal 
        )
        
   
        ltl_obs = self.get_obs(new_state)
        done = self.is_terminal(new_state, params) 
        
        total_reward = reward + ltl_reward 

        return ltl_obs, new_state, total_reward, done, info

    def get_obs(self, state: EnvState) -> dict:
        """Gets the observation from the current state."""
        # Get base observation from LetterEnv
        base_obs = self.env._get_observation(state.env_state)
        
        # Get LTL graph observation
        graph = self.sampler.build_ast(state.ltl_goal)
        
        # Combine them
        ltl_obs = Observation(image=base_obs, senders=graph['senders'], receivers=graph['receivers'], n_node=graph['n_node'], nodes=graph['nodes'], edge_types=graph['edge_types'])
        return ltl_obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.env_state.time >= params.max_steps_in_episode
        
        return jnp.logical_or(done_steps, state.terminal)

    
    def get_events(self, env_state: LetterEnvState) -> chex.Array:
        """Get current propositions from the underlying env using its state."""
        return self.env.get_events(env_state)
    

    

    def action_space(self) -> spaces.Discrete:
        """Action space of the LTL-wrapped environment."""
        return self.env.action_space()
   
    def get_env_metrics(self, state: EnvState) -> dict:
        assert False, "Not implemented"
   
def make_level_generator(
    env,
    sampler
    ):
    """
    Creates a level generator function for the LTLEnv.
    """

    # 1. Instantiate the map sampler (the base LetterEnv)
    #    using the provided static config.
    map_sampler_env = env
    ltl_sampler=sampler

    
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
        ltl_formula, num_conjunctions, avg_levels = ltl_sampler.sample(key_ltl)

        agent_pos = jnp.array([0, 0], dtype=jnp.int32)

        level = Level(
            letter_map=sampled_map,
            agent_pos=agent_pos,
            ltl_formula=ltl_formula,
            num_conjuncts=num_conjunctions,\
            avg_levels=avg_levels
            
        )
        return level

    return sample


# def make_level_mutator_minimax(max_num_edits: int) -> Callable[[chex.PRNGKey, Level, int], Level]:
#     """
#     Creates a mutator function that permutes propositions and letter maps.

#     The returned mutator applies a cyclical shift to all propositions
#     (e.g., 'a' -> 'b', ..., 'l' -> 'a') and the corresponding letter
#     representations in the grid.

#     The number of shifts (edits) is sampled randomly from [1, max_num_edits]
#     using the provided PRNG key.
#     """
    
#     # Pre-calculate constants for the inner function closure
#     _PROP_OFFSET = self.sampler.PROP_OFFSET
#     _NUM_PROPS = len(self.sampler.propositions)
    
#     @jax.jit
#     def _mutator(key: chex.PRNGKey, level: Level, solve_steps: int) -> Level:
#         """
#         Applies a random number of +1 proposition/letter permutations.
        
#         Args:
#             key: JAX PRNG key to sample the number of edits.
#             level: The level to mutate.
#             solve_steps: Ignored. Included for a consistent UED mutator API.
#         """
        
#         # 1. Sample the number of shifts (edits) to apply.
#         # We sample n from [1, max_num_edits].
#         shift_amount = jax.random.randint(
#             key, 
#             shape=(), 
#             minval=1, 
#             maxval=max_num_edits + 1
#         )

#         # --- 2. Mutate LTL Formula ---
        
#         # Get the token IDs (first column of the formula array)
#         tokens = level.ltl_formula[:, 0]
        
#         # Create a mask to identify which tokens are propositions
#         is_prop = (tokens >= _PROP_OFFSET) & (tokens < _PROP_OFFSET + _NUM_PROPS)
        
#         # Convert proposition token IDs to zero-based indices (0 for 'a', 1 for 'b', ...)
#         # This is the 'neighbour' value from the prompt
#         prop_indices = tokens - _PROP_OFFSET
        
#         # Apply the circular shift: (neighbour + n) % num_props
#         new_prop_indices = (prop_indices + shift_amount) % _NUM_PROPS
        
#         # Convert back to token IDs
#         new_tokens = new_prop_indices + _PROP_OFFSET
        
#         # Only apply the mutation to tokens that were propositions
#         mutated_tokens = jnp.where(is_prop, new_tokens, tokens)
        
#         # Update the formula array. .at[...].set() is the JAX way to update.
#         mutated_formula = level.ltl_formula.at[:, 0].set(mutated_tokens)

#         # --- 3. Mutate Letter Map ---
        
#         # The letter_map has shape [grid, grid, num_unique_letters].
#         # num_unique_letters should be equal to _NUM_PROPS.
#         # We apply the same circular shift to the last axis.
#         mutated_letter_map = jnp.roll(
#             level.letter_map, 
#             shift=shift_amount, 
#             axis=-1 # The last axis is the one-hot letter dimension
#         )
        
#         # --- 4. Return new Level ---
#         return level.replace(
#             letter_map=mutated_letter_map,
#             ltl_formula=mutated_formula
#         )

#     return _mutator

def make_level_mutator_minimax(
    sampler: JaxUntilTaskSampler, 
    max_num_edits: int
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    """
    Creates a JIT-compiled level mutator function for minimax-style UED.

    The mutator makes a small number of random edits to the LTL formula's
    ConjunctionState by swapping active propositions.

    Args:
        sampler: An instance of `JaxUntilTaskSampler`, used to access
                 the list of available proposition indices.
        max_num_edits: The maximum number of single-proposition
                       swaps to perform on the level.

    Returns:
        A JIT-compiled function with the signature:
        (key: PRNGKey, level: Level, agent_return: int) -> Level
        The `agent_return` is ignored in this implementation.
    """
    
    # Get shape constants from sampler
    N = sampler.max_conjunctions
    M = sampler.max_levels
    
    @jax.jit
    def _mutate_conjunction_state(
        key: chex.PRNGKey, 
        conj_state: ConjunctionState,
        n_edits: int
    ) -> Tuple[chex.PRNGKey, ConjunctionState]:
        """
        Applies `n_edits` random mutations to a ConjunctionState.
        Uses a fori_loop for JIT-compatibility.
        """
        
        def _edit_loop_body(i: int, loop_state: Tuple[chex.PRNGKey, ConjunctionState]):
            """Body of the fori_loop, performs a single edit."""
            key, state = loop_state
            # Split key for all random choices in this iteration
            key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)

            # 1. Find all *active* (n, m) indices.
            # These are indices within the actual depth and active conjuncts.
            depth_mask = jnp.arange(M) < state.depths[:, None]  # (N, M)
            conjunction_mask = ~state.already_true                # (N,)
            active_mask = depth_mask & conjunction_mask[:, None]  # (N, M)
            
            # 2. Get all valid indices and the total count
            num_active = jnp.sum(active_mask)
            # Get valid indices, padded with 0s if not enough
            valid_n, valid_m = jnp.where(active_mask, size=N * M, fill_value=0)

            def _apply_mutation(key_tuple):
                """Closure to apply one mutation, used in lax.cond."""
                key, subkey1, subkey2, subkey3, subkey4 = key_tuple
                
                # 3. Sample *which* active index to mutate
                # We use jnp.maximum(1, num_active) to avoid randint error if num_active is 0
                rand_idx = jax.random.randint(
                    subkey1, 
                    shape=(), 
                    minval=0, 
                    maxval=jnp.maximum(1, num_active)
                )
                n_idx = valid_n[rand_idx]
                m_idx = valid_m[rand_idx]

                # 4. Sample *what* to mutate (avoid or progress)
                mutate_avoid = jax.random.bernoulli(subkey2)

                # 5. Sample the *new proposition*
                new_prop_idx = jax.random.choice(subkey3, sampler.prop_indices)
                
                # 6. Apply the mutation
                to_avoid = state.to_avoid
                to_progress = state.to_progress

                # Create the mutated arrays
                new_to_avoid = to_avoid.at[n_idx, m_idx].set(new_prop_idx)
                new_to_progress = to_progress.at[n_idx, m_idx].set(new_prop_idx)
                
                # Selectively apply the mutation based on `mutate_avoid`
                mutated_to_avoid = jax.lax.cond(
                    mutate_avoid,
                    lambda: new_to_avoid,
                    lambda: to_avoid
                )
                mutated_to_progress = jax.lax.cond(
                    mutate_avoid,
                    lambda: to_progress,
                    lambda: new_to_progress
                )
                
                return state._replace(
                    to_avoid=mutated_to_avoid, 
                    to_progress=mutated_to_progress
                )

            # 7. Only apply mutation if there's at least one active prop
            #    Otherwise, return the state unchanged for this iteration.
            new_state = jax.lax.cond(
                num_active > 0,
                # Pass keys as a tuple to the true_fun
                lambda: _apply_mutation((key, subkey1, subkey2, subkey3, subkey4)),
                # false_fun: just return the state
                lambda: state 
            )
            
            return (key, new_state)

        # --- End of _edit_loop_body ---

        # Run the mutation loop `n_edits` times
        key, final_state = jax.lax.fori_loop(
            0, n_edits, _edit_loop_body, (key, conj_state)
        )
        
        return key, final_state

    @jax.jit
    def mutate_level(
        key: chex.PRNGKey, 
        level: Level, 
        agent_return: int  # Standard UED/PAIRED arg, ignored here
    ) -> Level:
        """
        The returned mutator function.
        
        It samples a number of edits and applies them to the level's
        LTL formula.
        """
        # Split key to sample n_edits
        key, subkey = jax.random.split(key)
        
        # Sample the number of edits to perform, from 1 to max_num_edits
        n_edits = jax.random.randint(
            subkey, 
            shape=(), 
            minval=1, 
            maxval=max_num_edits + 1
        )
        
        # Get the LTL formula (ConjunctionState)
        current_conj_state = level.ltl_formula
        
        # Mutate it
        key, new_conj_state = _mutate_conjunction_state(
            key, 
            current_conj_state, 
            n_edits
        )
        
        # Return the *new* level with the mutated formula
        new_level = level.replace(ltl_formula=new_conj_state)
        
        return new_level

    # Return the JIT-compiled mutator function
    return mutate_level