import jax
import jax.numpy as jnp
from gymnax.environments import spaces
import chex
from dataclasses import replace
from functools import partial
from sfl.envs.ltl_env.utils import encode_letters, VOCAB_INV
from flax.struct import PyTreeNode


class LetterEnvState(PyTreeNode):
    agent: chex.Array
    map: chex.Array
    time: int
    num_episodes: int
    key: chex.Array

class LetterEnv():
    """
    JAX port of LetterEnv: Grid with random letters, agent movement, edge wrapping.
    Ensures a clean path to all *free* cells. Observations are
    (grid_size, grid_size, num_unique_letters + 1).
    """
    def __init__(self,
                 grid_size=7,
                 letters="aabbccddeeffgghhiijjkkll",
                 use_fixed_map=False,
                 use_agent_centric_view=True,
                 max_steps_in_episode=100
                ):
        
        self.grid_size = grid_size
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.max_steps_in_episode = max_steps_in_episode # For state_space
        
        self.actions = jnp.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        
        self.encoded_letters = encode_letters(letters) # Python list
        self.num_unique_letters = len(set(self.encoded_letters))
        
        self.letters_arr = jnp.array(self.encoded_letters, dtype=jnp.int32)
        self.unique_letter_ids = jnp.unique(self.letters_arr, size=self.num_unique_letters)
        self.local_letter_ids = jnp.searchsorted(self.unique_letter_ids, self.letters_arr)
        self.locations = jnp.array([(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if (i, j) != (0, 0)])
        self.len_locations=len(self.locations)
        self.len_letters=len(self.letters_arr)
    @partial(jax.jit, static_argnames=("self"))
    def reset_env(self, key: chex.Array) -> tuple[chex.Array, LetterEnvState]:
        """Reset environment, sample valid map, place agent at (0,0)."""
        def sample_map(rng):
            rng, subkey = jax.random.split(rng)
            perm = jax.random.permutation(subkey, self.len_locations)
            indices = perm[:self.len_letters]
            chosen = self.locations[indices]
            map_array = jnp.zeros((self.grid_size, self.grid_size, self.num_unique_letters), dtype=jnp.uint8)
            map_array = map_array.at[chosen[:, 0], chosen[:, 1], self.local_letter_ids].set(1)
            return map_array, rng

        def try_sample_map(rng):
            map_array, new_rng = sample_map(rng)
            is_valid = _is_valid_map(map_array, self.grid_size)
            return map_array, is_valid, new_rng

        empty_map = jnp.zeros((self.grid_size, self.grid_size, self.num_unique_letters), dtype=jnp.uint8)

        def cond_fn(state):
            _, is_valid, _ = state
            return ~is_valid

        def body_fn(state):
            _, _, rng = state
            rng, sub = jax.random.split(rng)
            return try_sample_map(sub)

        init_key = key if self.use_fixed_map else jax.random.split(key)[0]
        map_array, _, rng = jax.lax.while_loop(cond_fn, body_fn, (empty_map, False, init_key))

        state = LetterEnvState(
            agent=jnp.array([0, 0]), map=map_array, time=0, num_episodes=0, key=rng
        )
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnames=("self"))
    def step_env(self, key: chex.Array, state: LetterEnvState, action: int) -> tuple[chex.Array, LetterEnvState, float, bool, dict]:
        di, dj = self.actions[action]
        agent_i = (state.agent[0] + di + self.grid_size) % self.grid_size
        agent_j = (state.agent[1] + dj + self.grid_size) % self.grid_size
        new_agent = jnp.array([agent_i, agent_j])
        new_time = state.time + 1
        new_state = replace(state, agent=new_agent, time=new_time, key=key)

        obs = self._get_observation(new_state)
        reward = 0.0
        done = new_time > self.max_steps_in_episode
        info = {}
        return obs, new_state, reward, done, info, 

    def _get_observation(self, state: LetterEnvState) -> chex.Array:
        obs = jnp.zeros((self.grid_size, self.grid_size, self.num_unique_letters + 1), dtype=jnp.uint8)
        c_map, agent = state.map, state.agent

        if self.use_agent_centric_view:
            c_map, agent = self._get_centric_map(state)

        obs = obs.at[:, :, :-1].set(c_map)
        obs = obs.at[agent[0], agent[1], -1].set(1)
        return obs

    def _get_centric_map(self, state: LetterEnvState) -> tuple[chex.Array, chex.Array]:
        center = self.grid_size // 2
        agent = jnp.array([center, center])
        delta = center - state.agent
        grid_coords = jnp.arange(self.grid_size)
        ii, jj = jnp.meshgrid(grid_coords, grid_coords, indexing='ij')
        src_ii = (ii - delta[0]) % self.grid_size
        src_jj = (jj - delta[1]) % self.grid_size
        c_map = state.map[src_ii, src_jj]
        return c_map, agent

    @partial(jax.jit, static_argnames=("self"))
    def get_events(self, state: LetterEnvState) -> str:
        agent_pos = state.agent
        cell = state.map[agent_pos[0], agent_pos[1]]
        is_letter_pred = jnp.any(cell)


        def true_fn(_):
            local_idx = jnp.argmax(cell)
            return self.unique_letter_ids[local_idx]

        def false_fn(_):
            return -1

        return jax.lax.cond(is_letter_pred, true_fn, false_fn, operand=None)

    def get_propositions(self) -> list:
        unique_ids = sorted(list(set(self.encoded_letters)))
        return [VOCAB_INV[int(id_)] for id_ in unique_ids]
    
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.actions), dtype=jnp.uint32)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, self.num_unique_letters + 1),
            dtype=jnp.uint8
        )

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "agent": spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=jnp.int32),
            "map": spaces.Box(
                low=0,
                high=1,
                shape=(self.grid_size, self.grid_size, self.num_unique_letters),
                dtype=jnp.uint8
            ),
            "time": spaces.Discrete(self.max_steps_in_episode + 1),
            "num_episodes": spaces.Box(low=0, high=jnp.iinfo(jnp.int32).max, shape=(), dtype=jnp.int32)
        })


def _is_valid_map(map_array: chex.Array, grid_size: int) -> bool:
    free = ~jnp.any(map_array, axis=2)
    total_free = jnp.sum(free)
    reachable = jnp.zeros_like(free)
    reachable = reachable.at[0, 0].set(True)

    def body(_, reach):
        up, down = jnp.roll(reach, 1, axis=0), jnp.roll(reach, -1, axis=0)
        left, right = jnp.roll(reach, 1, axis=1), jnp.roll(reach, -1, axis=1)
        new_reach = reach | ((up | down | left | right) & free)
        return new_reach

    max_iters = grid_size * grid_size
    reachable = jax.lax.fori_loop(0, max_iters, body, reachable)
    return (jnp.sum(reachable) == total_free) & free[0, 0]

