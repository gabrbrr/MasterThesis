import jax
import jax.numpy as jnp
from gymnax.environments import spaces
import chex
from dataclasses import replace
from functools import partial
from flax.struct import PyTreeNode
import matplotlib.pyplot as plt
import numpy as np
import random 
# --- Mocking dependencies from sfl.envs.ltl_env.utils ---
# Based on the default letters="aabbccddeeffgghhiijjkkll"
# This assumes letters 'a' through 'l'.
# _LETTERS_STR = "abcdefghijklmnopqrstuvwyz"
# LTL_BASE_VOCAB = {c: i for i, c in enumerate(_LETTERS_STR)}
# VOCAB_INV = {i: c for i, c in enumerate(_LETTERS_STR)}

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
                 max_steps_in_episode=75
                ):
        
        self.grid_size = grid_size
        self.use_fixed_map = use_fixed_map
        self.use_agent_centric_view = use_agent_centric_view
        self.max_steps_in_episode = max_steps_in_episode # For state_space
        self.actions = jnp.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
        self.letters=letters
        self.encoded_letters = tuple(ord(l) - ord('a') for l in letters)
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
    def get_events(self, state: LetterEnvState) -> chex.Array:
        """Return a boolean array of letter activations at the agent's current cell."""
        agent_pos = state.agent
        cell = state.map[agent_pos[0], agent_pos[1]]  # shape: (num_unique_letters,)
    
        # Ensure it's a boolean array
        events = cell.astype(bool)
        return events

    def get_propositions(self) -> list:
        l=sorted(list(set(self.letters)))
        return l
    
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

@partial(jax.jit, static_argnames=("grid_size",))
def _is_valid_map(map_array: chex.Array, grid_size: int) -> bool:
    """
    Checks map validity using a JAX-compatible flood fill (BFS).

    This function ports the logic of the original non-JAX _is_valid_map:
    1. Start a search from (0,0), which is guaranteed to be empty.
    2. The search can *only* expand from *empty* cells.
    3. The search *stops* at letter cells (it visits them but doesn't expand
       from them).
    4. A map is "valid" if this search manages to visit *every single cell*
       (i.e., len(closed_list) == grid_size * grid_size).
    """
    
    # 1. 'is_empty' mask: True for empty cells, False for letter cells.
    # A cell is empty if all its letter channels are 0.
    is_empty = ~jnp.any(map_array, axis=-1) 

    # 2. 'visited' (closed_list) & 'frontier' (open_list)
    # We use a BFS-style iterative expansion.
    visited = jnp.zeros((grid_size, grid_size), dtype=bool)
    # Start the frontier/open_list at (0,0)
    frontier = jnp.zeros((grid_size, grid_size), dtype=bool).at[0, 0].set(True)

    # 3. Loop state for jax.lax.while_loop
    # (visited_mask, frontier_mask, is_empty_mask)
    init_state = (visited, frontier, is_empty)

    def cond_fn(state):
        """Continue looping as long as the frontier is not empty."""
        _, frontier, _ = state
        return jnp.any(frontier)

    def body_fn(state):
        """Execute one step of the BFS."""
        visited, current_frontier, is_empty = state
        
        # 1. Add all cells from the current frontier to the visited set.
        new_visited = visited | current_frontier

        # 2. Find cells in the frontier that we can *expand* from.
        # Per original logic, we only expand from *empty* cells.
        expandable_cells = current_frontier & is_empty

        # 3. Find all neighbors of the expandable cells (with wrapping).
        # This corresponds to the 4 actions.
        neighbors_up = jnp.roll(expandable_cells, shift=-1, axis=0)
        neighbors_down = jnp.roll(expandable_cells, shift=1, axis=0)
        neighbors_left = jnp.roll(expandable_cells, shift=-1, axis=1)
        neighbors_right = jnp.roll(expandable_cells, shift=1, axis=1)
        
        all_neighbors = neighbors_up | neighbors_down | neighbors_left | neighbors_right

        # 4. The new frontier is all neighbors that have not been
        #    visited before (which is the 'new_visited' mask).
        new_frontier = all_neighbors & ~new_visited

        return (new_visited, new_frontier, is_empty)

    # 5. Run the loop
    final_visited, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # 6. Final check: The map is valid if the total number of visited
    #    cells equals the total grid size, as in the original logic.
    total_visited = jnp.sum(final_visited)
    
    return total_visited == (grid_size * grid_size)


def visualize_map_terminal(env: LetterEnv, state: LetterEnvState):
    """
    Prints a text-based visualization of the map to the terminal.
    """
    # Convert JAX arrays to NumPy
    grid_map = np.array(state.map)
    agent_pos = np.array(state.agent)
    unique_ids = np.array(env.unique_letter_ids)
    
    # Create a 2D array for terminal output, filled with '.'
    terminal_grid = np.full((env.grid_size, env.grid_size), fill_value=".", dtype=object)
    
    # Place letters on the grid
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            cell = grid_map[r, c]
            if np.any(cell):
                local_idx = np.argmax(cell)
                global_id = unique_ids[local_idx]
                char = VOCAB_INV[int(global_id)]
                terminal_grid[r, c] = char
                
    # Place the agent on the grid
    # The agent symbol '@' will overwrite a letter if they are on the same cell
    agent_r, agent_c = agent_pos
    terminal_grid[agent_r, agent_c] = "@"
    
    # Print the terminal visualization
    print("\n--- Terminal Map Visualization ---")
    print(f"Grid Size: {env.grid_size}x{env.grid_size} | Agent '@' at ({agent_r}, {agent_c})")
    border_line = "-" * (env.grid_size * 2 + 1)
    print(border_line)
    for r in range(env.grid_size):
        # Add spaces for better readability
        print("|" + " ".join(terminal_grid[r]) + "|")
    print(border_line)
    print("Key: '.' = Empty, '@' = Agent, 'a'-'l' = Letters")


if __name__ == "__main_ss_":
    print("Initializing environment...")
    # You can change grid_size or letters here
    env = LetterEnv(
        grid_size=6, 
        letters="abcdefghijklmnopqrstuvwyz",
        use_fixed_map=False # Set to True to get the same map every time (with same key)
    )
    
    # Use a fixed key for reproducibility
    key = jax.random.PRNGKey(random.randint(1,100) )
    
    print("Resetting environment to generate map...")
    key, reset_key = jax.random.split(key)
    obs, state = env.reset_env(reset_key)
    
    print(f"Map generated. Agent at: {state.agent}")
    
    # Visualize the generated map
    visualize_map_terminal(env, state)
