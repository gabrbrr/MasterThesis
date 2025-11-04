import jax
import jax.numpy as jnp
import numpy as np  # Use numpy for atlas creation
import chex
from functools import partial

# --- Assumed imports from your environment code ---
# These classes are defined in your prompt.
# We just need their type definitions for the renderer.
from sfl.envs.ltl_env.letter_env_wrap import LTLEnv, EnvState, Level, EnvParams

# --- Color Palette for Letters ---
# A list of distinct RGB colors for the letter tiles.
COLOR_PALETTE = [
    (31, 119, 180),   # 1. Blue
    (44, 160, 44),    # 2. Green
    (255, 127, 14),   # 3. Orange
    (148, 103, 189),  # 4. Purple
    (140, 86, 75),    # 5. Brown
    (227, 119, 194),  # 6. Pink
    (188, 189, 34),   # 7. Olive
    (23, 190, 207),   # 8. Cyan
    (214, 39, 40),    # 9. Red (Distinct from agent)
    (127, 127, 127),  # 10. Gray
    (174, 199, 232),  # 11. Light Blue
    (152, 223, 138),  # 12. Light Green
    (255, 187, 120),  # 13. Light Orange
    (197, 176, 213),  # 14. Light Purple
]

# --- Atlas Generation Functions ---
# These are helper functions (using numpy) to draw the tiles.

def _fill_coords(img, fn, color):
    """Fills pixels in img with color where fn(x, y) is True."""
    new_img = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]  # Normalized Y
            xf = (x + 0.5) / img.shape[1]  # Normalized X
            if fn(xf, yf):
                new_img[y, x] = color
    return new_img

def _point_in_rect(xmin, xmax, ymin, ymax):
    """Returns a function that checks if a point is in a rectangle."""
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def _point_in_circle(cx, cy, r_sq):
    """Returns a function that checks if a point is in a circle (using r^2)."""
    def fn(x, y):
        return (x - cx)**2 + (y - cy)**2 < r_sq
    return fn

def _make_ltl_tile_atlas(tile_size, num_unique_letters):
    """Creates the tile atlas for the LTLEnv."""
    
    # Total tiles = 1 (empty) + num_unique_letters + 1 (agent)
    num_tiles = num_unique_letters + 2
    atlas = np.empty((num_tiles, tile_size, tile_size, 3), dtype=np.uint8)
    
    def add_border(tile, color=(50, 50, 50), width=0.031):
        """Helper to add a dark border to a tile."""
        tile = _fill_coords(tile, _point_in_rect(0, width, 0, 1), color)
        tile = _fill_coords(tile, _point_in_rect(1 - width, 1, 0, 1), color)
        tile = _fill_coords(tile, _point_in_rect(0, 1, 0, width), color)
        tile = _fill_coords(tile, _point_in_rect(0, 1, 1 - width, 1), color)
        return tile

    # --- Tile 0: Empty ---
    # Black background, dark grey border
    empty_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    atlas[0] = add_border(empty_tile)
    
    # --- Tiles 1 to num_unique_letters: Letters ---
    if num_unique_letters > len(COLOR_PALETTE):
        print(
            f"Warning: Not enough unique colors for {num_unique_letters} letters."
            " Colors will repeat."
        )
        
    for i in range(num_unique_letters):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        letter_tile = np.tile(color, (tile_size, tile_size, 1))
        atlas[i + 1] = add_border(letter_tile)
        
    # --- Last Tile (index num_unique_letters + 1): Agent ---
    agent_idx = num_unique_letters + 1
    # Black background, red circle
    agent_tile = np.tile([0, 0, 0], (tile_size, tile_size, 1))
    agent_tile = _fill_coords(agent_tile, _point_in_circle(0.5, 0.5, 0.4**2), [255, 0, 0]) # Red circle
    atlas[agent_idx] = add_border(agent_tile)
    
    return atlas

# --- The Renderer Class ---

class LTLEnvRenderer:
    """This class renders the LTL gridworld for visual logging, compatible with jit.

    Args:
        env (LTLEnv): The LTL Environment instance.
        tile_size (int, optional): The number of pixels each tile should take up.
    """
    def __init__(self, env: LTLEnv, tile_size: int = 32):
        self.env = env  # Used to get config params
        self.tile_size = tile_size
        
        # Extract key parameters from the env config
        self.grid_size = self.env.grid_size
        self.num_unique_letters = len(set(self.env.letters_str))
        
        # Create the atlas and store it as a jax array
        self._atlas = jnp.array(
            _make_ltl_tile_atlas(self.tile_size, self.num_unique_letters)
        )
        
        # Pre-calculate the agent's index in the atlas for use in JIT functions
        self.agent_idx = self.num_unique_letters + 1
        
    @partial(jax.jit, static_argnums=(0,))
    def render_level(self, level: Level, env_params: EnvParams) -> chex.Array:
        """Renders a static Level representation."""
        # A Level contains the static map and agent start position
        return (self._render_grid(level.letter_map, level.agent_pos),level.ltl_goal,level.ltl_root_idx,level.ltl_num_nodes)

    @partial(jax.jit, static_argnums=(0,))
    def render_state(self, env_state: EnvState, env_params: EnvParams) -> chex.Array:
        """Renders a full dynamic EnvState."""
        # The EnvState contains the inner LetterEnv state, which has the map and agent pos
        return (self._render_grid(
            env_state.env_state.map, 
            env_state.env_state.agent
        ),env_state.ltl_goal,env_state.root_idx,env_state.num_nodes)

    @partial(jax.jit, static_argnums=(0,))
    def _render_grid(self, letter_map: chex.Array, agent_pos: chex.Array) -> chex.Array:
        """JIT-compiled core rendering logic."""
        
        tile_size = self.tile_size
        nrows = self.grid_size
        ncols = self.grid_size
        height_px = nrows * tile_size
        width_px = ncols * tile_size
        
        # 1. Create integer grid for letters
        # letter_map is (H, W, num_unique_letters) and one-hot
        
        # Find where letters are and get their *local* index
        is_letter = jnp.any(letter_map, axis=2)
        letter_indices = jnp.argmax(letter_map, axis=2)
        
        # Map to atlas indices: Empty=0, Letter 0=1, Letter 1=2, ...
        cells = jnp.where(is_letter, letter_indices + 1, 0)
        
        # 2. Place agent on top
        # The agent's index is (num_unique_letters + 1)
        # Use JAX's immutable update
        cells = cells.at[agent_pos[0], agent_pos[1]].set(self.agent_idx)
        
        # 3. Atlas lookup and reshape
        # This is the magic step:
        # (nrows, ncols) -> (nrows, ncols, tile_size, tile_size, 3)
        img = self._atlas[cells]
        
        # Reshape to final image:
        # (nrows, ncols, tile_size, tile_size, 3) -> (nrows, tile_size, ncols, tile_size, 3) -> (height_px, width_px, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)
        
        return img