import numpy as np
import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

# Pillow is used for text rendering in the tile atlas, which is run once in numpy.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Pillow is required for rendering. Please install with: pip install Pillow")

from .ltl_env import LTLEnv, LTLEnvState, LTLLevel, LTLEnvParams
from .utils import VOCAB_INV # To convert AST nodes back to strings

class LTLEnvRenderer:
    """
    Renders the LTLEnv grid for visual logging, compatible with JAX JIT.

    This class creates a visual representation of the environment's state,
    including the grid, letters, and the agent's position. The LTL formula
    can be converted to a string separately using the `ltl_to_string` method.

    Args:
        env (LTLEnv): An instance of the LTLEnv to render.
        tile_size (int): The number of pixels for each grid cell.
    """
    def __init__(self, env: LTLEnv, tile_size: int = 32):
        self.env = env
        self.tile_size = tile_size
        
        # The atlas is pre-built using NumPy/Pillow and then converted to a JAX array.
        self._atlas = jnp.array(_make_ltl_tile_atlas(
            tile_size=tile_size,
            letters=env.propositions # e.g., ['a', 'b', 'c']
        ))

    @partial(jax.jit, static_argnums=(0,))
    def render_level(self, level: LTLLevel, env_params: LTLEnvParams) -> chex.Array:
        """
        Renders a static level by creating and rendering its initial state.
        
        Since a level can map to many trajectories, we render the canonical
        starting state defined by the level.
        """
        # Create a dummy key, as the state is fully determined by the level
        dummy_key = jax.random.PRNGKey(0)
        initial_state = self.env.init_state_from_level(dummy_key, level)
        return self.render_state(initial_state, env_params)

    @partial(jax.jit, static_argnums=(0,))
    def render_state(self, env_state: LTLEnvState, env_params: LTLEnvParams) -> chex.Array:
        """Renders a specific dynamic state of the environment."""
        tile_size = self.tile_size
        grid_size = self.env.config_params.grid_size
        width_px = height_px = grid_size * tile_size

        # --- Create a 2D integer grid representing the scene ---
        
        # Start with an empty grid (ID 0)
        cells = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
        
        # Place letters (IDs 2, 3, ...)
        # Find where letters exist and what their indices are
        letter_map = env_state.env_state.map
        has_letter = jnp.any(letter_map, axis=-1)
        letter_indices = jnp.argmax(letter_map, axis=-1)
        
        # Add 2 to the indices to shift them past empty (0) and agent (1)
        cells = jnp.where(has_letter, letter_indices + 2, cells)
        
        # Place agent (ID 1), overwriting any letter at its position
        agent_pos = env_state.env_state.agent
        cells = cells.at[agent_pos[0], agent_pos[1]].set(1)

        # --- Convert the integer grid to an image using the atlas ---
        img = self._atlas[cells]
        img = img.transpose(0, 2, 1, 3, 4).reshape(height_px, width_px, 3)
        
        return img.astype(jnp.uint8)
        
    def ltl_to_string(self, ltl_goal: chex.Array, root_idx: int) -> str:
        """
        Helper function to convert an LTL formula AST back to a human-readable string.
        This runs in Python and is not JIT-compatible.
        """
        nodes = ltl_goal
        
        # Handle terminal nodes (True, False, or a proposition)
        op_code = int(nodes[root_idx][0])
        op_char = VOCAB_INV.get(op_code, '?')
        if op_char in self.env.propositions or op_char in ['True', 'False']:
            return op_char

        # Handle unary operators (Not, Eventually, Always)
        left_child_idx = int(nodes[root_idx][1])
        if op_char in ['!', 'F', 'G']:
            left_str = self.ltl_to_string(ltl_goal, left_child_idx)
            return f"{op_char}({left_str})"

        # Handle binary operators (And, Or, Until)
        right_child_idx = int(nodes[root_idx][2])
        if op_char in ['&', '|', 'U']:
            left_str = self.ltl_to_string(ltl_goal, left_child_idx)
            right_str = self.ltl_to_string(ltl_goal, right_child_idx)
            return f"({left_str} {op_char} {right_str})"
        
        return "<?>"


def _make_ltl_tile_atlas(tile_size: int, letters: list) -> np.ndarray:
    """
    Creates the tile atlas using Pillow for rendering text and shapes.
    This function is pure NumPy/Pillow and runs only once during initialization.
    """
    num_letters = len(letters)
    # Atlas size: 1 for empty, 1 for agent, N for letters
    atlas = np.zeros((2 + num_letters, tile_size, tile_size, 3), dtype=np.uint8)
    
    # --- Helper to draw a border for grid lines ---
    def add_border(tile_array, color=(50, 50, 50)):
        tile_array[0, :, :] = color
        tile_array[-1, :, :] = color
        tile_array[:, 0, :] = color
        tile_array[:, -1, :] = color
        return tile_array

    # --- Tile 0: Empty Cell ---
    empty_tile = np.full((tile_size, tile_size, 3), (20, 20, 20), dtype=np.uint8)
    atlas[0] = add_border(empty_tile)

    # --- Tile 1: Agent ---
    # Draw a circle for the agent
    agent_tile_pil = Image.fromarray(empty_tile.copy())
    draw = ImageDraw.Draw(agent_tile_pil)
    padding = tile_size // 5
    draw.ellipse(
        (padding, padding, tile_size - padding, tile_size - padding),
        fill=(200, 50, 50) # Red
    )
    atlas[1] = add_border(np.array(agent_tile_pil))
    
    # --- Tiles 2 onwards: Letters ---
    try:
        # Use a common, simple font. DroidSansMono is often available on Colab/Linux.
        font = ImageFont.truetype("DroidSansMono.ttf", size=int(tile_size * 0.7))
    except IOError:
        font = ImageFont.load_default()

    for i, letter in enumerate(letters):
        letter_tile_pil = Image.fromarray(empty_tile.copy())
        draw = ImageDraw.Draw(letter_tile_pil)
        
        # Center the text
        text_bbox = draw.textbbox((0,0), letter, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        position = (
            (tile_size - text_width) / 2,
            (tile_size - text_height) / 2 - tile_size * 0.1 # slight vertical lift
        )
        
        draw.text(position, letter, fill=(220, 220, 220), font=font)
        atlas[i + 2] = add_border(np.array(letter_tile_pil))

    return atlas
