import os
import pickle
import jax
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file
import typing

def save_params(params: typing.Dict, filename: typing.Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

class CheckpointManager:
    def __init__(self, base_path, run_name):
        self.save_dir = os.path.join(base_path, run_name)
        self.ckpt_path = os.path.join(self.save_dir, "checkpoint_state.pkl")
        self.weights_path = os.path.join(self.save_dir, "model.safetensors")
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, train_state, rng, update_step, wandb_run_id):
        """Saves full training state, including WandB ID."""
        print(f"Saving full checkpoint to {self.ckpt_path}...")
        
        # 1. Save Weights (safetensors)
        save_params(train_state.params, self.weights_path)

        # 2. Save Full State (pickle)
        state_byte_data = serialization.to_bytes(train_state)
        
        meta_data = {
            "train_state_bytes": state_byte_data,
            "rng": rng,
            "update_step": update_step,
            "wandb_run_id": wandb_run_id  # <--- SAVE ID
        }
        
        with open(self.ckpt_path, "wb") as f:
            pickle.dump(meta_data, f)
        print("Checkpoint saved.")

    def restore_or_initialize(self, base_train_state, base_rng):
        """
        Returns: (train_state, rng, start_step, wandb_run_id)
        """
        wandb_run_id = None # Default if new run

        # 1. Try Full Resume
        if os.path.exists(self.ckpt_path):
            print(f"Found full checkpoint at {self.ckpt_path}. Resuming...")
            try:
                with open(self.ckpt_path, "rb") as f:
                    meta_data = pickle.load(f)
                
                restored_state = serialization.from_bytes(base_train_state, meta_data["train_state_bytes"])
                restored_rng = meta_data["rng"]
                start_step = meta_data["update_step"]
                wandb_run_id = meta_data.get("wandb_run_id", None) # <--- LOAD ID
                
                return restored_state, restored_rng, start_step, wandb_run_id
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Falling back...")

        # 2. Try Weights Only
        if os.path.exists(self.weights_path):
            print(f"Found weights at {self.weights_path}. Loading weights into fresh state...")
            try:
                flat_params = load_file(self.weights_path)
                params = unflatten_dict(flat_params, sep=',')
                restored_state = base_train_state.replace(params=params)
                return restored_state, base_rng, 0, None
            except Exception as e:
                print(f"Failed to load weights: {e}.")

        # 3. Scratch
        print("Starting from scratch.")
        return base_train_state, base_rng, 0, None