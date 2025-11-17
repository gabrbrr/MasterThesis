import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from typing import List, Tuple, Dict, Any, Optional

# --- JIT-Compilable Core Functions ---

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def _sample_jit(
    key: jax.random.PRNGKey,
    n_propositions: int,
    min_levels: int,
    max_levels: int,
    min_conjunctions: int,
    max_conjunctions: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compilable function to sample a new Until-Task task.

    Returns:
        task_matrix (jnp.ndarray): Shape (max_levels, max_conjunctions, 2).
                                  Contains proposition indices, -1 for padding.
        levels_array (jnp.ndarray): Shape (max_conjunctions,).
                                   Contains the number of levels for each conjunct, 0 for padding.
    """
    key, subkey_prop, subkey_n_conj, subkey_levels = jax.random.split(key, 4)

    # 1. Sample all propositions needed upfront, without replacement
    n_props_needed = 2 * max_levels * max_conjunctions
    if n_props_needed > n_propositions:
        raise ValueError(
            f"Not enough propositions. Need {n_props_needed}, "
            f"but only {n_propositions} are available."
        )
    
    prop_pool = jax.random.choice(
        subkey_prop, n_propositions, shape=(n_props_needed,), replace=False
    )
    
    # Reshape into the full matrix structure
    task_matrix = prop_pool.reshape((max_levels, max_conjunctions, 2))

    # 2. Sample the number of conjuncts
    n_conjs = jax.random.randint(
        subkey_n_conj, shape=(), minval=min_conjunctions, maxval=max_conjunctions + 1
    )

    # 3. Sample the number of levels for all *potential* conjuncts
    levels_array_full = jax.random.randint(
        subkey_levels, shape=(max_conjunctions,), minval=min_levels, maxval=max_levels + 1
    )

    # 4. Create masks to handle variable numbers of conjuncts and levels
    
    # Mask for active conjuncts
    conjunct_indices = jnp.arange(max_conjunctions)
    conjunct_mask = (conjunct_indices < n_conjs).astype(jnp.int32)
    
    # Set levels for inactive conjuncts to 0
    final_levels_array = levels_array_full * conjunct_mask
    
    # Mask for active levels within each conjunct
    level_indices = jnp.arange(max_levels)
    
    # Broadcast to (max_levels, max_conjunctions)
    level_indices_matrix = jnp.broadcast_to(level_indices[:, None], (max_levels, max_conjunctions))
    active_levels_matrix = jnp.broadcast_to(final_levels_array[None, :], (max_levels, max_conjunctions))
    
    level_mask = (level_indices_matrix < active_levels_matrix)
    
    # Broadcast mask to 3D: (max_levels, max_conjunctions, 2)
    level_mask_3d = jnp.broadcast_to(level_mask[..., None], (max_levels, max_conjunctions, 2))
    
    # 5. Apply mask. Inactive slots are set to -1.
    final_task_matrix = jnp.where(level_mask_3d, task_matrix, -1)

    return final_task_matrix, final_levels_array

@jax.jit
def _progress_jit(
    task_matrix: jnp.ndarray,
    levels_array: jnp.ndarray,
    propositions_state: jnp.ndarray
) -> jnp.ndarray:
    """
    JIT-compilable function to check the satisfaction of the Until-Tasks.

    Args:
        task_matrix: The task matrix from _sample_jit.
        levels_array: The levels array from _sample_jit.
        propositions_state: A boolean array (shape [n_propositions,])
                            indicating the truth value of each proposition.

    Returns:
        satisfied (jnp.ndarray): A boolean array (shape [max_conjunctions,])
                                 where satisfied[i] is True if conjunct i is satisfied.
    """
    max_levels, n_conjs = task_matrix.shape[0], task_matrix.shape[1]
    
    # `scan_fn` computes the 'done' status for level `j`,
    # given the 'done' status of the inner task `carry_done`.
    def scan_fn(carry_done: jnp.ndarray, j: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            carry_done: bool[n_conjs], 'done' status of the task *inside* this one.
            j: int, current level index (from max_levels-1 down to 0).
        """
        # Get prop indices for level j
        avoid_idx = task_matrix[j, :, 0] # (n_conjs,)
        reach_idx = task_matrix[j, :, 1] # (n_conjs,)
        
        # Mask for valid (non-padded) levels
        valid_mask = (avoid_idx != -1) # (n_conjs,)
        
        # Get truth values, using index 0 for padded slots (will be masked out)
        safe_reach_idx = jnp.where(valid_mask, reach_idx, 0)
        reach_true = propositions_state[safe_reach_idx].astype(jnp.bool_)
        
        # Check if this is the highest level (base case) for each conjunct
        is_highest_level = (levels_array == j + 1)
        
        # The 'reach part' is just 'b' for the highest level.
        # It's '(and b, inner_task_done)' for all other levels.
        reach_part_done = jnp.where(
            is_highest_level,
            reach_true,
            reach_true & carry_done
        )
        
        # The 'until' task U(not a, reach_part) is satisfied if reach_part_done is true.
        current_level_done = reach_part_done
        
        # This 'done' status is the carry for the next level up (j-1).
        # Only update the carry if this level was valid; otherwise, pass False.
        new_carry = jnp.where(valid_mask, current_level_done, False)
        
        # We don't need the stacked 'y' values, just the final carry.
        return new_carry, new_carry

    # Iterate from the highest level (max_levels - 1) down to 0
    levels_to_scan = jnp.arange(max_levels - 1, -1, -1)
    
    # Initial 'carry' is False. This is safe because the 'is_highest_level'
    # branch ignores the carry for the base case.
    init_carry = jnp.zeros(n_conjs, dtype=jnp.bool_)
    
    # Run the scan
    final_carry, _ = lax.scan(scan_fn, init_carry, levels_to_scan)
    
    # `final_carry` is the 'done' status after processing level 0.
    # Also, conjuncts that have 0 levels (i.e., were never active)
    # are vacuously satisfied.
    vacuously_satisfied = (levels_array == 0)
    
    return final_carry | vacuously_satisfied


# --- Encoder, Decoder, and Wrapper Class ---

def encode_until_task(
    ltl_tuple: Tuple,
    propositions: List[str],
    max_levels: int,
    max_conjunctions: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encodes a Python tuple LTL formula into the JAX matrix representation.
    """
    prop_map = {prop: i for i, prop in enumerate(propositions)}
    task_matrix = jnp.full((max_levels, max_conjunctions, 2), -1, dtype=jnp.int32)
    levels_array = jnp.zeros(max_conjunctions, dtype=jnp.int32)

    def get_conjuncts(ltl: Tuple) -> List[Tuple]:
        conjuncts = []
        if ltl and ltl[0] == 'and':
            conjuncts.extend(get_conjuncts(ltl[1]))
            conjuncts.extend(get_conjuncts(ltl[2]))
        elif ltl:
            conjuncts.append(ltl)
        return conjuncts

    conjunct_list = get_conjuncts(ltl_tuple)
    
    if len(conjunct_list) > max_conjunctions:
        raise ValueError(
            f"LTL has {len(conjunct_list)} conjuncts, "
            f"but max_conjunctions is {max_conjunctions}"
        )

    for i, task in enumerate(conjunct_list):
        level = 0
        current_task = task
        
        while level < max_levels and current_task and current_task[0] == 'until':
            # ('until', ('not', avoid_prop), reach_part)
            avoid_prop = current_task[1][1]
            reach_part = current_task[2]
            
            avoid_idx = prop_map[avoid_prop]
            
            if isinstance(reach_part, tuple) and reach_part[0] == 'and':
                # ('and', reach_prop, next_task)
                reach_prop = reach_part[1]
                next_task = reach_part[2]
            else:
                # Base case: ('until', ('not', avoid_prop), reach_prop)
                reach_prop = reach_part
                next_task = None
            
            reach_idx = prop_map[reach_prop]
            
            task_matrix = task_matrix.at[level, i, 0].set(avoid_idx)
            task_matrix = task_matrix.at[level, i, 1].set(reach_idx)
            level += 1
            
            if next_task is None:
                break
            current_task = next_task
        
        levels_array = levels_array.at[i].set(level)

    return task_matrix, levels_array

def decode_until_task(
    task_matrix: jnp.ndarray,
    levels_array: jnp.ndarray,
    propositions: List[str]
) -> Optional[Tuple]:
    """
    Decodes the JAX matrix representation back into a Python tuple LTL formula.
    """
    conjunct_tasks = []
    n_conjs = jnp.sum(levels_array > 0).item()

    for i in range(n_conjs):
        n_levels = levels_array[i].item()
        if n_levels == 0:
            continue
            
        current_ltl = None
        # Build from the inside out (highest level first)
        for j in range(n_levels - 1, -1, -1):
            avoid_prop = propositions[task_matrix[j, i, 0].item()]
            reach_prop = propositions[task_matrix[j, i, 1].item()]
            
            if current_ltl is None:
                # Base case (highest level)
                current_ltl = ('until', ('not', avoid_prop), reach_prop)
            else:
                # Nested case
                current_ltl = ('until', ('not', avoid_prop), ('and', reach_prop, current_ltl))
                
        if current_ltl:
            conjunct_tasks.append(current_ltl)
            
    if not conjunct_tasks:
        return None
    
    # Combine conjuncts with 'and'
    ltl = conjunct_tasks[0]
    for i in range(1, len(conjunct_tasks)):
        ltl = ('and', conjunct_tasks[i], ltl)
        
    return ltl


class JaxUntilTaskSampler:
    """
    A wrapper class to manage the JIT-compiled functions and proposition mapping.
    """
    def __init__(
        self,
        propositions: List[str],
        min_levels: int = 1,
        max_levels: int = 2,
        min_conjunctions: int = 1,
        max_conjunctions: int = 2
    ):
        self.propositions = propositions
        self.prop_map = {prop: i for i, prop in enumerate(propositions)}
        self.n_propositions = len(propositions)
        
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)
        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        
        n_props_needed = 2 * self.max_levels * self.max_conjunctions
        if n_props_needed > self.n_propositions:
            raise ValueError(
                f"The domain does not have enough propositions! "
                f"Needs: {n_props_needed} (2 * {self.max_levels} * {self.max_conjunctions}), "
                f"Available: {self.n_propositions}"
            )

        # Create and JIT the sampling function
        self._sample_logic = partial(
            _sample_jit,
            n_propositions=self.n_propositions,
            min_levels=self.min_levels,
            max_levels=self.max_levels,
            min_conjunctions=self.min_conjunctions,
            max_conjunctions=self.max_conjunctions
        )
        self.sample_jit = jax.jit(self._sample_logic)
        
        # JIT the progress function
        self.progress_jit = _progress_jit # Already JIT-compiled
        
    def sample(self, key: jax.random.PRNGKey) -> Optional[Tuple]:
        """
        Samples a new task and returns the human-readable LTL tuple.
        """
        task_matrix, levels_array = self.sample_jit(key)
        return self.decode(task_matrix, levels_array)

    def encode(self, ltl_tuple: Tuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encodes an LTL tuple into the JAX matrix representation.
        """
        return encode_until_task(
            ltl_tuple, self.propositions, self.max_levels, self.max_conjunctions
        )
    
    def decode(
        self, task_matrix: jnp.ndarray, levels_array: jnp.ndarray
    ) -> Optional[Tuple]:
        """
        Decodes the JAX matrix representation into an LTL tuple.
        """
        return decode_until_task(task_matrix, levels_array, self.propositions)


# --- Example Usage ---

if __name__ == "__main__":
    propositions_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    # --- Test 1: Sampling and Encode/Decode ---
    print("--- Test 1: Sampling and Encode/Decode ---")
    sampler = JaxUntilTaskSampler(
        propositions_list,
        min_levels=1,
        max_levels=2,
        min_conjunctions=1,
        max_conjunctions=2
    )
    
    key = jax.random.PRNGKey(42)
    key, sample_key = jax.random.split(key)
    
    # Sample using the JIT-compiled function
    print("Sampling task...")
    task_matrix, levels_array = sampler.sample_jit(sample_key)
    
    print("\nTask Matrix (Prop Indices):")
    print(task_matrix)
    print("\nLevels Array:")
    print(levels_array)
    
    # Decode back to Python tuple
    ltl_formula = sampler.decode(task_matrix, levels_array)
    print(f"\nDecoded LTL Formula:\n{ltl_formula}")
    
    # Re-encode
    if ltl_formula:
        matrix_rt, levels_rt = sampler.encode(ltl_formula)
        print("\nRe-encoded Matrix matches:", jnp.array_equal(task_matrix, matrix_rt))
        print("Re-encoded Levels matches:", jnp.array_equal(levels_array, levels_rt))

    # --- Test 2: Progress Function (Manual Case) ---
    print("\n\n--- Test 2: Progress Function (Manual Case) ---")
    # Task: ('and', U(not c, (and d, U(not a, b))), U(not e, f))
    # 'a'=0, 'b'=1, 'c'=2, 'd'=3, 'e'=4, 'f'=5
    # Conjunct 0: U(not c, (and d, U(not a, b))) -> 2 levels
    # Conjunct 1: U(not e, f) -> 1 level
    
    test_sampler = JaxUntilTaskSampler(propositions_list, max_levels=2, max_conjunctions=2)
    
    manual_task_matrix = jnp.full((2, 2, 2), -1, dtype=jnp.int32)
    manual_levels_array = jnp.array([2, 1], dtype=jnp.int32)
    
    # Conjunct 0, Level 0 (outer)
    manual_task_matrix = manual_task_matrix.at[0, 0, 0].set(2) # 'c'
    manual_task_matrix = manual_task_matrix.at[0, 0, 1].set(3) # 'd'
    # Conjunct 0, Level 1 (inner)
    manual_task_matrix = manual_task_matrix.at[1, 0, 0].set(0) # 'a'
    manual_task_matrix = manual_task_matrix.at[1, 0, 1].set(1) # 'b'

    # Conjunct 1, Level 0 (outer)
    manual_task_matrix = manual_task_matrix.at[0, 1, 0].set(4) # 'e'
    manual_task_matrix = manual_task_matrix.at[0, 1, 1].set(5) # 'f'
    
    ltl_manual = test_sampler.decode(manual_task_matrix, manual_levels_array)
    print(f"Manual LTL:\n{ltl_manual}")
    
    # --- Test progress states ---
    
    # State 1: Nothing true
    state1 = jnp.zeros(test_sampler.n_propositions, dtype=jnp.bool_)
    satisfied1 = test_sampler.progress_jit(manual_task_matrix, manual_levels_array, state1)
    print(f"\nState: All False\nSatisfied: {satisfied1}") # Expect [False False]

    # State 2: 'f' is true (satisfies Conjunct 1)
    state2 = state1.at[5].set(True) # 'f' = True
    satisfied2 = test_sampler.progress_jit(manual_task_matrix, manual_levels_array, state2)
    print(f"\nState: 'f' = True\nSatisfied: {satisfied2}") # Expect [False  True]

    # State 3: 'b' is true (satisfies inner part of Conjunct 0, but not outer)
    state3 = state1.at[1].set(True) # 'b' = True
    satisfied3 = test_sampler.progress_jit(manual_task_matrix, manual_levels_array, state3)
    print(f"\nState: 'b' = True\nSatisfied: {satisfied3}") # Expect [False False]
    
    # State 4: 'b' and 'd' are true (satisfies all of Conjunct 0)
    state4 = state1.at[1].set(True).at[3].set(True) # 'b' = True, 'd' = True
    satisfied4 = test_sampler.progress_jit(manual_task_matrix, manual_levels_array, state4)
    print(f"\nState: 'b' = True, 'd' = True\nSatisfied: {satisfied4}") # Expect [ True False]

    # State 5: 'b', 'd', and 'f' are true (satisfies all)
    state5 = state4.at[5].set(True) # 'f' = True
    satisfied5 = test_sampler.progress_jit(manual_task_matrix, manual_levels_array, state5)
    print(f"\nState: 'b'=T, 'd'=T, 'f'=T\nSatisfied: {satisfied5}") # Expect [ True  True]