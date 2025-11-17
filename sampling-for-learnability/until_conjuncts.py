import jax
import jax.numpy as jnp
from jax import lax
import re
from typing import List, Tuple, Dict, Optional, NamedTuple

# --- Define the Formula State Structures ---

class SingleFormulaState(NamedTuple):
    """
    Holds the state for *one* nested 'Until' formula.
    Used as the input/output for the vmapped function.
    
    active_pointers: Boolean array, True if sub-formula at depth `i` is active.
    to_avoid: Integer array of propositions 'a' in `!a U ...`.
    to_progress: Integer array of propositions 'b' in `... U (b & ...)`.
    """
    active_pointers: jnp.ndarray
    to_avoid: jnp.ndarray
    to_progress: jnp.ndarray

class ConjunctionState(NamedTuple):
    """
    Holds the batched state for a *conjunction* of N formulas.
    All arrays are padded to GLOBAL_MAX_DEPTH.
    
    active_pointers: (N, GLOBAL_MAX_DEPTH) bool
    to_avoid: (N, GLOBAL_MAX_DEPTH) int
    to_progress: (N, GLOBAL_MAX_DEPTH) int
    depths: (N,) int - The *actual* depth of each formula.
    already_true: (N,) bool - Mask of formulas that have resolved to True.
    """
    active_pointers: jnp.ndarray
    to_avoid: jnp.ndarray
    to_progress: jnp.ndarray
    depths: jnp.ndarray
    already_true: jnp.ndarray

# --- Core JAX Functions ---

@jax.jit
def _progress_single_formula(
    state: SingleFormulaState,
    current_props: jnp.ndarray,
    depth: jnp.ndarray # The *actual* depth of this formula
) -> Tuple[SingleFormulaState, jnp.ndarray, jnp.ndarray]:
    """
    Progresses a *single* nested 'Until' formula.
    This function is designed to be vmapped.
    """
    MAX_DEPTH = state.to_avoid.shape[0] # This is the GLOBAL_MAX_DEPTH
    new_active = jnp.zeros(MAX_DEPTH, dtype=bool)
    is_true_overall = jnp.array(False)

    for i in range(MAX_DEPTH):
        # --- 1. 'Stays Active' (from active_pointers[i]) ---
        is_active_i = state.active_pointers[i]
        phi_progresses_i = ~current_props[state.to_avoid[i]]
        stays_active = is_active_i & phi_progresses_i

        # --- 2. 'Gets Activated' (from active_pointers[i-1]) ---
        gets_activated = jnp.array(False)
        if i > 0:
            is_active_prev = state.active_pointers[i-1]
            psi_progresses_prev = current_props[state.to_progress[i-1]]
            gets_activated = is_active_prev & psi_progresses_prev

        new_active = new_active.at[i].set(stays_active | gets_activated)

        # --- 3. Check for immediate 'True' satisfaction ---
        # This now uses the 'depth' argument to check at the correct level.
        is_last_real_prop = (i == depth - 1)
        is_active_at_depth = state.active_pointers[i]
        psi_progresses_at_depth = current_props[state.to_progress[i]]
        
        # If this is the last *real* proposition for this formula,
        # and it's active and its 'psi' progresses, the formula resolves to True.
        becomes_true = is_active_at_depth & psi_progresses_at_depth & is_last_real_prop
        is_true_overall = is_true_overall | becomes_true

    # The entire formula is False if no branches are active in the new state
    # AND it didn't just become True.
    is_false_overall = ~jnp.any(new_active) & ~is_true_overall

    # If formula resolved to True or False, clear all active pointers.
    # This prevents resolved formulas from progressing further.
    final_active_pointers = lax.cond(
        is_true_overall | is_false_overall,
        lambda: jnp.zeros_like(new_active),
        lambda: new_active
    )
    new_state = state._replace(active_pointers=final_active_pointers)
    
    return new_state, is_true_overall, is_false_overall

# --- Vmapped and Public Functions ---

# Create a vmapped version of the single-formula progress function.
# We map over:
#   - SingleFormulaState (all fields, axis 0)
#   - current_props (None, broadcast)
#   - depths (axis 0)
_vmapped_progress = jax.vmap(
    _progress_single_formula,
    in_axes=(SingleFormulaState(0, 0, 0), None, 0),
    out_axes=(SingleFormulaState(0, 0, 0), 0, 0)
)

@jax.jit
def progress_conjunction(
    state: ConjunctionState,
    current_props: jnp.ndarray
) -> Tuple[ConjunctionState, jnp.ndarray, jnp.ndarray]:
    """
    Progresses an entire conjunction of N formulas in parallel.
    """
    
    # 1. Create a mask of tasks that are *still progressing*
    #    (i.e., not already True).
    progressing_mask = ~state.already_true # Shape (N,)

    # 2. We only want to progress tasks that are still progressing.
    #    We "mask" the input `active_pointers`.
    #    `active_pointers` for non-progressing tasks are set to all-False.
    input_active_pointers = state.active_pointers * jnp.expand_dims(progressing_mask, -1)
    
    vmap_input_state = SingleFormulaState(
        input_active_pointers, state.to_avoid, state.to_progress
    )

    # 3. Run the vmapped progress function
    vmap_output_state, is_true_this_step, is_false_this_step = \
        _vmapped_progress(vmap_input_state, current_props, state.depths)
    
    # 4. Update the `already_true` mask
    # A task is now true if it was *already* true OR it *just became* true.
    new_already_true = state.already_true | is_true_this_step
    
    # 5. Determine overall conjunction status
    # Conjunction is False if any *progressing* task *just became* False.
    is_false_overall = jnp.any(is_false_this_step)
    
    # Conjunction is True if *all* tasks are now in the `already_true` state.
    is_true_overall = jnp.all(new_already_true)
    
    # 6. Create the new state.
    # The new `active_pointers` are what vmap returned.
    # (Tasks that just finished are zeroed by `_progress_single_formula`).
    # (Tasks that *were* finished were zeroed on input).
    new_conjunction_state = ConjunctionState(
        active_pointers=vmap_output_state.active_pointers,
        to_avoid=state.to_avoid,
        to_progress=state.to_progress,
        depths=state.depths,
        already_true=new_already_true
    )
    
    return new_conjunction_state, is_true_overall, is_false_overall


# --- Verification and Test Harness ---

# Helper to map proposition characters ('a'-'z') to integer indices (0-25)
PROP_TO_INT = {chr(ord('a') + i): i for i in range(26)}
INT_TO_PROP = {i: chr(ord('a') + i) for i in range(26)}
NUM_PROPS = 26

def _parse_single_formula(
    formula_str: str
) -> Tuple[jnp.ndarray, jnp.ndarray, int, str]:
    """Helper to parse one formula string. (Modified from old parse_formula_log)"""
    avoids = re.findall(r"!\(([a-z])\)\sU", formula_str)
    progresses = re.findall(r"U\s\(([a-z])", formula_str)
    
    last_prog = re.search(r"U\s([a-z])\)\)+$", formula_str)
    if not last_prog:
        # Fallback for simple formula like `(!b U j)`
        last_prog = re.search(r"U\s([a-z])\)", formula_str)
        
    if last_prog:
        progresses.append(last_prog.group(1))

    depth = len(avoids)
    if len(progresses) != depth or depth == 0:
        raise ValueError(f"Mismatch or parse error: {formula_str}")

    to_avoid_arr = jnp.array([PROP_TO_INT[p] for p in avoids], dtype=jnp.int32)
    to_progress_arr = jnp.array([PROP_TO_INT[p] for p in progresses], dtype=jnp.int32)

    def build_str(d):
        if d == depth - 1:
            return f"(!({avoids[d]}) U {progresses[d]})"
        return f"(!({avoids[d]}) U ({progresses[d]} & {build_str(d+1)}))"
    recon_str = build_str(0)
    
    return to_avoid_arr, to_progress_arr, depth, recon_str

def build_conjunction_state(
    formula_strings: List[str]
) -> Tuple[ConjunctionState, int, List[str]]:
    """
    Parses a list of formula strings and builds a batched, padded ConjunctionState.
    """
    N = len(formula_strings)
    parsed_formulas = []
    max_depth = 0
    recon_strings = []
    
    for f_str in formula_strings:
        avoid, prog, depth, recon = _parse_single_formula(f_str)
        parsed_formulas.append((avoid, prog, depth))
        recon_strings.append(recon)
        if depth > max_depth:
            max_depth = depth
            
    GLOBAL_MAX_DEPTH = max_depth
    
    # Initialize padded arrays
    all_active = jnp.zeros((N, GLOBAL_MAX_DEPTH), dtype=bool)
    all_avoid = jnp.zeros((N, GLOBAL_MAX_DEPTH), dtype=jnp.int32)
    all_progress = jnp.zeros((N, GLOBAL_MAX_DEPTH), dtype=jnp.int32)
    all_depths = jnp.zeros((N,), dtype=jnp.int32)
    all_already_true = jnp.zeros((N,), dtype=bool)

    # Fill the arrays
    for i, (avoid, prog, depth) in enumerate(parsed_formulas):
        all_active = all_active.at[i, 0].set(True) # Set initial pointer
        all_avoid = all_avoid.at[i, :depth].set(avoid)
        all_progress = all_progress.at[i, :depth].set(prog)
        all_depths = all_depths.at[i].set(depth)

    initial_state = ConjunctionState(
        active_pointers=all_active,
        to_avoid=all_avoid,
        to_progress=all_progress,
        depths=all_depths,
        already_true=all_already_true
    )
    
    return initial_state, GLOBAL_MAX_DEPTH, recon_strings

def props_to_bool_array(props_list: List[str]) -> jnp.ndarray:
    """Converts a list of prop strings like ['c'] to a boolean array."""
    arr = jnp.zeros(NUM_PROPS, dtype=bool)
    for p in props_list:
        if p in PROP_TO_INT:
            arr = arr.at[PROP_TO_INT[p]].set(True)
    return arr

def _format_single_state_str(
    state: SingleFormulaState,
    is_true: bool,
    is_false: bool,
    depth: int,
) -> str:
    """Formats the output state of a single formula."""
    if is_true:
        return "True"
    if is_false:
        return "False"

    def build_str(d):
        avoid_p = INT_TO_PROP[state.to_avoid[d].item()]
        prog_p = INT_TO_PROP[state.to_progress[d].item()]
        if d == depth - 1:
            return f"(!({avoid_p}) U {prog_p})"
        return f"(!({avoid_p}) U ({prog_p} & {build_str(d+1)}))"

    active_formulas = []
    for i in range(depth):
        if state.active_pointers[i]:
            active_formulas.append(f"({build_str(i)})")
    
    if not active_formulas:
        return "False" # Should be caught by is_false
        
    return " | ".join(active_formulas)

def format_conjunction_state(
    state: ConjunctionState,
    is_true_overall: jnp.ndarray,
    is_false_overall: jnp.ndarray,
    recon_strings: List[str]
) -> str:
    """Formats the state of the entire conjunction."""
    N = state.depths.shape[0]
    output = []
    
    if is_false_overall:
        output.append("--- OVERALL CONJUNCTION: FALSE ---")
    elif is_true_overall:
        output.append("--- OVERALL CONJUNCTION: TRUE ---")
    else:
        output.append("--- OVERALL CONJUNCTION: Progressing ---")
        
    for i in range(N):
        # Re-create a temporary SingleFormulaState for the formatter
        single_state = SingleFormulaState(
            state.active_pointers[i],
            state.to_avoid[i],
            state.to_progress[i]
        )
        depth = state.depths[i].item()
        
        # Check individual status
        is_true = state.already_true[i]
        is_false = ~is_true & ~jnp.any(state.active_pointers[i])
        
        formula_str = _format_single_state_str(
            single_state, is_true, is_false, depth
        )
        output.append(f"    Task {i} [{recon_strings[i]}]:\n        -> {formula_str}")
    
    return "\n".join(output)
    
# --- Main execution ---
def run_conjunction_test():
    """
    Runs a new test case for the conjunction of two formulas.
    """
    
    # Test Case:
    # Task 0: (!(a) U b)
    # Task 1: (!(c) U (d & (!(e) U f)))
    # Task 2: (!(j) U k)
    
    log_data = [
        "(!(a) U b)",
        "(!(c) U (d & (!(e) U f)))",
        "(!(j) U k)"
    ]
    
    print(f"--- 🔄 Initializing Conjunction of {len(log_data)} Tasks ---")
    
    try:
        current_state, GLOBAL_MAX_DEPTH, recon_strings = \
            build_conjunction_state(log_data)
        for i, s in enumerate(recon_strings):
            print(f"  Task {i}: {s}")
            
    except Exception as e:
        print(f"--- 🚫 Failed to build state (Parse Error: {e}) ---")
        return
            
    # Define the progression steps
    steps = [['c'], ['a'], ['d'], ['e', 'b'], ['f', 'k']]
    
    for step_props in steps:
        print(f"\n--- Progressing with {step_props} ---")
        
        # Create the boolean proposition array for this step
        current_props = props_to_bool_array(step_props)
        
        # Run the jitted conjunction progression function
        new_state, is_true, is_false = progress_conjunction(
            current_state, current_props
        )
        
        # Format the output
        simplification_str = format_conjunction_state(
            new_state, is_true, is_false, recon_strings
        )
        print(simplification_str)

        # The new state becomes the current state for the next step
        current_state = new_state
        
        # If formula becomes True or False, stop progressing
        if is_true or is_false:
            break
            
    # --- Second Test: Based on user log data ---
    # This test will run all formulas from the log in parallel
    print("\n\n--- 🔄 Initializing Conjunction of ALL Log Formulas ---")
    
    all_log_formulas = [
        "(!(f) U (c & (!(i) U (e & (!(b) U j)))))",
        "(!(e) U (c & (!(b) U (k & (!(h) U g)))))",
        "(!(i) U (l & (!(e) U (g & (!(j) U h)))))",
        "(!(d) U (i & (!(b) U (g & (!(e) U h)))))",
        "(!(c) U (i & (!(d) U (j & (!(b) U f)))))",
        "(!(j) U (b & (!(k) U (a & (!(d) U f)))))",
        "(!(a) U (e & (!(c) U (f & (!(k) U l)))))",
        "(!(k) U (e & (!(l) U (f & (!(g) U a)))))",
        "(!(l) U (e & (!(i) U (k & (!(d) U g)))))",
        "(!(f) U (a & (!(k) U (h & (!(l) U j)))))",
        "(!(e) U (b & (!(h) U (a & (!(j) U d))))))",
        "(!(j) U (l & (!(e) U (g & (!(c) U a)))))",
        "(!(l) U (f & (!(i) U (h & (!(b) U c)))))",
        "(!(e) U (d & (!(b) U (h & (!(l) U a)))))",
        "(!(k) U (d & (!(b) U (j & (!(a) U f)))))",
        "(!(j) U (f & (!(g) U (l & (!(d) U h)))))",
        "(!(j) U (k & (!(l) U (g & (!(h) U a)))))",
        "(!(b) U (i & (!(h) U (d & (!(g) U e)))))",
    ]

    try:
        current_state, GLOBAL_MAX_DEPTH, recon_strings = \
            build_conjunction_state(all_log_formulas)
        print(f"  Loaded {len(recon_strings)} tasks. Max depth: {GLOBAL_MAX_DEPTH}")
            
    except Exception as e:
        print(f"--- 🚫 Failed to build state (Parse Error: {e}) ---")
        return
        
    # Progress with a few arbitrary steps
    steps = [['c'], ['e'], ['j'], ['l', 'f', 'a'], ['g', 'b', 'k']]
    
    for step_props in steps:
        print(f"\n--- Progressing ALL formulas with {step_props} ---")
        current_props = props_to_bool_array(step_props)
        
        new_state, is_true, is_false = progress_conjunction(
            current_state, current_props
        )
        
        simplification_str = format_conjunction_state(
            new_state, is_true, is_false, recon_strings
        )
        print(simplification_str)

        current_state = new_state
        if is_true or is_false:
            print("\n--- Conjunction has resolved. ---")
            break

if __name__ == "__main__":
    # run_log_verification() # This function is now deprecated
    run_conjunction_test()