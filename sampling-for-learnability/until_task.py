import jax
import jax.numpy as jnp
from jax import lax
import re
from typing import List, Tuple, Dict, Optional, NamedTuple

# --- Define the Formula State Structure ---
class FormulaState(NamedTuple):
    """
    Holds the complete state of a nested 'Until' formula progression.
    
    active_pointers: Boolean array, True if the sub-formula at depth `i` is active.
    to_avoid: Integer array of propositions 'a' in `!a U ...`.
    to_progress: Integer array of propositions 'b' in `... U (b & ...)`.
    """
    active_pointers: jnp.ndarray
    to_avoid: jnp.ndarray
    to_progress: jnp.ndarray

# --- Core JAX Function ---

@jax.jit
def progress_formula(
    state: FormulaState,
    current_props: jnp.ndarray
) -> Tuple[FormulaState, jnp.ndarray, jnp.ndarray]:
    """
    Progresses a nested 'Until' formula based on the current true propositions.

    The formula is represented as a disjunction of nested 'Until' sub-formulas,
    tracked by `active_pointers`.
    Formula structure:
    active[0] -> (!to_avoid[0] U (to_progress[0] & ...))
    active[1] ->   (!to_avoid[1] U (to_progress[1] & ...))
    ...
    active[n] ->     (!to_avoid[n] U to_progress[n])

    Progression rule for `phi U psi` (where phi = !avoid, psi = progress & sub):
    progress(phi U psi, P) = progress(psi, P) | (progress(phi, P) & (phi U psi))

    Args:
        state: A FormulaState object containing the active pointers and
               the static formula definition (to_avoid, to_progress).
        current_props: A boolean array indexed by proposition integer,
                       where `current_props[p]` is true if proposition `p` holds.

    Returns:
        A tuple of:
        (new_state, is_true_overall, is_false_overall)
        - new_state: The updated FormulaState object.
        - is_true_overall: True if the entire disjunction simplifies to True.
        - is_false_overall: True if the entire disjunction simplifies to False.
    """
    MAX_DEPTH = state.to_avoid.shape[0]
    new_active = jnp.zeros(MAX_DEPTH, dtype=bool)
    is_true_overall = jnp.array(False)

    # We iterate through each possible depth `i` of the formula.
    # JAX unrolls this loop as MAX_DEPTH is known at compile time.
    for i in range(MAX_DEPTH):
        # A branch `i` in new_active can become True from two sources:
        # 1. It 'stays active' from active_pointers[i]
        # 2. It 'gets activated' from active_pointers[i-1]

        # --- 1. 'Stays Active' (from active_pointers[i]) ---
        # This corresponds to the `(progress(phi, P) & (phi U psi))` part of the rule.
        # The branch `i` must be active, and its `phi` part must progress.
        is_active_i = state.active_pointers[i]
        # phi = !to_avoid[i]. progress(phi, P) = !current_props[to_avoid[i]]
        phi_progresses_i = ~current_props[state.to_avoid[i]]
        stays_active = is_active_i & phi_progresses_i

        # --- 2. 'Gets Activated' (from active_pointers[i-1]) ---
        # This corresponds to the `progress(psi, P)` part of the rule
        # applied at depth `i-1`.
        # The branch `i-1` must be active, and its `psi` part must progress.
        # `psi = to_progress[i-1] & (sub-formula)`.
        # `progress(psi, P)` = `current_props[to_progress[i-1]] & progress(sub, P)`
        # This simplifies to `(sub-formula)`, which *is* the branch at depth `i`.
        
        # We use a standard Python `if` which JAX handles during unrolling.
        if i == 0:
            # The first branch (i=0) cannot be activated by a previous one.
            gets_activated = jnp.array(False)
        else:
            is_active_prev = state.active_pointers[i-1]
            # psi_progresses_prev = current_props[to_progress[i-1]]
            psi_progresses_prev = current_props[state.to_progress[i-1]]
            gets_activated = is_active_prev & psi_progresses_prev

        # The new branch `i` is active if it stays active OR gets activated.
        new_active = new_active.at[i].set(stays_active | gets_activated)

        # --- 3. Check for immediate 'True' satisfaction ---
        # This only happens at the *last* depth (i == MAX_DEPTH - 1).
        # If the last branch is active and its `psi` part progresses,
        # the formula `!avoid[n] U progress[n]` is satisfied.
        # `progress(psi, P)` becomes `True`.
        if i == MAX_DEPTH - 1:
            is_active_last = state.active_pointers[i]
            psi_progresses_last = current_props[state.to_progress[i]]
            # If this part becomes True, the whole disjunction becomes True.
            becomes_true = is_active_last & psi_progresses_last
            is_true_overall = is_true_overall | becomes_true

    # The entire formula is False if no branches are active in the new state
    # AND it didn't just become True.
    is_false_overall = ~jnp.any(new_active) & ~is_true_overall

    # Create the new state by replacing the active_pointers
    new_state = state._replace(active_pointers=new_active)

    return new_state, is_true_overall, is_false_overall


# --- Verification and Test Harness ---

# Helper to map proposition characters ('a'-'z') to integer indices (0-25)
PROP_TO_INT = {chr(ord('a') + i): i for i in range(26)}
INT_TO_PROP = {i: chr(ord('a') + i) for i in range(26)}
NUM_PROPS = 26

def parse_formula_log(
    formula_str: str
) -> Tuple[FormulaState, int, str]:
    """
    Parses the log string into `to_avoid` and `to_progress` arrays
    and returns an initial FormulaState.
    """
    # Example: (!(f) U (c & (!(i) U (e & (!(b) U j)))))
    
    # Find all (!prop) U ...
    avoids = re.findall(r"!\(([a-z])\)\sU", formula_str)
    # Find all prop & ... or ... U prop)
    progresses = re.findall(r"U\s\(([a-z])", formula_str)
    
    # Find the last proposition
    last_prog = re.search(r"U\s([a-z])\)\)+$", formula_str)
    if last_prog:
        progresses.append(last_prog.group(1))

    depth = len(avoids)
    if len(progresses) != depth:
        # Fallback for simple formula like `(!b U j)`
        if depth == 1 and not progresses:
             last_prog = re.search(r"U\s([a-z])\)", formula_str)
             if last_prog:
                progresses.append(last_prog.group(1))
        else:
            raise ValueError(f"Mismatch in parsing: {formula_str}")

    # Convert to integer arrays
    to_avoid_arr = jnp.array([PROP_TO_INT[p] for p in avoids], dtype=jnp.int32)
    to_progress_arr = jnp.array([PROP_TO_INT[p] for p in progresses], dtype=jnp.int32)

    # Reconstruct formula for printing
    def build_str(d):
        if d == depth - 1:
            return f"(!({avoids[d]}) U {progresses[d]})"
        return f"(!({avoids[d]}) U ({progresses[d]} & {build_str(d+1)}))"

    recon_str = build_str(0)
    
    # Initial state: Only the outermost formula is active
    active_pointers = jnp.zeros(depth, dtype=bool).at[0].set(True)
    
    initial_state = FormulaState(
        active_pointers=active_pointers,
        to_avoid=to_avoid_arr,
        to_progress=to_progress_arr
    )
    
    return initial_state, depth, recon_str

def props_to_bool_array(props_list: List[str]) -> jnp.ndarray:
    """Converts a list of prop strings like ['c'] to a boolean array."""
    arr = jnp.zeros(NUM_PROPS, dtype=bool)
    for p in props_list:
        arr = arr.at[PROP_TO_INT[p]].set(True)
    return arr

def format_state(
    state: FormulaState,
    is_true: jnp.ndarray,
    is_false: jnp.ndarray,
    depth: int
) -> str:
    """Formats the output state to match the log's 'true simplification'."""
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
        # This should be caught by is_false, but as a fallback
        return "False"
        
    return " | ".join(active_formulas)

# --- Main execution ---
def run_log_verification():
    """
    Runs a verification against the progression steps provided in the user log.
    """
    
    # A list of (formula_string, list_of_progress_steps)
    # Parsed from the user's log
    log_data = [
        (
            "(!(f) U (c & (!(i) U (e & (!(b) U j)))))",
            [['c'], ['e'], ['j']]
        ),
        (
            "(!(e) U (c & (!(b) U (k & (!(h) U g)))))",
            [['e']]
        ),
        (
            "(!(i) U (l & (!(e) U (g & (!(j) U h)))))",
            [['j'], ['i']]
        ),
        (
            "(!(d) U (i & (!(b) U (g & (!(e) U h)))))",
            [['g'], ['d']]
        ),
        (
            "(!(c) U (i & (!(d) U (j & (!(b) U f)))))",
            [['c']]
        ),
        (
            "(!(j) U (b & (!(k) U (a & (!(d) U f)))))",
            [['b'], ['f'], ['a'], ['d'], ['j'], ['k']]
        ),
        (
            "(!(a) U (e & (!(c) U (f & (!(k) U l)))))",
            [['a']]
        ),
        (
            "(!(k) U (e & (!(l) U (f & (!(g) U a)))))",
            [['f'], ['k']]
        ),
        (
            "(!(l) U (e & (!(i) U (k & (!(d) U g)))))",
            [['l']]
        ),
        (
            "(!(f) U (a & (!(k) U (h & (!(l) U j)))))",
            [['k'], ['h'], ['l'], ['a'], ['j'], ['f'], ['j'], ['l'], ['k']]
        ),
        (
            "(!(e) U (b & (!(h) U (a & (!(j) U d)))))):", # Cleaned trailing ':'
            [['d'], ['h'], ['a'], ['e']]
        ),
        (
            "(!(j) U (l & (!(e) U (g & (!(c) U a)))))",
            [['j']]),
        (
            "(!(l) U (f & (!(i) U (h & (!(b) U c)))))",
            [['l']]
        ),
        (
            "(!(e) U (d & (!(b) U (h & (!(l) U a)))))",
            [['l'], ['e']]
        ),
        (
            "(!(k) U (d & (!(b) U (j & (!(a) U f)))))",
            [['d'], ['j'], ['k'], ['a'], ['f'], ['b']]
        ),
        (
            "(!(j) U (f & (!(g) U (l & (!(d) U h)))))",
            [['j']]
        ),
        (
            "(!(j) U (k & (!(l) U (g & (!(h) U a)))))",
            [['h'], ['a'], ['l'], ['k'], ['g'], ['j'], ['h'], ['g'], ['l'], ['a']]
        ),
        (
            "(!(b) U (i & (!(h) U (d & (!(g) U e)))))",
            [['g'], ['i'], ['b'], ['e'], ['d'], ['h'], ['e']]
        ),
    ]

    for formula_str_raw, steps in log_data:
        formula_str = formula_str_raw.strip().replace(":", "")
        try:
            initial_state, depth, recon_str = parse_formula_log(formula_str)
        except Exception as e:
            print(f"--- 🚫 Skipping Formula (Parse Error: {e}): {formula_str} ---")
            continue
            
        print(f"\n--- 🔄 Processing New Formula ( {recon_str} ): ---")
        
        # Initial state is now returned by the parser
        current_state = initial_state

        for step_props in steps:
            print(f"    -> Progressing with {step_props}:")
            
            # Create the boolean proposition array for this step
            current_props = props_to_bool_array(step_props)
            
            # Run the jitted progression function
            new_state, is_true, is_false = progress_formula(
                current_state, current_props
            )
            
            # Format the output to match the log
            simplification = format_state(
                new_state, is_true, is_false, depth
            )
            
            print(f"        true simplification :  {simplification}")

            # The new state becomes the current state for the next step
            current_state = new_state
            
            # If formula becomes True or False, stop progressing this one
            if is_true or is_false:
                break
        
        print("    ")


if __name__ == "__main__":
    run_log_verification()