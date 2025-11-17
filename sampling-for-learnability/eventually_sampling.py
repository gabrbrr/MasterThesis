import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial

@jax.jit
def _check_proposition_implication(p1_i, p2_i, p1_j, p2_j):
    """
    Checks if (p1_i | p2_i) => (p1_j | p2_j).
    
    This is true if the set of propositions {p1_i, p2_i} (ignoring 0s)
    is a subset of {p1_j, p2_j} (ignoring 0s).
    """
    
    # Check if p1_i is in the j-set {p1_j, p2_j}
    # p1_i is "ok" if it's 0 (inactive) or if it's present in j.
    p1_i_in_j = (p1_i == p1_j) | (p1_i == p2_j)
    p1_i_ok = (p1_i == 0) | p1_i_in_j

    # Check if p2_i is in the j-set {p1_j, p2_j}
    # p2_i is "ok" if it's 0 (inactive) or if it's present in j.
    p2_i_in_j = (p2_i == p1_j) | (p2_i == p2_j)
    p2_i_ok = (p2_i == 0) | p2_i_in_j

    # Both must be satisfied for the subset relationship to hold.
    return p1_i_ok & p2_i_ok


@partial(jax.jit, static_argnames=['max_depth'])
def _check_conjunct_implication(conj_i, conj_j, max_depth):
    """
    Checks if conjunct C_i (implier) implies C_j (implied).
    
    This uses a 'subset-subsequence' check.
    
    Args:
        conj_i (jnp.ndarray): Shape (2, max_depth), the implier conjunct.
        conj_j (jnp.ndarray): Shape (2, max_depth), the implied conjunct.
        max_depth (int): Static argument for loop bounds.
        
    Returns:
        bool: True if C_i implies C_j.
    """

    # We scan over the depths of C_j (the "implied" sequence).
    # The carry `i_idx_carry` is the depth we are searching *from* in C_i.
    
    def scan_body(i_idx_carry, j_depth):
        # Get the propositions for the current depth of C_j
        p1_j = conj_j[0, j_depth]
        p2_j = conj_j[1, j_depth]
        
        # Is this depth of C_j active? (i.e., p1_j > 0)
        is_j_active = p1_j > 0
        
        # --- Inner loop: Find a matching i-depth ---
        # We need to find the *first* depth k >= i_idx_carry in C_i
        # such that C_i[k] is active and implies C_j[j_depth].
        
        def find_i_loop_cond(val):
            # val is (k, found_match)
            k, found_match = val
            # Continue looping if we're still within bounds and haven't found a match
            return (k < max_depth) & ~found_match

        def find_i_loop_body(val):
            k, found_match = val
            p1_i = conj_i[0, k]
            p2_i = conj_i[1, k]
            
            # Is this depth of C_i active?
            is_i_active = p1_i > 0
            
            # Check for proposition-level implication
            i_implies_j = _check_proposition_implication(p1_i, p2_i, p1_j, p2_j)
            
            # We have a match if C_i[k] is active AND it implies C_j[j_depth]
            is_match = is_i_active & i_implies_j
            
            # Return the next k and whether we found a match
            return (k + 1, is_match)

        # Start the inner-loop search from `i_idx_carry`
        initial_val = (i_idx_carry, jnp.array(False))
        
        # `final_k` will be the (k_of_match + 1) or max_depth
        # `final_found` will be True if a match was found
        final_k, final_found = jax.lax.while_loop(
            find_i_loop_cond, 
            find_i_loop_body, 
            initial_val
        )
        
        # This j_depth is satisfied if:
        # 1. It was not active in the first place (~is_j_active)
        # 2. It was active, and we found a matching i_depth (final_found)
        j_level_satisfied = ~is_j_active | final_found
        
        # The new carry for the *next* scan iteration is `final_k`.
        # This ensures our subsequence search in C_i only moves forward.
        # The output for this scan step is whether this j_level was satisfied.
        return final_k, j_level_satisfied

    # Initial carry: Start searching C_i from depth 0
    init_carry = 0 
    j_depths = jnp.arange(max_depth)
    
    # Run the scan over all of C_j's depths
    (final_i_idx, all_j_levels_satisfied) = jax.lax.scan(
        scan_body, init_carry, j_depths
    )
    
    # The implication holds only if *all* of C_j's depths were satisfied.
    return jnp.all(all_j_levels_satisfied)


# --- Main Simplification Function ---

@partial(jax.jit, static_argnames=['max_depth', 'max_conjuncts'])
def simplify_conjuncts(formula_matrix, max_depth, max_conjuncts):
    """
    Simplifies the formula matrix by removing redundant conjuncts.
    
    A conjunct C_j is removed if any *other* active conjunct C_i implies it.
    
    Args:
        formula_matrix (jnp.ndarray): Shape (2, max_depth, max_conjuncts).
        max_depth (int): Static argument.
        max_conjuncts (int): Static argument.
        
    Returns:
        jnp.ndarray: A new formula matrix with redundant conjuncts zeroed out.
    """
    
    # --- 1. Vmap the conjunct implication check ---
    
    # We want to run _check_conjunct_implication for all pairs (i, j).
    # We will vmap over the *last* axis (axis=2) of two matrices.
    vmapped_implies = jax.vmap(
        _check_conjunct_implication, 
        in_axes=(2, 2, None), # (conj_i, conj_j, max_depth)
        out_axes=0
    )
    
    # --- 2. Create all-pairs (i, j) ---
    
    # We need to compare every conjunct with every other conjunct.
    i_indices = jnp.arange(max_conjuncts)
    j_indices = jnp.arange(max_conjuncts)
    
    # Create all (i, j) pairs
    # i_pairs = [0, 0, ..., 1, 1, ..., N, N, ...]
    i_pairs = jnp.repeat(i_indices, max_conjuncts)
    # j_pairs = [0, 1, ..., N, 0, 1, ..., N, ...]
    j_pairs = jnp.tile(j_indices, max_conjuncts)

    # Get the data for all pairs.
    # all_conj_i[..., k] will be the i-conjunct for the k-th pair.
    # all_conj_j[..., k] will be the j-conjunct for the k-th pair.
    # Shape of both is (2, max_depth, max_conjuncts * max_conjuncts)
    all_conj_i = formula_matrix[:, :, i_pairs]
    all_conj_j = formula_matrix[:, :, j_pairs]

    # --- 3. Run the implication check for all pairs ---
    
    # This is a 1D array of shape (max_conjuncts * max_conjuncts,)
    implication_flat = vmapped_implies(all_conj_i, all_conj_j, max_depth)
    
    # Reshape to a 2D matrix: (i, j)
    # implication_matrix[i, j] is True if C_i implies C_j
    implication_matrix = implication_flat.reshape(max_conjuncts, max_conjuncts)
    
    # --- 4. Identify conjuncts to remove ---
    
    # A conjunct `j` is active if its p1 at depth 0 is > 0
    is_active = formula_matrix[0, 0, :] > 0
    
    # Broadcast active masks for i and j
    is_i_active = is_active[:, None]  # Shape (max_conjuncts, 1)
    is_j_active = is_active[None, :]  # Shape (1, max_conjuncts)

    # A conjunct cannot imply itself for removal purposes
    i_not_eq_j = ~jnp.eye(max_conjuncts, dtype=bool)
    
    # `valid_implication[i, j]` is True if:
    # 1. i != j
    # 2. C_i is active
    # 3. C_j is active
    # 4. C_i implies C_j
    valid_implication = i_not_eq_j & is_i_active & is_j_active & implication_matrix
    
    # We want to remove any conjunct `j` (the "implied") if *any* *other*
    # conjunct `i` (the "implier") implies it.
    # We check for `any` along axis 0 (the `i` axis).
    # This gives a 1D mask, True for each `j` that should be removed.
    to_remove_mask_1d = jnp.any(valid_implication, axis=0) # Shape (max_conjuncts,)
    
    # --- 5. Create the new simplified matrix ---
    
    # We want a "keep" mask, which is the opposite
    keep_mask_1d = ~to_remove_mask_1d
    
    # Broadcast the 1D keep_mask to the 3D matrix shape
    keep_mask_broadcast = keep_mask_1d[None, None, :]
    
    # Zero out the conjuncts that are marked for removal
    simplified_matrix = jnp.where(
        keep_mask_broadcast,
        formula_matrix,
        0
    )
    
    return simplified_matrix

@partial(jax.jit, static_argnames=['max_depth'])
def progress_formula(formula_matrix, propositions_true, max_depth):
    """
    Progresses the LTL formula matrix based on the propositions that are true.

    Args:
        formula_matrix (jnp.ndarray): The formula representation, shape (2, max_depth, max_conjuncts).
        propositions_true (jnp.ndarray): Boolean array of shape (num_propositions + 1,).
                                         propositions_true[i] is True if proposition i is true.
                                         propositions_true[0] is always False (padding).
        max_depth (int): Static argument, max depth of a sequence.

    Returns:
        jnp.ndarray: The new formula_matrix after one step of progression.
    """
    # Get the propositions at the current depth (depth 0) for all conjuncts
    p1s = formula_matrix[0, 0, :]  # Shape (max_conjuncts,)
    p2s = formula_matrix[1, 0, :]  # Shape (max_conjuncts,)

    # Check if the conditions are met
    # propositions_true[0] is False, so if p1s or p2s is 0, this will be False.
    p1s_met = propositions_true[p1s]
    p2s_met = propositions_true[p2s]
    
    # A condition is met if it's active (p1s > 0) and either p1 or p2 is true
    active_mask = p1s > 0
    condition_met_mask = active_mask & (p1s_met | p2s_met) # Shape (max_conjuncts,)

    # Create the "progressed" matrix (roll depth dimension up by 1)
    # This simulates F(p & F(...)) -> F(...)
    progressed_matrix = jnp.roll(formula_matrix, shift=-1, axis=1)
    # Zero out the last depth-level after rolling
    progressed_matrix = progressed_matrix.at[:, max_depth - 1, :].set(0)

    # If the condition was not met, the formula remains the same (retained_matrix)
    retained_matrix = formula_matrix

    # Use jnp.where to select between progressed and retained for each conjunct
    # We need to broadcast the 1D mask to the 3D matrix shape
    progress_mask_broadcast = jnp.broadcast_to(
        condition_met_mask[None, None, :], 
        formula_matrix.shape
    )

    new_formula_matrix = jnp.where(
        progress_mask_broadcast,
        progressed_matrix,
        retained_matrix
    )
    
    return new_formula_matrix

def _sample_sequence_jax(key, num_levels, max_depth, num_propositions, disjunction_prob):
    """
    JAX-jittable function to sample a single conjunct sequence using lax.scan.
    This function is vmapped in the main sampler.
    """
    
    def scan_body(carry, d):
        """
        Body of the scan function, samples one level (depth).
        'd' is the current depth index (from 0 to max_depth-1).
        Uses rejection sampling to ensure new props are not in 'last'.
        """
        key, last_p1, last_p2 = carry
        key, k_p1_loop, k_p2_loop, k_disj = random.split(key, 4)

        # 1. Sample p1, ensuring it's not the same as last_p1 or last_p2
        def p1_loop_cond(val):
            # Condition to *continue* looping
            k, p1 = val
            return (p1 == last_p1) | (p1 == last_p2)

        def p1_loop_body(val):
            # Sample a new p1
            k, p1_old = val
            k, subkey = random.split(k)
            p1_new = random.randint(subkey, (), 1, num_propositions + 1)
            return k, p1_new

        # Initialize with dummy value 0 (which will trigger loop if last_p1/p2 is 0)
        # or a valid prop. Rejection sampling handles all cases.
        key, p1_init = p1_loop_body((k_p1_loop, last_p1)) 
        _, p1 = jax.lax.while_loop(p1_loop_cond, p1_loop_body, (key, p1_init))


        # 2. Sample if this is a disjunction
        is_disj = random.bernoulli(k_disj, disjunction_prob)

        # 3. Sample p2, ensuring it's not p1, last_p1, or last_p2
        def p2_loop_cond(val):
            # Condition to *continue* looping
            k, p2 = val
            return (p2 == p1) | (p2 == last_p1) | (p2 == last_p2)

        def p2_loop_body(val):
            # Sample a new p2
            k, p2_old = val
            k, subkey = random.split(k)
            p2_new = random.randint(subkey, (), 1, num_propositions + 1)
            return k, p2_new

        # Initialize with dummy value
        key, p2_init = p2_loop_body((k_p2_loop, p1))
        _, p2 = jax.lax.while_loop(p2_loop_cond, p2_loop_body, (key, p2_init))
        
        # 4. Apply masks
        # Is this depth level active for this conjunct?
        is_active = d < num_levels
        
        final_p1 = jnp.where(is_active, p1, 0)
        final_p2 = jnp.where(is_active & is_disj, p2, 0)
        
        new_carry = (key, final_p1, final_p2) # Pass new "last" props
        output = (final_p1, final_p2)       # The matrix row for this depth
        return new_carry, output

    init_carry = (key, 0, 0) # (key, last_p1, last_p2)
    depth_indices = jnp.arange(max_depth)
    
    # Run the scan over all depths
    (final_key, _, _), (p1s, p2s) = jax.lax.scan(scan_body, init_carry, depth_indices)
    
    # p1s and p2s have shape (max_depth,)
    return jnp.stack([p1s, p2s], axis=0) # Shape (2, max_depth)


@partial(jax.jit, static_argnames=['max_depth', 'max_conjuncts', 'num_propositions', 'disjunction_prob'])
def sample_eventually_jax(
    key, 
    min_levels, max_levels, 
    min_conjunctions, max_conjunctions, 
    max_depth, max_conjuncts, 
    num_propositions, disjunction_prob
):
    """
    JAX-jittable sampler for the LTL formula matrix.
    """
    key, k_conj, k_levels, k_seqs = random.split(key, 4)

    # 1. Sample number of conjuncts
    num_conjs = random.randint(k_conj, (), min_conjunctions, max_conjunctions + 1)
    
    # 2. Create mask for active conjuncts
    conj_mask = jnp.arange(max_conjuncts) < num_conjs
    
    # 3. Sample number of levels (depth) for each *potential* conjunct
    levels = random.randint(k_levels, (max_conjuncts,), min_levels, max_levels + 1)
    
    # 4. Sample all sequences in parallel using vmap
    seq_keys = random.split(k_seqs, max_conjuncts)
    
    # vmap the sequence sampler
    # in_axes = (key, num_levels, max_depth, num_propositions, disjunction_prob)
    vmap_sampler = jax.vmap(
        _sample_sequence_jax, 
        in_axes=(0, 0, None, None, None),
        out_axes=2 # Output shape (2, max_depth, max_conjuncts)
    )
    
    formula_matrix = vmap_sampler(
        seq_keys, levels, max_depth, num_propositions, disjunction_prob
    )
    
    # 5. Apply the conjunct mask
    final_matrix = formula_matrix * conj_mask[None, None, :]
    
    return final_matrix


# --- Non-JAX Encoding/Decoding Functions ---

def encode_formula(formula_tuple, prop_to_int, max_depth, max_conjuncts):
    """
    Encodes a Python tuple representation of a formula into a NumPy matrix.
    Not JAX-jittable.
    """
    matrix = np.zeros((2, max_depth, max_conjuncts), dtype=np.int32)
    
    # 1. Collect top-level conjuncts
    conjuncts = []
    def collect_conjuncts(f):
        if not f:
            return
        if f[0] == 'and':
            collect_conjuncts(f[1])
            collect_conjuncts(f[2])
        else:
            conjuncts.append(f)
            
    collect_conjuncts(formula_tuple)
    
    # 2. Iterate over conjuncts and fill matrix
    for c_idx, task in enumerate(conjuncts):
        if c_idx >= max_conjuncts:
            break # Too many conjuncts for our matrix
        
        d_idx = 0
        current_task = task
        
        # 3. Unroll the 'eventually (and ...)' structure
        while current_task and current_task[0] == 'eventually' and d_idx < max_depth:
            term = current_task[1]
            
            if term[0] == 'and':
                prop_part = term[1]
                current_task = term[2] # This is the next 'eventually'
            else:
                prop_part = term       # This is the last term in the sequence
                current_task = None    # Stop unrolling
            
            # 4. Encode the proposition part (atom or disjunction)
            if isinstance(prop_part, str):
                if prop_part in prop_to_int:
                    matrix[0, d_idx, c_idx] = prop_to_int[prop_part]
                    matrix[1, d_idx, c_idx] = 0
            elif prop_part[0] == 'or':
                if prop_part[1] in prop_to_int:
                    matrix[0, d_idx, c_idx] = prop_to_int[prop_part[1]]
                if prop_part[2] in prop_to_int:
                    matrix[1, d_idx, c_idx] = prop_to_int[prop_part[2]]
            
            d_idx += 1
            
    return jnp.array(matrix)


def decode_formula(formula_matrix, int_to_prop):
    """
    Decodes a formula matrix (NumPy or JAX array) back into a Python tuple.
    Not JAX-jittable.
    """
    conjuncts = []
    # Ensure we're working with NumPy for easy iteration
    matrix = np.asarray(formula_matrix)
    max_depth, max_conjuncts = matrix.shape[1], matrix.shape[2]
    
    for c_idx in range(max_conjuncts):
        # 1. Check if this conjunct is active
        if matrix[0, 0, c_idx] == 0:
            continue
            
        # 2. Read the sequence of propositions for this conjunct
        seq_parts = []
        for d_idx in range(max_depth):
            p1_int = matrix[0, d_idx, c_idx]
            p2_int = matrix[1, d_idx, c_idx]
            
            if p1_int == 0:
                break # End of this sequence
            
            if p2_int == 0:
                # Single proposition
                term = int_to_prop.get(p1_int, f"P{p1_int}?")
            else:
                # Disjunction
                term = ('or', 
                        int_to_prop.get(p1_int, f"P{p1_int}?"), 
                        int_to_prop.get(p2_int, f"P{p2_int}?")
                       )
            seq_parts.append(term)
        
        if not seq_parts:
            continue
            
        # 3. Rebuild the nested 'eventually' structure from the sequence
        # (This logic is from your _get_sequence helper)
        task = None
        for term in reversed(seq_parts):
            if task is None:
                # This is the innermost 'eventually'
                task = ('eventually', term)
            else:
                # This is an outer 'eventually'
                task = ('eventually', ('and', term, task))
        
        if task:
            conjuncts.append(task)
            
    # 4. Combine all conjuncts with 'and'
    if not conjuncts:
        return True 
        
    # 4. Combine all conjuncts with 'and' in a consistent order
    #    (Builds (and, C1, (and, C2, C3)))
    ltl = conjuncts[-1]
    for i in range(len(conjuncts) - 2, -1, -1):
        ltl = ('and', conjuncts[i], ltl)
        
    return ltl


# --- Example Usage ---
if __name__ == "__main__":
    
    # --- Setup ---
    PROPOSITIONS = ['a', 'b', 'c', 'd']
    PROP_TO_INT = {prop: i + 1 for i, prop in enumerate(PROPOSITIONS)}
    INT_TO_PROP = {i + 1: prop for i, prop in enumerate(PROPOSITIONS)}
    NUM_PROPOSITIONS = len(PROPOSITIONS)

    # Add 0-mapping for convenience (maps to False)
    INT_TO_PROP[0] = "NULL" 
    
    MAX_DEPTH = 5
    MAX_CONJUNCTS = 5
    
    # JIT the progress function for our specific max_depth
    jitted_progress = partial(progress_formula, max_depth=MAX_DEPTH)

    # --- 1. Test Encoding and Decoding ---
    print("--- 1. Test Encoding/Decoding ---")
    
    # (eventually (a and eventually c)) and (eventually b) and (eventually (d or a))
    formula_tuple = ('and',
        ('eventually', ('and', 'a', ('eventually', 'c'))),
        ('and',
            ('eventually', 'b'),
            ('eventually', ('or', 'd', 'a'))
        )
    )
    
    print(f"Original Tuple:\n{formula_tuple}\n")
    encoded_matrix = encode_formula(formula_tuple, PROP_TO_INT, MAX_DEPTH, MAX_CONJUNCTS)
    print(f"Encoded Matrix (shape={encoded_matrix.shape}):\n{encoded_matrix}\n")
    
    decoded_tuple = decode_formula(encoded_matrix, INT_TO_PROP)
    print(f"Decoded Tuple:\n{decoded_tuple}\n")
    
    # Test for consistency
    assert str(decoded_tuple) == str(formula_tuple) # Simple way to check nested tuple equality
    print("Encode/Decode successful.\n")

    # --- 2. Test Progression ---
    print("--- 2. Test Progression ---")
    
    # Start with the same formula: F(a & F(c)) & F(b) & F(d | a)
    # Re-order for matrix: [F(a & F(c)), F(b), F(d | a), <empty>]
    # Matrix P1: [[1, 2, 4, 0], [3, 0, 0, 0], [0, 0, 0, 0], ...]
    # Matrix P2: [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], ...]
    current_matrix = encoded_matrix
    print(f"Start: {decode_formula(current_matrix, INT_TO_PROP)}")

    # Step 1: 'a' is true.
    # F(a & F(c)) -> progresses to F(c)
    # F(b) -> no progress
    # F(d | a) -> progresses to <True> (empty)
    props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT['a']].set(True)
    current_matrix = jitted_progress(current_matrix, props_true)
    current_matrix=simplify_conjuncts(current_matrix,MAX_DEPTH,MAX_CONJUNCTS)
    print(f"After 'a': {current_matrix}")
    print(f"After 'a': {decode_formula(current_matrix, INT_TO_PROP)}")
    # Expected: F(c) & F(b)

    # Step 2: 'b' is true.
    # F(c) -> no progress
    # F(b) -> progresses to <True> (empty)
    props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT['b']].set(True)
    current_matrix = jitted_progress(current_matrix, props_true)
    current_matrix=simplify_conjuncts(current_matrix,MAX_DEPTH,MAX_CONJUNCTS)
    print(f"After 'b': {current_matrix}")
    print(f"After 'b': {decode_formula(current_matrix, INT_TO_PROP)}")
    # Expected: F(c)

    # Step 3: 'c' is true.
    # F(c) -> progresses to <True> (empty)
    props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT['c']].set(True)
    current_matrix = jitted_progress(current_matrix, props_true)
    current_matrix=simplify_conjuncts(current_matrix,MAX_DEPTH,MAX_CONJUNCTS)
    print(f"After 'c': {current_matrix}")
    print(f"After 'c': {decode_formula(current_matrix, INT_TO_PROP)}")
    # Expected: None (all conjuncts satisfied)
    print("\nProgression successful.\n")

    # --- 3. Test JAX Sampler ---
    print("--- 3. Test JAX Sampler ---")
    
    # JIT the sampler with static args
    jitted_sampler = partial(sample_eventually_jax,
        max_depth=MAX_DEPTH,
        max_conjuncts=MAX_CONJUNCTS,
        num_propositions=NUM_PROPOSITIONS,
        disjunction_prob=0.25
    )
    
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    sampled_matrix = jitted_sampler(
        subkey,
        min_levels=1, max_levels=4,
        min_conjunctions=2, max_conjunctions=3
    )
    
    print(f"Sampled Matrix (shape={sampled_matrix.shape}):\n{sampled_matrix}\n")
    
    decoded_sample = decode_formula(sampled_matrix, INT_TO_PROP)
    print(f"Decoded Sample:\n{decoded_sample}\n")