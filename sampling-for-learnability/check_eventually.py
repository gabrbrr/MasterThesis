


import  spot
import logging
import random
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial




def canonical_form(formula):
    """
    Recursively converts an LTL formula tuple into a canonical form.
    Applies commutativity AND associativity for 'and'/'or'.
    - Flattens nested 'and'/'or' operators.
    - Sorts operands.
    - Rebuilds as a left-associative chain.
    """
    # Base case: Proposition (string)
    if not isinstance(formula, tuple):
        return formula
    
    op = formula[0]
    operands = formula[1:]
    
    # Recursively canonicalize all operands first
    canonical_operands = [canonical_form(opnd) for opnd in operands]
    
    if op in ('and', 'or'):
        # --- Associativity Step: Flatten (FIXED) ---
        # This now handles arbitrarily deep nesting, e.g., (A | (B | (C | D)))
        flattened_ops = []
        queue = list(canonical_operands) # Start with the immediate operands

        while queue:
            opnd = queue.pop(0) # Get the next item to check
            
            if isinstance(opnd, tuple) and opnd[0] == op:
                # It's a nested op of the same type.
                # Add its children to the queue to be processed.
                queue.extend(opnd[1:])
            else:
                # It's a base case (a different op, a str, etc.)
                # Add it to our final flat list.
                flattened_ops.append(opnd)
        
        # --- Commutativity Step: Sort ---
        def sort_key(item):
            # Sort by type (str vs tuple) first, then by string representation
            is_tuple = isinstance(item, tuple)
            return (is_tuple, str(item))
        
        sorted_ops = sorted(flattened_ops, key=sort_key)
        
        # --- Rebuild as a left-associative chain ---
        if not sorted_ops:
            return 'True' if op == 'and' else 'False'
        
        result = sorted_ops[0]
        for i in range(1, len(sorted_ops)):
            result = (op, result, sorted_ops[i])
            
        return result

    elif op in ('not', 'eventually', 'always', 'next'):
        # Unary operators
        return (op, canonical_operands[0])
    
    elif op == 'until':
        # Binary, non-commutative
        return (op, canonical_operands[0], canonical_operands[1])
    
    else:
        # Unknown operator
        return (op,) + tuple(canonical_operands)

def progress_and_clean(ltl_formula, truth_assignment):
    ltl = progress(ltl_formula, truth_assignment)
    # I am using spot to simplify the resulting ltl formula
    ltl_spot = _get_spot_format(ltl)
    f = spot.formula(ltl_spot)
    f = spot.simplify(f)
    ltl_spot = f.__format__("l")
    ltl_std,r = _get_std_format(ltl_spot.split(' '))
    assert len(r) == 0, "Format error" + str(ltl_std) + " " + str(r)
    return ltl_std


def spotify(ltl_formula):
    ltl_spot = _get_spot_format(ltl_formula)
    f = spot.formula(ltl_spot)
    f = spot.simplify(f)
    ltl_spot = f.__format__("l")
    # return ltl_spot
    return f#.to_str('latex')


def _get_spot_format(ltl_std):
    ltl_spot = str(ltl_std).replace("(","").replace(")","").replace(",","")
    ltl_spot = ltl_spot.replace("'until'","U").replace("'not'","!").replace("'or'","|").replace("'and'","&")
    ltl_spot = ltl_spot.replace("'next'","X").replace("'eventually'","F").replace("'always'","G").replace("'True'","t").replace("'False'","f").replace("\'","\"")
    return ltl_spot

def _get_std_format(ltl_spot):

    s = ltl_spot[0]
    r = ltl_spot[1:]

    if s in ["X","U","&","|"]:
        v1,r1 = _get_std_format(r)
        v2,r2 = _get_std_format(r1)
        if s == "X": op = 'next'
        if s == "U": op = 'until'
        if s == "&": op = 'and'
        if s == "|": op = 'or'
        return (op,v1,v2),r2

    if s in ["F","G","!"]:
        v1,r1 = _get_std_format(r)
        if s == "F": op = 'eventually'
        if s == "G": op = 'always'
        if s == "!": op = 'not'
        return (op,v1),r1

    if s == "f":
        return 'False', r

    if s == "t":
        return 'True', r

    if s[0] == '"':
        return s.replace('"',''), r

    assert False, "Format error in spot2std"

def progress(ltl_formula, truth_assignment):
    if type(ltl_formula) == str:
        # True, False, or proposition
        if len(ltl_formula) == 1:
            # ltl_formula is a proposition
            if ltl_formula in truth_assignment:
                return 'True'
            else:
                return 'False'
        return ltl_formula

    if ltl_formula[0] == 'not':
        # negations should be over propositions only according to the cosafe ltl syntactic restriction
        result = progress(ltl_formula[1], truth_assignment)
        if result == 'True':
            return 'False'
        elif result == 'False':
            return 'True'
        else:
            raise NotImplementedError("The following formula doesn't follow the cosafe syntactic restriction: " + str(ltl_formula))

    if ltl_formula[0] == 'and':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)
        if res1 == 'True' and res2 == 'True': return 'True'
        if res1 == 'False' or res2 == 'False': return 'False'
        if res1 == 'True': return res2
        if res2 == 'True': return res1
        if res1 == res2:   return res1
        #if _subsume_until(res1, res2): return res2
        #if _subsume_until(res2, res1): return res1
        return ('and',res1,res2)

    if ltl_formula[0] == 'or':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)
        if res1 == 'True'  or res2 == 'True'  : return 'True'
        if res1 == 'False' and res2 == 'False': return 'False'
        if res1 == 'False': return res2
        if res2 == 'False': return res1
        if res1 == res2:    return res1
        #if _subsume_until(res1, res2): return res1
        #if _subsume_until(res2, res1): return res2
        return ('or',res1,res2)

    if ltl_formula[0] == 'next':
        return progress(ltl_formula[1], truth_assignment)

    if ltl_formula[0] == 'eventually':
        res = progress(ltl_formula[1], truth_assignment)
        return ("or", res, ltl_formula)

    if ltl_formula[0] == 'always':
        res = progress(ltl_formula[1], truth_assignment)
        return ("and", ltl_formula, res)

    if ltl_formula[0] == 'until':
        res1 = progress(ltl_formula[1], truth_assignment)
        res2 = progress(ltl_formula[2], truth_assignment)

        if res1 == 'False':
            f1 = 'False'
        elif res1 == 'True':
            f1 = ('until', ltl_formula[1], ltl_formula[2])
        else:
            f1 = ('and', res1, ('until', ltl_formula[1], ltl_formula[2]))

        if res2 == 'True':
            return 'True'
        if res2 == 'False':
            return f1

        # Returning ('or', res2, f1)
        #if _subsume_until(f1, res2): return f1
        #if _subsume_until(res2, f1): return res2
        return ('or', res2, f1)





import random

class LTLSampler():
    def __init__(self, propositions):
        self.propositions = propositions

    def sample(self):
        raise NotImplementedError



class UntilTaskSampler(LTLSampler):
    def __init__(self, propositions, min_levels=1, max_levels=2, min_conjunctions=1 , max_conjunctions=2):
        super().__init__(propositions)
        self.levels       = (int(min_levels), int(max_levels))
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        assert 2*int(max_levels)*int(max_conjunctions) <= len(propositions), "The domain does not have enough propositions!"

    def sample(self):
        # Sampling a conjuntion of *n_conjs* (not p[0]) Until (p[1]) formulas of *n_levels* levels
        n_conjs = random.randint(*self.conjunctions)
        p = random.sample(self.propositions,2*self.levels[1]*n_conjs)
        ltl = None
        b = 0
        for i in range(n_conjs):
            n_levels = random.randint(*self.levels)
            # Sampling an until task of *n_levels* levels
            until_task = ('until',('not',p[b]),p[b+1])
            b +=2
            for j in range(1,n_levels):
                until_task = ('until',('not',p[b]),('and', p[b+1], until_task))
                b +=2
            # Adding the until task to the conjunction of formulas that the agent have to solve
            if ltl is None: ltl = until_task
            else:           ltl = ('and',until_task,ltl)
        return ltl


class EventuallySampler(LTLSampler):
    def __init__(self, propositions, min_levels = 1, max_levels=4, min_conjunctions=1, max_conjunctions=3):
        super().__init__(propositions)
        assert(len(propositions) >= 3)
        self.conjunctions = (int(min_conjunctions), int(max_conjunctions))
        self.levels = (int(min_levels), int(max_levels))

    def sample(self):
        conjs = random.randint(*self.conjunctions)
        ltl = None

        for i in range(conjs):
            task = self.sample_sequence()
            if ltl is None:
                ltl = task
            else:
                ltl = ('and',task,ltl)
        return ltl


    def sample_sequence(self):
        length = random.randint(*self.levels)
        seq = []

        last = []
        while len(seq) < length:
            # Randomly replace some propositions with a disjunction to make more complex formulas
            population = [p for p in self.propositions if p not in last]

            if random.random() < 0.25:
                c = random.sample(population, 2)
            else:
                c = random.sample(population, 1)

            seq.append(c)
            last = c

        ret = self._get_sequence(seq)

        return ret

    def _get_sequence(self, seq):
        term = seq[0][0] if len(seq[0]) == 1 else ('or', seq[0][0], seq[0][1])
        if len(seq) == 1:
            return ('eventually',term)
        return ('eventually',('and', term, self._get_sequence(seq[1:])))


    
def get_propositions_in_formula(ltl_formula):
    """
    Recursively finds all unique propositions (atomic strings)
    in a formula represented as a nested tuple.
    """
    props = set()

    if isinstance(ltl_formula, str):
        # Is a string, check if it's a proposition
        if ltl_formula != 'True' and ltl_formula != 'False':
            props.add(ltl_formula)
        return props

    if isinstance(ltl_formula, tuple):
        # Is a tuple, e.g., ('and', 'a', 'b')
        # The first element is the operator, skip it.
        # Recursively check all other elements in the tuple.
        for sub_formula in ltl_formula[1:]:
            props.update(get_propositions_in_formula(sub_formula))

    return props


def ltl_tuple_to_string(ltl_tuple):
    """
    Recursively converts a nested LTL tuple into a human-readable string.

    Args:
        ltl_tuple: The LTL formula represented as a string (for atomic
                   propositions) or a tuple (for logical operators).

    Returns:
        A string representation of the LTL formula.
    """
    
    # Base Case: If the input is not a tuple, it's an atomic proposition (string).
    if not isinstance(ltl_tuple, tuple):
        return str(ltl_tuple)

    # Recursive Step: The input is a tuple, so process the operator.
    operator = ltl_tuple[0]
    
    # --- Unary Operators (1 operand) ---
    if operator == 'not':
        # Format: !(operand)
        operand_str = ltl_tuple_to_string(ltl_tuple[1])
        return f"!({operand_str})"
        
    elif operator == 'eventually':
        # Format: F(operand)
        operand_str = ltl_tuple_to_string(ltl_tuple[1])
        return f"F({operand_str})"
        
    elif operator == 'globally':
        # Format: G(operand)
        operand_str = ltl_tuple_to_string(ltl_tuple[1])
        return f"G({operand_str})"
        
    elif operator == 'next':
        # Format: X(operand)
        operand_str = ltl_tuple_to_string(ltl_tuple[1])
        return f"X({operand_str})"

    # --- Binary Operators (2 operands) ---
    elif operator == 'and':
        # Format: (left & right)
        left_str = ltl_tuple_to_string(ltl_tuple[1])
        right_str = ltl_tuple_to_string(ltl_tuple[2])
        return f"({left_str} & {right_str})"
        
    elif operator == 'or':
        # Format: (left | right)
        left_str = ltl_tuple_to_string(ltl_tuple[1])
        right_str = ltl_tuple_to_string(ltl_tuple[2])
        return f"({left_str} | {right_str})"
        
    elif operator == 'until':
        # Format: (left U right)
        left_str = ltl_tuple_to_string(ltl_tuple[1])
        right_str = ltl_tuple_to_string(ltl_tuple[2])
        return f"({left_str} U {right_str})"
        
    # Fallback for any unknown operator
    else:
        # Just return the tuple as a string if the operator is not recognized
        return str(ltl_tuple)

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
def progress_jax(formula_matrix, propositions_true, max_depth):
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
    is_true = jnp.all(new_formula_matrix[0, 0, :] == 0)
    
    return new_formula_matrix, is_true, jnp.array(False)


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

class JaxEventuallyTaskSampler:
    """
    JAX-based sampler for LTL 'Eventually' formulas.

    This class stores the static sampling configuration and provides a
    JIT-compiled `sample` method to generate formula matrices.

    The formula structure is a conjunction of 'Eventually' clauses,
    where each clause can be nested and include disjunctions.
    """

    def __init__(
        self,
        num_propositions: int,
        min_levels: int,
        max_levels: int,
        min_conjunctions: int,
        max_conjunctions: int,
        max_depth: int,
        max_conjuncts: int,
        disjunction_prob: float = 0.3
    ):
        """
        Initializes the sampler with static configuration.

        Args:
            num_propositions: Number of available atomic propositions (p1, p2...).
            min_levels: Minimum nesting depth (F) for a conjunct.
            max_levels: Maximum nesting depth (F) for a conjunct.
            min_conjunctions: Minimum number of conjuncts (G(...)) in the formula.
            max_conjunctions: Maximum number of conjuncts (G(...)) in the formula.
            max_depth: Static max depth for the formula matrix (for JIT).
            max_conjuncts: Static max conjuncts for the formula matrix (for JIT).
            disjunction_prob: Probability of a clause being a disjunction F(p1 | p2).
        """
        # Store all configuration parameters as instance attributes
        self.num_propositions = num_propositions
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.min_conjunctions = min_conjunctions
        self.max_conjunctions = max_conjunctions
        self.max_depth = max_depth
        self.max_conjuncts = max_conjuncts
        self.disjunction_prob = disjunction_prob
   
    @staticmethod
    def _sample_sequence_jax(self, key, num_levels, max_depth, num_propositions, disjunction_prob):
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
    
    
    @partial(jax.jit, static_argnames=['self'])
    def sample(self,
        key,
    ):
        """
        JAX-jittable sampler for the LTL formula matrix.
        """
        key, k_conj, k_levels, k_seqs = random.split(key, 4)
    
        # 1. Sample number of conjuncts
        num_conjs = random.randint(k_conj, (), self.min_conjunctions, self.max_conjunctions + 1)
        
        # 2. Create mask for active conjuncts
        conj_mask = jnp.arange(self.max_conjuncts) < num_conjs
        
        # 3. Sample number of levels (depth) for each *potential* conjunct
        levels = random.randint(k_levels, (self.max_conjuncts,), self.min_levels, self.max_levels + 1)
        
        # 4. Sample all sequences in parallel using vmap
        seq_keys = random.split(k_seqs, self.max_conjuncts)
        
        # vmap the sequence sampler
        # in_axes = (key, num_levels, max_depth, num_propositions, disjunction_prob)
        vmap_sampler = jax.vmap(
            self._sample_sequence_jax, 
            in_axes=(0, 0, None, None, None),
            out_axes=2 # Output shape (2, max_depth, max_conjuncts)
        )
        
        formula_matrix = vmap_sampler(
            seq_keys, levels, self.max_depth, self.num_propositions, self.disjunction_prob
        )
        
        # 5. Apply the conjunct mask
        final_matrix = formula_matrix * conj_mask[None, None, :]
        
        return final_matrix

OP_PAD = 0
OP_AND = 1
OP_OR = 2
OP_EVENTUALLY = 3
OP_NEXT = 4
OP_ALWAYS = 5
OP_UNTIL = 6
OP_NOT = 7
OP_TRUE = 8
OP_FALSE = 9
# Propositions will be mapped to indices starting from PROP_START_IDX
PROP_START_IDX = 10

# Edge types
EDGE_SELF = 0
EDGE_ARG = 1  # For unary operators
EDGE_ARG1 = 2 # For binary operators, left
EDGE_ARG2 = 3 # For binary operators, right

# --- 2. State Definition ---

# This holds the state of our graph as it's being built.
# We use a namedtuple as it is a JAX-compatible pytree.
GraphState = namedtuple('GraphState', [
    'nodes',       # (max_nodes, feature_dim + 1) array
    'senders',     # (max_edges,) array
    'receivers',   # (max_edges,) array
    'edge_types',  # (max_edges,) array
    'n_node',      # int counter for nodes
    'n_edge'       # int counter for edges
])

# --- 3. Jittable Helper Functions ---

@jax.jit
def _add_node(b_state: GraphState, feat_idx: int, is_root: float, feature_map: jnp.ndarray) -> (GraphState, int):
    """
    Adds a new node and its self-loop, returning the updated state and the new node's index.
    The last column of the feature_map is assumed to be for the 'is_root' feature.
    """
    # Get the index for the new node
    idx = b_state.n_node
    
    # Get features from the map and set the is_root flag
    node_feature = feature_map[feat_idx]
    
    # We assume the 'nodes' array has shape (max_nodes, feature_dim + 1)
    # where the last column is 'is_root'
    full_node_feature = node_feature.at[-1].set(is_root)
    
    # Add the node to the array
    nodes = b_state.nodes.at[idx].set(full_node_feature)
    
    # Increment node count
    n_node = idx + 1
    
    # Add the required self-loop (as per original ASTBuilder)
    edge_idx = b_state.n_edge
    senders = b_state.senders.at[edge_idx].set(idx)
    receivers = b_state.receivers.at[edge_idx].set(idx)
    edge_types = b_state.edge_types.at[edge_idx].set(EDGE_SELF)
    n_edge = edge_idx + 1
    
    # Return updated state and the new node's index
    new_state = b_state._replace(
        nodes=nodes, 
        senders=senders, 
        receivers=receivers, 
        edge_types=edge_types, 
        n_node=n_node, 
        n_edge=n_edge
    )
    return new_state, idx

@jax.jit
def _add_edge(b_state: GraphState, sender_idx: int, receiver_idx: int, edge_type: int) -> GraphState:
    """Adds a single directed edge to the graph state."""
    idx = b_state.n_edge
    
    senders = b_state.senders.at[idx].set(sender_idx)
    receivers = b_state.receivers.at[idx].set(receiver_idx)
    edge_types = b_state.edge_types.at[idx].set(edge_type)
    n_edge = idx + 1
    
    return b_state._replace(
        senders=senders, 
        receivers=receivers, 
        edge_types=edge_types, 
        n_edge=n_edge
    )

# --- 4. Core Jittable Logic ---

@jax.jit
def _build_term(b_state: GraphState, p1: int, p2: int, feature_map: jnp.ndarray) -> (GraphState, int):
    """
    Builds the sub-graph for a term (e.g., 'p1' or '(or p1 p2)')
    Returns (updated_state, root_index_of_term_subgraph)
    """
    # Check if this is a disjunction ('or')
    is_or = p2 > 0
    
    # --- Branch 1: Build ('or', p1, p2) ---
    def build_or_branch(state):
        # Add the 'or' node
        state, or_idx = _add_node(state, OP_OR, 0.0, feature_map)
        # Add the 'p1' node
        state, p1_idx = _add_node(state, PROP_START_IDX + p1, 0.0, feature_map)
        # Add the 'p2' node
        state, p2_idx = _add_node(state, PROP_START_IDX + p2, 0.0, feature_map)
        # Add edges
        state = _add_edge(state, p1_idx, or_idx, EDGE_ARG1)
        state = _add_edge(state, p2_idx, or_idx, EDGE_ARG2)
        return state, or_idx

    # --- Branch 2: Build (p1) ---
    def build_prop_branch(state):
        # Add the 'p1' node
        state, p1_idx = _add_node(state, PROP_START_IDX + p1, 0.0, feature_map)
        return state, p1_idx

    # Conditionally execute the correct branch
    return lax.cond(
        is_or,
        build_or_branch,
        build_prop_branch,
        b_state
    )

@partial(jax.jit, static_argnames=['max_depth'])
def _build_conjunct_subtree(conj_c: jnp.ndarray, b_state: GraphState, 
                            feature_map: jnp.ndarray, max_depth: int) -> (GraphState, int):
    """
    Builds the full 'eventually(and(term, ...))' chain for a single conjunct.
    Uses lax.scan, iterating backwards from the innermost term.
    """
    
    def scan_body(carry, d_idx):
        """
        Builds one 'eventually(and(term, child))' layer.
        carry = (GraphState, child_root_idx)
        d_idx = current depth
        """
        b_state_carry, child_idx = carry
        p1 = conj_c[0, d_idx]
        p2 = conj_c[1, d_idx]
        
        is_active = p1 > 0
        
        def build_step(state_child_tuple):
            state, child_root_idx = state_child_tuple
            
            # 1. Build the term (P or P1|P2)
            state, term_root_idx = _build_term(state, p1, p2, feature_map)
            
            # 2. Check if there is a child (from a deeper 'eventually')
            has_child = child_root_idx > 0 # 0 is padding node
            
            # --- Branch 2a: ('and', term, child) ---
            def link_child_branch(state_term_child_idx):
                state_link, term_idx, child_link_idx = state_term_child_idx
                # Add 'and' node
                state_link, and_idx = _add_node(state_link, OP_AND, 0.0, feature_map)
                # Link term
                state_link = _add_edge(state_link, term_idx, and_idx, EDGE_ARG1)
                # Link child
                state_link = _add_edge(state_link, child_link_idx, and_idx, EDGE_ARG2)
                return state_link, and_idx

            # --- Branch 2b: (term) ---
            def no_link_child_branch(state_term_child_idx):
                state_nolink, term_idx, _ = state_term_child_idx
                return state_nolink, term_idx

            # Conditionally create 'and' node
            state, subtree_root_idx = lax.cond(
                has_child,
                link_child_branch,
                no_link_child_branch,
                (state, term_root_idx, child_root_idx)
            )

            # 3. Add the 'eventually' node
            state, eventually_idx = _add_node(state, OP_EVENTUALLY, 0.0, feature_map)
            # Link 'eventually' to its argument
            state = _add_edge(state, subtree_root_idx, eventually_idx, EDGE_ARG)
            
            return state, eventually_idx

        # Only build this layer if it's active
        # Otherwise, pass the carry through unchanged
        return lax.cond(
            is_active,
            build_step,
            lambda x: x, # Identity function
            (b_state_carry, child_idx)
        ), None # No scan output needed, just final carry


    # Initial carry: start with the given graph state and no child (idx=0)
    init_carry = (b_state, 0) 
    
    # Iterate from max_depth-1 down to 0
    d_indices = jnp.arange(max_depth - 1, -1, -1)
    
    # Run the scan
    (final_b_state, final_root_idx), _ = lax.scan(scan_body, init_carry, d_indices)
    
    return final_b_state, final_root_idx

# --- 5. Main Jittable Function ---

@partial(jax.jit, static_argnames=[
    'max_depth', 'max_conjuncts', 'feature_dim', 'max_nodes', 'max_edges'
])
def build_ast_from_matrix(formula_matrix: jnp.ndarray, 
                          feature_map: jnp.ndarray, 
                          max_depth: int, 
                          max_conjuncts: int,
                          feature_dim: int,
                          max_nodes: int, 
                          max_edges: int):
    """
    JAX-jittable function to build a graph AST from a formula matrix.
    
    Args:
        formula_matrix (jnp.ndarray): Shape (2, max_depth, max_conjuncts)
        feature_map (jnp.ndarray): Shape (num_ops + num_props, feature_dim + 1).
                                   Maps int indices to feature vectors. The last
                                   column is assumed to be the 'is_root' feature,
                                   which will be overwritten.
        max_depth (int): Static arg, from matrix shape.
        max_conjuncts (int): Static arg, from matrix shape.
        feature_dim (int): Static arg, dimension of features (e.g., 22).
                           The `feature_map` and `nodes` array will have
                           `feature_dim + 1` columns to include `is_root`.
        max_nodes (int): Static arg, conservative allocation for nodes.
        max_edges (int): Static arg, conservative allocation for edges.
                           
    Returns:
        OrderedDict: A dictionary of JAX arrays representing the graph.
    """
    
    # --- 1. Initialize GraphState ---
    # We add 1 to feature_dim for the 'is_root' column
    nodes = jnp.zeros((max_nodes, feature_dim + 1), dtype=jnp.float32)
    # Padded edges are (0, 0, EDGE_SELF) by default
    senders = jnp.zeros((max_edges,), dtype=jnp.int32)
    receivers = jnp.zeros((max_edges,), dtype=jnp.int32)
    edge_types = jnp.full((max_edges,), EDGE_SELF, dtype=jnp.int32)
    
    # Node 0 is the padding node. Start counting from 1.
    n_node = 1
    # Add its self-loop (at edge 0)
    n_edge = 1 
    
    b_state_init = GraphState(nodes, senders, receivers, edge_types, n_node, n_edge)

    # --- 2. Define Outer Loop (over conjuncts) ---
    
    # This loop state needs to track the graph and the root of the
    # *previous* conjunct's tree, so we can link it with 'and'.
    OuterLoopState = namedtuple('OuterLoopState', ['b_state', 'last_conjunct_root_idx'])
    init_loop_state = OuterLoopState(b_state=b_state_init, last_conjunct_root_idx=0)

    def outer_loop_body(c_idx, loop_state):
        """
        Processes one conjunct, builds its subtree, and links it to the
        main 'and' chain.
        """
        b_state_loop = loop_state.b_state
        prev_root_idx = loop_state.last_conjunct_root_idx
        
        conj_c = formula_matrix[:, :, c_idx]
        is_active = conj_c[0, 0] > 0
        
        def build_and_link(state_prev_idx_tuple):
            state, prev_idx = state_prev_idx_tuple
            
            # 1. Build the subtree for this conjunct
            state, new_conj_root_idx = _build_conjunct_subtree(
                conj_c, state, feature_map, max_depth
            )
            
            # 2. Check if this is the first conjunct
            is_first_conjunct = (prev_idx == 0)

            # --- Branch 2a: Link to previous root with 'and' ---
            def link_conjuncts(state_roots_tuple):
                state_link, conj_root, prev_conj_root = state_roots_tuple
                
                # Add 'and' node. This is the new main root.
                state_link, and_idx = _add_node(state_link, OP_AND, 1.0, feature_map)
                
                # Link new conjunct
                state_link = _add_edge(state_link, conj_root, and_idx, EDGE_ARG1)
                # Link previous chain
                state_link = _add_edge(state_link, prev_conj_root, and_idx, EDGE_ARG2)
                
                # Un-set the root flag on the previous root
                nodes_link = state_link.nodes.at[prev_conj_root, -1].set(0.0)
                state_link = state_link._replace(nodes=nodes_link)
                
                return state_link, and_idx

            # --- Branch 2b: This is the first conjunct ---
            def first_conjunct(state_roots_tuple):
                state_first, conj_root, _ = state_roots_tuple
                # Set this conjunct's root as the main root
                nodes_first = state_first.nodes.at[conj_root, -1].set(1.0)
                state_first = state_first._replace(nodes=nodes_first)
                return state_first, conj_root
            
            # Conditionally link
            final_b_state, final_root_idx = lax.cond(
                is_first_conjunct,
                first_conjunct,
                link_conjuncts,
                (state, new_conj_root_idx, prev_idx)
            )
            
            return OuterLoopState(final_b_state, final_root_idx)
        
        # Only do *anything* if the conjunct is active
        return lax.cond(
            is_active,
            build_and_link,
            lambda x: x, # Identity function
            (b_state_loop, prev_root_idx)
        )

    # --- 3. Run the Loop ---
    final_loop_state = lax.fori_loop(0, max_conjuncts, outer_loop_body, init_loop_state)
    
    # --- 4. Finalize and Return ---
    final_b_state = final_loop_state.b_state
    final_n_node = final_b_state.n_node
    final_n_edge = final_b_state.n_edge

    # Conditionally update nodes:
    # If no conjuncts were active, the last_conjunct_root_idx is 0.
    # Set node 0 to be root in that case.
    is_root_col_idx = -1
    final_nodes = lax.cond(
        final_loop_state.last_conjunct_root_idx == 0,
        lambda n: n.at[0, is_root_col_idx].set(1.0), # Set padding node as root
        lambda n: n, # Root is already set correctly
        final_b_state.nodes
    )

    return OrderedDict([
        ('nodes', final_nodes),
        ('senders', final_b_state.senders),
        ('receivers', final_b_state.receivers),
        ('n_node', final_n_node.reshape(1)),
        ('n_edge', final_n_edge.reshape(1)),
        ('edge_types', final_b_state.edge_types)
    ])


logging.basicConfig(
    filename='sampler.txt',     # log file name
    filemode='w',               # append mode (use 'w' to overwrite each run)
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def test_sampler():
    """
    Main test function to sample, progress, and check for simplifications.
    
    MODIFIED:
    1. Injects a new sample formula every 5 progression iterations.
    2. Adds the *spot simplified* formula to the worklist for future processing.
    """
    # This list is now just used for *sampling*
    propositions_for_sampling = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l']
    PROP_TO_INT = {prop: i + 1 for i, prop in enumerate(propositions_for_sampling)}
    INT_TO_PROP = {i + 1: prop for i, prop in enumerate(propositions_for_sampling)}
    NUM_PROPOSITIONS = len(propositions_for_sampling)
    
    # Add 0-mapping for convenience (maps to False)
    INT_TO_PROP[0] = "NULL" 
    
    MAX_DEPTH = 3
    MAX_CONJUNCTS = 3
    jitted_progress = partial(progress_jax, max_depth=MAX_DEPTH)
    
    # sampler = EventuallySampler(propositions_for_sampling, 
    #                             min_levels=MAX_DEPTH, 
    #                             max_levels=MAX_DEPTH, 
    #                             min_conjunctions=MAX_CONJUNCTS, 
    #                             max_conjunctions=MAX_CONJUNCTS)

    # sampler= UntilTaskSampler(propositions_for_sampling, 
    #                             min_levels=2, 
    #                             max_levels=2, 
    #                             min_conjunctions=1 ,
    #                             max_conjunctions=1)

    
    

    
    step_count = 0
    current_formula = sampler.sample()
    while step_count < 1000: # Safety break
        
        # --- MODIFICATION 1: Re-sample every 5 steps ---
        if step_count > 0 and step_count % 50 == 0:
            logging.info("\n" + "*"*20)
            logging.info(f"🔥 Reached {step_count} iterations. Sampling a new formula.")
            logging.info("*"*20)
            current_formula = sampler.sample()
            logging.info(f"\n--- 🔄 Processing New Formula ( {ltl_tuple_to_string(current_formula)}): ---")
            
        
        # This check is now more important, as we might pop 'True'/'False'
        if not isinstance(current_formula, tuple):
            current_formula = sampler.sample()
           
            logging.info(f"\n--- 🔄 Processing New Formula ( {ltl_tuple_to_string(current_formula)}): ---")
        
        

        formula_props = get_propositions_in_formula(current_formula)
        all_single_prop_assignments = [[p] for p in random.sample(formula_props, len(formula_props))]

        for ta in all_single_prop_assignments:
            if (current_formula=="False" or current_formula=="True"):
                continue
            logging.info(f"  -> Progressing with {ta}:")
            
            try:
                # # 1. Get the raw progressed formula
                # custom_progressed_formula = progress_eventually(current_formula, ta)
                

                encoded_formula=encode_formula(current_formula,PROP_TO_INT, MAX_DEPTH, MAX_CONJUNCTS)
                props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT[ta[0]]].set(True)
                jax_progressed_formula, _ , _=jitted_progress(encoded_formula,props_true)
                jax_progressed_formula=simplify_conjuncts(jax_progressed_formula,MAX_DEPTH,MAX_CONJUNCTS)
                jax_progressed_formula=decode_formula(jax_progressed_formula,INT_TO_PROP)

                true_progressed_formula= progress(current_formula, ta)
                ltl_spot = _get_spot_format(true_progressed_formula)
                f = spot.formula(ltl_spot)
                f = spot.simplify(f)
                ltl_spot = f.__format__("l")
                true_simplified_formula,r = _get_std_format(ltl_spot.split(' '))
                # custom_simplified_formula= simplify(custom_progressed_formula)
                # custom_simplified_formula=simplify_conjunctions(custom_simplified_formula)

                # 3. Check if simplification occurred
                can_custom_formula=canonical_form(jax_progressed_formula)
                can_true_formula=canonical_form(true_simplified_formula)
                if  can_custom_formula!= can_true_formula:
                    logging.info("    [DIFFERENT SIMPLIFICATION DETECTED! ]")
                    
                    # logging.info(f"      Not complete: {ltl_tuple_to_string(progressed_formula)}")
                    # logging.info(f"      Complete:  {ltl_tuple_to_string(simplified_formula)}")
                    logging.info(f"      custom simplification: {ltl_tuple_to_string(jax_progressed_formula)}")
                    logging.info(f"      true simplification :  {ltl_tuple_to_string(true_simplified_formula)}")
                else:
                    logging.info(f"      simplification is same which is : {ltl_tuple_to_string(jax_progressed_formula)}")
                current_formula=true_simplified_formula
                    
            except Exception as e:
                print(f"     [ERROR] Error progressing formula: {e}")
                print(f"     Formula was: {current_formula}")
                print(f"     Truth Assignment was: {ta}")

        step_count += 1
        
    print("\n" + "="*50)
    print("✅ Exploration complete.")

if __name__ == "__main__":
    # Ensure spot is installed (pip install spot)
    # And that all helper functions (progress, spotify, etc.) are defined
    test_sampler()