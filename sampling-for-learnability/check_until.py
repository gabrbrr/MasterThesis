


import  spot
import logging
import random
import jax
import jax.numpy as jnp
from jax import random
from jax import lax
import numpy as np
from functools import partial
import re
from typing import List, Tuple, Dict, Optional, NamedTuple, Any, Union
from jax.random import PRNGKey




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
def jitted_progress(
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

def _build_tree(op: str, clauses: List[Any]) -> Any:
    """Recursively builds a nested 'and' or 'or' tuple tree."""
    if not clauses:
        return 'True' if op == 'and' else 'False'
    if len(clauses) == 1:
        return clauses[0]
    return (op, clauses[0], _build_tree(op, clauses[1:]))

def _parse_single_until(
    formula_tuple: tuple, 
    prop_map: Dict[str, int], 
    max_depth: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Parses one !(a) U (b & !(c) U d) formula into JAX arrays."""
    to_avoid_list = []
    to_progress_list = []
    curr = formula_tuple
    
    depth = 0
    while True:
        depth += 1
        # ('until', ('not', A), B)
        _, not_a, b_and_rest = curr
        
        # A = ('not', prop)
        avoid_prop = not_a[1]
        to_avoid_list.append(prop_map[avoid_prop])
        
        # B = ('and', prop, rest_formula)
        if isinstance(b_and_rest, tuple) and b_and_rest[0] == 'and':
            _, progress_prop, rest_formula = b_and_rest
            to_progress_list.append(prop_map[progress_prop])
            curr = rest_formula
        # B = prop (base case)
        else:
            progress_prop = b_and_rest
            to_progress_list.append(prop_map[progress_prop])
            break # Reached the end of the nesting

    # --- Padding ---
    # We use -1 for padding, assuming prop indices are >= 0
    pad_len = max_depth - depth
    to_avoid = jnp.array(to_avoid_list + [-1] * pad_len, dtype=jnp.int32)
    to_progress = jnp.array(to_progress_list + [-1] * pad_len, dtype=jnp.int32)
    
    # Initial state: only the outermost formula (index 0) is active
    active_pointers = jnp.zeros(max_depth, dtype=bool).at[0].set(True)
    
    return active_pointers, to_avoid, to_progress, depth

def _reconstruct_single_formula(
    avoid_arr: jnp.ndarray, 
    prog_arr: jnp.ndarray, 
    depth: int, 
    prop_list: List[str]
) -> Any:
    """Helper to rebuild the *original* nested tuple from JAX arrays."""
    avoid_arr=np.array(avoid_arr)
    prog_arr=np.array(prog_arr)
    # Start from the innermost formula
    # !avoid[depth-1] U prog[depth-1]
    curr = ('until', 
            ('not', prop_list[avoid_arr[depth-1]]), 
            prop_list[prog_arr[depth-1]])
    
    # Wrap it outwards
    for i in range(depth - 2, -1, -1):
        # !(avoid[i]) U (prog[i] & curr)
        curr = ('until',
                ('not', prop_list[avoid_arr[i]]),
                ('and', prop_list[prog_arr[i]], curr))
    return curr

# --- Public Encode/Decode Functions ---

def _parse_fresh_until(
    formula_tuple: tuple, 
    prop_map: Dict[str, int], 
    max_depth: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Parses one *fresh* !(a) U (b & ...) formula into JAX arrays.
    This is the original parsing logic, used as a helper.
    """
    to_avoid_list = []
    to_progress_list = []
    curr = formula_tuple
    
    depth = 0
    while True:
        depth += 1
        _, not_a, b_and_rest = curr
        avoid_prop = not_a[1]
        to_avoid_list.append(prop_map[avoid_prop])
        
        if isinstance(b_and_rest, tuple) and b_and_rest[0] == 'and':
            _, progress_prop, rest_formula = b_and_rest
            to_progress_list.append(prop_map[progress_prop])
            curr = rest_formula
        else:
            progress_prop = b_and_rest
            to_progress_list.append(prop_map[progress_prop])
            break 

    pad_len = max_depth - depth
    to_avoid = jnp.array(to_avoid_list + [-1] * pad_len, dtype=jnp.int32)
    to_progress = jnp.array(to_progress_list + [-1] * pad_len, dtype=jnp.int32)
    active_pointers = jnp.zeros(max_depth, dtype=bool).at[0].set(True)
    
    return active_pointers, to_avoid, to_progress, depth

def _find_subformula_depth(full_formula: tuple, sub_formula: tuple) -> int:
    """Finds the depth of sub_formula within full_formula. (0-indexed)"""
    if str(full_formula) == str(sub_formula):
        return 0
    
    if (isinstance(full_formula, tuple) and 
        full_formula[0] == 'until' and len(full_formula) == 3 and
        isinstance(full_formula[2], tuple) and
        full_formula[2][0] == 'and' and len(full_formula[2]) == 3):
        
        rest_formula = full_formula[2][2]
        depth = _find_subformula_depth(rest_formula, sub_formula)
        if depth != -1:
            return depth + 1
            
    return -1 # Not found

def _parse_formula_state(
    formula_tuple: Union[tuple, str], 
    prop_map: Dict[str, int], 
    max_depth: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, bool]:
    """
    Parses a single formula, which can be 'True', 'False', 
    a fresh 'until' tuple, or an 'or' tuple representing a progressed state.
    
    Returns: (active_pointers, to_avoid, to_progress, depth, already_true)
    """
    
    # --- Handle resolved states ---
    if formula_tuple == 'True':
        return (jnp.zeros(max_depth, dtype=bool), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                1, True) # depth=1 (dummy)
        
    if formula_tuple == 'False':
        return (jnp.zeros(max_depth, dtype=bool), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                1, False) # depth=1 (dummy)

    # --- 1. Find all 'until' clauses ---
    or_clauses = []
    if formula_tuple[0] == 'or':
        queue = [formula_tuple]
        while queue:
            item = queue.pop()
            if isinstance(item, tuple) and item[0] == 'or':
                queue.append(item[1])
                queue.append(item[2])
            elif isinstance(item, tuple) and item[0] == 'until':
                or_clauses.append(item)
    elif formula_tuple[0] == 'until':
        or_clauses = [formula_tuple]
    else:
         raise ValueError(f"Cannot parse formula state: {formula_tuple}")

    if not or_clauses: # e.g., an 'or' tree that resolved to 'False'
        return (jnp.zeros(max_depth, dtype=bool), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                jnp.full(max_depth, -1, dtype=jnp.int32), 
                1, False)

    # --- 2. Find the "base" formula (the one that contains all others) ---
    parsed_formula_tuples = [f for f in or_clauses]
    base_formula_tuple = None
    for f_i in parsed_formula_tuples:
        is_base = True
        for f_j in parsed_formula_tuples:
            if _find_subformula_depth(f_i, f_j) == -1:
                is_base = False # f_i does not contain f_j
                break
        if is_base:
            base_formula_tuple = f_i
            break
            
    if base_formula_tuple is None:
        raise ValueError("Cannot encode 'or' of disjoint formulas. "
                         "The 'or' must represent a single formula's progression.")

    # --- 3. Parse the base formula to get canonical arrays/depth ---
    _, to_avoid, to_progress, depth = _parse_fresh_until(
        base_formula_tuple, prop_map, max_depth
    )

    # --- 4. Set active pointers based on 'or' clauses ---
    active_pointers = jnp.zeros(max_depth, dtype=bool)
    for sub_tuple in parsed_formula_tuples:
        relative_depth = _find_subformula_depth(base_formula_tuple, sub_tuple)
        active_pointers = active_pointers.at[relative_depth].set(True)

    return active_pointers, to_avoid, to_progress, depth, False # Not 'already_true'

def encode_formula(
    ltl_tuple: Union[tuple, str],
    prop_map: Dict[str, int],
    max_conjunctions: int,
    max_depth: int
) -> ConjunctionState:
    """
    Encodes a Python tuple representation of LTL formulas into a JAX ConjunctionState.
    
    Handles:
    - 'True', 'False'
    - A single 'until' formula (fresh)
    - A single 'or' formula (progressed state)
    - An 'and' of the above
    """
    
    # 1. Collect all formulas from the 'and' tree
    formulas_to_parse = []
    if not isinstance(ltl_tuple, tuple): # 'True' or 'False'
        formulas_to_parse = [ltl_tuple]
    elif ltl_tuple[0] == 'and':
        queue = [ltl_tuple]
        while queue:
            item = queue.pop()
            if isinstance(item, tuple) and item[0] == 'and':
                queue.append(item[1])
                queue.append(item[2])
            else: # 'until', 'or', 'True', 'False'
                formulas_to_parse.append(item)
    else: # A single 'until' or 'or' formula
        formulas_to_parse = [ltl_tuple]

    num_conjunctions = len(formulas_to_parse)
    if num_conjunctions > max_conjunctions:
        raise ValueError(f"Found {num_conjunctions} formulas, but max_conjunctions={max_conjunctions}")

    # 2. Parse each formula state
    all_actives, all_avoids, all_progress, all_depths, all_already_true = [], [], [], [], []
    
    for f_tuple in formulas_to_parse:
        act, avoid, prog, depth, already_true = _parse_formula_state(
            f_tuple, prop_map, max_depth
        )
        all_actives.append(act)
        all_avoids.append(avoid)
        all_progress.append(prog)
        all_depths.append(depth)
        all_already_true.append(already_true)
    
    # 3. Pad the *batch* of formulas up to max_conjunctions
    pad_n = max_conjunctions - num_conjunctions
    all_already_true.extend([True] * pad_n) # Padded tasks are "already true"

    pad_active = jnp.zeros(max_depth, dtype=bool)
    pad_avoid = jnp.full(max_depth, -1, dtype=jnp.int32)
    pad_progress = jnp.full(max_depth, -1, dtype=jnp.int32)
    pad_depth = 1 # Dummy depth
    
    all_actives.extend([pad_active] * pad_n)
    all_avoids.extend([pad_avoid] * pad_n)
    all_progress.extend([pad_progress] * pad_n)
    all_depths.extend([pad_depth] * pad_n)

    # 4. Stack into a single ConjunctionState
    return ConjunctionState(
        active_pointers=jnp.stack(all_actives),
        to_avoid=jnp.stack(all_avoids),
        to_progress=jnp.stack(all_progress),
        depths=jnp.array(all_depths, dtype=jnp.int32),
        already_true=jnp.array(all_already_true, dtype=bool)
    )


def decode_formula(
    state: ConjunctionState, 
    prop_list: List[str]
) -> Any:
    """
    Decodes a JAX ConjunctionState back into a human-readable Python tuple.
    
    This represents the *current* state, handling 'ORs' from active pointers.
    """
    num_conjunctions = state.depths.shape[0]
    final_conjunctions = []

    for n in range(num_conjunctions):
        if state.already_true[n]:

            continue
            
        depth = int(state.depths[n])
        
        # Reconstruct the original full formula for this index
        original_formula = _reconstruct_single_formula(
            state.to_avoid[n], state.to_progress[n], depth, prop_list
        )
        
        # Find which sub-formulas are active
        active_indices = jnp.where(state.active_pointers[n, :depth])[0]
        
        if active_indices.shape[0] == 0:
            return 'False'

        # Create an 'OR' list of all active sub-formulas
        or_clauses = []
        for start_depth in active_indices.tolist():
            # "Drill down" to the correct sub-formula
            sub_formula = original_formula
            for _ in range(start_depth):
                # ('until', A, ('and', B, Rest)) -> get Rest
                sub_formula = sub_formula[2][2]
            or_clauses.append(sub_formula)
            
        final_conjunctions.append(_build_tree('or', or_clauses))
        
    # Combine all conjunctions with 'AND'
    return _build_tree('and', final_conjunctions)

class JaxUntilTaskSampler:
    """
    A JAX-based LTL sampler for nested 'Until' formulas.

    This class samples a conjunction of 'Until' tasks and directly
    encodes them into a JAX-compatible `ConjunctionState` for use
    with jitted progress functions.
    """
    def __init__(self, 
                 propositions: List[str], 
                 min_levels: int = 1, 
                 max_levels: int = 2, 
                 min_conjunctions: int = 1, 
                 max_conjunctions: int = 2,
                ):
        """
        Initializes the sampler.

        Args:
            propositions: List of all available proposition names.
            min_levels: Minimum nesting depth of a single 'Until' formula.
            max_levels: Maximum nesting depth (GLOBAL_MAX_DEPTH).
            min_conjunctions: Minimum number of formulas in the conjunction.
            max_conjunctions: Maximum number of formulas (GLOBAL_MAX_CONJUNCTIONS).
        """
        self.min_levels = int(min_levels)
        self.max_levels = int(max_levels)
        self.min_conjunctions = int(min_conjunctions)
        self.max_conjunctions = int(max_conjunctions)
        self.propositions = sorted(list(set(propositions)))
        # Map from proposition name to its integer index
        PROP_TO_INT = {prop: i + 1 for i, prop in enumerate(self.propositions)}
        self.prop_map=PROP_TO_INT
        
   
        
        # Get all proposition indices as a JAX array for sampling
        self.prop_indices = jnp.array(list(self.prop_map.values()), dtype=jnp.int32)
        
        # Calculate the absolute maximum number of propositions we could need
        self.max_props_needed = 2 * self.max_levels * self.max_conjunctions
        
        # Assert that we have enough unique propositions
        if self.max_props_needed > len(self.propositions):
            raise ValueError(
                f"Not enough propositions! Need at most {self.max_props_needed} "
                f"(2 * max_levels * max_conjunctions), but only have {len(self.propositions)}."
            )

    @partial(jax.jit, static_argnames=['self'])
    def sample(self, key: PRNGKey) -> ConjunctionState:
        """
        Samples a new LTL task and returns its JAX ConjunctionState.

        This function is designed to be JIT-compiled.

        Args:
            key: A JAX PRNGKey.

        Returns:
            A `ConjunctionState` tuple containing the encoded formula.
        """
        # Define global padding sizes from config
        N = self.max_conjunctions # Max number of conjunctions
        M = self.max_levels      # Max nesting depth
        
        # Split key for all random operations
        key, n_conjs_key, depths_key, choice_key = jax.random.split(key, 4)

        # 1. Sample the number of *actual* conjunctions
        n_conjs = jax.random.randint(
            n_conjs_key, 
            shape=(), 
            minval=self.min_conjunctions, 
            maxval=N + 1 # maxval is exclusive
        )
        
        # 2. Create masks based on the number of conjunctions
        # (N,) bool: True for active, False for padded
        conjunction_mask = jnp.arange(N) < n_conjs 
        # (N,) bool: False for active, True for padded (initial state)
        already_true = ~conjunction_mask 
        
        # 3. Sample depths for all N slots
        # (N,) int: Sampled depths
        sampled_depths = jax.random.randint(
            depths_key, 
            shape=(N,), 
            minval=self.min_levels, 
            maxval=M + 1 # maxval is exclusive
        )
        # Use real depths for active conjunctions, dummy depth (e.g., 1) for padded
        depths = jnp.where(conjunction_mask, sampled_depths, 1)

        # 4. Create initial active_pointers
        # (N, M) bool: All zeros
        active_pointers = jnp.zeros((N, M), dtype=bool)
        
        # --- THIS IS THE FIX ---
        # Set the *entire* first column (a static slice [:, 0]) 
        # to the *dynamic* conjunction_mask. This is JIT-compatible.
        active_pointers = active_pointers.at[:, 0].set(conjunction_mask)
        # --- END FIX ---
        
        # 5. Sample all propositions needed
        # We sample enough for the *entire* (N, M) block
        sampled_indices = jax.random.choice(
            choice_key, 
            self.prop_indices, 
            shape=(self.max_props_needed,), 
            replace=False
        )
        
        # Reshape into (N, M, 2)
        all_props = sampled_indices.reshape((N, M, 2))
        
        # Separate into full 'to_avoid' and 'to_progress' arrays
        to_avoid_full = all_props[..., 0]   # (N, M)
        to_progress_full = all_props[..., 1] # (N, M)
        
        # 6. Mask the prop arrays based on actual depths
        # (N, M) bool: True for slots within the *actual* depth
        depth_mask = jnp.arange(M) < depths[:, None]
        
        # Set all padded slots (outside actual depth) to -1
        to_avoid = jnp.where(depth_mask, to_avoid_full, -1)
        to_progress = jnp.where(depth_mask, to_progress_full, -1)
        
        # 7. Construct and return the state
        return ConjunctionState(
            active_pointers=active_pointers,
            to_avoid=to_avoid,
            to_progress=to_progress,
            depths=depths,
            already_true=already_true
        )


logging.basicConfig(
    filename='sampler.txt',     # log file name
    filemode='w',               # append mode (use 'w' to overwrite each run)
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def test_samplero():
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
    MAX_CONJUNCTS = 2
    
    # sampler = EventuallySampler(propositions_for_sampling, 
    #                             min_levels=MAX_DEPTH, 
    #                             max_levels=MAX_DEPTH, 
    #                             min_conjunctions=MAX_CONJUNCTS, 
    #                             max_conjunctions=MAX_CONJUNCTS)

    sampler= UntilTaskSampler(propositions_for_sampling, 
                                min_levels=1, 
                                max_levels=MAX_DEPTH, 
                                min_conjunctions=1 ,
                                max_conjunctions=MAX_CONJUNCTS)

    
    

    
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
        all_single_prop_assignments = [[p] for p in random.sample(list(formula_props), len(formula_props))]

        for ta in all_single_prop_assignments:
            if (current_formula=="False" or current_formula=="True"):
                continue
            logging.info(f"  -> Progressing with {ta}:")
            
            # try:
                # # 1. Get the raw progressed formula
                # custom_progressed_formula = progress_eventually(current_formula, ta)
                

            encoded_formula=encode_formula(current_formula,PROP_TO_INT, MAX_CONJUNCTS,MAX_DEPTH)
            props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT[ta[0]]].set(True)
            jax_progressed_formula, _, _=jitted_progress(encoded_formula,props_true)
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
                
            # except Exception as e:
            #     print(f"     [ERROR] Error progressing formula: {e}")
            #     print(f"     Formula was: {current_formula}")
            #     print(f"     Truth Assignment was: {ta}")

        step_count += 1
        
    print("\n" + "="*50)
    print("✅ Exploration complete.")

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
    
    MAX_DEPTH = 6
    MAX_CONJUNCTS = 1
    sampler=JaxUntilTaskSampler(propositions_for_sampling, 
                                min_levels=MAX_DEPTH, 
                                max_levels=MAX_DEPTH, 
                                min_conjunctions=1 ,
                                max_conjunctions=MAX_CONJUNCTS)

    
    step_count = 0
    rng = jax.random.PRNGKey(random.randint(1,200))
    current_formula = sampler.sample(rng)
    current_formula=decode_formula(current_formula,INT_TO_PROP)
    
    while step_count < 1000: # Safety break
        
        # --- MODIFICATION 1: Re-sample every 5 steps ---
        if step_count > 0 and step_count % 50 == 0:
            logging.info("\n" + "*"*20)
            logging.info(f"🔥 Reached {step_count} iterations. Sampling a new formula.")
            logging.info("*"*20)
            rng = jax.random.PRNGKey(random.randint(1,200))
            current_formula = sampler.sample(rng)
            current_formula=decode_formula(current_formula,INT_TO_PROP)
            logging.info(f"\n--- 🔄 Processing New Formula ( {ltl_tuple_to_string(current_formula)}): ---")
            
        
        # This check is now more important, as we might pop 'True'/'False'
        if not isinstance(current_formula, tuple):
            rng = jax.random.PRNGKey(random.randint(1,200))
            current_formula = sampler.sample(rng)
            current_formula=decode_formula(current_formula,INT_TO_PROP)
           
            logging.info(f"\n--- 🔄 Processing New Formula ( {ltl_tuple_to_string(current_formula)}): ---")
        
        

        formula_props = get_propositions_in_formula(current_formula)
        all_single_prop_assignments = [[p] for p in random.sample(list(formula_props), len(formula_props))]

        for ta in all_single_prop_assignments:
            if (current_formula=="False" or current_formula=="True"):
                continue
            logging.info(f"  -> Progressing with {ta}:")
            
            # try:
                # # 1. Get the raw progressed formula
                # custom_progressed_formula = progress_eventually(current_formula, ta)
                

            encoded_formula=encode_formula(current_formula,PROP_TO_INT, MAX_CONJUNCTS,MAX_DEPTH)
            props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT[ta[0]]].set(True)
            jax_progressed_formula, _, _=jitted_progress(encoded_formula,props_true)
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
                
            # except Exception as e:
            #     print(f"     [ERROR] Error progressing formula: {e}")
            #     print(f"     Formula was: {current_formula}")
            #     print(f"     Truth Assignment was: {ta}")

        step_count += 1
        
    print("\n" + "="*50)
    print("✅ Exploration complete.")

if __name__ == "__main__":
    # Ensure spot is installed (pip install spot)
    # And that all helper functions (progress, spotify, etc.) are defined
    test_sampler()
