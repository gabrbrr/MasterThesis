import jax
import jax.numpy as jnp
from jax import lax
from jax.random import PRNGKey
from collections import OrderedDict
from typing import NamedTuple, Tuple, List, Dict, Any, Union
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import random

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



# --- Core JAX Functions ---














# Define a BuilderState to pass through loops
class BuilderState(NamedTuple):
    """Holds the state of the graph construction."""
    node_idx: jnp.ndarray
    edge_idx: jnp.ndarray
    nodes: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_types: jnp.ndarray

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

        LTL_BASE_VOCAB = {
            "and": 0, "or": 1, "not": 2, "next": 3, "until": 4,
            "always": 5, "eventually": 6, "True": 7, "False": 8,
        }
        
        self.PROPS_OFFSET = len(LTL_BASE_VOCAB)
        for i, el in enumerate(self.propositions):
            LTL_BASE_VOCAB[el] = self.PROPS_OFFSET + i
        self.prop_map=LTL_BASE_VOCAB
   
        
        # Get all proposition indices as a JAX array for sampling
        self.prop_indices = jnp.array(list(self.prop_map.values()), dtype=jnp.int32)
        
        # Calculate the absolute maximum number of propositions we could need
        self.max_props_needed = 2 * self.max_levels * self.max_conjunctions
        BASE_FEATURE_DIM = len(LTL_BASE_VOCAB)
        # Add 1 for the 'is_root' flag
        FEATURE_DIM = BASE_FEATURE_DIM + 1
        
        
        
        # Define edge types from ASTBuilder context
        self.EDGE_SELF = 0
        self.EDGE_ARG = 1
        self.EDGE_ARG1 = 2
        self.self.EDGE_ARG2 = 3

        self._vmapped_progress = jax.vmap(
        self._progress_single_formula,
        in_axes=(SingleFormulaState(0, 0, 0), None, 0),
        out_axes=(SingleFormulaState(0, 0, 0), 0, 0)
    )
        # Assert that we have enough unique propositions
        if self.max_props_needed > len(self.propositions):
            raise ValueError(
                f"Not enough propositions! Need at most {self.max_props_needed} "
                f"(2 * max_levels * max_conjunctions), but only have {len(self.propositions)}."
            )
   ############################### SAMPLE #######################################################################
    
    @staticmethod
    @jax.jit
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

    @staticmethod
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
    
    @staticmethod
    @jax.jit
    def progress(
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
            self._vmapped_progress(vmap_input_state, current_props, state.depths)
        
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

    
    
    
    
    
    ###################################### AST TREE ###############################################################

    def _one_hot_base(token_id: jnp.ndarray) -> jnp.ndarray:
        """Creates the base one-hot vector (size 22)."""
        return jax.nn.one_hot(token_id, BASE_FEATURE_DIM, dtype=jnp.float32)

    def _add_node(
        builder_state: BuilderState, 
        token_id: jnp.ndarray
    ) -> Tuple[BuilderState, jnp.ndarray]:
        """
        Adds a node to the graph buffers and returns its index.
        The 'is_root' flag (last feature) defaults to 0.0.
        """
        idx = builder_state.node_idx
        
        # Create base features
        base_features = _one_hot_base(token_id)
        
        # Create the 'is_root' feature, defaulting to 0.0
        is_root_feature = jnp.array([0.0], dtype=jnp.float32)
        
        # Concat base features and is_root feature
        node_feature_vector = jnp.concatenate(
            [base_features, is_root_feature], 
            axis=0
        )
        
        # Add node feature
        nodes = builder_state.nodes.at[idx].set(node_feature_vector)
        
        # Add self-loop edge
        senders = builder_state.senders.at[builder_state.edge_idx].set(idx)
        receivers = builder_state.receivers.at[builder_state.edge_idx].set(idx)
        edge_types = builder_state.edge_types.at[builder_state.edge_idx].set(self.EDGE_SELF)
        
        new_state = builder_state._replace(
            node_idx=idx + 1,
            edge_idx=builder_state.edge_idx + 1,
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            edge_types=edge_types
        )
        return new_state, idx
    
    def _add_edge(
        builder_state: BuilderState, 
        sender: jnp.ndarray, 
        receiver: jnp.ndarray, 
        edge_type: int
    ) -> BuilderState:
        """Adds a directional edge to the graph buffers."""
        idx = builder_state.edge_idx
        senders = builder_state.senders.at[idx].set(sender)
        receivers = builder_state.receivers.at[idx].set(receiver)
        edge_types = builder_state.edge_types.at[idx].set(edge_type)
        
        return builder_state._replace(
            edge_idx=idx + 1,
            senders=senders,
            receivers=receivers,
            edge_types=edge_types
        )
    
    def _build_binary_tree(
        builder_state: BuilderState,
        leaf_indices: jnp.ndarray,
        leaf_mask: jnp.ndarray,
        op_token: int,
    ) -> Tuple[BuilderState, jnp.ndarray]:
        """
        Builds a JAX-native binary tree (like _build_tree) from a list of leaf node indices.
        
        Returns:
            (new_builder_state, root_node_index)
            Returns root_node_index = -1 if there are 0 active leaves.
        """
        
        # --- 1. Filter active leaf indices ---
        active_indices = jnp.where(leaf_mask, leaf_indices, -1)
        is_active = (active_indices != -1)
        active_count = jnp.sum(is_active)
        sorted_indices = jnp.argsort(jnp.where(is_active, jnp.arange(leaf_mask.shape[0]), leaf_mask.shape[0]))
        
        # Statically-sized array, active indices at the front.
        compressed_leaves = active_indices[sorted_indices]
    
        # --- 2. Build the binary tree from the compressed list ---
        
        # --- THIS IS THE FIX ---
        # The signature must be (loop_index, carry_state)
        def build_tree_recursive(
            start_idx: int,                          # loop_index 'i'
            state: Tuple[BuilderState, jnp.ndarray]  # carry_state 'val'
        ) -> Tuple[BuilderState, jnp.ndarray]:
        # --- END FIX ---
            """Iteratively combines leaf nodes under an operation."""
            builder_state, root_idx = state
            right_child_idx = compressed_leaves[start_idx]
            
            # Add new parent node (is_root defaults to 0.0)
            builder_state, new_root_idx = _add_node(builder_state, jnp.array(op_token))
            
            # Add edges: child -> parent
            builder_state = _add_edge(builder_state, root_idx, new_root_idx, self.EDGE_ARG1)
            builder_state = _add_edge(builder_state, right_child_idx, new_root_idx, self.EDGE_ARG2)
            
            return (builder_state, new_root_idx)
    
        # --- 3. Handle 0, 1, or N active leaves ---
        def case_zero():
            # *** MODIFIED ***
            # 0 active leaves. Do not create a True/False node.
            # Return a null index (-1) and the unmodified state.
            return builder_state, jnp.array(-1)
    
        def case_one():
            # One leaf, just return its index. No new nodes.
            root_idx = compressed_leaves[0]
            return builder_state, root_idx
    
        def case_many():
            # N > 1 leaves. Build the binary tree.
            initial_root_idx = compressed_leaves[0]
            initial_state = (builder_state, initial_root_idx)
            
            final_state, final_root_idx = lax.fori_loop(
                1, 
                active_count, 
                build_tree_recursive, # This function now has the correct signature
                initial_state
            )
            return final_state, final_root_idx
    
        # Main logic
        branch_index = jnp.minimum(active_count, 2)
        
        return lax.switch(
            branch_index,
            [case_zero, case_one, case_many]
        )
    def _build_subformula_until(
        builder_state: BuilderState,
        avoid_props: jnp.ndarray,
        prog_props: jnp.ndarray,
        formula_depth: jnp.ndarray,
        start_depth: jnp.ndarray
    ) -> Tuple[BuilderState, jnp.ndarray]:
        """
        Builds the static AST for a *single* sub-formula:
        !avoid[i] U (prog[i] & (!avoid[i+1] U (...)))
        """
        
        # 1. Build the innermost (base case) formula
        last_idx = formula_depth - 1
        
        b_state, until_idx = _add_node(builder_state, jnp.array(self.LTL_BASE_VOCAB["until"]))
        b_state, not_idx = _add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["not"]))
        
        avoid_token = avoid_props[last_idx] + self.PROPS_OFFSET
        b_state, avoid_idx = _add_node(b_state, avoid_token)
        
        prog_token = prog_props[last_idx] + self.PROPS_OFFSET
        b_state, prog_idx = _add_node(b_state, prog_token)
        
        b_state = _add_edge(b_state, not_idx, until_idx, self.EDGE_ARG1)
        b_state = _add_edge(b_state, prog_idx, until_idx, self.EDGE_ARG2)
        b_state = _add_edge(b_state, avoid_idx, not_idx, self.EDGE_ARG)
        
        current_root_idx = until_idx
    
        # 2. Loop to wrap the formula outwards
        def wrap_loop_body(m, state):
            b_state, current_root_idx = state
            
            b_state, new_until_idx = _add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["until"]))
            b_state, new_not_idx = _add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["not"]))
            
            avoid_token = avoid_props[m] + self.PROPS_OFFSET
            b_state, new_avoid_idx = _add_node(b_state, avoid_token)
            
            b_state, new_and_idx = _add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["and"]))
            
            prog_token = prog_props[m] + self.PROPS_OFFSET
            b_state, new_prog_idx = _add_node(b_state, prog_token)
            
            b_state = _add_edge(b_state, new_not_idx, new_until_idx, self.EDGE_ARG1)
            b_state = _add_edge(b_state, new_and_idx, new_until_idx, self.EDGE_ARG2)
            b_state = _add_edge(b_state, new_avoid_idx, new_not_idx, self.EDGE_ARG)
            b_state = _add_edge(b_state, new_prog_idx, new_and_idx, self.EDGE_ARG1)
            b_state = _add_edge(b_state, current_root_idx, new_and_idx, self.EDGE_ARG2)
            
            return (b_state, new_until_idx)
    
        # Re-parametrizing for lax.fori_loop (which increments)
        def loop_wrapper(i, state):
            m = (formula_depth - 2) - i
            return wrap_loop_body(m, state)
    
        n_wraps = (formula_depth - 1) - start_depth
        
        final_state, final_root_idx = lax.cond(
            n_wraps > 0,
            lambda: lax.fori_loop(0, n_wraps, loop_wrapper, (b_state, current_root_idx)),
            lambda: (b_state, current_root_idx)
        )
    
        return final_state, final_root_idx
    
    
    def _build_formula_or_tree(
        builder_state: BuilderState,
        formula_idx: int,
        state: ConjunctionState
    ) -> Tuple[BuilderState, jnp.ndarray]:
        """
        Builds the 'OR' tree for a single formula 'n'.
        
        *** MODIFIED ***
        Returns root_node_index = -1 if formula is 'True' or 'False'.
        """
        M = state.active_pointers.shape[1]
        
        is_already_true = state.already_true[formula_idx]
        active_mask = state.active_pointers[formula_idx] # (M,) bool
        is_false = ~jnp.any(active_mask) & ~is_already_true
        
        avoid_props = state.to_avoid[formula_idx]
        prog_props = state.to_progress[formula_idx]
        formula_depth = state.depths[formula_idx]
    
        # --- 1. Handle simple TRUE/FALSE cases ---
        def case_true():
            # *** MODIFIED ***
            # Do not add a node. Return null index.
            return builder_state, jnp.array(-1)
    
        def case_false():
            # *** MODIFIED ***
            # Do not add a node. Return null index.
            return builder_state, jnp.array(-1)
    
        # --- 2. Handle active (OR) case ---
        def case_active():
            # Use lax.scan to iterate over all possible start_depths (0..M-1).
            def scan_body(carry_state, m):
                b_state = carry_state
                is_active = active_mask[m] & (m < formula_depth)
                
                def build_it():
                    return _build_subformula_until(
                        b_state, avoid_props, prog_props, formula_depth, m
                    )
                
                def dont_build_it():
                    return b_state, jnp.array(-1)
    
                b_state, root_idx = lax.cond(
                   is_active,
                    build_it,
                    dont_build_it
                )
                
                final_root_idx = jnp.where(is_active, root_idx, -1)
                return b_state, final_root_idx
    
            # Run the scan.
            final_b_state, all_leaf_indices = lax.scan(
                scan_body,
                builder_state,
                jnp.arange(M)
            )
            
            # Create an 'OR' tree from the resulting leaf indices.
            active_leaf_mask = (all_leaf_indices != -1)
            
            # This will hit case_one() or case_many().
            # It will NOT hit case_zero() because is_false check
            # in the outer switch already handled the 0-active-pointer case.
            return _build_binary_tree(
                final_b_state,
                all_leaf_indices,
                active_leaf_mask,
                self.LTL_BASE_VOCAB["or"]
            )
    
        # --- 3. Main Switch ---
        branch_index = jnp.where(is_already_true, 0, jnp.where(is_false, 1, 2))
        
        return lax.switch(
            branch_index,
            [case_true, case_false, case_active]
        )
    
    @staticmethod
    @jax.jit
    def build_ast(state: ConjunctionState) -> OrderedDict:
        """
        JIT-compilable function to convert a ConjunctionState into a binary AST graph.
        
        The graph is returned in a JAX-friendly OrderedDict (GraphTuple) format
        with statically-sized buffers.
        
        The last feature of the node tensor is 'is_root'.
        
        *** MODIFIED ***
        This function no longer creates nodes for 'True' or 'False'.
        If the entire formula resolves to True/False, the graph will be empty
        (n_node=1 for padding) and no node will be marked as root.
        """
        
        N, M = state.active_pointers.shape
        
        # --- 1. Define Conservative Static Buffer Sizes ---
        MAX_NODES_PER_UNTIL_TREE = 5 * M - 1 
        MAX_NODES_PER_FORMULA = (M - 1) + (M * MAX_NODES_PER_UNTIL_TREE) + 1
        MAX_NODES = 1 + (N - 1) + (N * MAX_NODES_PER_FORMULA)
        MAX_EDGES = MAX_NODES * 3
    
        # --- 2. Initialize Buffers ---
        nodes = jnp.zeros((MAX_NODES, STATIC_FEATURE_DIM), dtype=jnp.float32)
        senders = jnp.full((MAX_EDGES,), 0, dtype=jnp.int32)
        receivers = jnp.full((MAX_EDGES,), 0, dtype=jnp.int32)
        edge_types = jnp.full((MAX_EDGES,),self.EDGE_SELF, dtype=jnp.int32)
    
        builder_state = BuilderState(
            node_idx=jnp.array(1),
            edge_idx=jnp.array(0),
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            edge_types=edge_types
        )
        
    
        # --- 4. Build the 'AND' tree of all N formulas ---
        def global_and_scan_body(carry_state, n):
            b_state = carry_state
            
            # This now returns -1 for True/False formulas
            b_state, formula_root_idx = _build_formula_or_tree(b_state, n, state)
            
            return b_state, formula_root_idx
    
        final_b_state, all_formula_leaf_indices = lax.scan(
            global_and_scan_body,
            builder_state,
            jnp.arange(N)
        )
    
        # --- 5. Build the final 'AND' tree from the formula roots ---
        
        # *** MODIFIED ***
        # The mask is now based on which indices are *not* null (-1).
        valid_leaf_mask = (all_formula_leaf_indices != -1)
        
        # This builds the final 'AND' tree.
        # If valid_leaf_mask is all False, this returns global_root_idx = -1
        final_b_state, global_root_idx = _build_binary_tree(
            final_b_state,
            all_formula_leaf_indices,
            valid_leaf_mask, # The modified mask
            self.LTL_BASE_VOCAB["and"]
        )
        
        # --- 6. Set the 'is_root' flag on the global root node ---
        
        # *** MODIFIED ***
        # Only set the 'is_root' flag if the global_root_idx is valid (not -1).
        final_nodes = lax.cond(
            global_root_idx != -1,
            lambda: final_b_state.nodes.at[global_root_idx, -1].set(1.0),
            lambda: final_b_state.nodes
        )
        
        # --- 7. Finalize and Return GraphTuple ---
        final_n_node = final_b_state.node_idx
        final_n_edge = final_b_state.edge_idx
        
        return OrderedDict([
            ('nodes', final_nodes), # Use the conditionally updated nodes
            ('senders', final_b_state.senders),
            ('receivers', final_b_state.receivers),
            ('n_node', final_n_node.reshape(1)),
            ('n_edge', final_n_edge.reshape(1)),
            ('edge_types', final_b_state.edge_types)
        ])

    
    
    
    ##################################### DECODE ##################################################################
    @staticmethod
    def decode_formula(
        state: ConjunctionState, 
    ) -> Any:
        """
        Decodes a JAX ConjunctionState back into a human-readable Python tuple.
        
        This represents the *current* state, handling 'ORs' from active pointers.
        """
        
        def _build_tree(op: str, clauses: List[Any]) -> Any:
            """Recursively builds a nested 'and' or 'or' tuple tree."""
            if not clauses:
                return 'True' if op == 'and' else 'False'
            if len(clauses) == 1:
                return clauses[0]
            return (op, clauses[0], _build_tree(op, clauses[1:]))
       
        def _reconstruct_single_formula(
            avoid_arr: jnp.ndarray, 
            prog_arr: jnp.ndarray, 
            depth: int, 
        ) -> Any:
            """Helper to rebuild the *original* nested tuple from JAX arrays."""
            avoid_arr=np.array(avoid_arr)
            prog_arr=np.array(prog_arr)
            # Start from the innermost formula
            # !avoid[depth-1] U prog[depth-1]
            curr = ('until', 
                    ('not', self.PROPS[avoid_arr[depth-1]]), 
                    self.PROPS[prog_arr[depth-1]])
            
            # Wrap it outwards
            for i in range(depth - 2, -1, -1):
                # !(avoid[i]) U (prog[i] & curr)
                curr = ('until',
                        ('not', self.PROPS[avoid_arr[i]]),
                        ('and', self.PROPS[prog_arr[i]], curr))
            return curr
        
        num_conjunctions = state.depths.shape[0]
        final_conjunctions = []
    
        for n in range(num_conjunctions):
            if state.already_true[n]:
    
                continue
                
            depth = int(state.depths[n])
            
            # Reconstruct the original full formula for this index
            original_formula = _reconstruct_single_formula(
                state.to_avoid[n], state.to_progress[n], depth, self.PROPS
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
    
    
#########################     ENCODE ########################################################################################
    
  
    @staticmethod
    def encode_formula(
        ltl_tuple: Union[tuple, str],
    ) -> ConjunctionState:
        """
        Encodes a Python tuple representation of LTL formulas into a JAX ConjunctionState.
        
        Handles:
        - 'True', 'False'
        - A single 'until' formula (fresh)
        - A single 'or' formula (progressed state)
        - An 'and' of the above
        """
       
        def _parse_single_until(
            formula_tuple: tuple,  
        ) -> Tuple[ jnp.ndarray, jnp.ndarray, int]:
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
                to_avoid_list.append(self.prop_map[avoid_prop])
                
                # B = ('and', prop, rest_formula)
                if isinstance(b_and_rest, tuple) and b_and_rest[0] == 'and':
                    _, progress_prop, rest_formula = b_and_rest
                    to_progress_list.append(self.prop_map[progress_prop])
                    curr = rest_formula
                # B = prop (base case)
                else:
                    progress_prop = b_and_rest
                    to_progress_list.append(self.prop_map[progress_prop])
                    break # Reached the end of the nesting
        
            # --- Padding ---
            # We use -1 for padding, assuming prop indices are >= 0
            pad_len = self.max_depth - depth
            to_avoid = jnp.array(to_avoid_list + [-1] * pad_len, dtype=jnp.int32)
            to_progress = jnp.array(to_progress_list + [-1] * pad_len, dtype=jnp.int32)
            
            # Initial state: only the outermost formula (index 0) is active
            active_pointers = jnp.zeros(self.max_depth, dtype=bool).at[0].set(True)
            
            return active_pointers, to_avoid, to_progress, depth
        def _parse_fresh_until(
            formula_tuple: tuple, 
        ) -> Tuple[ jnp.ndarray, jnp.ndarray, int]:
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
                to_avoid_list.append(self.prop_map[avoid_prop])
                
                if isinstance(b_and_rest, tuple) and b_and_rest[0] == 'and':
                    _, progress_prop, rest_formula = b_and_rest
                    to_progress_list.append(self.prop_map[progress_prop])
                    curr = rest_formula
                else:
                    progress_prop = b_and_rest
                    to_progress_list.append(self.prop_map[progress_prop])
                    break 
        
            pad_len = self.max_depth - depth
            to_avoid = jnp.array(to_avoid_list + [-1] * pad_len, dtype=jnp.int32)
            to_progress = jnp.array(to_progress_list + [-1] * pad_len, dtype=jnp.int32)
            active_pointers = jnp.zeros(self.max_depth, dtype=bool).at[0].set(True)
            
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
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, bool]:
            """
            Parses a single formula, which can be 'True', 'False', 
            a fresh 'until' tuple, or an 'or' tuple representing a progressed state.
            
            Returns: (active_pointers, to_avoid, to_progress, depth, already_true)
            """
            
            # --- Handle resolved states ---
            if formula_tuple == 'True':
                return (jnp.zeros(self.max_depth, dtype=bool), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
                        1, True) # depth=1 (dummy)
                
            if formula_tuple == 'False':
                return (jnp.zeros(self.max_depth, dtype=bool), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
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
                return (jnp.zeros(self.max_depth, dtype=bool), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
                        jnp.full(self.max_depth, -1, dtype=jnp.int32), 
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
                base_formula_tuple, self.prop_map, self.max_depth
            )
        
            # --- 4. Set active pointers based on 'or' clauses ---
            active_pointers = jnp.zeros(self.max_depth, dtype=bool)
            for sub_tuple in parsed_formula_tuples:
                relative_depth = _find_subformula_depth(base_formula_tuple, sub_tuple)
                active_pointers = active_pointers.at[relative_depth].set(True)
        
            return active_pointers, to_avoid, to_progress, depth, False # Not 'already_true'
    
        
       
        
        
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
        if num_conjunctions > self.max_conjunctions:
            raise ValueError(f"Found {num_conjunctions} formulas, but self.max_conjunctions={self.max_conjunctions}")
    
        # 2. Parse each formula state
        all_actives, all_avoids, all_progress, all_depths, all_already_true = [], [], [], [], []
        
        for f_tuple in formulas_to_parse:
            act, avoid, prog, depth, already_true = _parse_formula_state(
                f_tuple, self.prop_map, self.max_depth
            )
            all_actives.append(act)
            all_avoids.append(avoid)
            all_progress.append(prog)
            all_depths.append(depth)
            all_already_true.append(already_true)
        
        # 3. Pad the *batch* of formulas up to self.max_conjunctions
        pad_n = self.max_conjunctions - num_conjunctions
        all_already_true.extend([True] * pad_n) # Padded tasks are "already true"
    
        pad_active = jnp.zeros(self.max_depth, dtype=bool)
        pad_avoid = jnp.full(self.max_depth, -1, dtype=jnp.int32)
        pad_progress = jnp.full(self.max_depth, -1, dtype=jnp.int32)
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






# --- JAX-native Helper Functions ---




def visualize_ast(
    graph_tuple: OrderedDict, 
    props_list: List[str], 
    title: str
):
    """
    Draws the AST graph using NetworkX and Matplotlib.
    
    You may need to install graphviz for the 'dot' layout:
    pip install pygraphviz (or pydot)
    """
    
    # Define reverse maps for labels
    TOKEN_MAP = {
        0: "PAD",
        1: "UNTIL",
        2: "AND",
        3: "OR",
        4: "NOT",
        5: "TRUE",  # Will not be created, but map exists
        6: "FALSE", # Will not be created, but map exists
    }
    
    EDGE_TYPE_MAP = {
        0: "self",
        1: "arg",
        2: "arg1",
        3: "arg2",
    }
    
    # Extract data from graph tuple
    n_node = graph_tuple['n_node'][0]
    n_edge = graph_tuple['n_edge'][0]
    
    if n_node <= 1:
        print(f"Visualization for '{title}': Graph is empty (n_node={n_node}).")
        return

    nodes = np.array(graph_tuple['nodes'])[:n_node]
    senders = np.array(graph_tuple['senders'])[:n_edge]
    receivers = np.array(graph_tuple['receivers'])[:n_edge]
    edge_types = np.array(graph_tuple['edge_types'])[:n_edge]
    
    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    
    # Add nodes with labels and colors
    for i in range(n_node):
        feat = nodes[i]
        token_id = np.argmax(feat[:-1]) # Get token from base features
        is_root = feat[-1] == 1.0
        
        label = ""
        if token_id in TOKEN_MAP:
            label = TOKEN_MAP[token_id]
        else:
            # It's a proposition
            prop_index = token_id - 7 # (self.PROPS_OFFSET + 1)
            if 0 <= prop_index < len(props_list):
                label = props_list[prop_index]
            else:
                label = f"PROP_?"

        G.add_node(i)
        node_labels[i] = label
        node_colors.append("lightcoral" if is_root else "lightblue")

    # Add edges with labels
    edge_labels = {}
    for i in range(n_edge):
        u, v = int(senders[i]), int(receivers[i])
        edge_type_id = int(edge_types[i])
        
        # Don't draw self-loops to reduce clutter
        if u == v:
            continue
            
        G.add_edge(u, v)
        if edge_type_id in EDGE_TYPE_MAP:
            edge_labels[(u, v)] = EDGE_TYPE_MAP[edge_type_id]
            
    # Draw the graph
    plt.figure(figsize=(16, 10))
    
    # Use graphviz 'dot' layout for hierarchical trees if available
    try:
        # Note: Requires `pip install pygraphviz` or `pip install pydot`
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    except ImportError:
        print("Graphviz (pydot/pygraphviz) not found. Using spring layout.")
        print("For a better tree layout, run: pip install pydot")
        pos = nx.spring_layout(G)
        
    nx.draw(
        G, 
        pos, 
        node_color=node_colors, 
        labels=node_labels, 
        with_labels=True, 
        node_size=3000, 
        font_size=10, 
        font_weight="bold",
        arrows=True,
        arrowsize=20
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title, fontsize=20)
    filename = title.lower()
    filename = re.sub(r'\n+', '_', filename) # Replace newlines with underscore
    filename = re.sub(r'[\s:]+', '_', filename) # Replace spaces and colons
    filename = re.sub(r'[^\w_]+', '', filename) # Remove other non-alphanumeric
    filename = re.sub(r'_+', '_', filename).strip('_') # Consolidate underscores
    
    if not filename: # Handle empty or invalid titles
        filename = "ast_visualization"
        
    filename = f"{filename}.png"
    
    try:
        # Save the figure to the current directory
        plt.savefig(filename, bbox_inches='tight')
        print(f"✅ Saved graph to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving graph to '{filename}': {e}")


def test_and_visualize():
    """
    Runs several test scenarios and visualizes the generated ASTs.
    """
    
    # --- Setup ---
    PROPS = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l']
    MAX_CONJUNCTIONS = 1
    MAX_DEPTH = 6
    
    sampler = JaxUntilTaskSampler(
        propositions=PROPS,
        min_levels=MAX_DEPTH,
        max_levels=MAX_DEPTH,
        min_conjunctions=MAX_CONJUNCTIONS,
        max_conjunctions=MAX_CONJUNCTIONS
    )
    
    key = jax.random.PRNGKey(random.randint(1,300))
    
    # --- Test 1: Fresh State ---
    print("--- Running Test 1: Fresh State ---")
    key, sample_key = jax.random.split(key)
    state = sampler.sample(sample_key)
    print(ltl_tuple_to_string(decode_formula(state,PROPS)))
    print("Sampled State (already_true):", state.already_true)
    graph = state_to_ast(state)
    visualize_ast(graph, PROPS, "Test 1: Fresh Sampled State (N=2, M=3)")

TOKEN_MAP = {
    0: "PAD",
    1: "UNTIL",
    2: "AND",
    3: "OR",
    4: "NOT",
    5: "TRUE",
    6: "FALSE",
}

EDGE_TYPE_MAP = {
    0: "self",
    1: "arg",
    2: "arg1",
    3: "arg2",
}

# --- NEW HELPER FUNCTION ---
def _draw_ast_on_ax(
    ax: plt.Axes, 
    graph_tuple: OrderedDict, 
    props_list: List[str], 
    title: str
):
    """
    Draws the AST graph onto a provided Matplotlib Axes object.
    (This is the core logic from your `visualize_ast` refactored for subplots)
    """
    
    # Extract data from graph tuple
    n_node = graph_tuple['n_node'][0]
    n_edge = graph_tuple['n_edge'][0]
    
    if n_node <= 1:
        ax.text(0.5, 0.5, f"Graph is empty (n_node={n_node})", 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        return

    nodes = np.array(graph_tuple['nodes'])[:n_node]
    senders = np.array(graph_tuple['senders'])[:n_edge]
    receivers = np.array(graph_tuple['receivers'])[:n_edge]
    edge_types = np.array(graph_tuple['edge_types'])[:n_edge]
    
    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    
    # Add nodes
    for i in range(n_node):
        feat = nodes[i]
        token_id = np.argmax(feat[:-1]) # Get token
        is_root = feat[-1] == 1.0
        
        label = ""
        if token_id in TOKEN_MAP:
            label = TOKEN_MAP[token_id]
        else:
            prop_index = token_id - 7 # (self.PROPS_OFFSET + 1)
            if 0 <= prop_index < len(props_list):
                label = props_list[prop_index]
            else:
                label = f"PROP_?"

        G.add_node(i)
        node_labels[i] = label
        node_colors.append("lightcoral" if is_root else "lightblue")

    # Add edges
    edge_labels = {}
    for i in range(n_edge):
        u, v = int(senders[i]), int(receivers[i])
        edge_type_id = int(edge_types[i])
        
        if u == v: continue # Skip self-loops
            
        G.add_edge(u, v)
        if edge_type_id in EDGE_TYPE_MAP:
            edge_labels[(u, v)] = EDGE_TYPE_MAP[edge_type_id]
            
    # Draw the graph
    try:
        pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    except ImportError:
        print("Warning: Graphviz not found. Using spring layout.")
        pos = nx.spring_layout(G)
        
    nx.draw(
        G, 
        pos, 
        ax=ax,  # <-- Draw on the provided axis
        node_color=node_colors, 
        labels=node_labels, 
        with_labels=True, 
        node_size=2500, 
        font_size=9, 
        font_weight="bold",
        arrows=True,
        arrowsize=20
    )
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color='red')
    ax.set_title(title, fontsize=12, pad=10) # Set subplot title


# --- NEW GRID VISUALIZATION FUNCTION ---
def visualize_progression_grid(
    progression_steps: List[Tuple[OrderedDict, str]],
    props_list: List[str],
    grid_title: str,
    max_cols: int = 3
):
    """
    Creates and saves a grid of AST visualizations representing a formula's progression.
    
    Args:
        progression_steps: A list of (graph_tuple, formula_string) tuples.
        props_list: The list of proposition names.
        grid_title: The main title for the entire grid.
        max_cols: Maximum number of columns in the grid.
    """
    
    n_steps = len(progression_steps)
    if n_steps == 0:
        print("No progression steps to visualize.")
        return

    # Calculate grid dimensions
    n_cols = min(n_steps, max_cols)
    n_rows = (n_steps + n_cols - 1) // n_cols # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6.5))
    
    # Ensure axes is always a 2D array for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
        
    axes_flat = axes.flatten()

    print(f"Generating grid '{grid_title}' with {n_steps} steps...")

    # Draw each step on its subplot
    for i, (graph_tuple, formula_str) in enumerate(progression_steps):
        ax = axes_flat[i]
        step_title = f"Step {i}\n{formula_str}"
        if i == 0:
            step_title = f"Step 0 (Start)\n{formula_str}"
        _draw_ast_on_ax(ax, graph_tuple, props_list, step_title)
        
    # Hide any unused subplots
    for i in range(n_steps, len(axes_flat)):
        axes_flat[i].axis('off')
        
    # Add main title
    fig.suptitle(grid_title, fontsize=24, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for main title
    
    # --- Filename Generation ---
    filename = grid_title.lower()
    filename = re.sub(r'\n+', '_', filename) # Replace newlines
    filename = re.sub(r'[\s:]+', '_', filename) # Replace spaces/colons
    filename = re.sub(r'[^\w_]+', '', filename) # Remove non-alphanumeric
    filename = re.sub(r'_+', '_', filename).strip('_') # Consolidate
    
    if not filename:
        filename = "formula_progression"
    
    # Add a suffix to denote it's a grid
    filename = f"grid_{filename}.png"

    # --- Save Figure ---
    try:
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        print(f"✅ Saved progression grid to '{filename}'")
    except Exception as e:
        print(f"❌ Error saving grid to '{filename}': {e}")
    
    plt.close(fig) # Close the figure to free memory


# --- MODIFIED MAIN FUNCTION ---
def test_and_visualize_progression(
    n_formulas_to_test: int = 3,
    max_progression_steps: int = 5
):
    """
    Samples N formulas and generates a visualization grid for the
    progression of each one.
    """
    
    # --- Setup ---
    PROPS = ['a', 'b', 'c','d','e','f','g','h','i','j','k','l']
    PROP_TO_INT = {prop: i  for i, prop in enumerate(PROPS)}
    INT_TO_PROP = {i : prop for i, prop in enumerate(PROPS)}
    NUM_PROPOSITIONS = len(PROPS)
    
    MAX_DEPTH = 2
    MAX_CONJUNCTIONS = 3
    
    sampler = JaxUntilTaskSampler(
        propositions=PROPS,
        min_levels=MAX_DEPTH,
        max_levels=MAX_DEPTH,
        min_conjunctions=MAX_CONJUNCTIONS,
        max_conjunctions=MAX_CONJUNCTIONS
    )
    
    key = jax.random.PRNGKey(1)
    el=["d","f","g"]
    # --- Main Loop ---
    for i in range(n_formulas_to_test):
        print("\n" + "="*50)
        print(f"🧪 Sampling and Progressing Formula {i+1}/{n_formulas_to_test}")
        print("="*50)
        
        # 1. Sample a new formula state
        key, sample_key = jax.random.split(key)
        current_state = sampler.sample(sample_key)
        
        # This list will store all (graph, title) tuples for the grid
        progression_steps: List[Tuple[OrderedDict, str]] = []

        initial_formula_str = ""
        
        # 2. Loop for progression steps
        #for step in range(max_progression_steps + 1): # +1 to include initial state
        for step, elem in enumerate(el):   
            # Decode the current state for its string representation
            formula_tuple = decode_formula(current_state, INT_TO_PROP)
            formula_str = ltl_tuple_to_string(formula_tuple)
            
            if step == 0:
                initial_formula_str = formula_str
                print(f"  Start: {formula_str}")
            else:
                print(f"  Step {step}: {formula_str}")

            # Get the graph representation (AST) from the state
            graph_tuple = state_to_ast(current_state)
            
            # Add the state's graph and formula to our list for visualization
            progression_steps.append((graph_tuple, formula_str))
            
            # --- Check for termination ---
            if formula_tuple == "True" or formula_tuple == "False":
                print(f"  Formula terminated.")
                break
                
           # if step >= max_progression_steps:
           #     break # Reached max steps for this formula
                
            current_props = get_propositions_in_formula(formula_tuple)
            current_props = [p for p in current_props if p != 'NULL']
            
            if not current_props:
                # Formula has no propositions (e.g., "X(True)")
                # We can't meaningfully progress, so we stop
                print("  No propositions found. Stopping progression.")
                break
            
            # Randomly pick one proposition to be true
            ta_prop = random.choice(list(current_props))
            ta_prop=elem
            # Create the truth assignment vector for the jitted function
            print("stepping",ta_prop)
            props_true = jnp.zeros(NUM_PROPOSITIONS + 1, dtype=bool).at[PROP_TO_INT[ta_prop]].set(True)
            
            # 3. Progress the state
            # jitted_progress returns (new_state, reward, done)
            current_state, _, _ = jitted_progress(current_state, props_true)
            
        # 4. All steps for this formula are collected, visualize the grid
        grid_title = f"Formula {i+1} Progression\nStart: {initial_formula_str}"
        visualize_progression_grid(progression_steps, PROPS, grid_title)


if __name__ == "__main__":
    # Ensure you have graphviz installed for the 'dot' layout:
    # pip install pygraphviz
    # or
    # pip install pydot
    
    # Run the new test function
    test_and_visualize_progression(
        n_formulas_to_test=1,      # How many different formulas to sample
        max_progression_steps=5  # Max progression steps per formula (total grid size = 1 + this)
    )