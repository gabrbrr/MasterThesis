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
import random as py_random
from jax import random
from functools import partial
from collections import OrderedDict
import graphviz
from PIL import Image, ImageDraw, ImageFont
import os


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

        self.LTL_BASE_VOCAB = {
            "and": 0, "or": 1, "not": 2, "next": 3, "until": 4,
            "always": 5, "eventually": 6, "True": 7, "False": 8,
        }
        
        self.PROPS_OFFSET = len(self.LTL_BASE_VOCAB)
        for i, el in enumerate(self.propositions):
            self.LTL_BASE_VOCAB[el] = self.PROPS_OFFSET + i
        self.prop_map=self.LTL_BASE_VOCAB
   
        
        # Get all proposition indices as a JAX array for sampling
        
        
        # Calculate the absolute maximum number of propositions we could need
        self.max_props_needed = 2 * self.max_levels * self.max_conjunctions
        self.vocab_size = len(self.LTL_BASE_VOCAB)
        self.feature_size = self.vocab_size + 1 # One-hot + is_root

        self.prop_indices = jnp.arange(len(self.propositions), dtype=jnp.int32)
        
        self.EDGE_TYPES = {
            "self": 0,
            "arg": 1,    # For unary operators
            "arg1": 2,   # For binary operator, arg 1
            "arg2": 3,   # For binary operator, arg 2
        }
        # Create the inverse mapping for labeling
        self.INV_EDGE_TYPES = {v: k for k, v in self.EDGE_TYPES.items()}
        self.INV_LTL_BASE_VOCAB = {v: k for k, v in self.LTL_BASE_VOCAB.items()}
        

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
        ), n_conjs, jnp.average(sampled_depths)

    @partial(jax.jit, static_argnames=['self'])
    def _progress_single_formula(
        self,
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
    
    @partial(jax.jit, static_argnames=['self'])
    def progress(
        self, 
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
        is_false_overall = jnp.any(is_false_this_step & progressing_mask)
        
        return new_conjunction_state, is_true_overall, is_false_overall

    
    
    
    
    
    ###################################### AST TREE ###############################################################
    @partial(jax.jit, static_argnames=['self'])
    def _one_hot_base(self, token_id: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.one_hot(token_id, self.vocab_size, dtype=jnp.float32)
    @partial(jax.jit, static_argnames=['self'])
    def _add_node(
        self,
        builder_state: BuilderState, 
        token_id: jnp.ndarray
    ) -> Tuple[BuilderState, jnp.ndarray]:
        """
        Adds a node to the graph buffers and returns its index.
        The 'is_root' flag (last feature) defaults to 0.0.
        """
        idx = builder_state.node_idx
        
        # Create base features
        base_features = self._one_hot_base(token_id)
        
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
        edge_types = builder_state.edge_types.at[builder_state.edge_idx].set(self.EDGE_TYPES["self"])
        
        new_state = builder_state._replace(
            node_idx=idx + 1,
            edge_idx=builder_state.edge_idx + 1,
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            edge_types=edge_types
        )
        return new_state, idx
    @partial(jax.jit, static_argnames=['self'])
    def _add_edge(
        self,
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
    @partial(jax.jit, static_argnames=['self'])
    def _build_binary_tree(
        self,
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
            builder_state, new_root_idx = self._add_node(builder_state, jnp.array(op_token))
            
            # Add edges: child -> parent
            builder_state = self._add_edge(builder_state, root_idx, new_root_idx, self.EDGE_TYPES["arg1"])
            builder_state = self._add_edge(builder_state, right_child_idx, new_root_idx, self.EDGE_TYPES["arg2"])
            
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
    @partial(jax.jit, static_argnames=['self'])
    def _build_subformula_until(
        self,
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
        
        b_state, until_idx = self._add_node(builder_state, jnp.array(self.LTL_BASE_VOCAB["until"]))
        b_state, not_idx = self._add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["not"]))
        
        avoid_token = avoid_props[last_idx] + self.PROPS_OFFSET
        b_state, avoid_idx = self._add_node(b_state, avoid_token)
        
        prog_token = prog_props[last_idx] + self.PROPS_OFFSET
        b_state, prog_idx = self._add_node(b_state, prog_token)
        
        b_state = self._add_edge(b_state, not_idx, until_idx, self.EDGE_TYPES["arg1"])
        b_state = self._add_edge(b_state, prog_idx, until_idx, self.EDGE_TYPES["arg2"])
        b_state = self._add_edge(b_state, avoid_idx, not_idx, self.EDGE_TYPES["arg"])
        
        current_root_idx = until_idx
    
        # 2. Loop to wrap the formula outwards
        def wrap_loop_body(m, state):
            b_state, current_root_idx = state
            
            b_state, new_until_idx = self._add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["until"]))
            b_state, new_not_idx = self._add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["not"]))
            
            avoid_token = avoid_props[m] + self.PROPS_OFFSET
            b_state, new_avoid_idx = self._add_node(b_state, avoid_token)
            
            b_state, new_and_idx = self._add_node(b_state, jnp.array(self.LTL_BASE_VOCAB["and"]))
            
            prog_token = prog_props[m] + self.PROPS_OFFSET
            b_state, new_prog_idx = self._add_node(b_state, prog_token)
            
            b_state = self._add_edge(b_state, new_not_idx, new_until_idx, self.EDGE_TYPES["arg1"])
            b_state = self._add_edge(b_state, new_and_idx, new_until_idx, self.EDGE_TYPES["arg2"])
            b_state = self._add_edge(b_state, new_avoid_idx, new_not_idx, self.EDGE_TYPES["arg"])
            b_state = self._add_edge(b_state, new_prog_idx, new_and_idx, self.EDGE_TYPES["arg1"])
            b_state = self._add_edge(b_state, current_root_idx, new_and_idx, self.EDGE_TYPES["arg2"])
            
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
    
    @partial(jax.jit, static_argnames=['self'])
    def _build_formula_or_tree(self, 
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
                    return self._build_subformula_until(
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
            return self._build_binary_tree(
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
    
    
    @partial(jax.jit, static_argnames=['self'])
    def build_ast(self, state: ConjunctionState) -> OrderedDict:
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
        nodes = jnp.zeros((MAX_NODES, self.feature_size), dtype=jnp.float32)
        senders = jnp.full((MAX_EDGES,), 0, dtype=jnp.int32)
        receivers = jnp.full((MAX_EDGES,), 0, dtype=jnp.int32)
        edge_types = jnp.full((MAX_EDGES,),self.EDGE_TYPES["self"], dtype=jnp.int32)
    
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
            b_state, formula_root_idx = self._build_formula_or_tree(b_state, n, state)
            
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
        final_b_state, global_root_idx = self._build_binary_tree(
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
    
    def decode_formula(self, 
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
                    ('not', self.propositions[avoid_arr[depth-1]]), 
                    self.propositions[prog_arr[depth-1]])
            
            # Wrap it outwards
            for i in range(depth - 2, -1, -1):
                # !(avoid[i]) U (prog[i] & curr)
                curr = ('until',
                        ('not', self.propositions[avoid_arr[i]]),
                        ('and', self.propositions[prog_arr[i]], curr))
            return curr
        
        num_conjunctions = state.depths.shape[0]
        final_conjunctions = []
    
        for n in range(num_conjunctions):
            if state.already_true[n]:
    
                continue
                
            depth = int(state.depths[n])
            
            # Reconstruct the original full formula for this index
            original_formula = _reconstruct_single_formula(
                state.to_avoid[n], state.to_progress[n], depth,
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
    
  
    
    def encode_formula(self, 
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
        
            return active_pointers, to_avoid, to_progress, depth, False 
    
        
       
        
        
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

#############################################  VISUALIZE AST    ###########################################

    def visualize_ast(self, ast_data: OrderedDict, img_size=(1290, 1080)):
        """
        Visualizes an LTL AST graph from a data dictionary using
        Networkx and Matplotlib.

        Args:
            ast_data (OrderedDict): An ordered dictionary containing the graph data.
            img_size (tuple): The target (width, height) in pixels for the
                              output image.

        Returns:
            PIL.Image.Image: A PIL Image object of the rendered graph, or None
                             if plotting fails.
        """
        
        try:
            # 1. Extract data
            nodes = ast_data['nodes']
            senders = ast_data['senders']
            receivers = ast_data['receivers']
            
            # <-- MODIFIED: Use .item() to get Python scalars from JAX arrays
            # This is safer and more direct than .reshape(1)[0]
            n_node = ast_data['n_node'].item()
            n_edge = ast_data['n_edge'].item()
            
            edge_types = ast_data['edge_types']
            
            # 2. Slice data
            nodes = nodes[:n_node]
            senders = senders[:n_edge]
            receivers = receivers[:n_edge]
            edge_types = edge_types[:n_edge]

        except KeyError as e:
            print(f"Error: Missing key in ast_data: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error extracting data: {e}", file=sys.stderr)
            return None

        # 3. Build the Networkx Graph
        G = nx.DiGraph()
        node_labels = {}
        node_colors = []
        edge_labels = {}
        
        # 4. Add all nodes
        for i in range(n_node):
            node_id = str(i) # Use string IDs for consistency
            node_features = nodes[i]
            
            try:
                # <-- MODIFIED: .item() converts unhashable JAX scalar to Python int
                vocab_index = np.argmax(node_features[:-1]).item() 
                label = self.INV_LTL_BASE_VOCAB.get(vocab_index, f"UNK_{vocab_index}")
            except Exception as e:
                label = f"Error: {e}"

            # <-- MODIFIED: .item() converts JAX bool to Python bool
            is_root = (node_features[-1] == 1).item() 
            
            G.add_node(node_id)
            
            if is_root:
                node_colors.append('lightgreen')
                node_labels[node_id] = f"{label}\n(ROOT)"
            else:
                node_colors.append('lightblue')
                node_labels[node_id] = label

        # 5. Add all edges
        for j in range(n_edge):
            # Use .item() to convert numpy types to standard python types
            sender_id = str(senders[j].item()) 
            receiver_id = str(receivers[j].item())
            
            edge_type_index = edge_types[j]
            if hasattr(edge_type_index, 'item'):
                edge_type_index = edge_type_index.item()
                
            edge_label = self.INV_EDGE_TYPES.get(edge_type_index, f"UNK_{edge_type_index}")
            
            # Skip "self" loops
            if edge_label == "self":
                continue
                
            G.add_edge(sender_id, receiver_id)
            edge_labels[(sender_id, receiver_id)] = edge_label

        # 6. Plot with Matplotlib
        dpi = 100
        figsize = (img_size[0] / dpi, img_size[1] / dpi)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        try:
            # 7. Get layout
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
            except ImportError:
                print("Warning: pydot not found. Using spring_layout.", file=sys.stderr)
                print("Install pydot for a hierarchical tree layout.", file=sys.stderr)
                pos = nx.spring_layout(G, seed=42)
            
            # 8. Draw graph components
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color=node_colors, 
                node_size=50, node_shape='s'
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax, arrowstyle='->', arrowsize=20, 
                node_size=50, connectionstyle='arc3,rad=0.1'
            )
            nx.draw_networkx_labels(
                G, pos, ax=ax, labels=node_labels, font_size=9
            )
            nx.draw_networkx_edge_labels(
                G, pos, ax=ax, edge_labels=edge_labels, font_size=7
            )
            
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # 9. Save plot to in-memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            
            # 10. Open as PIL Image and resize
            img = Image.open(buf)
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            
            return img

        except Exception as e:
            print(f"Error during Networkx/Matplotlib plotting: {e}", file=sys.stderr)
            return None
        finally:
            plt.close(fig)

def main():
    """
    Test main function to:
    1. Sample a task.
    2. Progress it randomly for several steps, **sampling ONE letter at a time**.
    3. Build the AST for each step.
    4. Decode the formula for each step.
    5. Save a grid of the ASTs showing the progression.
    """
    
    # --- 1. Setup ---
    print("Setting up sampler...")
    PROPS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','l','m']
    MAX_LEVELS = 3
    MAX_CONJUNCTIONS = 2
    NUM_STEPS = 4 # Max number of steps to show
    
    # Image grid settings
    IMG_SIZE = (400, 300) # Size for each individual plot
    PADDING = 20
    TITLE_HEIGHT = 70      # Space above each plot for the title
    GRID_COLS = 3

    sampler = JaxUntilTaskSampler(
        propositions=PROPS,
        min_levels=MAX_LEVELS,
        max_levels=MAX_LEVELS,
        min_conjunctions=MAX_CONJUNCTIONS,
        max_conjunctions=MAX_CONJUNCTIONS
    )
    
    key = jax.random.PRNGKey(42)
    
    # --- 2. Sample Initial Task ---
    print("Sampling initial task...")
    key, sample_key = jax.random.split(key)
    state, c, l = sampler.sample(sample_key)
    print("avg",int(c),float(l))

    images = []
    titles = []
    sampled_letter = "None" # For first title

    # --- 3. Simulation Loop ---
    print(f"Starting simulation loop for max {NUM_STEPS} steps...")
    for t in range(NUM_STEPS):
        print(f"--- Step {t} ---")
        
        # 3a. Decode formula for title
        formula_str = ltl_tuple_to_string(sampler.decode_formula(state))
        title = f"Step {t} (Sampled: {sampled_letter})\n{formula_str}"
        titles.append(title)
        print(f"Formula: {formula_str}")

        ast_data = sampler.build_ast(state)
        try:
            # Use new function
            img = sampler.visualize_ast(ast_data, img_size=IMG_SIZE)
            
            if img is None:
                 raise Exception("visualize_ast returned None (plotting failed)")

            images.append(img)
        except Exception as e:
            print(f"Final state rendering failed: {e}")
            img = Image.new('RGB', IMG_SIZE, 'white') # Placeholder
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Plotting failed\n{e}", fill='red')
            images.append(img)
        
        # *** START MODIFICATION ***
        # 3d. Progress the state randomly (ONE letter at a time)
        key, step_key, prop_key = jax.random.split(key, 3)
        
        # Sample a single proposition index to set to True
        # sampler.prop_indices already has the offset (e.g., 8, 9, 10...)
        sampled_prop_index = jax.random.choice(
            prop_key,
            sampler.prop_indices,
            shape=(1,) # Sample one index
        )[0] # Get the scalar index
        print(sampler.prop_indices[sampled_prop_index])
        # Create the full boolean array (all False)
        current_props_np = np.zeros(len(PROPS), dtype=bool)
        # Set only the sampled proposition to True
        current_props_np[sampled_prop_index] = True 
        current_props = jnp.array(current_props_np)
        
        # Get the name of the letter for logging and the next title
        sampled_letter = sampler.INV_LTL_BASE_VOCAB[int(sampled_prop_index)+sampler.PROPS_OFFSET]
        print(f"Sampling letter: {sampled_letter}")
        # *** END MODIFICATION ***

        # Run the progress step
        state, is_conj_true, is_conj_false = sampler.progress(state, current_props)
        print(is_conj_true,is_conj_false)
        
        # 3e. Check for terminal state
        if is_conj_true:
            print("Conjunction resolved to TRUE. Stopping.")
            break
        if is_conj_false:
            print("Conjunction resolved to FALSE. Stopping.")
            break
    
    # --- 4. Handle Final State (if loop broke) ---
    if (t < NUM_STEPS - 1): # Loop broke early
        print(f"--- Final State (Step {t+1}) ---")
        formula_str = ltl_tuple_to_string(sampler.decode_formula(state))
        title = f"Step {t+1} (Final State)\n{formula_str}"
        titles.append(title)
        print(f"Formula: {formula_str}")
        
        ast_data = sampler.build_ast(state)
        try:
            # Use new function
            img = sampler.visualize_ast(ast_data, img_size=IMG_SIZE)
            
            if img is None:
                 raise Exception("visualize_ast returned None (plotting failed)")

            images.append(img)
        except Exception as e:
            print(f"Final state rendering failed: {e}")
            img = Image.new('RGB', IMG_SIZE, 'white') # Placeholder
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Plotting failed\n{e}", fill='red')
            images.append(img)

    # --- 5. Create Image Grid ---
    if not images:
        print("No images were generated.")
        return

    print("Creating image grid...")
    num_images = len(images)
    grid_rows = (num_images + GRID_COLS - 1) // GRID_COLS
    
    grid_width = (IMG_SIZE[0] * GRID_COLS) + (PADDING * (GRID_COLS + 1))
    grid_height = ((IMG_SIZE[1] + TITLE_HEIGHT) * grid_rows) + (PADDING * (grid_rows + 1))
    
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_image)
    
    try:
        # Try to load a nicer font
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        print("Arial font not found, using default.")
        font = ImageFont.load_default()

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // GRID_COLS
        col = i % GRID_COLS
        
        # Calculate top-left corner for this cell
        x_offset = PADDING + col * (IMG_SIZE[0] + PADDING)
        y_offset = PADDING + row * (IMG_SIZE[1] + TITLE_HEIGHT + PADDING)
        
        # Draw Title
        draw.text((x_offset + 5, y_offset + 5), title, fill='black', font=font)
        
        # Paste Image
        grid_image.paste(img, (x_offset, y_offset + TITLE_HEIGHT))

    # --- 6. Save Final Image ---
    output_filename = "ast_progression_grid_single_letter.png"
    grid_image.save(output_filename)
    print(f"\nSuccess! Saved AST progression grid to {output_filename}")


if __name__ == "__main__":
    main()