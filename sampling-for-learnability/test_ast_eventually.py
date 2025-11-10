

import logging
import random as py_random
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from functools import partial
from jax import lax
from jax.random import PRNGKey
from collections import OrderedDict
from typing import NamedTuple, Tuple, List, Dict, Any, Union
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import matplotlib.pyplot as plt
import graphviz
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import io
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



class AstBuildState(NamedTuple):
    nodes: jnp.ndarray       # (MAX_NODES, feature_size)
    senders: jnp.ndarray     # (MAX_EDGES,)
    receivers: jnp.ndarray   # (MAX_EDGES,)
    edge_types: jnp.ndarray  # (MAX_EDGES,)
    node_idx: jnp.ndarray    # Scalar array, (1,)
    edge_idx: jnp.ndarray    # Scalar array, (1,)






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
        propositions: List,
        min_levels: int,
        max_levels: int,
        min_conjunctions: int,
        max_conjunctions: int,
        disjunction_prob: float = 0.25
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
        self.propositions = sorted(list(set(propositions)))
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.min_conjunctions = min_conjunctions
        self.max_conjunctions = max_conjunctions
        self.max_depth = max_levels
        self.max_conjuncts = max_conjunctions
        self.disjunction_prob = disjunction_prob
        self.EDGE_TYPES = {
        "self": 0,
        "arg": 1,   # For unary operators
        "arg1": 2,  # For binary operator, arg 1
        "arg2": 3,  # For binary operator, arg 2
         }
        self.INV_EDGE_TYPES = {v: k for k, v in self.EDGE_TYPES.items()}
        self.LTL_BASE_VOCAB = {
            "and": 0, "or": 1, "not": 2, "next": 3, "until": 4,
            "always": 5, "eventually": 6, "True": 7, "False": 8,
        }
        
        self.PROPS_OFFSET = len(self.LTL_BASE_VOCAB)
        for i, el in enumerate(self.propositions):
            self.LTL_BASE_VOCAB[el] = self.PROPS_OFFSET + i
        self.num_propositions=len(self.propositions)
        self.INV_LTL_BASE_VOCAB = {v: k for k, v in self.LTL_BASE_VOCAB.items()}
        self.PROP_TO_INDEX = {prop: i+1 for i, prop in enumerate(self.propositions)}

        self.vocab_size = len(self.LTL_BASE_VOCAB)
        self.feature_size = self.vocab_size + 1 # One-hot + is_root
        
        D = self.max_depth + 1
        K = self.max_conjuncts + 1
        
        # Max nodes: K * (5*D + 1) covers subtrees + linkers + 'True' node
        self.MAX_NODES = K * (5 * D + 1) 
        
        # Max edges: Each node has 1 'self' edge and at most 2 child edges.
        self.MAX_EDGES = self.MAX_NODES * 3
        
        # Store token IDs for JAX functions
        self.TOKEN_ID_AND = jnp.array(self.LTL_BASE_VOCAB['and'])
        self.TOKEN_ID_OR = jnp.array(self.LTL_BASE_VOCAB['or'])
        self.TOKEN_ID_F = jnp.array(self.LTL_BASE_VOCAB['eventually'])
        self.TOKEN_ID_TRUE = jnp.array(self.LTL_BASE_VOCAB['True'])
        
   
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
            k_p1_loop, p1_init = p1_loop_body((k_p1_loop, last_p1)) 
            _, p1 = jax.lax.while_loop(p1_loop_cond, p1_loop_body, (k_p1_loop, p1_init))
    
    
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
            k_p2_loop, p2_init = p2_loop_body((k_p2_loop, p1))
            _, p2 = jax.lax.while_loop(p2_loop_cond, p2_loop_body, (k_p2_loop, p2_init))
            
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
        
        return final_matrix, num_conjs, jnp.average(levels)


    ###########################    PROGRESS     ########################################################

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
    
    @partial(jax.jit, static_argnames=['self'])
    def progress(self, formula_matrix, propositions_true):
        """
        Progresses the LTL formula matrix based on the propositions that are true.
    
        Args:
            formula_matrix (jnp.ndarray): The formula representation, shape (2, max_depth, max_conjuncts).
            propositions_true (jnp.ndarray): Boolean array of shape (num_propositions).
                                             propositions_true[i] is True if proposition i is true.
                                             propositions_true[0] is always False (padding).
    
        Returns:
            jnp.ndarray: The new formula_matrix after one step of progression.
        """
        padding = jnp.array([False])
        propositions_true = jnp.concatenate([padding, propositions_true])
        
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
        progressed_matrix = progressed_matrix.at[:, self.max_depth - 1, :].set(0)
    
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


    #########################   AST BUILDER ##########################################################




    @partial(jax.jit, static_argnames=['self'])
    def _one_hot_feat(self, token_id, is_root):
        """Helper to create a node feature vector."""
        token_vec = jax.nn.one_hot(token_id, self.vocab_size, dtype=jnp.float32)
        root_vec = jnp.array([is_root], dtype=jnp.float32)
        return jnp.concatenate([token_vec, root_vec])

    @partial(jax.jit, static_argnames=['self'])
    def _add_edge(self, state: AstBuildState, sender_idx, receiver_idx, edge_type):
        """Helper to add an edge to the graph state."""
        e_idx = state.edge_idx[0]
        new_senders = state.senders.at[e_idx].set(sender_idx)
        new_receivers = state.receivers.at[e_idx].set(receiver_idx)
        new_edge_types = state.edge_types.at[e_idx].set(edge_type)
        return state._replace(
            senders=new_senders,
            receivers=new_receivers,
            edge_types=new_edge_types,
            edge_idx=state.edge_idx + 1
        )

    @partial(jax.jit, static_argnames=['self'])
    def _add_node(self, state: AstBuildState, token_id, is_root=0.0):
        """Helper to add a node and its 'self' edge."""
        n_idx = state.node_idx[0]
        feat = self._one_hot_feat(token_id, is_root)
        new_nodes = state.nodes.at[n_idx].set(feat)
        
        new_state = state._replace(
            nodes=new_nodes,
            node_idx=state.node_idx + 1
        )
        # Add the 'self' edge
        new_state = self._add_edge(new_state, n_idx, n_idx, self.EDGE_TYPES["self"])
        return new_state, n_idx

    @partial(jax.jit, static_argnames=['self'])
    def _build_conjunct_scan_body(self, carry, x_slice):
        """
        lax.scan body to build one conjunct tree, iterating *backward* over depth.
        This function is called *inside* another scan.
        """
        state, child_root_idx = carry
        p1, p2 = x_slice # p1, p2 are prop_nums (1-indexed) or 0
        
        # --- 1. Check if this depth-level is active ---
        is_active = p1 > 0
        
        def build_nodes_fn(carry_in):
            """Function for lax.cond if p1 > 0."""
            state_in, child_root_idx_in = carry_in
            
            # --- 2. Build the 'term' subtree (p1 or (or p1 p2)) ---
            is_disj = p2 > 0
            
            def build_disj_term(state_t):
                # Create 'or', 'p1', 'p2' nodes
                state_t, or_idx = self._add_node(state_t, self.TOKEN_ID_OR)
                # Convert 1-indexed prop_num to 0-indexed vocab ID
                p1_token_id = p1 + self.PROPS_OFFSET - 1
                p2_token_id = p2 + self.PROPS_OFFSET - 1
                state_t, p1_idx = self._add_node(state_t, p1_token_id)
                state_t, p2_idx = self._add_node(state_t, p2_token_id)
                # Add edges
                state_t = self._add_edge(state_t, p1_idx, or_idx, self.EDGE_TYPES["arg1"])
                state_t = self._add_edge(state_t, p2_idx, or_idx, self.EDGE_TYPES["arg2"])
                return state_t, or_idx
                
            def build_atom_term(state_t):
                # Create 'p1' node
                p1_token_id = p1 + self.PROPS_OFFSET - 1
                state_t, p1_idx = self._add_node(state_t, p1_token_id)
                return state_t, p1_idx

            state_in, term_root_idx = jax.lax.cond(
                is_disj, build_disj_term, build_atom_term, state_in
            )

            # --- 3. Build the 'F' or 'F(&...)' structure ---
            is_innermost = child_root_idx_in < 0

            def build_innermost_f(state_f):
                # Structure: F -> term
                state_f, f_idx = self._add_node(state_f, self.TOKEN_ID_F)
                state_f = self._add_edge(state_f, term_root_idx, f_idx, self.EDGE_TYPES["arg"])
                return state_f, f_idx
            
            def build_outer_f_and(state_f):
                # Structure: F -> & -> term
                #                    -> child_root (from prev step)
                state_f, f_idx = self._add_node(state_f, self.TOKEN_ID_F)
                state_f, and_idx = self._add_node(state_f, self.TOKEN_ID_AND)
                # F -> &
                state_f = self._add_edge(state_f, and_idx, f_idx, self.EDGE_TYPES["arg"])
                # & -> term
                state_f = self._add_edge(state_f, term_root_idx, and_idx, self.EDGE_TYPES["arg1"])
                # & -> child
                state_f = self._add_edge(state_f, child_root_idx_in, and_idx, self.EDGE_TYPES["arg2"])
                return state_f, f_idx

            state_out, new_root_idx = jax.lax.cond(
                is_innermost, build_innermost_f, build_outer_f_and, state_in
            )
            
            return state_out, new_root_idx

        def no_op_fn(carry_in):
            """Function for lax.cond if p1 == 0. No change."""
            return carry_in

        # Run the conditional build
        final_state, final_root_idx = jax.lax.cond(
            is_active, build_nodes_fn, no_op_fn, (state, child_root_idx)
        )
        
        # Return new carry, and the root of the subtree built *at this step*
        # The final carry will have the root of the *entire* conjunct tree.
        return (final_state, final_root_idx), final_root_idx


    @partial(jax.jit, static_argnames=['self'])
    def _link_conjuncts_scan_body(self, carry, conj_root_idx):
        """
        lax.scan body to link the roots of the conjunct subtrees
        together with 'and' nodes.
        """
        state, root_so_far_idx = carry
        
        # Check if this conjunct was active (root_idx >= 0)
        is_active = conj_root_idx >= 0

        def handle_active_node(carry_in):
            """lax.cond helper: This conjunct is active."""
            state_in, root_so_far_idx_in = carry_in
            
            is_first_active_node = root_so_far_idx_in < 0
            
            def add_first_node(state_f):
                # This is the first active conjunct. Its root is the new root-so-far.
                # No new nodes/edges are added yet.
                return state_f, conj_root_idx
            
            def add_subsequent_node(state_s):
                # This is not the first. Link it with the previous root-so-far
                # using a new 'and' node.
                # Structure: & -> root_so_far
                #              -> conj_root_idx
                state_s, and_idx = self._add_node(state_s, self.TOKEN_ID_AND)
                state_s = self._add_edge(state_s, root_so_far_idx_in, and_idx, self.EDGE_TYPES["arg1"])
                state_s = self._add_edge(state_s, conj_root_idx, and_idx, self.EDGE_TYPES["arg2"])
                # The new 'and' node is now the root-so-far.
                return state_s, and_idx

            new_state, new_root_idx = jax.lax.cond(
                is_first_active_node, add_first_node, add_subsequent_node, state_in
            )
            return new_state, new_root_idx

        def handle_inactive_node(carry_in):
            """lax.cond helper: This conjunct is inactive. Do nothing."""
            return carry_in

        # Run the conditional logic
        final_state, final_root_so_far = jax.lax.cond(
            is_active, handle_active_node, handle_inactive_node, carry
        )

        return (final_state, final_root_so_far), None # No scan output needed


    @partial(jax.jit, static_argnames=['self'])
    def build_ast(self, formula_matrix):
        """
        JAX-jittable method to build an AST graph from a formula matrix.
        
        Args:
            formula_matrix (jnp.ndarray): Shape (2, max_depth, max_conjuncts)
            
        Returns:
            OrderedDict: A dict representing the graph in jraph-compatible format.
        """
        
        
             
        init_state = AstBuildState(
            nodes=jnp.zeros((self.MAX_NODES, self.feature_size), dtype=jnp.float32),
            senders=jnp.full((self.MAX_EDGES,), -1, dtype=jnp.int32),
            receivers=jnp.full((self.MAX_EDGES,), -1, dtype=jnp.int32),
            edge_types=jnp.full((self.MAX_EDGES,), -1, dtype=jnp.int32),
            node_idx=jnp.array([1]),
            edge_idx=jnp.array([0])
        )

        # --- 1. Build Conjunct Forest ---
        # We scan over the conjuncts (axis 2 of the matrix)
        # For each one, we run an *inner scan* over depth.
        
        def build_forest_scan_body(state, c_idx):
            """Outer scan body (over conjuncts)"""
            # Get the (p1s, p2s) for this conjunct
            p1s = formula_matrix[0, :, c_idx]
            p2s = formula_matrix[1, :, c_idx]
            
            # We must scan backward over depth (D-1 down to 0)
            depth_indices_reversed = jnp.arange(self.max_depth - 1, -1, -1)
            
            # Pack p1s, p2s for the scan
            # We only need the values at the reversed depth indices
            xs_inner = (p1s[depth_indices_reversed], p2s[depth_indices_reversed])
            
            # This is the 'carry' for the *inner* scan
            init_inner_carry = (state, jnp.array(-1)) # (state, child_root_idx)
            
            # Run the inner scan to build one conjunct subtree
            (final_state, final_conj_root_idx), _ = jax.lax.scan(
                self._build_conjunct_scan_body, init_inner_carry, xs_inner
            )
            
            # Return the updated state, and the root ID of the tree we just built
            return final_state, final_conj_root_idx
        
        # Run the outer scan over all conjuncts
        # `final_forest_state` has all nodes/edges for all disjoint trees
        # `all_conjunct_roots` has shape (max_conjuncts,)
        final_forest_state, all_conjunct_roots = jax.lax.scan(
            build_forest_scan_body, init_state, jnp.arange(self.max_conjuncts)
        )

        # --- 2. Link Conjuncts ---
        # Now, link the roots in `all_conjunct_roots` with 'and' nodes
        init_link_carry = (final_forest_state, jnp.array(-1)) # (state, root_so_far)
        
        (final_linked_state, final_root_idx), _ = jax.lax.scan(
            self._link_conjuncts_scan_body, init_link_carry, all_conjunct_roots
        )
        
        # --- 3. Handle 'True' case (no active conjuncts) ---
        is_empty_formula = final_root_idx < 0
        
        def add_true_node_fn(state):
            """If formula was empty, add a 'True' node."""
            new_state, true_idx = self._add_node(state, self.TOKEN_ID_TRUE)
            return new_state, true_idx
            
        def keep_existing_root_fn(state):
            """Formula was not empty, just pass state and root."""
            return state, final_root_idx
            
        final_state, final_root_idx_safe = jax.lax.cond(
            is_empty_formula, add_true_node_fn, keep_existing_root_fn, final_linked_state
        )

        # --- 4. Set 'is_root' feature ---
        # The 'is_root' feature is the last one in the feature vector
        final_nodes = final_state.nodes.at[final_root_idx_safe, -1].set(1.0)
        final_state = final_state._replace(nodes=final_nodes)

        # --- 5. Format Output ---
        # Get final counts
        final_n_node = final_state.node_idx
        final_n_edge = final_state.edge_idx
        
        return OrderedDict([
            ('nodes', final_state.nodes),
            ('senders', final_state.senders),
            ('receivers', final_state.receivers),
            ('n_node', final_n_node.reshape(1)),
            ('n_edge', final_n_edge.reshape(1)),
            ('edge_types', final_state.edge_types)
        ])


    ####################################     ENCODE     DECODE   #################################################
    
    def encode_formula(self, formula_tuple):
        """
        Encodes a Python tuple representation of a formula into a NumPy matrix.
        Not JAX-jittable.
        """
        matrix = np.zeros((2, self.max_depth, self.max_conjuncts), dtype=np.int32)
        
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
            if c_idx >= self.max_conjuncts:
                break # Too many conjuncts for our matrix
            
            d_idx = 0
            current_task = task
            
            # 3. Unroll the 'eventually (and ...)' structure
            while current_task and current_task[0] == 'eventually' and d_idx < self.max_depth:
                term = current_task[1]
                
                if term[0] == 'and':
                    prop_part = term[1]
                    current_task = term[2] # This is the next 'eventually'
                else:
                    prop_part = term       # This is the last term in the sequence
                    current_task = None    # Stop unrolling
                
                # 4. Encode the proposition part (atom or disjunction)
                if isinstance(prop_part, str):
                    if prop_part in PROP_TO_INDEX:
                        matrix[0, d_idx, c_idx] = PROP_TO_INDEX[prop_part]
                        matrix[1, d_idx, c_idx] = 0
                elif prop_part[0] == 'or':
                    if prop_part[1] in LTL_BASE_VOCAB:
                        matrix[0, d_idx, c_idx] = PROP_TO_INDEX[prop_part[1]]
                    if prop_part[2] in LTL_BASE_VOCAB:
                        matrix[1, d_idx, c_idx] = PROP_TO_INDEX[prop_part[2]]
                
                d_idx += 1
                
        return jnp.array(matrix)
    

    def decode_formula(self, formula_matrix):
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
                    term = self.INV_LTL_BASE_VOCAB.get(p1_int+self.PROPS_OFFSET-1, f"P{p1_int}?")
                else:
                    # Disjunction
                    term = ('or', 
                           self.INV_LTL_BASE_VOCAB.get(p1_int+self.PROPS_OFFSET-1, f"P{p1_int}?"), 
                            self.INV_LTL_BASE_VOCAB.get(p2_int+self.PROPS_OFFSET-1, f"P{p2_int}?")
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
    MAX_LEVELS = 5
    MAX_CONJUNCTIONS = 5
    NUM_STEPS = 40 # Max number of steps to show
    
    # Image grid settings
    IMG_SIZE = (400, 300) # Size for each individual plot
    PADDING = 20
    TITLE_HEIGHT = 70      # Space above each plot for the title
    GRID_COLS = 3

    sampler = JaxEventuallyTaskSampler(
        propositions=PROPS,
        min_levels=2,
        max_levels=MAX_LEVELS,
        min_conjunctions=2,
        max_conjunctions=MAX_CONJUNCTIONS
    )
    
    key = jax.random.PRNGKey(py_random.randint(1,300))
    
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
        print(np.arange(3))
        sampled_prop_index = jax.random.choice(
            prop_key,
            jnp.arange(len(PROPS)),
            shape=(1,) # Sample one index
        )[0] # Get the scalar index
        # Create the full boolean array (all False)
        current_props_np = np.zeros(len(PROPS), dtype=bool)
        # Set only the sampled proposition to True
        current_props_np[sampled_prop_index] = True 
        current_props = jnp.array(current_props_np)
        print(sampled_prop_index)
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