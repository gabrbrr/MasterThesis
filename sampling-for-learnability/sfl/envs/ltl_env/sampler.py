import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from sfl.envs.ltl_env.utils import *
from jax import random
class JaxUntilTaskSampler():
    def __init__(self, propositions, min_levels=1, max_levels=3, min_conjunctions=1, max_conjunctions=2):
        self.prop_tokens = jnp.array([LTL_BASE_VOCAB[p] for p in propositions],dtype=jnp.int32)
        self.num_props = len(propositions)
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.min_conjunctions = min_conjunctions
        self.max_conjunctions = max_conjunctions
        # Probability of choosing a disjunction ('or') of two propositions
        self.disjunction_prob = 0.25
        self.max_props_needed = 2 * max_levels * max_conjunctions
        assert self.max_props_needed <= len(self.prop_tokens), "Not enough propositions for the given max settings!"


    @partial(jax.jit, static_argnames=("self"))
    def sample(self, key):
        """
        JAX-compatible function to sample a complex 'Until' task.

        This function builds a formula of the form:
        (Task_1) AND (Task_2) AND ... AND (Task_n_conjs)
        where each Task_i is a nested 'Until' formula:
        U(!p1, p2 AND U(!p3, p4 AND ...))

        Args:
            key: A jax.random.PRNGKey for random operations.
            min_levels, max_levels: Min/max nesting depth for each 'Until' sub-formula.
            min_conjunctions, max_conjunctions: Min/max number of 'Until' sub-formulas to be joined by 'AND'.

        Returns:
            A tuple containing:
            - formula_array (jnp.ndarray): The encoded formula in a (MAX_NODES, 3) array.
            - num_nodes (int): The number of valid nodes used in the array.
            - root_idx (int): The index of the root node of the final formula.
            - num_conjuncts (int): The number of conjunctions in the formula.
            - num_levels (int): The total number of levels across all conjunctions.
        """
        # --- 1. Initial Setup and Random Sampling ---
        key, n_conjs_key, p_key = jax.random.split(key, 3)

        # Sample the number of conjunctions
        n_conjs = jax.random.randint(n_conjs_key, (), self.min_conjunctions, self.max_conjunctions + 1)

        # Sample all propositions needed upfront without replacement.
        # We must sample the maximum possible number to ensure a static shape for JIT.

        p = jax.random.choice(p_key, self.prop_tokens, shape=(self.max_props_needed,), replace=False)

        # --- 2. Define Loop Bodies for jax.lax.fori_loop ---

        def build_nested_until_task(key, formula_array, start_node_idx, start_prop_idx, n_levels):
            """Builds one nested 'Until' sub-formula."""

            # Base case: U(not p[b], p[b+1])
            # This requires 4 nodes: p[b], p[b+1], NOT, and UNTIL.
            p1_idx = start_node_idx
            p0_idx = start_node_idx + 1
            not_p0_idx = start_node_idx + 2
            until_root_idx = start_node_idx + 3

            formula_array = formula_array.at[p1_idx].set(jnp.array([p[start_prop_idx + 1], 0, 0]))
            formula_array = formula_array.at[p0_idx].set(jnp.array([p[start_prop_idx], 0, 0]))
            formula_array = formula_array.at[not_p0_idx].set(jnp.array([NOT, p0_idx, 0]))
            formula_array = formula_array.at[until_root_idx].set(jnp.array([UNTIL, not_p0_idx, p1_idx]))

            prop_idx = start_prop_idx + 2
            node_idx = start_node_idx + 4

            # Inner loop state: (key, formula_array, node_idx, prop_idx, current_until_root)
            initial_inner_carry = (key, formula_array, node_idx, prop_idx, until_root_idx)

            def inner_loop_body(j, carry):
                """Adds one level of nesting: U(not p_new, (and p_next, old_until_task))"""
                l_key, l_formula_array, l_node_idx, l_prop_idx, l_until_root_idx = carry

                # This requires 5 new nodes: p_next, p_new, NOT, AND, UNTIL
                p_next_idx      = l_node_idx
                p_new_idx       = l_node_idx + 1
                not_p_new_idx   = l_node_idx + 2
                and_idx         = l_node_idx + 3
                new_until_root  = l_node_idx + 4

                # p_next
                l_formula_array = l_formula_array.at[p_next_idx].set(jnp.array([p[l_prop_idx + 1], 0, 0]))
                # p_new
                l_formula_array = l_formula_array.at[p_new_idx].set(jnp.array([p[l_prop_idx], 0, 0]))
                # not p_new
                l_formula_array = l_formula_array.at[not_p_new_idx].set(jnp.array([NOT, p_new_idx, 0]))
                # p_next AND old_until_task
                l_formula_array = l_formula_array.at[and_idx].set(jnp.array([AND, p_next_idx, l_until_root_idx]))
                # U(not p_new, (...))
                l_formula_array = l_formula_array.at[new_until_root].set(jnp.array([UNTIL, not_p_new_idx, and_idx]))

                return (l_key, l_formula_array, l_node_idx + 5, l_prop_idx + 2, new_until_root)

            # Loop n_levels - 1 times to add the nested layers
            key, formula_array, node_idx, prop_idx, until_root_idx = jax.lax.fori_loop(
                0, n_levels - 1, inner_loop_body, initial_inner_carry
            )
            return key, formula_array, node_idx, prop_idx, until_root_idx

        def outer_loop_body(i, carry):
            """Builds one 'Until' task and ANDs it with the main formula."""
            key, formula_array, node_idx, prop_idx, ltl_root_idx, total_levels = carry
            key, n_levels_key, build_key = jax.random.split(key, 3)
            
            # Sample levels for this specific sub-formula
            n_levels = jax.random.randint(n_levels_key, (), self.min_levels, self.max_levels + 1)
           
            new_total_levels = total_levels + n_levels
            
            # Build the sub-formula
            build_key, formula_array, new_node_idx, new_prop_idx, until_task_root = build_nested_until_task(
                build_key, formula_array, node_idx, prop_idx, n_levels
            )

            # If this is the first task, it becomes the root.
            # Otherwise, create an AND node to join it with the existing formula.
            def first_task_fn(_):
                return until_task_root, formula_array, new_node_idx

            def subsequent_task_fn(_):
                and_node_idx = new_node_idx
                new_array = formula_array.at[and_node_idx].set(jnp.array([AND, until_task_root, ltl_root_idx]))
                return and_node_idx, new_array, new_node_idx + 1

            new_ltl_root, formula_array, node_idx = jax.lax.cond(
                ltl_root_idx == -1,  # Use -1 as a sentinel for the first task
                first_task_fn,
                subsequent_task_fn,
                operand=None
            )

            return key, formula_array, node_idx, new_prop_idx, new_ltl_root, new_total_levels

        # --- 3. Execute Main Loop ---

        # Initial state for the main loop
        # carry = (key, formula_array, node_idx, prop_idx, ltl_root_idx)
        initial_carry = (
            key,
            jnp.full((MAX_NODES, 3), -1, dtype=jnp.int32), # Formula array
            0,                                             # Next available node index
            0,                                             # Next available proposition index
            -1,
             0,                                                                                         # Root of the combined formula
        )

        # Run the loop for n_conjs iterations
        _, final_array, num_nodes, _, root_idx, total_levels = jax.lax.fori_loop(
            0, n_conjs, outer_loop_body, initial_carry
        )
        avg_levels = total_levels.astype(jnp.float32) / n_conjs.astype(jnp.float32)
        return final_array, num_nodes, root_idx, n_conjs, avg_levels



class JaxEventuallySampler:
    """
    A class to generate complex LTL formulas using a JIT-compiled JAX sampler.
    The sampler's static configuration is provided during initialization, and
    the JIT compilation happens once.
    """
    def __init__(self, propositions, min_levels=1, max_levels=5, min_conjunctions=1, max_conjunctions=4):
        self.propositions = jnp.array([LTL_BASE_VOCAB[p] for p in propositions],dtype=jnp.int32)
        self.min_levels = min_levels
        self.max_levels = max_levels
        self.min_conjunctions = min_conjunctions
        self.max_conjunctions = max_conjunctions
        assert(len(propositions) >= 3)

        self._jitted_sampler = partial(
            jax.jit(self._static_sampler, static_argnames=(
                "min_levels", "max_levels", "min_conjunctions", "max_conjunctions"
            )),
            min_levels=self.min_levels,
            max_levels=self.max_levels,
            min_conjunctions=self.min_conjunctions,
            max_conjunctions=self.max_conjunctions,
            propositions=self.propositions
        )

    def sample(self, key):
        """
        Generates a new LTL formula sample.
        Args:
            key (jax.random.PRNGKey): The random key for this specific sample generation.
        Returns:
            A tuple of (formula_array, num_nodes, root_id).
        """
        return self._jitted_sampler(key=key)

    @staticmethod
    def _static_sampler(key, propositions, min_levels, max_levels, min_conjunctions, max_conjunctions):
        """
        The core JAX-jittable static method to generate LTL formulas.
        """
        formula_array = jnp.zeros((MAX_NODES, 3), dtype=jnp.int32)
        
        key, subkey = random.split(key)
        num_conjs = random.randint(subkey, shape=(), minval=min_conjunctions, maxval=max_conjunctions + 1)

        def _sample_sequence_task(carry, _):
            key, formula_array, next_node_idx = carry
            key, subkey = random.split(key)
            seq_length = random.randint(subkey, shape=(), minval=min_levels, maxval=max_levels + 1)

            def _generate_seq_body(i, state):
                key, formula_array, next_node_idx, last_prop_ids, seq_node_ids = state
                mask = jnp.all(propositions[:, None] != last_prop_ids[None, :], axis=1)
                safe_mask = jnp.where(mask.sum() == 0, jnp.ones_like(mask), mask)
                probs = safe_mask.astype(jnp.float32) / safe_mask.sum()
                
                key, subkey_cond, subkey_disj = random.split(key, 3)
                
                def _create_disjunction(k):
                    p1, p2 = random.choice(k, propositions, shape=(2,), replace=False, p=probs)
                    node_idx, p1_idx, p2_idx = next_node_idx, next_node_idx + 1, next_node_idx + 2
                    arr = formula_array.at[node_idx].set(jnp.array([OR, p1_idx, p2_idx]))
                    arr = arr.at[p1_idx].set(jnp.array([p1, 0, 0]))
                    arr = arr.at[p2_idx].set(jnp.array([p2, 0, 0]))
                    return arr, node_idx, next_node_idx + 3, jnp.array([p1, p2])
                
                def _create_single_prop(k):
                    p1 = random.choice(k, propositions, shape=(1,), p=probs)[0]
                    node_idx = next_node_idx
                    arr = formula_array.at[node_idx].set(jnp.array([p1, 0, 0]))
                    return arr, node_idx, next_node_idx + 1, jnp.array([p1, -1])
                
                arr, node_id, next_idx, new_last_props = jax.lax.cond(
                    random.uniform(subkey_cond) < 0.25, _create_disjunction, _create_single_prop, subkey_disj)
                seq_node_ids = seq_node_ids.at[i].set(node_id)
                return key, arr, next_idx, new_last_props, seq_node_ids

            init_seq_state = (key, formula_array, next_node_idx, jnp.array([-1, -1]), jnp.full((max_levels,), -1, dtype=jnp.int32))
            key, formula_array, next_node_idx, _, seq_node_ids = jax.lax.fori_loop(0, seq_length, _generate_seq_body, init_seq_state)
            
            def _build_nested_formula(i, state):
                rev_i = seq_length - 2 - i
                _, formula_array, next_node_idx, current_root_id = state
                prop_node_id = seq_node_ids[rev_i]
                and_node_id = next_node_idx
                formula_array = formula_array.at[and_node_id].set(jnp.array([AND, prop_node_id, current_root_id]))
                eventually_node_id = next_node_idx + 1
                formula_array = formula_array.at[eventually_node_id].set(jnp.array([EVENTUALLY, and_node_id, 0]))
                return key, formula_array, next_node_idx + 2, eventually_node_id

            last_prop_node_id = seq_node_ids[seq_length - 1]
            initial_root_id = next_node_idx
            formula_array = formula_array.at[initial_root_id].set(jnp.array([EVENTUALLY, last_prop_node_id, 0]))
            
            init_build_state = (key, formula_array, next_node_idx + 1, initial_root_id)
            
            _, formula_array, next_node_idx, final_root_id = jax.lax.cond(
                seq_length > 1,
                lambda: jax.lax.fori_loop(0, seq_length - 1, _build_nested_formula, init_build_state),
                lambda: init_build_state)

            return (key, formula_array, next_node_idx), final_root_id, seq_length

        def _main_conj_loop_body(i, state):
            key, formula_array, next_node_idx, overall_root_id, total_levels = state
            (key, formula_array, next_node_idx), new_task_root_id, seq_length = _sample_sequence_task((key, formula_array, next_node_idx), 0)
            new_total_levels = total_levels + seq_length
            def _combine_with_and(op):
                prev_root_id, new_root_id, arr, idx = op
                and_node_id = idx
                arr = arr.at[and_node_id].set(jnp.array([AND, new_root_id, prev_root_id]))
                return arr, idx + 1, and_node_id

            def _first_task(op):
                _, new_root_id, arr, idx = op
                return arr, idx, new_root_id

            formula_array, next_node_idx, overall_root_id = jax.lax.cond(
                i > 0, _combine_with_and, _first_task, (overall_root_id, new_task_root_id, formula_array, next_node_idx))
            return key, formula_array, next_node_idx, overall_root_id, new_total_levels

        init_main_state = (key, formula_array, 0, -1, 0)
        key, formula_array, num_nodes, root_id, total_levels= jax.lax.fori_loop(0, num_conjs, _main_conj_loop_body, init_main_state)
        avg_levels = total_levels.astype(jnp.float32) / num_conjs.astype(jnp.float32)
        return formula_array, num_nodes, root_id, num_conjs, avg_levels
    
    