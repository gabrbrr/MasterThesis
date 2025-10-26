import jax
import jax.numpy as jnp
import jax.lax as lax
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union
from functools import partial
import numpy as np
import spot
from sfl.envs.ltl_env.utils import *

def simplify_spot(array, node_index, num_valid_nodes):
    tuple_string_format=decode_array_to_formula(array, node_index, num_valid_nodes)
    array=np.array(array)
    ltl_spot = _get_spot_format(tuple_string_format)
    f = spot.formula(ltl_spot)
    f = spot.simplify(f)
    ltl_spot = f.__format__("l")
    ltl_std,r = _get_std_format(ltl_spot.split(' '))
    new_array = np.zeros_like(array)
    num_nodes=encode_formula_to_array(encode_formula(ltl_std), LTL_BASE_VOCAB, array)
    array=jnp.array(array)
    return array, 0, num_nodes

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



@jax.jit
def progress_and_clean_jax(formula_array, truth_assignment, root_index, num_nodes):
    """
    The main JIT-compiled function that orchestrates the workflow.
    It calls the core JAX logic and then the NumPy callback for simplification.
    """
    # 1. Run the initial JAX-compatible part of your logic
    dirty_root_idx, dirty_array, dirty_num_nodes = jax_static_iterative_progress_no_copy(
        formula_array, truth_assignment, root_index, num_nodes
    )
    

    # 2. Define the shapes and dtypes for the callback's output.
    #    This is the "contract" that JAX needs to compile the rest of the graph.
    result_shape_and_dtype = (
        jax.ShapeDtypeStruct(dirty_array.shape, dirty_array.dtype),
        jax.ShapeDtypeStruct(dirty_root_idx.shape, dirty_root_idx.dtype),
        jax.ShapeDtypeStruct(dirty_num_nodes.shape, dirty_num_nodes.dtype),
    )

    # 3. Call the Python function via jax.pure_callback
    simplified_array, simplified_root_idx, simplified_num_nodes = jax.pure_callback(
        simplify_spot, # The Python function to call
        result_shape_and_dtype,      # The "contract" for the output
        dirty_array,                 # Arguments to the callback
        dirty_root_idx,
        dirty_num_nodes,
        vmap_method='sequential'
    )

    return simplified_array, simplified_root_idx, simplified_num_nodes


@jax.jit
def jax_static_iterative_progress_no_copy(formula_array, truth_assignment, root_index, num_nodes):
    true_node_idx = num_nodes
    formula_array = formula_array.at[true_node_idx].set(jnp.array([TRUE_VAL, 0, 0]))
    num_nodes += 1

    false_node_idx = num_nodes
    formula_array = formula_array.at[false_node_idx].set(jnp.array([FALSE_VAL, 0, 0]))
    num_nodes += 1

    results = jnp.full(MAX_NODES, -1, dtype=jnp.int32)
    stack = jnp.zeros((MAX_NODES, 2), dtype=jnp.int32)
    stack_ptr = 0
    stack = stack.at[stack_ptr].set(jnp.array([root_index, 0]))
    stack_ptr += 1

    init_state = (formula_array, results, stack, stack_ptr, num_nodes, true_node_idx, false_node_idx)

    def main_loop_body(i, state):
        fa, res, st, sp, nn, true_idx, false_idx = state

        def process_stack_top(carry):
            fa, res, st, sp, nn, t_idx, f_idx = carry
            sp -= 1
            node_index, processed = st[sp]
            op, left_idx, right_idx = fa[node_index]

            is_atomic = (left_idx == 0) & (right_idx == 0) & ~(IS_UNARY_OP[op] | IS_BINARY_OP[op])


            def compute_node(compute_carry):
                fa, res, st, sp, nn, t_idx, f_idx = compute_carry

                def process_parent(parent_carry):
                    fa, res, st, sp, nn, t_idx, f_idx = parent_carry

                    def handle_unary(unary_carry):
                        fa_u, res_u, nn_u, sp_u = unary_carry
                        child_res_idx = res_u[left_idx]
                        child_res_val = fa_u[child_res_idx, 0]

                        def case_not(c):
                            res = c[0]
                            res_idx = jnp.where(child_res_val == TRUE_VAL, f_idx, t_idx)
                            return res.at[node_index].set(res_idx)

                        def case_next(c):
                            res = c[0]
                            return res.at[node_index].set(child_res_idx)

                        def case_temporal(op_val, c):
                            fa, res, nn = c
                            new_fa = fa.at[nn].set(jnp.array([op_val, node_index, child_res_idx]))
                            return res.at[node_index].set(nn), new_fa, nn + 1

                        res = lax.cond(
                            op == LTL_BASE_VOCAB["not"], case_not,
                            lambda c: lax.cond(op == LTL_BASE_VOCAB["next"], case_next, lambda x: x[0], c),
                            (res_u,)
                        )
                        res, fa, nn = lax.cond(
                            op == LTL_BASE_VOCAB["always"], lambda c: case_temporal(LTL_BASE_VOCAB["and"], c),
                            lambda c: lax.cond(
                                op == LTL_BASE_VOCAB["eventually"], lambda c2: case_temporal(LTL_BASE_VOCAB["or"], c2),
                                lambda c3: (c3[1], c3[0], c3[2]), c), # (res, fa, nn)
                            (fa_u, res, nn_u)
                        )
                        return fa, res, st, sp, nn, t_idx, f_idx

                    def handle_binary(binary_carry):
                        fa_b, res_b, nn_b, sp_b = binary_carry
                        left_res_idx, right_res_idx = res_b[left_idx], res_b[right_idx]
                        left_res_val, right_res_val = fa_b[left_res_idx, 0], fa_b[right_res_idx, 0]

                        def handle_and_or(carry):
                            fa_ao, res_ao, nn_ao, sp_ao = carry
                            is_and = op == LTL_BASE_VOCAB["and"]
                            term_val = jnp.where(is_and, FALSE_VAL, TRUE_VAL)
                            ident_val = jnp.where(is_and, TRUE_VAL, FALSE_VAL)

                            def make_new_op_node(c):
                                fa, nn = c
                                new_fa = fa.at[nn].set(jnp.array([op, left_res_idx, right_res_idx]))
                                return new_fa, nn + 1, nn

                            is_term = (left_res_val == term_val) | (right_res_val == term_val)
                            res_idx = jnp.where(is_and, f_idx, t_idx)

                            fa, nn, res_idx = lax.cond(
                               is_term, lambda c: (c[0],c[1], res_idx),
                               lambda c: lax.cond( (left_res_val==ident_val) & (right_res_val==ident_val), lambda c2: (c2[0],c2[1], jnp.where(is_and, t_idx, f_idx)),
                               lambda c2: lax.cond( left_res_val==ident_val, lambda c3: (c3[0],c3[1],right_res_idx),
                               lambda c3: lax.cond( right_res_val==ident_val, lambda c4: (c4[0],c4[1],left_res_idx), make_new_op_node,c3),c2),c),
                               (fa_ao, nn_ao))

                            res = res_ao.at[node_index].set(res_idx)
                            return fa, res, st, sp, nn, t_idx, f_idx

                        def handle_until(carry):
                            fa_u, res_u, nn_u, sp_u = carry

                            def make_and_for_f1(c):
                                fa, nn = c
                                new_fa = fa.at[nn].set(jnp.array([LTL_BASE_VOCAB["and"], left_res_idx, node_index]))
                                return new_fa, nn + 1, nn

                            fa, nn, f1_idx = lax.cond(
                                left_res_val == FALSE_VAL, lambda c: (c[0], c[1], f_idx),
                                lambda c: lax.cond(left_res_val == TRUE_VAL, lambda c2: (c2[0], c2[1], node_index), make_and_for_f1, c),
                                (fa_u, nn_u))

                            def make_or_for_final(c):
                                fa_in, nn_in = c
                                new_fa = fa_in.at[nn_in].set(jnp.array([LTL_BASE_VOCAB["or"], right_res_idx, f1_idx]))
                                return new_fa, nn_in + 1, nn_in

                            fa, nn, final_res_idx = lax.cond(
                                right_res_val == TRUE_VAL, lambda c: (c[0], c[1], t_idx),
                                lambda c: lax.cond(right_res_val == FALSE_VAL, lambda c2: (c2[0], c2[1], f1_idx), make_or_for_final, c),
                                (fa, nn))

                            res = res_u.at[node_index].set(final_res_idx)
                            return fa, res, st, sp, nn, t_idx, f_idx

                        return lax.cond(
                            (op == LTL_BASE_VOCAB["and"]) | (op == LTL_BASE_VOCAB["or"]),
                            handle_and_or,
                            handle_until,
                            binary_carry
                        )

                    return lax.cond(
                        IS_UNARY_OP[op],
                        handle_unary,
                        handle_binary,
                        (fa, res, nn, sp)
                    )

                def push_children(pre_carry):
                    fa, res, st, sp, nn, t_idx, f_idx = pre_carry
                    st = st.at[sp].set(jnp.array([node_index, 1]))
                    sp += 1
                    st, sp = lax.cond(IS_BINARY_OP[op], lambda c: (c[0].at[c[1]].set(jnp.array([right_idx, 0])), c[1] + 1), lambda c: c, (st, sp))
                    st = st.at[sp].set(jnp.array([left_idx, 0]))
                    sp += 1
                    return fa, res, st, sp, nn, t_idx, f_idx

                return lax.cond(processed == 1, process_parent, push_children, compute_carry)

            def process_atomic(atomic_carry):
                fa, res, st, sp, nn, t_idx, f_idx = atomic_carry
                is_true = jnp.any(op == truth_assignment)
                prop_res = jnp.where(is_true, t_idx, f_idx)
                final_res = jnp.where(op == TRUE_VAL, t_idx, jnp.where(op == FALSE_VAL, f_idx, prop_res))
                return fa, res.at[node_index].set(final_res), st, sp, nn, t_idx, f_idx

            return lax.cond(
                res[node_index] != -1, lambda x: x,
                lambda y: lax.cond(is_atomic, process_atomic, compute_node, y),
                (fa, res, st, sp, nn, true_idx, false_idx)
            )

        return lax.cond(sp > 0, process_stack_top, lambda x: x, state)

    final_state = lax.fori_loop(0, MAX_NODES * 2, main_loop_body, init_state)
    final_fa, final_res, _, _, final_nn, _, _ = final_state

    return final_res[root_index], final_fa, final_nn


