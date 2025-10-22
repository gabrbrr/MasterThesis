import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable, List, NamedTuple, Optional
from minimax.envs.ltl_env.utils import *
from collections import namedtuple, OrderedDict

edge_types = {"self": 0, "arg": 1, "arg1": 2, "arg2": 3}
NUM_EDGE_TYPES = len(edge_types)




class JaxASTBuilder:
    """
    Builds a GNN-compatible graph representation from an LTL formula array.
    This version is designed to be fully JAX-compilable and compatible with
    jax.ops.segment_sum by using a dedicated padding node at index 0.
    """
    def __init__(self, vocab_size: int, max_formula_nodes: int):
        self.vocab_size = vocab_size
        # The total number of nodes in the graph is max_formula_nodes + 1 (for the padding node)
        self.max_graph_nodes = max_formula_nodes + 1
        self.max_formula_nodes = max_formula_nodes


    """Creates and JIT-compiles the static graph building function."""
    @staticmethod
    @partial(jax.jit, static_argnames=['max_formula_nodes', 'vocab_size', 'max_graph_nodes'])
    def _build_graph_static(encoded_array, num_nodes: int, max_formula_nodes: int, vocab_size: int, max_graph_nodes: int):
        """
        JIT-compiled static method to construct the graph.
        This function returns PADDED graph arrays compatible with segment_sum.
        - Node 0 is a dedicated padding node.
        - Real formula nodes are indexed from 1 to num_nodes.
        - Padded edges are self-loops on node 0.
        """
        # === 1. Node Feature Construction ===
        # We have max_graph_nodes = max_formula_nodes + 1 total nodes. Node 0 is for padding.
        node_tokens = encoded_array[:max_formula_nodes, 0]
        one_hot_features = jax.nn.one_hot(node_tokens, num_classes=vocab_size, dtype=jnp.float32)

        # Initialize all node features to zero, including the padding node at index 0.
        node_features = jnp.zeros((max_graph_nodes, vocab_size), dtype=jnp.float32)
        # Place the one-hot features for the real nodes starting from index 1.
        node_features = node_features.at[1:].set(one_hot_features)

        # Mask out features for padded formula nodes (beyond num_nodes).
        # The mask starts from index 1, as node 0 is always a padding node.
        valid_node_mask = jnp.arange(max_graph_nodes) < (num_nodes + 1)
        node_features = node_features * valid_node_mask[:, None]

        # The root is now at index 1.
        is_root_feature = jnp.zeros((max_graph_nodes, 1), dtype=jnp.float32).at[1].set(1.0)
        node_features = jnp.concatenate([node_features, is_root_feature], axis=-1)

        # === 2. Edge Construction Loop ===
        def edge_construction_body(i, carry):
            senders, receivers, edge_type_indices = carry
            op, left_idx, right_idx = encoded_array[i]

            # **MODIFIED**: Shift all node indices by +1 to account for the padding node.
            current_node_idx = i + 1
            left_child_idx = left_idx + 1
            right_child_idx = right_idx + 1

            # Self-loop for the current real node.
            senders = senders.at[3 * i].set(current_node_idx)
            receivers = receivers.at[3 * i].set(current_node_idx)
            edge_type_indices = edge_type_indices.at[3 * i].set(edge_types['self'])

            # Unary op
            def unary_true_fn(vals):
                s, r, e = vals
                s = s.at[3 * i + 1].set(left_child_idx)
                r = r.at[3 * i + 1].set(current_node_idx)
                e = e.at[3 * i + 1].set(edge_types['arg'])
                return s, r, e
            senders, receivers, edge_type_indices = jax.lax.cond(
                IS_UNARY_OP[op], unary_true_fn, lambda vals: vals, (senders, receivers, edge_type_indices)
            )

            # Binary op
            def binary_true_fn(vals):
                s, r, e = vals
                s = s.at[3 * i + 1].set(left_child_idx)
                r = r.at[3 * i + 1].set(current_node_idx)
                e = e.at[3 * i + 1].set(edge_types['arg1'])
                s = s.at[3 * i + 2].set(right_child_idx)
                r = r.at[3 * i + 2].set(current_node_idx)
                e = e.at[3 * i + 2].set(edge_types['arg2'])
                return s, r, e
            senders, receivers, edge_type_indices = jax.lax.cond(
                IS_BINARY_OP[op], binary_true_fn, lambda vals: vals, (senders, receivers, edge_type_indices)
            )
            return senders, receivers, edge_type_indices

        # Initialize with a sentinel value to detect validly created edges.
        num_potential_edges = max_formula_nodes * 3
        init_val = jnp.full(num_potential_edges, -1, dtype=jnp.int32)
        senders_padded, receivers_padded, edge_types_padded = jax.lax.fori_loop(
            0, num_nodes, edge_construction_body, (init_val, init_val, init_val)
        )

        # === 3. Gather Valid Edges and Pad for segment_sum ===
        valid_edge_mask = senders_padded != -1
        num_valid_edges = valid_edge_mask.sum()

        # Use argsort to efficiently move all valid edges to the front.
        sort_key = jnp.where(valid_edge_mask, jnp.arange(num_potential_edges), num_potential_edges)
        permutation = jnp.argsort(sort_key)
        senders_clean = senders_padded[permutation]
        receivers_clean = receivers_padded[permutation]
        edge_types_clean = edge_types_padded[permutation]

        # **MODIFIED**: Create a mask for padded edges (those after the valid ones).
        # All padded senders and receivers should point to the padding node (index 0).
        is_padded_edge_mask = jnp.arange(num_potential_edges) >= num_valid_edges
        final_senders = jnp.where(is_padded_edge_mask, 0, senders_clean)
        final_receivers = jnp.where(is_padded_edge_mask, 0, receivers_clean)
        # We can also set the edge type to 'self' for padded edges.
        final_edge_types = jnp.where(is_padded_edge_mask, edge_types['self'], edge_types_clean)

        # # === 4. Edge Feature Construction ===
        # edge_features = jax.nn.one_hot(final_edge_types, num_classes=NUM_EDGE_TYPES, dtype=jnp.float32)
        # # Mask out features for padded edges to ensure they are zero.
        # edge_features = edge_features * (~is_padded_edge_mask)[:, None]

        return OrderedDict([
            ('nodes', node_features),
            ('senders', final_senders),
            ('receivers', final_receivers),
            ('n_node', jnp.array([num_nodes])),
            ('edge_types', final_edge_types)
        ])


    @partial(jax.jit, static_argnums=0)
    def __call__(self, encoded_array: jnp.ndarray, num_nodes: int):
        """
        Processes an encoded formula array to produce a padded graph representation
        that is compatible with jax.ops.segment_sum.
        """
        # The __call__ can't be jitted with self, as it would recompile for every instance.
        # The performance comes from jitting the static _build_graph_static method.
        return JaxASTBuilder._build_graph_static(encoded_array, num_nodes,max_formula_nodes=self.max_formula_nodes,max_graph_nodes=self.max_graph_nodes,vocab_size=VOCAB_SIZE)
