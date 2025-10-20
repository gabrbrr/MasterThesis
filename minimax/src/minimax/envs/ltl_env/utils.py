import jax
import jax.numpy as jnp
import numpy as np

propositions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l","m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

LTL_BASE_VOCAB = {
    "and": 0, "or": 1, "not": 2, "next": 3, "until": 4,
    "always": 5, "eventually": 6, "True": 7, "False": 8,
}

PROP_OFFSET = len(LTL_BASE_VOCAB)
for i, el in enumerate(propositions):
    LTL_BASE_VOCAB[el] = PROP_OFFSET + i
# Define constants AND OR ... for easier access
globals().update({k.upper(): v for k, v in LTL_BASE_VOCAB.items() if v < PROP_OFFSET})

TRUE_VAL = LTL_BASE_VOCAB["True"]
FALSE_VAL = LTL_BASE_VOCAB["False"]
VOCAB_SIZE = len(LTL_BASE_VOCAB)
VOCAB_INV = {v: k for k, v in LTL_BASE_VOCAB.items()}

# Use jax.numpy for constants to be used in JIT'd code
_is_unary_op_np = np.zeros(VOCAB_SIZE, dtype=bool)
_is_unary_op_np[NOT] = True
_is_unary_op_np[NEXT] = True
_is_unary_op_np[EVENTUALLY] = True
_is_unary_op_np[ALWAYS] = True
IS_UNARY_OP = jnp.array(_is_unary_op_np)

_is_binary_op_np = np.zeros(VOCAB_SIZE, dtype=bool)
_is_binary_op_np[AND] = True
_is_binary_op_np[OR] = True
_is_binary_op_np[UNTIL] = True
IS_BINARY_OP = jnp.array(_is_binary_op_np)
LTL_BASE_VOCAB["propositions"]=propositions

MAX_NODES = 200 # Maximum size of the formula array

def encode_letters(letter_str: str) -> tuple:
    """Helper function to encode a string of letters into a tuple of IDs."""
    return tuple(LTL_BASE_VOCAB[l] for l in letter_str)

def encode_formula(formula):
        """Recursively encodes a formula from string/tuple to integer representation."""
        if isinstance(formula, str): 
            return LTL_BASE_VOCAB[formula]
        if isinstance(formula, tuple): return tuple(encode_formula(f) for f in formula)
        raise ValueError(f"Unsupported element type: {formula}")


def encode_formula_to_array(formula, vocab, array, index=0):
    if isinstance(formula, int):
        array[index] = [formula, 0, 0]
        return index + 1

    op, children = formula[0], formula[1:]

    if len(children) == 1:
        array[index] = [op, index + 1, 0]
        return encode_formula_to_array(children[0], vocab, array, index + 1)
    elif len(children) == 2:
        left_index = index + 1
        right_start_index = encode_formula_to_array(children[0], vocab, array, left_index)
        array[index] = [op, left_index, right_start_index]
        return encode_formula_to_array(children[1], vocab, array, right_start_index)
    raise ValueError("Formulas must have 1 or 2 children")


def decode_array_to_formula(array, node_index, num_valid_nodes, visited_nodes=None):
    if visited_nodes is None:
        visited_nodes = set()
    node_index=int(node_index)
    if not (0 <= node_index < num_valid_nodes):
        return f"invalid_ref_{node_index}"

    # A cycle is detected only if we revisit a non-terminal node.
    # Shared terminal nodes like 'True' or 'False' are valid.
    op_val, left_idx, right_idx = array[node_index]

    op_val, left_idx, right_idx = int(op_val), int(left_idx), int(right_idx)
    is_terminal = not (IS_UNARY_OP[op_val] or IS_BINARY_OP[op_val])

    if not is_terminal and node_index in visited_nodes:
        return f"ref_{node_index}"

    visited_nodes.add(node_index)

    op_str = VOCAB_INV.get(op_val, f"p{op_val}")

    if is_terminal:
        # Once we decode a terminal, we can remove it from the visited set
        # to allow it to be decoded again if shared by another branch.
        # This is not strictly necessary with the above check, but is good practice.
        visited_nodes.remove(node_index)
        return op_str

    if IS_UNARY_OP[op_val]:
        child = decode_array_to_formula(array, int(left_idx), num_valid_nodes, visited_nodes)
        visited_nodes.remove(node_index)
        return (op_str, child)

    if IS_BINARY_OP[op_val]:
        left_child = decode_array_to_formula(array, int(left_idx), num_valid_nodes, visited_nodes)
        right_child = decode_array_to_formula(array, int(right_idx), num_valid_nodes, visited_nodes)
        visited_nodes.remove(node_index)
        return (op_str, left_child, right_child)

    # Should be unreachable if logic is correct
    visited_nodes.remove(node_index)
    return op_str
