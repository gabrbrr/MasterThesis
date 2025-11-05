


import  spot
import logging
import random



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

    # NOTE: What about release and other temporal operators?
    if ltl_formula[0] == 'eventually':
        res = progress(ltl_formula[1], truth_assignment)
        return ("or", ltl_formula, res)

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
    # sampler = EventuallySampler(propositions_for_sampling, 
    #                             min_levels=2, 
    #                             max_levels=3, 
    #                             min_conjunctions=2, 
    #                             max_conjunctions=3)

    sampler= UntilTaskSampler(propositions_for_sampling, 
                                min_levels=4, 
                                max_levels=4, 
                                min_conjunctions=1, 
                                max_conjunctions=1)

    
    def simplify(formula):
        """
        Recursively simplifies an LTL formula tuple.
        Applies associativity and idempotency for 'and'/'or'.
        - Flattens nested 'and'/'or' operators.
        - Removes duplicates (Idempotency: A | A = A).
        - Rebuilds as a left-associative chain (order is not guaranteed).
        """
        # Base case: Proposition (string)
        if not isinstance(formula, tuple):
            return formula
        
        op = formula[0]
        
        # 1. Recursively simplify all operands first
        operands = [simplify(opnd) for opnd in formula[1:]]
        
        if op in ('and', 'or'):
            # --- 2. Associativity Step: Flatten ---
            # Collect all operands from nested structures of the *same* operator
            flattened_ops = []
            for opnd in operands:
                if isinstance(opnd, tuple) and opnd[0] == op:
                    # ...add its operands (which are already simplified) to the list.
                    flattened_ops.extend(opnd[1:])
                else:
                    # ...otherwise, just add the operand itself.
                    flattened_ops.append(opnd)
            
            # --- 3. Idempotency Step: Remove Duplicates ---
            # Since the tuples/strings used in the formula are hashable, 
            # using set() is a fast way to remove all duplicates.
            unique_ops = list(set(flattened_ops))
            
            # --- 4. Rebuild as a left-associative chain ---
            # The order from list(set(...)) is arbitrary.
            # This is fine, as canonical_form() will sort it later.
            
            if not unique_ops:
                # An empty 'and' is True, an empty 'or' is False
                return 'True' if op == 'and' else 'False'
            
            # Build the chain from the arbitrarily ordered unique list
            result = unique_ops[0]
            for i in range(1, len(unique_ops)):
                result = (op, result, unique_ops[i])
                
            return result

        elif op in ('not', 'eventually', 'always', 'next'):
            # Unary operators
            return (op, operands[0])
        
        elif op == 'until':
            # Binary, non-commutative
            return (op, operands[0], operands[1])
        
        else:
            # Unknown operator
            return (op,) + tuple(operands)


    
        
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
            # --- Associativity Step: Flatten ---
            # Collect all operands from nested structures of the *same* operator
            flattened_ops = []
            for opnd in canonical_operands:
                # If operand is a tuple AND its operator is the same as the current one...
                if isinstance(opnd, tuple) and opnd[0] == op:
                    # ...add its operands (which are already canonical) to the list.
                    flattened_ops.extend(opnd[1:])
                else:
                    # ...otherwise, just add the operand itself.
                    flattened_ops.append(opnd)
            
            # --- Commutativity Step: Sort ---
            def sort_key(item):
                # Sort by type (str vs tuple) first, then by string representation
                is_tuple = isinstance(item, tuple)
                return (is_tuple, str(item))
            
            sorted_ops = sorted(flattened_ops, key=sort_key)
            
            # --- Rebuild as a left-associative chain ---
            # This ensures ('or', 'a', ('or', 'b', 'c')) and 
            # ('or', ('or', 'a', 'b'), 'c')
            # both become ('or', ('or', 'a', 'b'), 'c') after sorting.
            
            if not sorted_ops:
                # This case handles an empty conjunction (True) or disjunction (False)
                return 'True' if op == 'and' else 'False'
            
            # Start with the first operand
            result = sorted_ops[0]
            
            # Iteratively add the rest, building the chain
            for i in range(1, len(sorted_ops)):
                result = (op, result, sorted_ops[i])
                
            return result

        elif op in ('not', 'eventually', 'always', 'next'): # Expanded to include all unary ops
            # Unary operators
            return (op, canonical_operands[0])
        
        elif op == 'until':
            # Binary, non-commutative
            return (op, canonical_operands[0], canonical_operands[1])
        
        else:
            # Unknown operator
            return (op,) + tuple(canonical_operands)


    
    step_count = 0
    current_formula = sampler.sample()
    while step_count < 1000: # Safety break
        
        # --- MODIFICATION 1: Re-sample every 5 steps ---
        if step_count > 0 and step_count % 50 == 0:
            logging.info("\n" + "*"*20)
            logging.info(f"🔥 Reached {step_count} iterations. Sampling a new formula.")
            logging.info("*"*20)
            current_formula = sampler.sample()
            logging.info(f"new_formula {ltl_tuple_to_string(current_formula)}")
            
        
        # This check is now more important, as we might pop 'True'/'False'
        if not isinstance(current_formula, tuple):
            current_formula = sampler.sample()
            logging.info(f"Sampling new_formula {ltl_tuple_to_string(current_formula)}")
        
        logging.info(f"\n--- 🔄 Processing Formula (Step {step_count+1}): ---")

        formula_props = get_propositions_in_formula(current_formula)
        all_single_prop_assignments = [[p] for p in random.sample(formula_props, len(formula_props))]

        for ta in all_single_prop_assignments:
            if (current_formula=="False" or current_formula=="True"):
                continue
            logging.info(f"  -> Progressing with {ta}:")
            
            try:
                # 1. Get the raw progressed formula
                progressed_formula = progress(current_formula, ta)
                custom_simplified_formula= simplify(progressed_formula)

                ltl_spot = _get_spot_format(progressed_formula)
                f = spot.formula(ltl_spot)
                f = spot.simplify(f)
                ltl_spot = f.__format__("l")
                simplified_formula,r = _get_std_format(ltl_spot.split(' '))
                # 3. Check if simplification occurred
                can_progressed_formula=canonical_form(custom_simplified_formula)
                can_simpilfied_formula=canonical_form(simplified_formula)
                if  can_progressed_formula!= can_simpilfied_formula:
                    #     print("    [DIFFERENT SIMPLIFICATION DETECTED! ]")
                    
                    # logging.info(f"      Not complete: {ltl_tuple_to_string(progressed_formula)}")
                    # logging.info(f"      Complete:  {ltl_tuple_to_string(simplified_formula)}")
                    logging.info(f"      Not complete canonicalized: {ltl_tuple_to_string(can_progressed_formula)}")
                    logging.info(f"      Complete canonicalized :  {ltl_tuple_to_string(can_simpilfied_formula)}")
                else:
                    logging.info(f"      Complete == Not complete  is : {ltl_tuple_to_string(progressed_formula)}")
                current_formula=simplified_formula
                    
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