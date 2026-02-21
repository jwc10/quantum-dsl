# test_decorator.py

# Import the decorator and draw function (works both when run directly and as a module)
try:
    from .dsl_decorator import quantum, draw
except ImportError:
    from dsl_decorator import quantum, draw
import numpy as np
import pennylane as qml
from sympy import symbols, Implies, Equivalent
from sympy.logic.boolalg import to_cnf, Xor

# Test the decorator with a simple example
@quantum
def simple_test():
    # Classical Python code
    print("Starting quantum test...")
    
    # DSL quantum operations
    qubit(3)
    hadamard(0)
    xgate(1)
    cnot(0, 1)
    
    # More classical code
    print("Applied quantum gates")
    
    # Measurement
    result = measure_probs(0, 1, 2)
    print(f"Measurement result: {result}")
    
    return result


# Test with proper phase-flipping oracles
@quantum
def constant_oracle_test():
    # Constant oracle: always returns 1 (no phase flips)
    print("Constant Oracle: always returns 1 (no phase flips)")
    
    qubit(2)
    hadamard(0)
    hadamard(1)
    
    # Constant oracle: do nothing (identity)
    # This represents a function that always returns 1
    # No phase flips needed
    
    hadamard(0)
    hadamard(1)
    
    result = measure_probs(0, 1)
    print(f"Constant oracle result: {result}")
    
    return result

@quantum
def balanced_oracle_2qubits():
    cnot(0, 2)

@quantum
def balanced_oracle_test():
    # Single solution oracle: flip phase only for |11⟩
    print("2 (out of 4) solution oracle, flips phase for |11⟩ and |10⟩")
    
    qubit(3)
    hadamard_transform([0, 1])

    xgate(2)
    hadamard(2)
    
    # flips phase of the state of the first two qubits if the first two qubits form a solution state
    #this is the balanced oracle for 2 qubits
    balanced_oracle_2qubits()  # This creates the phase flip for |11⟩ and |10⟩ - phase kickback
    
    hadamard_transform([0, 1])
    
    result = measure_probs(0, 1)
    print(f"Balanced oracle result: {result}")
    
    return result

#DOESNT WORK - issue with not being able to just have subroutines/subcircuits
# def diffuser_operator():
#     hadamard_transform(0, 1)
#     multi_controlled_x([0, 1, 2])
#     hadamard_transform(0, 1)
#     return

# Test with classical data preparation
@quantum
def grover_test():
    # Quantum operations
    qubit(4)
    all_combos = 8 # for the 3 qubits in the actual qubit register
    optimal_steps = int(np.pi/4 * np.sqrt(all_combos))
    print(f"Optimal steps: {optimal_steps}")
    xgate(3)
    hadamard(3)
    hadamard_transform([0, 1, 2])

    for i in range(optimal_steps):
        #oracle
        multi_controlled_x([0, 1, 2, 3], control_vals=[1, 1, 1])

        #diffuser
        hadamard_transform([0, 1, 2])
        multi_controlled_x([0, 1, 2, 3], control_vals=[0, 0, 0])
        hadamard_transform([0, 1, 2])
        
    # Measurement
    result = measure_probs(0, 1, 2)
    print(f"Final result: {result}")
    
    return result





def expr_to_cnf(expr_str, inputs):
    """
    Convert a boolean expression string to CNF form.
    
    Input parameters:
    - expr_str: A string containing a boolean expression using logical operators
               Examples: 
                 - Basic: "(a & b) | (~c & d)", "a & (b | c)"
                 - With implications: "a >> b" (a implies b), "(a >> b) & (b >> c)", "Implies(a, b)"
                 - XOR: "Xor(a, b)", "(Xor(a, b) & c) | d"
                 - Equivalent (biconditional): "Equivalent(a, b)" (a if and only if b)
                 - Complex: "(a | b) & (~a | c) & (b | ~c)", "(a >> b) & (b >> c) >> (a >> c)"
               Operators supported: 
                 - & (AND), | (OR), ~ (NOT) - work directly
                 - >> (implies) - sympy Symbol objects override >> to mean Implies
                 - Xor(a, b), Implies(a, b), Equivalent(a, b) - function calls (all available)
               Note: sympy's to_cnf() will convert implications, XOR, biconditionals, and other 
                     operators to CNF form automatically
               Parentheses can be used for grouping
    - inputs: A list of variable names (strings) that appear in expr_str
             Example: ["a", "b", "c", "d"] for variables a, b, c, d
             These are the primary search qubits in the quantum circuit
    
    The simplify=True parameter in to_cnf():
    - When True: sympy applies logical simplification rules to reduce the CNF to a more compact form
                (e.g., removes redundant clauses, simplifies tautologies)
    - When False: Returns the raw CNF without simplification (can be larger but preserves structure)
    - Example: "(a | b) & (a | b)" with simplify=True becomes "a | b" (removes duplicate)
    
    Output structure:
    - If single clause: returns the clause directly (e.g., 'a | b')
    - If multiple clauses: returns an And object where each argument is a clause
    - Each clause is an Or of literals (variables or their negations)
    
    Example with 3 clauses:
    Input: expr_str = "(a | b) & (~a | c) & (b | ~c)", inputs = ["a", "b", "c"]
    Output: And(Or(a, b), Or(~a, c), Or(b, ~c))
    - This means: (a OR b) AND (NOT a OR c) AND (b OR NOT c)
    - Structure: output.func.__name__ == "And"
    - Clauses: output.args = [Or(a, b), Or(~a, c), Or(b, ~c)]  (3 clauses)
    - Each clause: first clause is Or(a, b), second is Or(~a, c), third is Or(b, ~c)
    
    To extract clauses: 
    - If output.func.__name__ == "And": clauses = output.args
    - Otherwise: clauses = [output]  (single clause)
    - Each clause can be checked: clause.func.__name__ == "Or" to get literals
    """
    # Create a dictionary mapping variable names to sympy Symbol objects
    # For inputs=["a", "b", "c"], this creates {"a": Symbol('a'), "b": Symbol('b'), "c": Symbol('c')}
    # These Symbol objects allow sympy to parse and manipulate the boolean expression
    sym = {n: symbols(n) for n in inputs}
    
    # Make sympy boolean functions available for use in expressions
    # Users can now use Xor(a, b), Implies(a, b), Equivalent(a, b) in their expr_str
    sym.update({'Xor': Xor, 'Implies': Implies, 'Equivalent': Equivalent})
    
    # Evaluate the expression string as Python code, but replace variable names with sympy Symbols
    # eval(expr_str, {}, sym) evaluates expr_str in a safe namespace where:
    #   - First {} = no built-ins/globals (empty globals dict)
    #   - sym = local variables (our Symbol dictionary + boolean functions)
    # So when expr_str contains "a & b", eval replaces "a" and "b" with Symbol('a') and Symbol('b')
    # Or "a >> b" uses sympy's Implies (Symbol objects override >>)
    # Or "Xor(a, b)" uses the Xor function from sympy
    # Result: expr becomes a sympy expression like And(Symbol('a'), Symbol('b'))
    expr = eval(expr_str, {}, sym)
    
    return to_cnf(expr, simplify=True)




# #might need to figure out how to handle the declaration of qubits bc we might not know ahead of time how many qubits we will need to declare in the funciton that calls this one

# # might need this to check what state the phase qubit is in to ensure phase kickback is working correctly
# @quantum
# def build_cnf_oracle(expr_str, inputs, start_qubit_idx=0):
#     """
#     Build a phase-flip oracle for a Boolean expression using DSL functions.
#     - expr_str: e.g. '(a & b) | (~c & d)'
#     - inputs: list of variable names (primary search qubits)
#     - start_qubit_idx: starting index for primary qubits (default 0)
    
#     Returns: (num_primary, num_ancilla, num_phase) - qubit allocation info
#     """
#     # Convert to CNF
#     cnf = expr_to_cnf(expr_str, inputs)
#     clauses = cnf.args if cnf.func.__name__ == "And" else [cnf]
    
#     num_primary = len(inputs)
#     num_ancilla = len(clauses)
#     num_phase = 1
    
#     # Map variable name → qubit index
#     idx = {name: start_qubit_idx + i for i, name in enumerate(inputs)}
#     ancilla_start = start_qubit_idx + num_primary
#     phase_qubit = ancilla_start + num_ancilla
    
#     # --- Compute each clause into its ancilla ---
#     for j, clause in enumerate(clauses):
#         ancilla_qubit = ancilla_start + j
        
#         # Extract literals
#         lits = clause.args if clause.func.__name__ == "Or" else [clause]
        
#         # Apply De Morgan's: For OR clause (b | c), use ~(~b & ~c)
#         # Strategy: Prepare ancilla in |1⟩, use multi_controlled_x to flip to |0⟩ when (~b & ~c)
#         # This leaves ancilla in |1⟩ when (b | c) is satisfied
#         if len(lits) > 1:
#             # OR clause: (b | c) = ~(~b & ~c)
#             # Prepare ancilla in |1⟩
#             xgate(ancilla_qubit)
            
#             # Build control qubits and control values using De Morgan's
#             # For (b | c): detect when (~b & ~c) is true, i.e., when b=0 AND c=0
#             # So use control_vals=[0, 0] to detect both |0⟩
#             control_qubits = []
#             control_vals = []
#             for lit in lits:
#                 if lit.func.__name__ == "Not":
#                     # For ~a: we want to detect when a=1 (since ~a means we want a to be false)
#                     # In De Morgan's: (~a | b) = ~(a & ~b), so we detect when a=1 AND ~b
#                     var_name = str(lit.args[0])
#                     qubit_idx = idx[var_name]
#                     control_qubits.append(qubit_idx)
#                     control_vals.append(1)  # Detect when a=1 (meaning ~a is false, contributing to AND)
#                 else:
#                     # For a: we want to detect when a=0 (since we're checking ~a)
#                     var_name = str(lit)
#                     qubit_idx = idx[var_name]
#                     control_qubits.append(qubit_idx)
#                     control_vals.append(0)  # Detect when a=0 (meaning ~a is true, contributing to AND)
            
#             # Multi-controlled X: flips ancilla when all controls match control_vals
#             # This detects when (~b & ~c) is true, flipping ancilla from |1⟩ to |0⟩
#             # Leaving ancilla as |1⟩ when (b | c) is true
#             if control_qubits:
#                 all_wires = control_qubits + [ancilla_qubit]
#                 multi_controlled_x(all_wires, control_vals)
#         else:
#             # Single literal clause: use multi_controlled_x directly
#             lit = lits[0]
#             if lit.func.__name__ == "Not":
#                 var_name = str(lit.args[0])
#                 qubit_idx = idx[var_name]
#                 multi_controlled_x([qubit_idx, ancilla_qubit], [0])
#             else:
#                 var_name = str(lit)
#                 qubit_idx = idx[var_name]
#                 multi_controlled_x([qubit_idx, ancilla_qubit], [1])
    
#     # --- Phase flip if all clauses satisfied ---
#     ancilla_wires = list(range(ancilla_start, ancilla_start + num_ancilla))
#     phase_wires = ancilla_wires + [phase_qubit]
    
#     # Multi-controlled X with all ancillas as controls - checks if all clauses are satisfied
#     # Since phase qubit is in |-> state: flipping |-> → -|-> creates phase kickback
#     # This works on superpositions: for each basis state where all ancillas are 1,
#     # the phase qubit gets flipped from |-> to -|->, which phase-flips that basis state
#     # No uncompute needed - the phase information is already kicked back to primary qubits
#     control_vals = [1] * num_ancilla
#     multi_controlled_x(phase_wires, control_vals) #ASSUMES PHASE QUBIT IS IN |-⟩ STATE
    
#     # --- Uncompute clauses (reverse order) ---
#     # Why uncompute? After phase kickback, the ancilla qubits are still entangled with primary qubits.
#     # Uncomputing the clauses:
#     # 1. Disentangles ancillas from primary qubits (ancillas go back to |0⟩)
#     # 2. Allows ancillas to be reused or measured cleanly without affecting primary qubits
#     # 3. Ensures the oracle only affects primary qubits via phase, not through entanglement
#     # Note: We DON'T uncompute the phase qubit - its phase information is already kicked back
#     # Same control_vals as computation (no X gates needed - control_vals handle inversions)
#     for j, clause in reversed(list(enumerate(clauses))):
#         ancilla_qubit = ancilla_start + j
        
#         # Extract literals
#         lits = clause.args if clause.func.__name__ == "Or" else [clause]
        
#         # Uncompute using the same De Morgan's approach (reverse of computation)
#         if len(lits) > 1:
#             # OR clause: uncompute by reversing the multi_controlled_x, then uncompute initial X
#             # Build same control pattern as computation (no X gates needed - control_vals handle it)
#             control_qubits = []
#             control_vals = []
#             for lit in lits:
#                 if lit.func.__name__ == "Not":
#                     var_name = str(lit.args[0])
#                     qubit_idx = idx[var_name]
#                     control_qubits.append(qubit_idx)
#                     control_vals.append(1)  # Same as computation
#                 else:
#                     var_name = str(lit)
#                     qubit_idx = idx[var_name]
#                     control_qubits.append(qubit_idx)
#                     control_vals.append(0)  # Same as computation
            
#             # Uncompute the multi_controlled_x (same operation, reversible)
#             if control_qubits:
#                 all_wires = control_qubits + [ancilla_qubit]
#                 multi_controlled_x(all_wires, control_vals)
            
#             # Uncompute initial X on ancilla (return it to |0⟩)
#             xgate(ancilla_qubit)
#         else:
#             # Single literal clause: uncompute multi_controlled_x
#             lit = lits[0]
#             if lit.func.__name__ == "Not":
#                 var_name = str(lit.args[0])
#                 qubit_idx = idx[var_name]
#                 multi_controlled_x([qubit_idx, ancilla_qubit], [0])
#             else:
#                 var_name = str(lit)
#                 qubit_idx = idx[var_name]
#                 multi_controlled_x([qubit_idx, ancilla_qubit], [1])
    
#     return (num_primary, num_ancilla, num_phase)

# Test CNF oracle builder
@quantum
def test_cnf_oracle():
    """Test the CNF oracle builder with a simple expression."""
    expr = "(a & b) | (~c & d)"
    inputs = ["a", "b", "c", "d"]
    
    # Calculate how many qubits we need
    cnf = expr_to_cnf(expr, inputs)
    clauses = cnf.args if cnf.func.__name__ == "And" else [cnf]
    num_primary = len(inputs)
    num_ancilla = len(clauses)
    num_phase = 1
    total_qubits = num_primary + num_ancilla + num_phase
    
    print(f"CNF form: {cnf}")
    print(f"Total qubits needed: {total_qubits} (primary: {num_primary}, ancilla: {num_ancilla}, phase: {num_phase})")
    
    # Declare qubits
    qubit(total_qubits)
    
    # Put primary qubits in superposition
    hadamard_transform(list(range(num_primary)))
    
    # Prepare phase qubit in |-> state
    xgate(num_primary + num_ancilla)
    hadamard(num_primary + num_ancilla)
    
    # Build and apply the CNF oracle
    build_cnf_oracle(expr, inputs, start_qubit_idx=0)
    
    # Apply Hadamard transform back on primary qubits
    hadamard_transform(list(range(num_primary)))
    
    # Measure primary qubits
    result = measure_probs(*range(num_primary))
    print(f"CNF oracle result: {result}")
    
    return result

# Test Grover's search with CNF oracle to solve a SAT problem
@quantum
def test_grover_cnf_sat():
    """Test Grover's algorithm with CNF oracle to solve a Boolean satisfiability problem."""
    
    # Define a SAT problem with 6 variables (64 possible states) and exactly 2 solutions
    # Formula: (a & b & c & d & e & f) | (~a & ~b & ~c & ~d & ~e & ~f)
    # This means: ALL variables are 1, OR ALL variables are 0
    # Exactly 2 solutions:
    #   - |111111⟩ (state 63): a=1,b=1,c=1,d=1,e=1,f=1
    #   - |000000⟩ (state 0): a=0,b=0,c=0,d=0,e=0,f=0
    expr = "(a & b & c & d & e & f) | (~a & ~b & ~c & ~d & ~e & ~f)"
    inputs = ["a", "b", "c", "d", "e", "f"]
    num_solutions = 2  # Known number of solutions
    
    # Calculate qubit requirements
    cnf = expr_to_cnf(expr, inputs)
    clauses = cnf.args if cnf.func.__name__ == "And" else [cnf]
    num_primary = len(inputs)
    num_ancilla = len(clauses)
    num_phase = 1
    total_qubits = num_primary + num_ancilla + num_phase
    
    print(f"SAT expression: {expr}")
    print(f"CNF form: {cnf}")
    print(f"Number of clauses: {len(clauses)}")
    print(f"Known number of solutions: {num_solutions}")
    print(f"Total qubits needed: {total_qubits} (primary: {num_primary}, ancilla: {num_ancilla}, phase: {num_phase})")
    
    # Declare all qubits
    qubit(total_qubits)
    
    # Define primary qubits (the search space - values of a, b, c, d, e, f)
    primary_qubits = list(range(num_primary))
    # Phase qubit is the last qubit (after primary + ancilla qubits)
    phase_qubit = num_primary + num_ancilla
    
    # Create oracle function that wraps build_cnf_oracle
    # This oracle uses the phase qubit pattern (accepts both primary_qubits and phase_qubit)
    def cnf_oracle(primary_qubits, phase_qubit):
        """Oracle that marks satisfying assignments using CNF oracle."""
        # Note: grover_search prepares phase_qubit in |-> state before calling this oracle
        # build_cnf_oracle expects phase_qubit to be in |-> state, which it already is
        # Build and apply CNF oracle (this will phase-flip satisfying assignments)
        build_cnf_oracle(expr, inputs, start_qubit_idx=0)
    
    # Calculate optimal number of Grover iterations
    # For n qubits with M solutions, optimal iterations = π/4 * sqrt(2^n / M)
    # With 6 primary qubits (64 possible assignments) and 2 solutions:
    search_space_size = 2 ** num_primary
    optimal_float = np.pi / 4 * np.sqrt(search_space_size / num_solutions)
    optimal_iterations = max(1, int(optimal_float))
    
    print(f"Search space size: {search_space_size} possible assignments")
    print(f"Optimal Grover iterations: {optimal_iterations}")
    
    # Run Grover's search algorithm
    grover_search(cnf_oracle, primary_qubits, phase_qubit, optimal_iterations)
    
    # Measure the result (measure primary qubits to see which assignment satisfies the formula)
    result = measure_probs(*primary_qubits)
    
    print(f"\nGrover SAT search result: {result}")
    print("The states with high probability are satisfying assignments for the SAT problem.")
    print("\nExpected solutions (states with high probability):")
    print("  State 0 (|000000⟩): a=0, b=0, c=0, d=0, e=0, f=0")
    print("  State 63 (|111111⟩): a=1, b=1, c=1, d=1, e=1, f=1")
    print("\nTo interpret: Each state index represents a binary assignment to [a, b, c, d, e, f]")
    print("For example, state index 0 = |000000⟩ means all variables are 0")
    
    return result

# Test mid-circuit measurements with conditional operations
@quantum
def test_mid_circuit_conditional():
    """Test mid-circuit measurements with conditional operations."""
    print("\n=== Mid-Circuit Conditional Test ===")
    
    qubit(3)
    
    # Prepare qubit 0 in superposition
    hadamard(0)
    
    # Mid-circuit measurement on qubit 0
    m = measure(0)
    
    # Conditional operation: if qubit 0 measured 1, apply X to qubit 1
    cond(m == 1, xgate, 1)
    
    # Apply another gate to qubit 2
    hadamard(2)
    
    # Final measurement
    result = measure_probs(0, 1, 2)
    print(f"Mid-circuit conditional result: {result}")
    print("Expected: qubit 1 should be flipped when qubit 0 was |1⟩")
    
    return result

# Test analytic measurement types (state, density_matrix)
@quantum
def test_analytic_measurements():
    """Test analytic measurement types: state, density_matrix."""
    print("\n=== Analytic Measurement Types Test ===")
    
    qubit(2)
    
    # Prepare a Bell state
    hadamard(0)
    cnot(0, 1)
    
    # Test state measurement (measures full device state)
    state_result = measure_state(0, 1)
    print(f"State measurement: {state_result}")
    
    # Test density matrix measurement
    density_result = measure_density_matrix(0)
    print(f"Density matrix (qubit 0): {density_result}")
    
    # Final probability measurement
    prob_result = measure_probs(0, 1)
    print(f"Final probability: {prob_result}")
    
    return [state_result, density_result, prob_result]

# Test sampling measurement types (sample, counts)
@quantum
def test_sampling_measurements():
    """Test sampling measurement types: sample, counts (requires shots)."""
    print("\n=== Sampling Measurement Types Test ===")
    
    # Set shots BEFORE declaring qubits, so the device is created with shots
    # This ensures the QNode uses a device configured with shots
    set_shots(1000)
    
    qubit(2)
    
    # Prepare a Bell state
    hadamard(0)
    cnot(0, 1)
    
    # Test sample and counts (requires shots on device)
    # Note: Can't mix sampling measurements with analytic measurements like probs
    sample_result = measure_sample(0, 1)
    counts_result = measure_counts(0, 1)
    
    print(f"Sample measurement: {sample_result}")
    print(f"Counts measurement: {counts_result}")
    print("Sample and counts should show ~50% |00⟩ and ~50% |11⟩ for Bell state")
    print("Note: Results will be actual samples/counts, not probabilities")
    
    return [sample_result, counts_result]

# Test conditional operations with multiple measurements
@quantum
def test_conditional_cascade():
    """Test conditional operations with multiple mid-circuit measurements."""
    print("\n=== Conditional Cascade Test ===")
    
    qubit(4)
    
    # Prepare first qubit in superposition
    hadamard(0)
    
    # Measure qubit 0
    m0 = measure(0)
    
    # If qubit 0 is 1, prepare qubit 1 in |1⟩
    cond(m0 == 1, xgate, 1)
    
    # Prepare qubit 2 in superposition
    hadamard(2)
    
    # Measure qubit 2
    m2 = measure(2)
    
    # If qubit 2 is 1, flip qubit 3
    cond(m2 == 1, xgate, 3)
    
    # Final measurement
    result = measure_probs(1, 3)
    print(f"Conditional cascade result: {result}")
    print("Qubits 1 and 3 should be correlated with qubits 0 and 2 measurements")
    
    return result

# Test cond() with different gate types (including multi-parameter gates)
@quantum
def test_cond_with_various_gates():
    """Test cond() with different types of gates."""
    print("\n=== Conditional with Various Gates Test ===")
    
    qubit(3)
    
    # Prepare qubit 0 in superposition
    hadamard(0)
    
    # Measure qubit 0
    m = measure(0)
    
    # Conditional operations with different gate types
    cond(m == 1, xgate, 1)  # Single-parameter gate
    cond(m == 0, ry, np.pi/2, 2)  # Multi-parameter gate (angle, qubit)
    
    # Final measurement
    result = measure_probs(0, 1, 2)
    print(f"Conditional with various gates result: {result}")
    print("Qubit 1 should be flipped if qubit 0 was |1⟩")
    print("Qubit 2 should have RY(π/2) if qubit 0 was |0⟩")
    
    return result

@quantum
def test_qubit_decl():
    """Test qubit declaration."""
    print("\n=== Qubit Declaration Test ===")
    
    qubit(25)
    
    # Prepare qubit 0 in superposition
    hadamard(24)
    
    # Final measurement
    result = measure_probs(24)
    print(f"Qubit declaration result: {result}")
    return result

@quantum
def test_qpe_hermitian():
    """
    Test Quantum Phase Estimation on an arbitrary Hermitian matrix to find its lowest eigenvalue.
    
    Creates a 2x2 Hermitian matrix H, computes U = exp(-iHt), prepares ground state,
    and runs QPE to estimate the phase (which corresponds to the eigenvalue).
    
    Returns: probability distribution over estimation wire states
    """
    # Define an arbitrary 2x2 Hermitian matrix
    H = np.array([[1.0, 0.5], [0.5, -1.0]])
    
    # Compute eigenvalues classically for comparison
    eigenvals, eigenvecs = np.linalg.eigh(H)
    lowest_eigenval = eigenvals[0]
    
    # Set up QPE parameters
    n_ancilla = 4  # 4 ancilla qubits gives 16 possible phase values
    n_eigenstate = 1
    
    # Expected phase: For U = e^(iH), if H|ψ⟩ = λ|ψ⟩, then U|ψ⟩ = e^(iλ)|ψ⟩ = e^(2πiφ)|ψ⟩
    # So: iλ = 2πiφ, which means φ = λ/(2π) mod 1
    expected_phase = lowest_eigenval / (2 * np.pi)
    expected_phase = expected_phase % 1.0
    if expected_phase < 0:
        expected_phase += 1.0
    
    # Wire indices
    eigenstate_wires = [0]
    ancilla_wires = list(range(1, 1 + n_ancilla))
    
    # Declare qubits
    total_qubits = n_ancilla + n_eigenstate
    qubit(total_qubits)
    
    # Prepare ground state (lowest eigenvector)
    ground_state = eigenvecs[:, 0]
    state_prep(ground_state, eigenstate_wires)
    
    # Apply QPE for Hermitian matrix (uses PennyLane's built-in QPE)
    # Creates U = e^(iH) and estimates phase
    qpe_hermitian(H, eigenstate_wires, ancilla_wires)
    
    # Measure estimation wires to get phase estimate
    result = measure_probs(*ancilla_wires)
    
    print(f"Expected phase: {expected_phase:.6f}")
    print(f"Corresponds to H eigenvalue: {lowest_eigenval:.6f}")
    print(f"Phase is encoded in estimation wires {ancilla_wires}")
    print(f"To decode: phase = (most_probable_state_index) / {2**n_ancilla}")
    print(f"Then: H_eigenvalue = 2π * phase")
    
    return result

@quantum
def test_qpe_with_eigenstate():
    """
    Test QPE by preparing target qubit in an eigenstate of a unitary matrix.
    
    Creates a phase gate U = [[1, 0], [0, e^(2πi*phase)]] with known phase.
    Prepares eigenstate |1⟩ and runs QPE to estimate the phase.
    
    The estimation wires (ancilla qubits) encode the phase in binary.
    After QPE, we measure these wires to read out the phase estimate.
    
    Returns: probability distribution over estimation wire states
    """
    # Set up parameters
    known_phase = 0.3  # Expected phase: 0.3
    n_ancilla = 4  # 4 ancilla qubits gives 16 possible phase values (resolution 1/16 = 0.0625)
    n_eigenstate = 1
    
    # Create unitary matrix with known phase
    U = np.array([[1, 0], [0, np.exp(2j * np.pi * known_phase)]])
    
    # Eigenstate is |1⟩ (the second basis state)
    eigenstate = np.array([0, 1])
    
    # Wire indices
    eigenstate_wires = [0]
    ancilla_wires = list(range(1, 1 + n_ancilla))
    
    # Declare all qubits needed
    total_qubits = max(max(eigenstate_wires), max(ancilla_wires)) + 1
    qubit(total_qubits)
    
    # Prepare the target qubit in the eigenstate |1⟩
    state_prep(eigenstate, eigenstate_wires)
    
    # Apply QPE to estimate the phase of the unitary matrix U
    # QPE will estimate φ where U|ψ⟩ = e^(2πiφ)|ψ⟩
    qpe(U, eigenstate_wires, ancilla_wires)
    
    # Measure the estimation wires (ancilla qubits) to get the phase estimate
    # The phase is encoded in binary in these qubits
    result = measure_probs(*ancilla_wires)
    
    print(f"Expected phase: {known_phase:.6f}")
    print(f"Phase is encoded in estimation wires {ancilla_wires}")
    print(f"To decode: phase = (most_probable_state_index) / {2**n_ancilla}")
    
    return result

@quantum
def test_period_finding_s_gate():
    """
    Test period finding on the S gate (phase gate) with superposition state.
    
    S gate = [[1, 0], [0, i]] = [[1, 0], [0, e^(iπ/2)]]
    S^4 = I, so the period is 4.
    
    Uses |+⟩ = (|0⟩ + |1⟩)/√2, which is NOT an eigenstate of S.
    This is a superposition of |0⟩ (eigenvalue 1, phase 0) and |1⟩ (eigenvalue i, phase 1/4).
    
    IMPORTANT: Even though the period is 4, the S gate only has 2 eigenvectors (since it's
    a 1-qubit gate). So we only see phases 0 and 1/4, NOT 2/4 and 3/4. To see all phases
    s/r for s = 0, 1, ..., r-1, you need an operator with eigenvectors for all those phases
    (like the modular multiplication operator in Shor's algorithm).
    
    QPE reveals both phases, and we can extract period 4 from them.
    
    Returns: probability distribution over ancilla wire states
    """
    # S gate matrix
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    
    # Prepare initial state |+⟩ = (|0⟩ + |1⟩)/√2
    # This is a superposition of the two eigenstates:
    # - |0⟩ has eigenvalue 1 (phase 0)
    # - |1⟩ has eigenvalue i = e^(iπ/2) (phase 1/4)
    initial_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    
    # Set up wires
    n_eigenstate = 1
    n_ancilla = 4
    total_qubits = n_eigenstate + n_ancilla
    qubit(total_qubits)
    
    eigenstate_wires = [0]
    ancilla_wires = list(range(1, 1 + n_ancilla))
    
    # Apply period finding
    period_finding(S, initial_state, eigenstate_wires, ancilla_wires)
    
    # Measure ancilla wires to get phase estimate
    result = measure_probs(*ancilla_wires)
    
    print(f"Testing period finding on S gate (expected period: 4)")
    print(f"Initial state: |+⟩ = (|0⟩ + |1⟩)/√2 (superposition of eigenstates)")
    print(f"Expected to see peaks at phases: 0 (from |0⟩) and 1/4 (from |1⟩)")
    print(f"Note: S gate only has 2 eigenvectors, so we only see 2 phases (not all 4)")
    
    return result

@quantum
def test_period_finding_superposition():
    """
    Test period finding with Shor's modular multiplication (actual superposition case).
    
    Uses the modular multiplication unitary U: |y⟩ → |a*y mod N⟩ for a=2, N=15.
    The period is 4 (since 2^4 = 16 ≡ 1 mod 15).
    The state |1⟩ is in the periodic subspace and creates a superposition of
    eigenvectors with phases 0, 1/4, 2/4, 3/4 when evolved by U.
    
    This should show multiple peaks in the probability distribution.
    
    Returns: probability distribution over ancilla wire states
    """
    # Shor's modular multiplication: U|y⟩ = |a*y mod N⟩
    a = 2
    N = 15
    dim = 16  # Need at least N qubits, use 16 for 4 qubits
    
    # Construct the modular multiplication unitary
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        if y < N:
            result = (a * y) % N
            U[result, y] = 1.0
        else:
            U[y, y] = 1.0
    
    # Prepare initial state |1⟩
    # The state |1⟩ is in the periodic subspace. QPE naturally decomposes it into
    # a superposition of eigenvectors with phases s/4 for s = 0, 1, 2, 3.
    # No explicit superposition preparation needed - the phase kickback happens automatically.
    initial_state = np.zeros(dim, dtype=complex)
    initial_state[1] = 1.0  # |1⟩ state
    
    # Set up wires
    n_eigenstate = 4  # 4 qubits for 16-dimensional space
    n_ancilla = 6  # More ancilla qubits for better resolution
    total_qubits = n_eigenstate + n_ancilla
    qubit(total_qubits)
    
    eigenstate_wires = [0, 1, 2, 3]
    ancilla_wires = list(range(4, 4 + n_ancilla))
    
    # Apply period finding
    period_finding(U, initial_state, eigenstate_wires, ancilla_wires)
    
    # Measure ancilla wires
    result = measure_probs(*ancilla_wires)
    
    print(f"Testing period finding with Shor's modular multiplication (a=2, N=15)")
    print(f"Expected period: 4 (since 2^4 ≡ 1 mod 15)")
    print(f"Initial state: |1⟩ (in periodic subspace, QPE naturally decomposes into phases s/4)")
    print(f"Expected to see peaks at phases: 0, 1/4, 2/4, 3/4")
    
    return result

def test_shor_factorization():
    """
    Test Shor's factorization algorithm on several examples.
    """
    from dsl_decorator import shor_factor
    
    test_cases = [
        15,  # 3 * 5
        21,  # 3 * 7
        35,  # 5 * 7
    ]
    
    for N in test_cases:
        print(f"\n--- Factoring N = {N} ---")
        try:
            factors = shor_factor(N, max_attempts=5)
            print(f"Factors found: {factors}")
            print(f"Verification: {factors[0]} * {factors[1]} = {factors[0] * factors[1]}")
            if factors[0] * factors[1] == N:
                print("✓ Factorization correct!")
            else:
                print("✗ Factorization incorrect!")
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# State Preparation Tests
# ============================================================================

def test_bell_states():
    """
    Test all four Bell states: |Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩.
    
    Bell states are maximally entangled 2-qubit states:
    - |Φ+⟩ = (|00⟩ + |11⟩)/√2
    - |Φ-⟩ = (|00⟩ - |11⟩)/√2
    - |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    - |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    
    Expected probabilities:
    - |Φ+⟩ and |Φ-⟩: 50% |00⟩, 50% |11⟩
    - |Ψ+⟩ and |Ψ-⟩: 50% |01⟩, 50% |10⟩
    """
    results = {}
    
    # Test |Φ+⟩ (default)
    @quantum
    def test_phi_plus():
        qubit(2)
        bell_state(0, 1, 'phi_plus')
        return measure_probs(0, 1)
    
    # Test |Φ-⟩
    @quantum
    def test_phi_minus():
        qubit(2)
        bell_state(0, 1, 'phi_minus')
        return measure_probs(0, 1)
    
    # Test |Ψ+⟩
    @quantum
    def test_psi_plus():
        qubit(2)
        bell_state(0, 1, 'psi_plus')
        return measure_probs(0, 1)
    
    # Test |Ψ-⟩
    @quantum
    def test_psi_minus():
        qubit(2)
        bell_state(0, 1, 'psi_minus')
        return measure_probs(0, 1)
    
    results['phi_plus'] = test_phi_plus()
    results['phi_minus'] = test_phi_minus()
    results['psi_plus'] = test_psi_plus()
    results['psi_minus'] = test_psi_minus()
    
    print(f"|Φ+⟩ result: {results['phi_plus']}")
    print(f"Expected: ~50% |00⟩, ~50% |11⟩")
    print(f"|Φ-⟩ result: {results['phi_minus']}")
    print(f"Expected: ~50% |00⟩, ~50% |11⟩")
    print(f"|Ψ+⟩ result: {results['psi_plus']}")
    print(f"Expected: ~50% |01⟩, ~50% |10⟩")
    print(f"|Ψ-⟩ result: {results['psi_minus']}")
    print(f"Expected: ~50% |01⟩, ~50% |10⟩")
    
    return results


@quantum
def test_ghz_state():
    """
    Test GHZ (Greenberger-Horne-Zeilinger) state preparation.
    
    GHZ state for n qubits: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    
    Expected probabilities:
    - 50% probability of measuring |000...0⟩
    - 50% probability of measuring |111...1⟩
    - 0% probability of all other states
    """
    qubit(4)
    ghz_state([0, 1, 2, 3])
    
    result = measure_probs(0, 1, 2, 3)
    
    return result


@quantum
def test_w_state():
    """
    Test W state preparation.
    
    W state for n qubits: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    
    For 3 qubits: |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
    
    Expected probabilities:
    - Equal probability (1/n) for each state with exactly one |1⟩
    - 0% probability for |000⟩ and |111⟩
    """
    qubit(3)
    w_state([0, 1, 2])
    
    result = measure_probs(0, 1, 2)
    
    return result


@quantum
def test_w_state_4qubits():
    """
    Test W state preparation with 4 qubits.
    
    For 4 qubits: |W⟩ = (|1000⟩ + |0100⟩ + |0010⟩ + |0001⟩)/2
    
    Expected probabilities:
    - Equal probability (1/4 = 25%) for each state with exactly one |1⟩:
      |1000⟩ (index 8), |0100⟩ (index 4), |0010⟩ (index 2), |0001⟩ (index 1)
    - 0% probability for |0000⟩ (index 0) and |1111⟩ (index 15)
    - 0% probability for all other states
    """
    qubit(4)
    w_state([0, 1, 2, 3])
    
    result = measure_probs(0, 1, 2, 3)
    
    return result


@quantum
def test_quantum_teleportation():
    """
    Test quantum teleportation protocol.
    
    Quantum teleportation transfers an unknown quantum state from Alice to Bob
    using shared entanglement and classical communication.
    
    Protocol:
    1. Prepare state to teleport on alice_qubit (e.g., |+⟩ = (|0⟩ + |1⟩)/√2)
    2. Create Bell pair between bell_qubit1 (Alice) and bell_qubit2 (Bob)
    3. Alice performs Bell measurement on (alice_qubit, bell_qubit1)
    4. Bob applies corrections based on measurement outcome
    5. bell_qubit2 now contains the original state
    
    Expected: After teleportation, bell_qubit2 should be in the same state
    as alice_qubit was initially (e.g., |+⟩).
    """
    qubit(3)
    
    # Prepare state to teleport: |+⟩ = (|0⟩ + |1⟩)/√2 on qubit 0
    # This is Alice's qubit with the state to teleport
    hadamard(0)
    
    print("Prepared |+⟩ state on qubit 0 (Alice's qubit to teleport)")
    
    # Teleport state from qubit 0 to qubit 2
    # qubit 1 is Alice's half of Bell pair, qubit 2 is Bob's half
    quantum_teleportation(0, 1, 2)
    
    # Measure qubit 2 (Bob's qubit) to verify it's in |+⟩ state
    # |+⟩ state should have 50% probability of |0⟩ and 50% of |1⟩
    result = measure_probs(2)
    
    return result


# ============================================================================
# Measurement Utilities Tests
# ============================================================================

@quantum
def test_measure_pauli_x_plus():
    """
    Test measurement in X basis for |+⟩ state.
    
    |+⟩ = (|0⟩ + |1⟩)/√2
    After Hadamard: H|+⟩ = |0⟩
    Expected result: [1.0, 0.0] (100% probability of |0⟩ in rotated basis, which means |+⟩ in original basis)
    """
    qubit(1)
    
    # Prepare |+⟩ state
    hadamard(0)
    # Measure in X basis
    # measure_pauli_x applies H then measures, so H|+⟩ = |0⟩
    result = measure_pauli_x(0)
    
    return result


@quantum
def test_measure_pauli_x_zero():
    """
    Test measurement in X basis for |0⟩ state.
    
    |0⟩ after Hadamard: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
    Expected result: [0.5, 0.5] (50% probability of each outcome)
    """
    qubit(1)
    
    # qubit is in |0⟩ state
    # Measure in X basis (measure_pauli_x applies H then measures)
    result = measure_pauli_x(0)
    
    return result


@quantum
def test_measure_pauli_y_plusi():
    """
    Test measurement in Y basis for |+i⟩ state.
    
    |+i⟩ = (|0⟩ + i|1⟩)/√2 = S|+⟩
    measure_pauli_y applies S† then H then measures
    S†|+i⟩ = S†S|+⟩ = |+⟩, then H|+⟩ = |0⟩
    Expected result: [1.0, 0.0] (deterministic)
    """
    qubit(1)
    
    # Prepare |+i⟩ = S|+⟩
    hadamard(0)
    s(0)  # S gate: |+⟩ → |+i⟩
    # Measure in Y basis (measure_pauli_y applies S† then H then measures)
    result = measure_pauli_y(0)
    
    return result


@quantum
def test_measure_pauli_y_zero():
    """
    Test measurement in Y basis for |0⟩ state.
    
    |0⟩ after S† then H: S†|0⟩ = |0⟩, then H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
    Expected result: [0.5, 0.5] (50% probability of each outcome)
    """
    qubit(1)
    
    # qubit is in |0⟩ state
    # Measure in Y basis (measure_pauli_y applies S† then H then measures)
    result = measure_pauli_y(0)
    
    return result


@quantum
def test_measure_pauli_z_zero():
    """
    Test measurement in Z basis for |0⟩ state.
    
    This is the standard measurement - just measures in computational basis.
    Expected result: [1.0, 0.0] (100% probability of |0⟩)
    """
    qubit(1)
    
    # qubit is in |0⟩ state
    # Measure in Z basis (direct measurement, no rotation needed)
    result = measure_pauli_z(0)
    
    return result


@quantum
def test_measure_pauli_z_one():
    """
    Test measurement in Z basis for |1⟩ state.
    
    Expected result: [0.0, 1.0] (100% probability of |1⟩)
    """
    qubit(1)
    
    # Prepare |1⟩ state
    xgate(0)
    # Measure in Z basis
    result = measure_pauli_z(0)
    
    return result


@quantum
def test_measure_pauli_z_plus():
    """
    Test measurement in Z basis for |+⟩ state.
    
    |+⟩ = (|0⟩ + |1⟩)/√2
    Expected result: [0.5, 0.5] (50% probability of |0⟩, 50% of |1⟩)
    """
    qubit(1)
    
    # Prepare |+⟩ state
    hadamard(0)
    # Measure in Z basis
    result = measure_pauli_z(0)
    
    return result


# ============================================================================
# Utility Functions Tests
# ============================================================================

@quantum
def test_reset_qubit():
    """
    Test reset_qubit function.
    
    Reset should work for both |0⟩ and |1⟩ states.
    """
    qubit(2)
    
    # Test 1: Reset qubit that's already |0⟩
    # qubit 0 is in |0⟩
    reset_qubit(0)
    
    # Test 2: Reset qubit that's in |1⟩
    xgate(1)  # Prepare |1⟩
    reset_qubit(1)
    
    # Measure both qubits - both should be |0⟩
    result = measure_probs(0, 1)
    
    return result


@quantum
def test_copy_basis_state():
    """
    Test copy_basis_state function.
    
    Copy should work for basis states |0⟩ and |1⟩.
    """
    qubit(2)
    
    # Test 1: Copy |0⟩ from qubit 0 to qubit 1
    # qubit 0 is in |0⟩
    copy_basis_state(0, 1)
    
    # Measure both qubits - both should be |00⟩
    result1 = measure_probs(0, 1)
    
    return result1


@quantum
def test_copy_basis_state_one():
    """
    Test copy_basis_state function with |1⟩ state.
    """
    qubit(2)
    
    # Prepare |1⟩ on qubit 0
    xgate(0)
    # Copy to qubit 1
    copy_basis_state(0, 1)
    
    # Measure both qubits - both should be |1⟩ (state |11⟩)
    result = measure_probs(0, 1)
    
    return result


@quantum
def test_copy_basis_state_superposition():
    """
    Test copy_basis_state function with superposition state.
    
    This demonstrates that copy_basis_state creates entanglement for superpositions,
    not a true copy. After CNOT, the qubits are in a Bell state.
    """
    qubit(2)
    
    # Prepare |+⟩ = (|0⟩ + |1⟩)/√2 on qubit 0
    hadamard(0)
    # Try to "copy" to qubit 1
    copy_basis_state(0, 1)
    
    # Measure both qubits
    # Expected: Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    # Probability: 50% |00⟩, 50% |11⟩, 0% |01⟩ or |10⟩
    result = measure_probs(0, 1)
    
    return result


@quantum
def test_swap_test_identical():
    """
    Test swap_test with identical states.
    
    If states are identical, probability of |0⟩ on ancilla should be 1.0.
    """
    qubit(3)
    
    # Prepare same state on both qubits: |+⟩ = (|0⟩ + |1⟩)/√2
    hadamard(0)
    hadamard(1)
    
    # Swap test: compare qubit 0 with qubit 1, using qubit 2 as ancilla
    result = swap_test([0], [1], 2)
    
    return result


@quantum
def test_swap_test_orthogonal():
    """
    Test swap_test with orthogonal states.
    
    If states are orthogonal (|0⟩ and |1⟩), probability of |0⟩ on ancilla should be 0.5.
    """
    qubit(3)
    
    # Prepare orthogonal states: |0⟩ and |1⟩
    # qubit 0 is |0⟩
    xgate(1)  # qubit 1 is |1⟩
    
    # Swap test: compare qubit 0 with qubit 1, using qubit 2 as ancilla
    result = swap_test([0], [1], 2)
    
    return result


@quantum
def test_swap_test_same_basis():
    """
    Test swap_test with same basis state.
    
    If both states are |0⟩, probability of |0⟩ on ancilla should be 1.0.
    """
    qubit(3)
    
    # Both qubits are in |0⟩
    # Swap test: compare qubit 0 with qubit 1, using qubit 2 as ancilla
    result = swap_test([0], [1], 2)
    
    return result


# ============================================================================
# Circuit Analysis Tests
# ============================================================================

def test_circuit_analysis():
    """
    Test circuit analysis functions: depth, gate_count, circuit_resources.
    """
    from dsl_decorator import circuit_depth, gate_count, circuit_resources
    
    @quantum
    def simple_circuit():
        qubit(2)
        hadamard(0)
        cnot(0, 1)
        measure_probs(0, 1)
    
    @quantum
    def deeper_circuit():
        qubit(3)
        hadamard(0)
        hadamard(1)
        cnot(0, 1)
        cnot(1, 2)
        rx(np.pi/4, 0)
        ry(np.pi/4, 1)
        measure_probs(0, 1, 2)
    
    print("\n=== Circuit Analysis Tests ===")
    
    # Test simple circuit
    depth1 = circuit_depth(simple_circuit)
    count1 = gate_count(simple_circuit)
    resources1 = circuit_resources(simple_circuit)
    
    print(f"Simple circuit:")
    print(f"  Depth: {depth1}")
    print(f"  Gate count: {count1}")
    print(f"  Resources: {resources1}")
    
    # Test deeper circuit
    depth2 = circuit_depth(deeper_circuit)
    count2 = gate_count(deeper_circuit)
    resources2 = circuit_resources(deeper_circuit)
    
    print(f"\nDeeper circuit:")
    print(f"  Depth: {depth2}")
    print(f"  Gate count: {count2}")
    print(f"  Resources: {resources2}")
    
    return {
        'simple': {'depth': depth1, 'count': count1, 'resources': resources1},
        'deeper': {'depth': depth2, 'count': count2, 'resources': resources2}
    }


@quantum
def test_quantum_teleportation_bell_state():
    """
    Test quantum teleportation with a more complex state.
    
    Teleport a Bell state component to verify the protocol works correctly.
    """
    qubit(4)
    
    # Prepare a more complex state on qubit 0: |1⟩
    xgate(0)
    
    print("Prepared |1⟩ state on qubit 0 (Alice's qubit to teleport)")
    
    # Teleport state from qubit 0 to qubit 3
    # qubits 1 and 2 form the Bell pair
    quantum_teleportation(0, 1, 2)
    
    # Measure qubit 2 to verify it's in |1⟩ state
    result = measure_probs(2)
    
    return result


# Run the tests
if __name__ == "__main__":
    print("=== Simple Test ===")
    result1 = simple_test()
    print(f"Simple test result: {result1}")
    
    print("\n=== Constant Oracle Test ===")
    result2 = constant_oracle_test()
    print(f"Constant oracle result: {result2}")
    
    print("\n=== Balanced Oracle Test ===")
    result3 = balanced_oracle_test()
    print(f"Balanced oracle result: {result3}")
    
    print("\n=== Grover Test ===")
    result5 = grover_test()
    print(f"Grover test result: {result5}")
    
    print("\n=== CNF Oracle Test ===")
    result6 = test_cnf_oracle()
    print(f"CNF oracle test result: {result6}")
    
    print("\n=== Grover SAT Search Test ===")
    result7 = test_grover_cnf_sat()
    print(f"Grover SAT search result: {result7}")
    
    print("\n=== Mid-Circuit Conditional Test ===")
    result8 = test_mid_circuit_conditional()
    print(f"Mid-circuit conditional result: {result8}")
    
    print("\n=== Analytic Measurement Types Test ===")
    result9a = test_analytic_measurements()
    print(f"Analytic measurements result: {result9a}")
    
    # Note: Sampling measurements (sample, counts) require device with shots
    # The QNode is created before set_shots() can take effect, so this test
    # is currently skipped. To use sampling measurements, you need to set shots
    # at the QNode level or device creation time.
    # print("\n=== Sampling Measurement Types Test ===")
    # result9b = test_sampling_measurements()
    # print(f"Sampling measurements result: {result9b}")
    print("\n=== Sampling Measurement Types Test ===")
    print("Skipped: Sampling measurements require shots to be set at device creation time.")
    print("This is a limitation of the current implementation - QNode captures device at creation.")
    
    print("\n=== Conditional Cascade Test ===")
    result10 = test_conditional_cascade()
    print(f"Conditional cascade result: {result10}")
    
    print("\n=== Conditional with Various Gates Test ===")
    result11 = test_cond_with_various_gates()
    print(f"Conditional with various gates result: {result11}")

    print("\n=== Qubit Declaration Test ===")
    result12 = test_qubit_decl()
    print(f"Qubit declaration result: {result12}")
    
    print("\n=== QPE with Eigenstate Test ===")
    result13a = test_qpe_with_eigenstate()
    
    # Process the result to find the most probable phase
    if result13a is not None:
        if hasattr(result13a, 'numpy'):
            probs = result13a.numpy()
        else:
            probs = np.array(result13a)
        
        n_ancilla = int(np.log2(len(probs)))
        phase_estimate_idx = np.argmax(probs)
        phase_estimate = phase_estimate_idx / (2 ** n_ancilla)
        known_phase = 0.3  # Expected phase from the test function
        
        print(f"Most probable state: |{phase_estimate_idx:0{n_ancilla}b}⟩ (index {phase_estimate_idx})")
        print(f"Estimated phase: {phase_estimate:.6f}")
        print(f"Phase error: {abs(phase_estimate - known_phase):.6f}")
    
    print(f"QPE with eigenstate result: {result13a}")
    
    print("\n=== QPE Hermitian Test ===")
    result13 = test_qpe_hermitian()
    
    # Process the result to find the most probable phase
    if result13 is not None:
        if hasattr(result13, 'numpy'):
            probs = result13.numpy()
        else:
            probs = np.array(result13)
        
        n_ancilla = int(np.log2(len(probs)))
        phase_estimate_idx = np.argmax(probs)
        phase_estimate = phase_estimate_idx / (2 ** n_ancilla)
        
        # Calculate expected phase from the Hermitian matrix
        H = np.array([[1.0, 0.5], [0.5, -1.0]])
        eigenvals, eigenvecs = np.linalg.eigh(H)
        lowest_eigenval = eigenvals[0]
        expected_phase = lowest_eigenval / (2 * np.pi)
        expected_phase = expected_phase % 1.0
        if expected_phase < 0:
            expected_phase += 1.0
        
        # Convert phase back to eigenvalue
        # For U = e^(iH), if H|ψ⟩ = λ|ψ⟩, then U|ψ⟩ = e^(iλ)|ψ⟩ = e^(2πiφ)|ψ⟩
        # So: iλ = 2πiφ, which means λ = 2πφ (but we need to account for phase wrapping)
        # Since phase is in [0, 1), we need to check if the eigenvalue should be negative
        estimated_eigenval_from_phase = phase_estimate * 2 * np.pi
        # If the phase is > 0.5, the eigenvalue might be negative (wrapped around)
        # Check which interpretation is closer to the expected value
        if abs(estimated_eigenval_from_phase - lowest_eigenval) > abs(estimated_eigenval_from_phase - 2 * np.pi - lowest_eigenval):
            estimated_eigenval = estimated_eigenval_from_phase - 2 * np.pi
        else:
            estimated_eigenval = estimated_eigenval_from_phase
        
        print(f"Most probable state: |{phase_estimate_idx:0{n_ancilla}b}⟩ (index {phase_estimate_idx})")
        print(f"Estimated phase: {phase_estimate:.6f}")
        print(f"Expected phase: {expected_phase:.6f}")
        print(f"Phase error: {abs(phase_estimate - expected_phase):.6f}")
        print(f"Estimated H eigenvalue: {estimated_eigenval:.6f}")
        print(f"Expected H eigenvalue: {lowest_eigenval:.6f}")
        print(f"Eigenvalue error: {abs(estimated_eigenval - lowest_eigenval):.6f}")
    
    print(f"QPE Hermitian result: {result13}")
    
    print("\n=== Period Finding with S Gate Test ===")
    result14 = test_period_finding_s_gate()
    
    # Process the result to extract period
    if result14 is not None:
        if hasattr(result14, 'numpy'):
            probs = result14.numpy()
        else:
            probs = np.array(result14)
        
        n_ancilla = int(np.log2(len(probs)))
        
        # Extract period from the full probability distribution
        # With |+⟩ state, we expect peaks at phases 0 (from |0⟩) and 1/4 (from |1⟩)
        # Both are consistent with period 4
        # Use PennyLane's approach: multiple measurements, keep highest r
        from dsl_decorator import extract_period_from_probabilities
        period_r = extract_period_from_probabilities(probs, n_ancilla, max_period=100, num_samples=20)
        
        # Also show the most probable phase
        phase_idx = np.argmax(probs)
        phase_estimate = phase_idx / (2 ** n_ancilla)
        
        print(f"Most probable phase: {phase_estimate:.6f} (index {phase_idx})")
        print(f"Expected period: 4 (since S^4 = I)")
        print(f"Extracted period from probability distribution: {period_r}")
        
        # Show all significant peaks (phases with probability > 1%)
        print("Significant phase peaks (should see 0 and 1/4 for |+⟩ state):")
        peaks_found = []
        for idx, prob in enumerate(probs):
            if prob > 0.01:
                phase = idx / (2 ** n_ancilla)
                peaks_found.append((phase, prob))
                print(f"  Phase {phase:.6f} (index {idx}): probability {prob:.6f}")
        
        if len(peaks_found) > 1:
            print(f"✓ Successfully detected {len(peaks_found)} peaks - superposition working!")
        else:
            print(f"Note: Only {len(peaks_found)} peak(s) detected")
    
    print(f"Period finding S gate result: {result14}")
    
    print("\n=== Period Finding with Superposition Test ===")
    result15 = test_period_finding_superposition()
    
    # Process the result to extract period and show multiple peaks
    if result15 is not None:
        if hasattr(result15, 'numpy'):
            probs = result15.numpy()
        else:
            probs = np.array(result15)
        
        n_ancilla = int(np.log2(len(probs)))
        
        # Extract period using PennyLane's approach: multiple measurements, keep highest r
        from dsl_decorator import extract_period_from_probabilities
        period_r = extract_period_from_probabilities(probs, n_ancilla, max_period=100, num_samples=20)
        
        print(f"Extracted period from probability distribution: {period_r}")
        print(f"Expected period: 4")
        
        # Show all significant peaks (phases with probability > 1%)
        print("Significant phase peaks (should see multiple for superposition):")
        peaks_found = []
        for idx, prob in enumerate(probs):
            if prob > 0.01:
                phase = idx / (2 ** n_ancilla)
                peaks_found.append((phase, prob, idx))
                print(f"  Phase {phase:.6f} (index {idx}): probability {prob:.6f}")
        
        if len(peaks_found) > 1:
            print(f"✓ Successfully detected {len(peaks_found)} peaks - superposition working!")
        else:
            print(f"Note: Only {len(peaks_found)} peak(s) detected")
    
    print(f"Period finding superposition result: {result15}")
    
    print("\n=== Shor's Factorization Test ===")
    test_shor_factorization()
    
    print("\n=== Bell States Test ===")
    result15 = test_bell_states()
    print(f"Bell states result: {result15}")
    
    print("\n=== GHZ State Test ===")
    result16 = test_ghz_state()
    print(f"GHZ state result: {result16}")
    
    print("\n=== W State Test (3 qubits) ===")
    result17 = test_w_state()
    print(f"W state (3 qubits) result: {result17}")
    
    print("\n=== W State Test (4 qubits) ===")
    result17b = test_w_state_4qubits()
    print(f"W state (4 qubits) result: {result17b}")
    
    print("\n=== Quantum Teleportation Test ===")
    result18 = test_quantum_teleportation()
    print(f"Quantum teleportation result: {result18}")
    
    print("\n=== Quantum Teleportation (Bell State Component) Test ===")
    result19 = test_quantum_teleportation_bell_state()
    print(f"Quantum teleportation (Bell state component) result: {result19}")
    
    print("\n=== Measurement Utilities Tests ===")
    result20a = test_measure_pauli_x_plus()
    if result20a is not None:
        if hasattr(result20a, 'numpy'):
            probs = result20a.numpy()
        else:
            probs = np.array(result20a)
        print(f"Measure Pauli X (|+⟩ state) result: {probs}")
        print(f"  Expected: [1.0, 0.0] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result20b = test_measure_pauli_x_zero()
    if result20b is not None:
        if hasattr(result20b, 'numpy'):
            probs = result20b.numpy()
        else:
            probs = np.array(result20b)
        print(f"Measure Pauli X (|0⟩ state) result: {probs}")
        print(f"  Expected: [0.5, 0.5] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 0.5) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result21a = test_measure_pauli_y_plusi()
    if result21a is not None:
        if hasattr(result21a, 'numpy'):
            probs = result21a.numpy()
        else:
            probs = np.array(result21a)
        print(f"Measure Pauli Y (|+i⟩ state) result: {probs}")
        print(f"  Expected: [1.0, 0.0] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result21b = test_measure_pauli_y_zero()
    if result21b is not None:
        if hasattr(result21b, 'numpy'):
            probs = result21b.numpy()
        else:
            probs = np.array(result21b)
        print(f"Measure Pauli Y (|0⟩ state) result: {probs}")
        print(f"  Expected: [0.5, 0.5] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 0.5) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result22a = test_measure_pauli_z_zero()
    if result22a is not None:
        if hasattr(result22a, 'numpy'):
            probs = result22a.numpy()
        else:
            probs = np.array(result22a)
        print(f"Measure Pauli Z (|0⟩ state) result: {probs}")
        print(f"  Expected: [1.0, 0.0] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result22b = test_measure_pauli_z_one()
    if result22b is not None:
        if hasattr(result22b, 'numpy'):
            probs = result22b.numpy()
        else:
            probs = np.array(result22b)
        print(f"Measure Pauli Z (|1⟩ state) result: {probs}")
        print(f"  Expected: [0.0, 1.0] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[1] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result22c = test_measure_pauli_z_plus()
    if result22c is not None:
        if hasattr(result22c, 'numpy'):
            probs = result22c.numpy()
        else:
            probs = np.array(result22c)
        print(f"Measure Pauli Z (|+⟩ state) result: {probs}")
        print(f"  Expected: [0.5, 0.5] - Got: [{probs[0]:.4f}, {probs[1]:.4f}]")
        if abs(probs[0] - 0.5) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    print("\n=== Utility Functions Tests ===")
    result24 = test_reset_qubit()
    if result24 is not None:
        if hasattr(result24, 'numpy'):
            probs = result24.numpy()
        else:
            probs = np.array(result24)
        print(f"Reset qubit result: {probs}")
        print(f"  Expected: [1.0, 0.0, 0.0, 0.0] (both qubits should be |0⟩)")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result25 = test_copy_basis_state()
    if result25 is not None:
        if hasattr(result25, 'numpy'):
            probs = result25.numpy()
        else:
            probs = np.array(result25)
        print(f"Copy state (|0⟩) result: {probs}")
        print(f"  Expected: [1.0, 0.0, 0.0, 0.0] (both qubits should be |0⟩)")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result26 = test_copy_basis_state_one()
    if result26 is not None:
        if hasattr(result26, 'numpy'):
            probs = result26.numpy()
        else:
            probs = np.array(result26)
        print(f"Copy state (|1⟩) result: {probs}")
        print(f"  Expected: [0.0, 0.0, 0.0, 1.0] (both qubits should be |1⟩, state |11⟩)")
        if abs(probs[3] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result26b = test_copy_basis_state_superposition()
    if result26b is not None:
        if hasattr(result26b, 'numpy'):
            probs = result26b.numpy()
        else:
            probs = np.array(result26b)
        print(f"Copy state (|+⟩ superposition) result: {probs}")
        print(f"  Expected: [0.5, 0.0, 0.0, 0.5] (Bell state |Φ+⟩, NOT a copy!)")
        print(f"  This demonstrates entanglement, not copying (no-cloning theorem)")
        if abs(probs[0] - 0.5) < 0.01 and abs(probs[3] - 0.5) < 0.01:
            print("  ✓ Shows entanglement (as expected for superposition)")
        else:
            print("  ✗ Unexpected result!")
    
    result27 = test_swap_test_identical()
    if result27 is not None:
        if hasattr(result27, 'numpy'):
            probs = result27.numpy()
        else:
            probs = np.array(result27)
        print(f"Swap test (identical states) result: {probs}")
        print(f"  Expected: [1.0, 0.0] (ancilla should be |0⟩ with probability 1.0)")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result28 = test_swap_test_orthogonal()
    if result28 is not None:
        if hasattr(result28, 'numpy'):
            probs = result28.numpy()
        else:
            probs = np.array(result28)
        print(f"Swap test (orthogonal states) result: {probs}")
        print(f"  Expected: [0.5, 0.5] (ancilla should be 50/50)")
        if abs(probs[0] - 0.5) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    result29 = test_swap_test_same_basis()
    if result29 is not None:
        if hasattr(result29, 'numpy'):
            probs = result29.numpy()
        else:
            probs = np.array(result29)
        print(f"Swap test (same basis state) result: {probs}")
        print(f"  Expected: [1.0, 0.0] (ancilla should be |0⟩ with probability 1.0)")
        if abs(probs[0] - 1.0) < 0.01:
            print("  ✓ Correct!")
        else:
            print("  ✗ Incorrect!")
    
    print("\n=== Circuit Analysis Tests ===")
    result23 = test_circuit_analysis()
    print(f"Circuit analysis result: {result23}")
    
    print("\n=== Circuit Drawing Test ===")
    try:
        import matplotlib.pyplot as plt
        # Draw a simple circuit
        fig, ax = draw(simple_test)
        print("Successfully drew simple_test circuit")
        # Use plt.show() instead of fig.show() - it's more reliable
        # Optionally save as well
        # fig.savefig('simple_circuit.png', dpi=150, bbox_inches='tight')
        plt.show()  # This will display the figure and block until window is closed
        print("Displayed simple_test circuit (close window to continue)")
        
        # Draw a more complex circuit (Grover SAT search)
        fig2, ax2 = draw(test_grover_cnf_sat)
        print("Successfully drew test_grover_cnf_sat circuit")
        # fig2.savefig('grover_cnf_sat_circuit.png', dpi=150, bbox_inches='tight')
        plt.show()  # This will display the second figure
        print("Displayed test_grover_cnf_sat circuit (close window to continue)")
        
        # Close figures after showing
        plt.close(fig)
        plt.close(fig2)
        
    except Exception as e:
        print(f"Drawing test failed: {e}")
        import traceback
        traceback.print_exc()
