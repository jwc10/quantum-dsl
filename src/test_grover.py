# test_grover.py

from dsl_decorator import quantum
import pennylane as qml
import numpy as np

@quantum
def test_grover_search():
    """Test Grover's search algorithm."""
    print("Testing Grover's Search Algorithm...")
    
    # Define the search space (3 primary qubits = 8 possible states)
    # Plus 1 phase qubit for phase kickback
    qubit(4)
    primary_qubits = [0, 1, 2]
    phase_qubit = 3
    
    # Define oracle: mark |101⟩ as the solution
    def oracle(primary_qubits, phase_qubit):
        """Oracle that marks |101⟩ as solution using phase qubit pattern."""
        # Apply X to qubits 0 and 2 to flip |101⟩ to |000⟩
        xgate(primary_qubits[0])
        xgate(primary_qubits[2])
        
        # Multi-controlled X: primary qubits as controls, phase qubit as target
        # When all primaries are |0⟩ (after X gates), phase qubit flips |-> → -|->
        # This creates phase kickback, flipping phase of the |101⟩ state
        phase_wires = primary_qubits + [phase_qubit]
        control_vals = [0, 0, 0]  # Expect all primaries to be |0⟩ after X gates
        multi_controlled_x(phase_wires, control_vals)
        
        # Apply X again to restore |101⟩ (return primaries to original state)
        xgate(primary_qubits[0])
        xgate(primary_qubits[2])
    
    # Calculate optimal number of iterations
    # For n qubits, optimal iterations = π/4 * sqrt(2^n / num_solutions)
    # With 3 qubits and 1 solution: π/4 * sqrt(8/1) ≈ 2.22 → 2 iterations
    num_iterations = 2
    
    print(f"Searching for |101⟩ in 3-qubit space with {num_iterations} iterations...")
    
    # Run Grover's algorithm
    grover_search(oracle, primary_qubits, phase_qubit, num_iterations)
    
    # Measure the result
    result = measure_probs(0, 1, 2)
    
    print(f"Grover search result: {result}")
    print("Expected: High probability for |101⟩ (state index 5, binary 101)")
    
    return result

@quantum
def test_grover_manual():
    """Test Grover's algorithm step by step."""
    print("Testing Grover's Algorithm Step by Step...")
    
    # Primary qubits for search space + phase qubit
    qubit(4)
    primary_qubits = [0, 1, 2]
    phase_qubit = 3
    
    # Step 1: Initialize superposition on primary qubits
    print("Step 1: Initialize superposition")
    hadamard_transform(primary_qubits)
    
    # Prepare phase qubit in |-> state (for diffusion operator)
    xgate(phase_qubit)
    hadamard(phase_qubit)
    
    # Step 2: Apply oracle (mark |111⟩ as solution)
    print("Step 2: Apply oracle")
    def simple_oracle(qubits):
        # Mark |111⟩ as solution
        multi_controlled_z(qubits)
    
    simple_oracle(primary_qubits)
    
    # Step 3: Apply diffusion operator
    print("Step 3: Apply diffusion operator")
    grover_diffusion(primary_qubits, phase_qubit)
    
    # Step 4: Measure
    print("Step 4: Measure")
    result = measure_probs(*primary_qubits)
    
    print(f"Manual Grover result: {result}")
    print("Expected: High probability for |111⟩ (state index 7, binary 111)")
    
    return result

@quantum
def test_grover_unitary_oracle():
    """Test Grover's algorithm with a unitary-based oracle."""
    print("\n--- Test 3: Grover with Unitary Oracle ---")
    print("Testing Grover's Search with Unitary Matrix Oracle...")
    
    # Define the search space (2 primary qubits = 4 possible states)
    # Plus 1 phase qubit for diffusion operator
    qubit(3)
    primary_qubits = [0, 1]
    phase_qubit = 2
    
    # Define oracle using unitary matrix: mark |11⟩ as solution
    # Matrix flips phase of |11⟩ state (state index 3, binary 11)
    def unitary_oracle(qubits):
        """Unitary-based oracle that marks |11⟩ as solution."""
        # Create 4x4 identity matrix
        oracle_matrix = np.identity(4)
        # Flip phase of |11⟩ (state index 3, binary 11)
        oracle_matrix[3, 3] = -1
        # Apply unitary to primary qubits
        qubitunitary(oracle_matrix, qubits)
    
    # Calculate optimal iterations: π/4 * sqrt(4/1) ≈ 1.57 → 1 iteration
    num_iterations = 1
    
    print(f"Searching for |11⟩ in 2-qubit space with {num_iterations} iteration...")
    
    # Run Grover's algorithm (oracle only accepts primary_qubits)
    grover_search(unitary_oracle, primary_qubits, phase_qubit, num_iterations)
    
    # Measure the result
    result = measure_probs(*primary_qubits)
    print(f"Unitary oracle result: {result}")
    print("Expected: High probability for |11⟩ (state index 3, binary 11)")
    
    return result

if __name__ == "__main__":
    print("=== Grover's Algorithm Tests ===")
    
    print("\n--- Test 1: Complete Grover Search ---")
    result1 = test_grover_search()
    
    print("\n--- Test 2: Manual Grover Steps ---")
    result2 = test_grover_manual()
    
    print("\n--- Test 3: Unitary Oracle ---")
    result3 = test_grover_unitary_oracle()
    
    print(f"\nFinal results:")
    print(f"Search result: {result1}")
    print(f"Manual result: {result2}")
    print(f"Unitary oracle result: {result3}")
