# test_simple_grover.py

from dsl_decorator import quantum
import pennylane as qml
import numpy as np

@quantum
def test_simple_grover():
    """Test a simple 2-qubit Grover search."""
    print("Testing Simple 2-Qubit Grover Search...")
    
    qubit(2)
    
    # Initialize superposition
    hadamard(0)
    hadamard(1)
    
    # Oracle: mark |11⟩ as solution
    # Apply multi-controlled Z to flip phase of |11⟩
    multi_controlled_z([0, 1])
    
    # Diffusion operator
    hadamard(0)
    hadamard(1)
    multi_controlled_z([0, 1])
    hadamard(0)
    hadamard(1)
    
    # Measure
    result = measure_probs(0, 1)
    
    print(f"2-qubit Grover result: {result}")
    print("Expected: High probability for |11⟩ (state 3 in binary)")
    print("States: |00⟩ |01⟩ |10⟩ |11⟩")
    
    return result

@quantum
def test_grover_with_manual_diffusion():
    """Test Grover with manually implemented diffusion."""
    print("Testing Grover with Manual Diffusion...")
    
    qubit(2)
    
    # Initialize superposition
    hadamard(0)
    hadamard(1)
    
    # Oracle: mark |11⟩ as solution
    multi_controlled_z([0, 1])
    
    # Manual diffusion: flip all states, then flip |00⟩, then flip all again
    xgate(0)
    xgate(1)
    multi_controlled_z([0, 1])
    xgate(0)
    xgate(1)
    
    # Measure
    result = measure_probs(0, 1)
    
    print(f"Manual diffusion result: {result}")
    print("Expected: High probability for |11⟩ (state 3 in binary)")
    
    return result

if __name__ == "__main__":
    print("=== Simple Grover Tests ===")
    
    print("\n--- Test 1: Standard Grover ---")
    result1 = test_simple_grover()
    
    print("\n--- Test 2: Manual Diffusion ---")
    result2 = test_grover_with_manual_diffusion()
    
    print(f"\nFinal results:")
    print(f"Standard: {result1}")
    print(f"Manual: {result2}")
