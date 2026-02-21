# test_all_gates.py

from dsl_decorator import quantum
import pennylane as qml
import numpy as np

@quantum
def test_all_gates():
    """Test all the implemented gates."""
    print("Testing all quantum gates...")
    
    # Declare qubits
    qubit(4)
    
    # Single-qubit gates
    print("Testing single-qubit gates...")
    hadamard(0)
    xgate(1)
    ygate(2)
    zgate(3)
    rx(np.pi/4, 0)
    ry(np.pi/3, 1)
    rz(np.pi/2, 2)
    phase_shift(np.pi/6, 3)
    s(0)
    t(1)
    u1(np.pi/8, 2)
    u2(np.pi/4, np.pi/2, 3)
    u3(np.pi/3, np.pi/4, np.pi/5, 0)
    
    # Two-qubit gates
    print("Testing two-qubit gates...")
    cnot(0, 1)
    cy(1, 2)
    cz(2, 3)
    ch(3, 0)
    swap(0, 2)
    cswap(1, 2, 3)
    
    # Three-qubit gates
    print("Testing three-qubit gates...")
    toffoli(0, 1, 2)
    
    # Multi-qubit gates
    print("Testing multi-qubit gates...")
    multi_rz(np.pi/4, [0, 1, 2])
    multi_controlled_x([0, 1, 2])
    multi_controlled_z([0, 1, 2])
    
    # State preparation
    print("Testing state preparation...")
    basis_state([1, 0, 1, 0], [0, 1, 2, 3])
    
    # Measurements
    print("Testing measurements...")
    probs = measure_probs(0, 1, 2, 3)
    expval_x = measure_expval(qml.PauliX, 0)
    var_z = measure_var(qml.PauliZ, 1)
    
    print(f"Probabilities: {probs}")
    print(f"Expectation value of X on qubit 0: {expval_x}")
    print(f"Variance of Z on qubit 1: {var_z}")
    
    return probs, expval_x, var_z

if __name__ == "__main__":
    print("=== Testing All Gates ===")
    result = test_all_gates()
    print(f"Final result: {result}")
