# test_controlled_gates.py
# Test file for controlled rotation gates and arbitrary controlled gates

from dsl_decorator import quantum
import pennylane as qml
import numpy as np

@quantum
def test_crx():
    """Test controlled RX gate."""
    qubit(2)
    
    # Prepare |+⟩ on control, |0⟩ on target
    hadamard(0)
    
    # Apply CRX(π/2) - should rotate target by π/2 when control is |1⟩
    crx(np.pi/2, 0, 1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_cry():
    """Test controlled RY gate."""
    qubit(2)
    
    # Prepare |+⟩ on control, |0⟩ on target
    hadamard(0)
    
    # Apply CRY(π/2) - should rotate target by π/2 when control is |1⟩
    cry(np.pi/2, 0, 1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_crz():
    """Test controlled RZ gate."""
    qubit(2)
    
    # Prepare |+⟩ on control, |+⟩ on target
    hadamard(0)
    hadamard(1)
    
    # Apply CRZ(π/2) - should add phase when control is |1⟩
    crz(np.pi/2, 0, 1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_controlled_unitary_with_control_values():
    """Test controlled unitary with arbitrary control values."""
    qubit(3)
    
    # Create a simple unitary (Hadamard as matrix)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    # Prepare |0⟩ on control, |1⟩ on control2, |0⟩ on target
    xgate(1)  # Set control2 to |1⟩
    
    # Apply controlled unitary: H on target when control is |0⟩ and control2 is |1⟩
    # control_values=[0, 1] means control on |0⟩ and control2 on |1⟩
    controlled_unitary([0, 1], H, [2], control_values=[0, 1])
    
    # Measure probabilities
    result = measure_probs(0, 1, 2)
    return result

@quantum
def test_controlled_gate_simple():
    """Test controlled_gate with simple gate (RX)."""
    qubit(2)
    
    # Prepare |+⟩ on control, |0⟩ on target
    hadamard(0)
    
    # Apply controlled RX(π/2) using controlled_gate
    controlled_gate(qml.RX, [0], None, np.pi/2, wires=1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_controlled_gate_with_control_values():
    """Test controlled_gate with control values (control on |0⟩)."""
    qubit(2)
    
    # Prepare |0⟩ on control, |+⟩ on target
    hadamard(1)
    
    # Apply controlled Hadamard when control is |0⟩
    controlled_gate(qml.Hadamard, [0], [0], wires=1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_controlled_gate_multi_control():
    """Test controlled_gate with multiple controls and mixed control values."""
    qubit(4)
    
    # Prepare |0⟩ on control0, |1⟩ on control1, |+⟩ on target
    xgate(1)  # Set control1 to |1⟩
    hadamard(3)  # Set target to |+⟩
    
    # Apply controlled RY(π/2) when control0 is |0⟩ and control1 is |1⟩
    controlled_gate(qml.RY, [0, 1], [0, 1], np.pi/2, wires=3)
    
    # Measure probabilities
    result = measure_probs(0, 1, 2, 3)
    return result

@quantum
def test_controlled_gate_rz():
    """Test controlled_gate with RZ gate."""
    qubit(2)
    
    # Prepare |+⟩ on control, |+⟩ on target
    hadamard(0)
    hadamard(1)
    
    # Apply controlled RZ(π/4) when control is |1⟩
    controlled_gate(qml.RZ, [0], None, np.pi/4, wires=1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_controlled_gate_phase_shift():
    """Test controlled_gate with PhaseShift gate."""
    qubit(2)
    
    # Prepare |+⟩ on control, |+⟩ on target
    hadamard(0)
    hadamard(1)
    
    # Apply controlled PhaseShift(π/3) when control is |1⟩
    controlled_gate(qml.PhaseShift, [0], None, np.pi/3, wires=1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

@quantum
def test_controlled_gate_control_on_zero():
    """Test controlled_gate with control on |0⟩ state."""
    qubit(2)
    
    # Prepare |0⟩ on control, |+⟩ on target
    hadamard(1)
    
    # Apply controlled X gate when control is |0⟩ (should flip target)
    controlled_gate(qml.PauliX, [0], [0], wires=1)
    
    # Measure probabilities
    result = measure_probs(0, 1)
    return result

if __name__ == "__main__":
    print("=== Testing Controlled Rotation Gates ===\n")
    
    print("1. Testing CRX gate:")
    result1 = test_crx()
    if hasattr(result1, 'numpy'):
        probs1 = result1.numpy()
    else:
        probs1 = np.array(result1)
    print(f"   Probabilities: {probs1}")
    print(f"   ✓ CRX test completed\n")
    
    print("2. Testing CRY gate:")
    result2 = test_cry()
    if hasattr(result2, 'numpy'):
        probs2 = result2.numpy()
    else:
        probs2 = np.array(result2)
    print(f"   Probabilities: {probs2}")
    print(f"   ✓ CRY test completed\n")
    
    print("3. Testing CRZ gate:")
    result3 = test_crz()
    if hasattr(result3, 'numpy'):
        probs3 = result3.numpy()
    else:
        probs3 = np.array(result3)
    print(f"   Probabilities: {probs3}")
    print(f"   ✓ CRZ test completed\n")
    
    print("=== Testing Controlled Unitary with Control Values ===\n")
    
    print("4. Testing controlled_unitary with control_values:")
    result4 = test_controlled_unitary_with_control_values()
    if hasattr(result4, 'numpy'):
        probs4 = result4.numpy()
    else:
        probs4 = np.array(result4)
    print(f"   Probabilities: {probs4}")
    print(f"   ✓ Controlled unitary with control values test completed\n")
    
    print("=== Testing Controlled Gate Function ===\n")
    
    print("5. Testing controlled_gate with RX:")
    result5 = test_controlled_gate_simple()
    if hasattr(result5, 'numpy'):
        probs5 = result5.numpy()
    else:
        probs5 = np.array(result5)
    print(f"   Probabilities: {probs5}")
    print(f"   ✓ Controlled gate (RX) test completed\n")
    
    print("6. Testing controlled_gate with control on |0⟩:")
    result6 = test_controlled_gate_with_control_values()
    if hasattr(result6, 'numpy'):
        probs6 = result6.numpy()
    else:
        probs6 = np.array(result6)
    print(f"   Probabilities: {probs6}")
    print(f"   ✓ Controlled gate with control values test completed\n")
    
    print("7. Testing controlled_gate with multiple controls:")
    result7 = test_controlled_gate_multi_control()
    if hasattr(result7, 'numpy'):
        probs7 = result7.numpy()
    else:
        probs7 = np.array(result7)
    print(f"   Probabilities: {probs7}")
    print(f"   ✓ Multi-control controlled gate test completed\n")
    
    print("8. Testing controlled_gate with RZ:")
    result8 = test_controlled_gate_rz()
    if hasattr(result8, 'numpy'):
        probs8 = result8.numpy()
    else:
        probs8 = np.array(result8)
    print(f"   Probabilities: {probs8}")
    print(f"   ✓ Controlled gate (RZ) test completed\n")
    
    print("9. Testing controlled_gate with PhaseShift:")
    result9 = test_controlled_gate_phase_shift()
    if hasattr(result9, 'numpy'):
        probs9 = result9.numpy()
    else:
        probs9 = np.array(result9)
    print(f"   Probabilities: {probs9}")
    print(f"   ✓ Controlled gate (PhaseShift) test completed\n")
    
    print("10. Testing controlled_gate with control on |0⟩ (PauliX):")
    result10 = test_controlled_gate_control_on_zero()
    if hasattr(result10, 'numpy'):
        probs10 = result10.numpy()
    else:
        probs10 = np.array(result10)
    print(f"   Probabilities: {probs10}")
    print(f"   ✓ Controlled gate (control on |0⟩) test completed\n")
    
    print("=== All Tests Completed Successfully! ===")

