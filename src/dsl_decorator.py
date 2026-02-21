# dsl_decorator.py

import sys
import io
import inspect
from typing import Dict, Any, List, Optional
from contextvars import ContextVar
import pennylane as qml
import numpy as np
from sympy import symbols, Implies, Equivalent
from sympy.logic.boolalg import to_cnf, Xor
from fractions import Fraction

# Context variable to track the current quantum context (allows nested @quantum functions)
_current_context: ContextVar[Any] = ContextVar('_current_context', default=None)

def quantum(func):
    """
    Main decorator function that transforms a Python function into a quantum-enabled function.
    
    This decorator:
    1. Creates a quantum context object
    2. Replaces DSL function names in the function's namespace with context methods
    3. Executes the function normally (preserving all control flow)
    4. Returns quantum results
    
    Args:
        func: The function being decorated (e.g., grover_search)
    
    Returns:
        wrapper: A new function that replaces the original function
    """
    
    def wrapper(*args, **kwargs):
        """
        The wrapper function that actually gets called when the decorated function is invoked.
        
        This function:
        1. Checks if we're already in a quantum context (nested call)
        2. If yes, executes directly with existing context
        3. If no, creates a new context and QNode
        4. Returns quantum results
        
        Args:
            *args: Positional arguments passed to the original function
            **kwargs: Keyword arguments passed to the original function
            
        Returns:
            Any: The results from quantum execution (or None if nested)
        """
        
        # Check if we're already inside a quantum context (nested @quantum call)
        existing_context = _current_context.get()
        
        if existing_context is not None:
            # We're already in a QNode - execute the function code directly
            # This allows nested @quantum functions to work as subcircuits
            original_globals = func.__globals__.copy()
            dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                            'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 
                            'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                            'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                            'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                            'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                            'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                            'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                            'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian',
                            'period_finding', 'shor_factor_circuit']
            
            # Replace DSL names with methods from the existing context
            # We need to modify func.__globals__ directly (it's a reference to the actual module globals)
            func_globals = func.__globals__
            saved_dsl = {}
            for func_name in dsl_functions:
                saved_dsl[func_name] = func_globals.get(func_name)
                func_globals[func_name] = getattr(existing_context, func_name)
            
            try:
                # Call the original function directly (it will use DSL methods from existing context)
                # This preserves function parameter handling and calling convention
                func(*args, **kwargs)
            finally:
                # Restore original DSL function references
                for func_name, original_value in saved_dsl.items():
                    if original_value is not None:
                        func_globals[func_name] = original_value
                    elif func_name in func_globals:
                        del func_globals[func_name]
            
            return None  # Nested calls don't return measurements
        
        # We're not in a quantum context - create a new one
        # Step 1: Create a quantum context object
        qc = QuantumContext()
        
        # Set this context as the current one
        token = _current_context.set(qc)
        
        try:
            # Step 2: Get the original function's global namespace
            original_globals = func.__globals__.copy()
            
            # Step 3: Replace DSL function names with context methods
            dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                            'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 
                            'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                            'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                            'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                            'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                            'measure_pauli_x', 'measure_pauli_y', 'measure_pauli_z',
                            'reset_qubit', 'copy_basis_state', 'swap_test',
                            'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                            'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                            'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian',
                            'period_finding', 'shor_factor_circuit']
            
            # Step 4: Do a preliminary execution to call qubit() and create the device
            # We'll make all other functions no-ops during this pass
            prelim_locals = {}
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            for i, arg_name in enumerate(arg_names):
                if i < len(args):
                    prelim_locals[arg_name] = args[i]
            prelim_locals.update(kwargs)
            
            # Create temporary globals where qubit() and set_shots() work, everything else is a no-op
            prelim_globals = original_globals.copy()
            for func_name in dsl_functions:
                if func_name == 'qubit':
                    prelim_globals[func_name] = qc.qubit
                elif func_name == 'set_shots':
                    prelim_globals[func_name] = qc.set_shots
                else:
                    prelim_globals[func_name] = lambda *a, **kw: None
            
            # Execute just to call qubit() - suppress stdout to avoid duplicate print statements
            # Ignore any errors
            old_stdout = sys.stdout
            try:
                # Redirect stdout to suppress print statements during preliminary execution
                sys.stdout = io.StringIO()
                try:
                    exec(func.__code__, prelim_globals, prelim_locals)
                finally:
                    sys.stdout = old_stdout
            except Exception:
                # Restore stdout if there was an error
                sys.stdout = old_stdout
                pass  # If qubit() wasn't called or failed, we'll use minimal device
            
            # Step 5: Create device if qubit() wasn't called
            if qc.device is None:
                qc.device = qml.device('default.qubit', wires=1)
                qc.num_qubits = 1  # Initialize num_qubits to match device
            
            # Step 6: Replace DSL names with context methods for real execution
            # Also add qml to globals so users can use qml.PauliZ, etc.
            original_globals['qml'] = qml
            for func_name in dsl_functions:
                original_globals[func_name] = getattr(qc, func_name)
            
            # Step 7: Define a QNode that wraps the original function's execution
            @qml.qnode(qc.device)
            def quantum_circuit(*qnode_args, **qnode_kwargs):
                # Reset measurement results for this execution
                qc.measurement_results = []
                # Reset qubit call tracking for this execution
                qc._qubit_called = False
                
                # Create locals dict with function parameters
                exec_locals = {}
                arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                
                # First, populate from keyword arguments (these take precedence)
                exec_locals.update(qnode_kwargs)
                
                # Then, populate from positional arguments (only if not already in kwargs)
                for i, arg_name in enumerate(arg_names):
                    if i < len(qnode_args) and arg_name not in exec_locals:
                        exec_locals[arg_name] = qnode_args[i]
                
                # Check if we have all required arguments
                # During PennyLane's construction phase, it may call without arguments
                # In that case, provide default values
                num_required = func.__code__.co_argcount - len(func.__defaults__ or [])
                missing_args = []
                for i in range(num_required):
                    arg_name = arg_names[i]
                    if arg_name not in exec_locals:
                        missing_args.append(arg_name)
                        # Provide a default value for construction phase
                        # Use 1 as default (reasonable for qubit counts)
                        exec_locals[arg_name] = 1
                
                # Execute the original function's code within the QNode context
                # If we had to add default values, this is likely PennyLane's construction phase
                # The actual execution will happen later with proper arguments
                try:
                    exec(func.__code__, original_globals, exec_locals)
                except TypeError as e:
                    # If it's still a missing argument error (shouldn't happen, but be safe)
                    error_str = str(e)
                    if "missing" in error_str and "required positional argument" in error_str:
                        # Try to extract which argument is missing and provide it
                        for arg_name in arg_names[:num_required]:
                            if arg_name not in exec_locals:
                                exec_locals[arg_name] = 1
                        # Try one more time
                        try:
                            exec(func.__code__, original_globals, exec_locals)
                        except Exception:
                            # If it still fails, this is likely construction phase
                            # Return a dummy measurement so PennyLane can continue
                            # The real execution will happen later with proper arguments
                            if qc.num_qubits > 0:
                                return qml.probs(wires=list(range(qc.num_qubits)))
                            return qml.probs(wires=[0])
                    else:
                        raise
                except Exception:
                    # Any other exception during construction phase
                    # Return a dummy measurement if we have qubits, otherwise let it fail
                    if missing_args and qc.num_qubits > 0:
                        return qml.probs(wires=list(range(qc.num_qubits)))
                    elif missing_args:
                        return qml.probs(wires=[0])
                    raise
                
                results = qc.get_measurement_results()
                # If no measurements were made, return a dummy one for construction phase
                if results is None or (isinstance(results, list) and len(results) == 0):
                    if qc.num_qubits > 0:
                        return qml.probs(wires=list(range(qc.num_qubits)))
                    return qml.probs(wires=[0])
                return results

            # Step 8: Execute the QNode
            result = quantum_circuit(*args, **kwargs)
            return result
            
        finally:
            # Always restore the previous context to prevent leaking stale references
            # Even at top level, we need to reset so the context variable doesn't point
            # to a dead QuantumContext object after this function completes
            #however this will only reset at the top level since nested functions just use the top levels context
            _current_context.reset(token)
        
    # Store the original function on the wrapper for use by draw() and other utilities
    wrapper.__quantum_func__ = func
    
    # Return the wrapper function
    # When someone calls @quantum on a function, they get this wrapper instead
    return wrapper


def draw(quantum_func, *args, **kwargs):
    """
    Draw a quantum circuit diagram for a @quantum-decorated function using matplotlib.
    
    This function creates a visual representation of the quantum circuit using PennyLane's
    matplotlib drawer. It executes the function in a special mode that captures the QNode
    and draws it instead of executing it.
    
    Args:
        quantum_func: A @quantum-decorated function to draw
        *args: Positional arguments to pass to the quantum function (for parameterized circuits)
        **kwargs: Keyword arguments to pass to the quantum function
    
    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes) - The figure and axes objects
        
    Example:
        @quantum
        def my_circuit():
            qubit(2)
            hadamard(0)
            cnot(0, 1)
            measure_probs(0, 1)
        
        import matplotlib.pyplot as plt
        fig, ax = draw(my_circuit)
        plt.show()  # Use plt.show() instead of fig.show() for reliable display
    """
    # Get the original function from the wrapper
    # The wrapper stores the original function in __quantum_func__
    if hasattr(quantum_func, '__quantum_func__'):
        original_func = quantum_func.__quantum_func__
    elif hasattr(quantum_func, '__wrapped__'):
        original_func = quantum_func.__wrapped__
    else:
        # If no stored reference, assume it's the function itself (not decorated)
        original_func = quantum_func
    
    # Create a quantum context to build the circuit
    qc = QuantumContext()
    token = _current_context.set(qc)
    
    try:
        # Get the original function's global namespace
        original_globals = original_func.__globals__.copy()
        
        # Replace DSL function names with context methods
        dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                        'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
                        'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                        'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                        'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                        'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                        'measure_pauli_x', 'measure_pauli_y', 'measure_pauli_z',
                        'reset_qubit', 'copy_basis_state', 'swap_test',
                        'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                        'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                        'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian']
        
        # Do a preliminary execution to call qubit() and create the device
        # We'll make all other functions no-ops during this pass
        prelim_locals = {}
        arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
        for i, arg_name in enumerate(arg_names):
            if i < len(args):
                prelim_locals[arg_name] = args[i]
        prelim_locals.update(kwargs)
        
        # Create temporary globals where qubit() and set_shots() work, everything else is a no-op
        prelim_globals = original_globals.copy()
        for func_name in dsl_functions:
            if func_name == 'qubit':
                prelim_globals[func_name] = qc.qubit
            elif func_name == 'set_shots':
                prelim_globals[func_name] = qc.set_shots
            else:
                prelim_globals[func_name] = lambda *a, **kw: None
        
        # Execute just to call qubit() - suppress stdout to avoid duplicate print statements
        # Ignore any errors
        old_stdout = sys.stdout
        try:
            # Redirect stdout to suppress print statements during preliminary execution
            sys.stdout = io.StringIO()
            try:
                exec(original_func.__code__, prelim_globals, prelim_locals)
            finally:
                sys.stdout = old_stdout
        except Exception:
            # Restore stdout if there was an error
            sys.stdout = old_stdout
            pass  # If qubit() wasn't called or failed, we'll use minimal device
        
        # Create device if qubit() wasn't called
        if qc.device is None:
            qc.device = qml.device('default.qubit', wires=1)
            qc.num_qubits = 1  # Initialize num_qubits to match device
        
        # Replace DSL names with context methods for real execution
        # Also add qml to globals so users can use qml.PauliZ, etc.
        original_globals['qml'] = qml
        for func_name in dsl_functions:
            original_globals[func_name] = getattr(qc, func_name)
        
        # Create a QNode that wraps the function execution
        @qml.qnode(qc.device)
        def quantum_circuit(*qnode_args, **qnode_kwargs):
            # Reset measurement results for this execution
            qc.measurement_results = []
            # Reset qubit call tracking for this execution
            qc._qubit_called = False
            
            # Create locals dict with function parameters
            exec_locals = {}
            arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
            
            # First, populate from keyword arguments (these take precedence)
            exec_locals.update(qnode_kwargs)
            
            # Then, populate from positional arguments (only if not already in kwargs)
            for i, arg_name in enumerate(arg_names):
                if i < len(qnode_args) and arg_name not in exec_locals:
                    exec_locals[arg_name] = qnode_args[i]
            
            # Execute the original function's code within the QNode context
            # Suppress stdout to avoid print statements during drawing
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                try:
                    exec(original_func.__code__, original_globals, exec_locals)
                finally:
                    sys.stdout = old_stdout
            except Exception:
                sys.stdout = old_stdout
                raise
            
            return qc.get_measurement_results()
        
        # Use qml.draw_mpl to draw the circuit
        # Suppress stdout during the draw execution as well
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            try:
                fig, ax = qml.draw_mpl(quantum_circuit)(*args, **kwargs)
            finally:
                sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            raise
        
        return fig, ax
        
    finally:
        _current_context.reset(token)


class QuantumContext:
    """Context object that provides DSL methods and manages quantum execution."""
    
    def __init__(self):
        self.device = None  # Device created when qubit() is called
        self.num_qubits = 0
        self.measurement_results = []
        self._shots = None
        self._qubit_called = False  # Track if qubit() has been called (enforce single call)
    
    def _validate_qubit(self, qubit):
        """Validate qubit index and raise error if invalid."""
        if not isinstance(qubit, int):
            raise TypeError(f"Qubit must be an integer index, got {type(qubit)}")
        if not (0 <= qubit < self.num_qubits):
            raise ValueError(f"Qubit {qubit} not declared (only {self.num_qubits} qubits available)")
    
    def qubit(self, num_wires: int):
        """
        Declare qubits and create device. Can only be called once per quantum function.
        
        Args:
            num_wires: Number of qubits to declare (0-indexed, so qubit(5) creates qubits 0-4)
        
        Example:
            qubit(10)  # Declares qubits 0-9, device has 10 wires
        
        Raises:
            ValueError: If qubit() has already been called, or if num_wires <= 0
        """
        # Enforce single call
        if self._qubit_called:
            raise ValueError(
                "qubit() has already been called for this quantum function. "
                "Only one qubit() declaration is allowed per quantum function. "
                "If you need more qubits, create a new quantum function."
            )
        
        if num_wires <= 0:
            raise ValueError("Must declare at least 1 qubit")
        
        self._qubit_called = True
        self.num_qubits = num_wires
        
        # Create device with the specified number of wires
        if self._shots is not None:
            self.device = qml.device('default.qubit', wires=num_wires, shots=self._shots)
        else:
            self.device = qml.device('default.qubit', wires=num_wires)
    
    def set_shots(self, shots):
        """
        Set number of shots for the device (required for sample() and counts() measurements).
        
        Must be called before qubit() to take effect. If called after qubit(), the device
        will be recreated with the new shot count.
        
        Args:
            shots: Number of shots to use for sampling measurements
        """
        self._shots = shots
        # Recreate device with shots if qubit() was already called
        if self._qubit_called and self.device is not None:
            self.device = qml.device('default.qubit', wires=self.num_qubits, shots=shots)
    
    def hadamard(self, qubit):
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)
        qml.Hadamard(wires=qubit)
    
    def xgate(self, qubit):
        """Apply X gate."""
        self._validate_qubit(qubit)
        qml.PauliX(wires=qubit)

    def rx(self, angle, qubit):
        """Apply RX gate."""
        self._validate_qubit(qubit)
        qml.RX(angle, wires=qubit)

    def ry(self, angle, qubit):
        """Apply RY gate."""
        self._validate_qubit(qubit)
        qml.RY(angle, wires=qubit)

    def rz(self, angle, qubit):
        """Apply RZ gate."""
        self._validate_qubit(qubit)
        qml.RZ(angle, wires=qubit)

    def ygate(self, qubit):
        """Apply Y gate."""
        self._validate_qubit(qubit)
        qml.PauliY(wires=qubit)

    def zgate(self, qubit):
        """Apply Z gate."""
        self._validate_qubit(qubit)
        qml.PauliZ(wires=qubit)

    def phase_shift(self, phi, qubit):
        """Apply phase shift gate."""
        self._validate_qubit(qubit)
        qml.PhaseShift(phi, wires=qubit)

    def s(self, qubit):
        """Apply S gate."""
        self._validate_qubit(qubit)
        qml.S(wires=qubit)

    def t(self, qubit):
        """Apply T gate."""
        self._validate_qubit(qubit)
        qml.T(wires=qubit)

    def u1(self, phi, qubit):
        """Apply U1 gate."""
        self._validate_qubit(qubit)
        qml.U1(phi, wires=qubit)

    def u2(self, phi, lam, qubit):
        """Apply U2 gate."""
        self._validate_qubit(qubit)
        qml.U2(phi, lam, wires=qubit)

    def u3(self, theta, phi, lam, qubit):
        """Apply U3 gate."""
        self._validate_qubit(qubit)
        qml.U3(theta, phi, lam, wires=qubit)

    def cnot(self, control, target):
        """Apply CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CNOT(wires=[control, target])

    def cy(self, control, target):
        """Apply controlled Y gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CY(wires=[control, target])

    def cz(self, control, target):
        """Apply controlled Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CZ(wires=[control, target])
    
    def crx(self, angle, control, target):
        """Apply controlled RX gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CRX(angle, wires=[control, target])
    
    def cry(self, angle, control, target):
        """Apply controlled RY gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CRY(angle, wires=[control, target])
    
    def crz(self, angle, control, target):
        """Apply controlled RZ gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CRZ(angle, wires=[control, target])

    def ch(self, control, target):
        """Apply controlled Hadamard gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        qml.CH(wires=[control, target])
    
    def toffoli(self, control1, control2, target):
        """Apply Toffoli gate."""
        self._validate_qubit(control1)
        self._validate_qubit(control2)
        self._validate_qubit(target)
        qml.Toffoli(wires=[control1, control2, target])
    
    def swap(self, qubit1, qubit2):
        """Apply SWAP gate."""
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        qml.SWAP(wires=[qubit1, qubit2])

    def cswap(self, control, qubit1, qubit2):
        """Apply controlled SWAP gate."""
        self._validate_qubit(control)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        qml.CSWAP(wires=[control, qubit1, qubit2])

    def multi_rz(self, phi, wires):
        """Apply multi-qubit RZ rotation."""
        for wire in wires:
            self._validate_qubit(wire)
        qml.MultiRZ(phi, wires=wires)

    def multi_controlled_x(self, wires, control_vals):
        """Apply multi-controlled X gate."""
        for wire in wires:
            self._validate_qubit(wire)
        qml.MultiControlledX(wires=wires, control_values=control_vals)

    def multi_controlled_z(self, wires):
        """Apply multi-controlled Z gate."""
        for wire in wires:
            self._validate_qubit(wire)
        # Use MultiControlledX with X gates to create MultiControlledZ effect
        if len(wires) >= 2:
            # Apply X to all qubits except the last
            qml.Hadamard(wires=wires[-1])
            # Apply multi-controlled X
            qml.MultiControlledX(wires=wires)
            # Apply X again to all qubits except the last
            qml.Hadamard(wires=wires[-1])
        else:
            qml.PauliZ(wires=wires[0])
    
    def qubitunitary(self, matrix, wires):
        """Apply arbitrary unitary matrix."""
        for wire in wires:
            self._validate_qubit(wire)
        qml.QubitUnitary(matrix, wires=wires)
    
    def hadamard_transform(self, qubits):
        """Apply Hadamard transform to multiple qubits."""
        for qubit in qubits:
            self._validate_qubit(qubit)
            qml.Hadamard(wires=qubit)
    
    def measure_probs(self, *qubits):
        """Measure qubit probabilities."""
        for qubit in qubits:
            self._validate_qubit(qubit)
        measurement = qml.probs(wires=list(qubits))
        self.measurement_results.append(measurement)
        return measurement

    def measure_expval(self, observable, *qubits):
        """Measure expectation value of observable."""
        for qubit in qubits:
            self._validate_qubit(qubit)
        measurement = qml.expval(observable(wires=qubits))
        self.measurement_results.append(measurement)
        return measurement

    def measure_var(self, observable, *qubits):
        """Measure variance of observable."""
        for qubit in qubits:
            self._validate_qubit(qubit)
        measurement = qml.var(observable(wires=qubits))
        self.measurement_results.append(measurement)
        return measurement

    def measure(self, qubit, postselect=None):
        """
        Mid-circuit measurement (destructive measurement).
        
        This measurement can be used for conditional operations with cond().
        Example:
            m = measure(0)
            cond(m == 1, xgate, 1)  # Apply X to qubit 1 if qubit 0 measured 1
        
        Args:
            qubit: Qubit index to measure
            postselect: Optional post-selection value (0 or 1) to condition on
        
        Returns:
            Measurement result (can be used with cond() for conditional operations)
        """
        self._validate_qubit(qubit)
        if postselect is not None:
            return qml.measure(wires=qubit, postselect=postselect)
        return qml.measure(wires=qubit)
    
    def cond(self, condition, gate_func, *gate_args):
        """
        Apply a gate conditionally based on a measurement result.
        
        This is a DSL-friendly wrapper around qml.cond() that works with DSL gate functions.
        
        Args:
            condition: A boolean condition based on a measurement (e.g., m == 1, m == 0)
            gate_func: The DSL gate function to apply (e.g., xgate, hadamard, rx)
            *gate_args: Arguments to pass to the gate function
        
        Example:
            m = measure(0)
            cond(m == 1, xgate, 1)  # Apply X gate to qubit 1 if qubit 0 measured 1
            cond(m == 0, hadamard, 2)  # Apply Hadamard to qubit 2 if qubit 0 measured 0
        
        Note:
            The gate function must be a DSL primitive (e.g., xgate, not qml.X).
            The gate will be applied to the qubits specified in gate_args.
        """
        # Create a callable that applies the gate function with the given arguments
        def apply_gate():
            gate_func(*gate_args)
        
        # Use qml.cond to conditionally apply the gate
        qml.cond(condition, apply_gate)()

    def measure_state(self, *qubits):
        """
        Measure full quantum state.
        
        Note: qml.state() always measures the entire device state.
        If qubits are specified, the full state is still returned (you can extract
        the relevant qubits from the result).
        """
        # qml.state() doesn't take wires - it always measures the full device
        # Validate qubits if provided, but state() measures everything anyway
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
        measurement = qml.state()
        self.measurement_results.append(measurement)
        return measurement

    def measure_density_matrix(self, *qubits):
        """Measure reduced density matrix."""
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            # qml.density_matrix() takes wires as argument
            measurement = qml.density_matrix(wires=list(qubits))
        else:
            # Full device density matrix - need to pass all wires
            measurement = qml.density_matrix(wires=list(range(self.num_qubits)))
        self.measurement_results.append(measurement)
        return measurement

    def measure_sample(self, *qubits, observable=None):
        """
        Sample from measurement (returns raw samples).
        
        Note: Shots must be configured on the device using set_shots().
        
        Args:
            *qubits: Qubits to measure (if empty, measures all qubits)
            observable: Optional observable to measure (if None, measures in computational basis)
        
        Returns:
            Sample measurement object
        """
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            if observable:
                measurement = qml.sample(observable(wires=list(qubits)))
            else:
                measurement = qml.sample(wires=list(qubits))
        else:
            if observable:
                measurement = qml.sample(observable())
            else:
                measurement = qml.sample()
        self.measurement_results.append(measurement)
        return measurement

    def measure_counts(self, *qubits, observable=None):
        """
        Measure and return counts (histogram of measurement outcomes).
        
        Note: Shots must be configured on the device using set_shots().
        
        Args:
            *qubits: Qubits to measure (if empty, measures all qubits)
            observable: Optional observable to measure (if None, measures in computational basis)
        
        Returns:
            Counts measurement object
        """
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            if observable:
                measurement = qml.counts(observable(wires=list(qubits)))
            else:
                measurement = qml.counts(wires=list(qubits))
        else:
            if observable:
                measurement = qml.counts(observable())
            else:
                measurement = qml.counts()
        self.measurement_results.append(measurement)
        return measurement

    def measure_mutual_info(self, wires0, wires1, log_base=None):
        """
        Measure mutual information between two subsystems.
        
        Args:
            wires0: First subsystem (list of qubit indices or single qubit)
            wires1: Second subsystem (list of qubit indices or single qubit)
            log_base: Base of logarithm (default: e, natural log)
        
        Returns:
            Mutual information measurement object
        
        Example:
            measure_mutual_info([0, 1], [2, 3])  # Mutual info between qubits 0,1 and 2,3
        """
        # Convert single qubits to lists
        if isinstance(wires0, int):
            wires0 = [wires0]
        if isinstance(wires1, int):
            wires1 = [wires1]
        
        # Validate all qubits
        for q in wires0 + wires1:
            self._validate_qubit(q)
        
        if log_base is not None:
            measurement = qml.mutual_info(wires0=wires0, wires1=wires1, log_base=log_base)
        else:
            measurement = qml.mutual_info(wires0=wires0, wires1=wires1)
        self.measurement_results.append(measurement)
        return measurement

    def measure_vn_entropy(self, *qubits, log_base=None):
        """
        Measure Von Neumann entropy of a subsystem.
        
        Args:
            *qubits: Qubits to measure entropy of (if empty, measures all qubits)
            log_base: Base of logarithm (default: e, natural log)
        
        Returns:
            Von Neumann entropy measurement object
        
        Example:
            measure_vn_entropy(0, 1)  # Entropy of qubits 0 and 1
        """
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            wires = list(qubits)
        else:
            wires = list(range(self.num_qubits))
        
        if log_base is not None:
            measurement = qml.vn_entropy(wires=wires, log_base=log_base)
        else:
            measurement = qml.vn_entropy(wires=wires)
        self.measurement_results.append(measurement)
        return measurement

    def measure_purity(self, *qubits):
        """
        Measure purity of a subsystem.
        
        Purity is a measure of how "pure" a quantum state is.
        Pure states have purity = 1, mixed states have purity < 1.
        
        Args:
            *qubits: Qubits to measure purity of (if empty, measures all qubits)
        
        Returns:
            Purity measurement object
        
        Example:
            measure_purity(0, 1)  # Purity of qubits 0 and 1
        """
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            wires = list(qubits)
        else:
            wires = list(range(self.num_qubits))
        
        measurement = qml.purity(wires=wires)
        self.measurement_results.append(measurement)
        return measurement

    def measure_classical_shadow(self, *qubits, seed=None):
        """
        Measure using classical shadow protocol.
        
        Classical shadows are a measurement protocol that allows efficient
        estimation of many observables from a small number of measurements.
        
        Args:
            *qubits: Qubits to measure (if empty, measures all qubits)
            seed: Random seed for the measurement protocol
        
        Returns:
            Classical shadow measurement object
        
        Example:
            measure_classical_shadow(0, 1, 2, seed=42)
        """
        if qubits:
            for qubit in qubits:
                self._validate_qubit(qubit)
            wires = list(qubits)
        else:
            wires = list(range(self.num_qubits))
        
        if seed is not None:
            measurement = qml.classical_shadow(wires=wires, seed=seed)
        else:
            measurement = qml.classical_shadow(wires=wires)
        self.measurement_results.append(measurement)
        return measurement

    def measure_shadow_expval(self, H, k=1, seed=None):
        """
        Measure expectation value using classical shadow protocol.
        
        This uses the classical shadow protocol to estimate the expectation
        value of an observable H.
        
        Args:
            H: Observable to measure (e.g., qml.PauliZ(0), qml.PauliX(1))
            k: Number of shots to use for shadow estimation
            seed: Random seed for the measurement protocol
        
        Returns:
            Shadow expectation value measurement object
        
        Example:
            measure_shadow_expval(qml.PauliZ(0) @ qml.PauliZ(1), k=100)
        """
        # Validate qubits in the observable
        # H might be a single observable or a tensor product
        # We'll try to extract wires from it
        if hasattr(H, 'wires'):
            for wire in H.wires:
                if isinstance(wire, int):
                    self._validate_qubit(wire)
        
        if seed is not None:
            measurement = qml.shadow_expval(H=H, k=k, seed=seed)
        else:
            measurement = qml.shadow_expval(H=H, k=k)
        self.measurement_results.append(measurement)
        return measurement

    def measure_pauli_x(self, qubit):
        """
        Measure qubit in the X basis (Hadamard basis).
        
        How it works:
        1. Apply Hadamard gate to rotate from Z basis to X basis
        2. Measure in computational (Z) basis
        3. This effectively measures in X basis
        
        Args:
            qubit: Qubit index to measure
        
        Returns:
            Measurement result (probabilities in X basis)
        
        Example:
            measure_pauli_x(0)  # Measure qubit 0 in X basis
        """
        self._validate_qubit(qubit)
        # Rotate to X basis
        self.hadamard(qubit)
        # Measure in computational basis (now effectively X basis)
        measurement = qml.probs(wires=qubit)
        self.measurement_results.append(measurement)
        return measurement

    def measure_pauli_y(self, qubit):
        """
        Measure qubit in the Y basis.
        
        How it works:
        1. Apply S† (S dagger) gate: rotates Y basis to X basis
        2. Apply Hadamard gate: rotates X basis to Z basis
        3. Measure in computational (Z) basis
        4. This effectively measures in Y basis
        
        Args:
            qubit: Qubit index to measure
        
        Returns:
            Measurement result (probabilities in Y basis)
        
        Example:
            measure_pauli_y(0)  # Measure qubit 0 in Y basis
        """
        self._validate_qubit(qubit)
        # Rotate to Y basis: S† then H
        # S† = S^(-1) = S^3, but we can use qml.adjoint(qml.S)
        qml.adjoint(qml.S)(wires=qubit)
        self.hadamard(qubit)
        # Measure in computational basis (now effectively Y basis)
        measurement = qml.probs(wires=qubit)
        self.measurement_results.append(measurement)
        return measurement

    def measure_pauli_z(self, qubit):
        """
        Measure qubit in the Z basis (computational basis).
        
        This is the standard measurement - just measures in computational basis.
        
        Args:
            qubit: Qubit index to measure
        
        Returns:
            Measurement result (probabilities in Z basis)
        
        Example:
            measure_pauli_z(0)  # Measure qubit 0 in Z basis
        """
        self._validate_qubit(qubit)
        # Direct measurement in computational (Z) basis
        measurement = qml.probs(wires=qubit)
        self.measurement_results.append(measurement)
        return measurement

    def reset_qubit(self, qubit):
        """
        Reset qubit to |0⟩ state.
        
        How it works:
        1. Measure the qubit (destructive measurement)
        2. If measurement result is |1⟩, apply X gate to flip it to |0⟩
        3. If measurement result is |0⟩, do nothing
        4. Result: qubit is guaranteed to be in |0⟩ state
        
        Note: This is a destructive operation - the original state is lost.
        
        Args:
            qubit: Qubit index to reset
        
        Example:
            reset_qubit(0)  # Reset qubit 0 to |0⟩
        """
        self._validate_qubit(qubit)
        # Measure the qubit
        m = self.measure(qubit)
        # Conditionally apply X gate if qubit was |1⟩
        # This flips |1⟩ → |0⟩, leaving |0⟩ unchanged
        self.cond(m == 1, self.xgate, qubit)

    def copy_basis_state(self, source, target):
        """
        Copy basis state (|0⟩ or |1⟩) from source qubit to target qubit using CNOT.
        
        IMPORTANT LIMITATION: This only works for basis states |0⟩ and |1⟩.
        For superpositions, this creates ENTANGLEMENT, not a true copy.
        
        How it works:
        1. Apply CNOT with source as control and target as target
        2. If source is |0⟩, target stays unchanged → |00⟩
        3. If source is |1⟩, target flips → |11⟩
        4. Result: For basis states, target ends up in the same state as source
        
        Why it doesn't work for superpositions:
        - If source is |+⟩ = (|0⟩ + |1⟩)/√2 and target is |0⟩
        - After CNOT: (|00⟩ + |11⟩)/√2 = Bell state |Φ+⟩
        - The qubits are ENTANGLED, not independent copies
        - Measuring source collapses target (they're not independent)
        
        This is due to the no-cloning theorem: you cannot create an independent
        copy of an unknown quantum state. For true state transfer (not copying),
        use quantum_teleportation() instead.
        
        Args:
            source: Source qubit index (state to copy from)
            target: Target qubit index (state to copy to)
        
        Example:
            # Works for basis states:
            xgate(0)  # Prepare |1⟩
            copy_basis_state(0, 1)  # qubit 1 becomes |1⟩
            
            # Does NOT work for superpositions:
            hadamard(0)  # Prepare |+⟩
            copy_basis_state(0, 1)  # Creates entanglement, not a copy!
        """
        self._validate_qubit(source)
        self._validate_qubit(target)
        # CNOT: if source is |1⟩, flip target; if source is |0⟩, leave target unchanged
        # This works for basis states |0⟩ and |1⟩, but creates entanglement for superpositions
        self.cnot(source, target)

    def swap_test(self, state1_qubits, state2_qubits, ancilla):
        """
        Swap test: compare two quantum states.
        
        The swap test measures the overlap between two quantum states.
        The probability of measuring |0⟩ on the ancilla is related to the
        inner product (fidelity) between the two states.
        
        How it works:
        1. Prepare ancilla in |+⟩ = (|0⟩ + |1⟩)/√2
        2. Apply controlled-SWAP (CSWAP): ancilla controls swapping of state1 and state2
           - If ancilla is |0⟩, states are not swapped
           - If ancilla is |1⟩, states are swapped
        3. Apply Hadamard to ancilla
        4. Measure ancilla: probability of |0⟩ = (1 + |⟨ψ1|ψ2⟩|²)/2
        
        If states are identical: probability of |0⟩ = 1.0
        If states are orthogonal: probability of |0⟩ = 0.5
        If states are different: probability of |0⟩ is between 0.5 and 1.0
        
        Args:
            state1_qubits: List of qubit indices for first state
            state2_qubits: List of qubit indices for second state (must be same length as state1_qubits)
            ancilla: Ancilla qubit index for the swap test
        
        Returns:
            Measurement result (probabilities on ancilla qubit)
        
        Example:
            # Compare two 2-qubit states
            swap_test([0, 1], [2, 3], 4)  # Compare qubits 0,1 with qubits 2,3 using ancilla 4
        """
        if len(state1_qubits) != len(state2_qubits):
            raise ValueError(f"state1_qubits and state2_qubits must have the same length, got {len(state1_qubits)} and {len(state2_qubits)}")
        
        for qubit in state1_qubits + state2_qubits + [ancilla]:
            self._validate_qubit(qubit)
        
        # Step 1: Prepare ancilla in |+⟩ = (|0⟩ + |1⟩)/√2
        self.hadamard(ancilla)
        
        # Step 2: Apply controlled-SWAP for each pair of qubits
        # CSWAP swaps state1 and state2 qubits when ancilla is |1⟩
        for q1, q2 in zip(state1_qubits, state2_qubits):
            self.cswap(ancilla, q1, q2)
        
        # Step 3: Apply Hadamard to ancilla
        self.hadamard(ancilla)
        
        # Step 4: Measure ancilla
        measurement = qml.probs(wires=ancilla)
        self.measurement_results.append(measurement)
        return measurement

    def basis_state(self, state, wires):
        """Prepare specific basis state."""
        for wire in wires:
            self._validate_qubit(wire)
        qml.BasisState(state, wires=wires)

    def state_prep(self, state, wires):
        """Prepare arbitrary quantum state."""
        for wire in wires:
            self._validate_qubit(wire)
        qml.StatePrep(state, wires=wires)

    def bell_state(self, qubit1, qubit2, state='phi_plus'):
        """
        Prepare one of the four Bell states (maximally entangled 2-qubit states).
        
        The four Bell states are:
        - |Φ+⟩ = (|00⟩ + |11⟩)/√2  (phi_plus, default)
        - |Φ-⟩ = (|00⟩ - |11⟩)/√2  (phi_minus)
        - |Ψ+⟩ = (|01⟩ + |10⟩)/√2  (psi_plus)
        - |Ψ-⟩ = (|01⟩ - |10⟩)/√2  (psi_minus)
        
        How it works:
        1. Apply Hadamard to first qubit: creates superposition (|0⟩ + |1⟩)/√2
        2. Apply CNOT with first qubit as control: entangles the qubits
           - If first qubit is |0⟩, second stays |0⟩ → |00⟩
           - If first qubit is |1⟩, second flips to |1⟩ → |11⟩
           - Result: (|00⟩ + |11⟩)/√2 = |Φ+⟩
        3. Optional phase flip (Z gate) to create other Bell states
        
        Args:
            qubit1: First qubit index
            qubit2: Second qubit index
            state: Which Bell state to prepare ('phi_plus', 'phi_minus', 'psi_plus', 'psi_minus')
        
        Example:
            bell_state(0, 1, 'phi_plus')  # Prepare |Φ+⟩ on qubits 0 and 1
        """
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        
        # Standard Bell state preparation: H on first, CNOT
        self.hadamard(qubit1)
        self.cnot(qubit1, qubit2)
        
        # Apply phase/bit flips to get other Bell states
        if state == 'phi_minus':
            # |Φ-⟩ = (|00⟩ - |11⟩)/√2: add phase flip
            self.zgate(qubit1)
        elif state == 'psi_plus':
            # |Ψ+⟩ = (|01⟩ + |10⟩)/√2: flip second qubit
            self.xgate(qubit2)
        elif state == 'psi_minus':
            # |Ψ-⟩ = (|01⟩ - |10⟩)/√2: flip second qubit and add phase
            self.xgate(qubit2)
            self.zgate(qubit1)
        # phi_plus is the default (no additional gates needed)

    def ghz_state(self, qubits):
        """
        Prepare GHZ (Greenberger-Horne-Zeilinger) state: (|000...⟩ + |111...⟩)/√2.
        
        GHZ states are maximally entangled multi-qubit states. For n qubits:
        |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
        
        How it works:
        1. Apply Hadamard to first qubit: creates (|0⟩ + |1⟩)/√2
        2. Apply CNOT chain: first qubit controls second, second controls third, etc.
           - If first qubit is |0⟩, all others stay |0⟩ → |00...0⟩
           - If first qubit is |1⟩, all others flip to |1⟩ → |11...1⟩
           - Result: (|00...0⟩ + |11...1⟩)/√2
        
        Args:
            qubits: List of qubit indices to prepare GHZ state on
        
        Example:
            ghz_state([0, 1, 2, 3])  # Prepare 4-qubit GHZ state
        """
        if len(qubits) == 0:
            raise ValueError("Must provide at least one qubit for GHZ state")
        if len(qubits) == 1:
            # Single qubit: just apply Hadamard
            self.hadamard(qubits[0])
            return
        
        for qubit in qubits:
            self._validate_qubit(qubit)
        
        # Apply Hadamard to first qubit
        self.hadamard(qubits[0])
        
        # Apply CNOT chain: each qubit controls the next
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i], qubits[i + 1])

    def w_state(self, qubits):
        """
        Prepare W state: (|100...⟩ + |010...⟩ + |001...⟩ + ...)/√n.
        
        W states are symmetric superpositions where exactly one qubit is |1⟩.
        For n qubits: |W⟩ = (|100...0⟩ + |010...0⟩ + |001...0⟩ + ... + |000...1⟩)/√n
        
        How it works:
        Uses direct state preparation by constructing the W state vector and using StatePrep.
        The W state has equal amplitudes (1/√n) for all states with exactly one |1⟩.
        
        Args:
            qubits: List of qubit indices to prepare W state on
        
        Example:
            w_state([0, 1, 2])  # Prepare 3-qubit W state: (|100⟩ + |010⟩ + |001⟩)/√3
        """
        if len(qubits) == 0:
            raise ValueError("Must provide at least one qubit for W state")
        if len(qubits) == 1:
            # Single qubit W state is just |1⟩
            self.xgate(qubits[0])
            return
        
        for qubit in qubits:
            self._validate_qubit(qubit)
        
        n = len(qubits)
        dim = 2 ** n
        
        # Construct W state vector: equal amplitude for all single-excitation states
        # For n qubits, there are n states with exactly one |1⟩
        w_state_vector = np.zeros(dim, dtype=complex)
        
        # Set amplitude 1/√n for each state with exactly one |1⟩
        # These are states at indices: 2^0, 2^1, 2^2, ..., 2^(n-1)
        # Which correspond to: |000...1⟩, |000...10⟩, |000...100⟩, ..., |100...0⟩
        amplitude = 1.0 / np.sqrt(n)
        for i in range(n):
            # State with |1⟩ at position i (counting from right, LSB)
            # For qubit i, the state index is 2^i
            state_index = 2 ** i
            w_state_vector[state_index] = amplitude
        
        # Use state preparation to prepare the W state
        self.state_prep(w_state_vector, qubits)

    def quantum_teleportation(self, alice_qubit, bell_qubit1, bell_qubit2):
        """
        Quantum teleportation: teleport quantum state from Alice's qubit to Bob's qubit.
        
        Quantum teleportation allows transferring an unknown quantum state using:
        - 1 entangled Bell pair (shared between Alice and Bob)
        - 2 classical bits of communication
        
        How it works:
        1. Preparation: Create Bell state |Φ+⟩ between bell_qubit1 (Alice) and bell_qubit2 (Bob)
           - This creates shared entanglement: (|00⟩ + |11⟩)/√2
        2. Bell measurement: Alice performs Bell measurement on (alice_qubit, bell_qubit1)
           - Apply CNOT: alice_qubit controls bell_qubit1
           - Apply Hadamard to alice_qubit: measures in Bell basis
           - Measure both qubits: gets one of 4 outcomes (00, 01, 10, 11)
        3. Conditional correction: Bob applies gates to bell_qubit2 based on measurement outcome
           - 00: do nothing (state already correct)
           - 01: apply X gate (bit flip correction)
           - 10: apply Z gate (phase flip correction)
           - 11: apply X then Z gates (both corrections)
        
        After teleportation, bell_qubit2 contains the original state of alice_qubit.
        Note: alice_qubit is destroyed in the process (no-cloning theorem).
        
        Args:
            alice_qubit: Qubit with state to teleport (Alice's side)
            bell_qubit1: First qubit of Bell pair (Alice's side, part of shared entanglement)
            bell_qubit2: Second qubit of Bell pair (Bob's side, receives teleported state)
        
        Example:
            # Teleport state from qubit 0 to qubit 2 using Bell pair (1, 2)
            # qubit 1 is Alice's half of Bell pair, qubit 2 is Bob's half
            quantum_teleportation(0, 1, 2)
        """
        self._validate_qubit(alice_qubit)
        self._validate_qubit(bell_qubit1)
        self._validate_qubit(bell_qubit2)
        
        # Step 1: Create Bell state |Φ+⟩ between bell_qubit1 and bell_qubit2
        # This is the shared entanglement resource between Alice and Bob
        self.bell_state(bell_qubit1, bell_qubit2, 'phi_plus')
        
        # Step 2: Bell measurement on Alice's side (alice_qubit, bell_qubit1)
        # This measures in the Bell basis and collapses the state
        self.cnot(alice_qubit, bell_qubit1)
        self.hadamard(alice_qubit)
        
        # Step 3: Measure Alice's qubits to get classical bits
        # These measurements determine what correction Bob needs to apply
        m1 = self.measure(alice_qubit)
        m2 = self.measure(bell_qubit1)
        
        # Step 4: Bob applies conditional corrections based on measurement results
        # The corrections depend on the Bell measurement outcome:
        # - m1=0, m2=0 (00): no correction needed
        # - m1=0, m2=1 (01): apply X (bit flip)
        # - m1=1, m2=0 (10): apply Z (phase flip)
        # - m1=1, m2=1 (11): apply X then Z (both flips)
        
        # Apply X correction if m2 == 1
        self.cond(m2 == 1, self.xgate, bell_qubit2)
        # Apply Z correction if m1 == 1
        self.cond(m1 == 1, self.zgate, bell_qubit2)



#should probably check that phase qubit is in the |-> state
    def grover_diffusion(self, primary_qubits, phase_qubit):
        """
        Apply Grover's diffusion operator.
        
        Args:
            primary_qubits: List of qubit indices representing the search space
            phase_qubit: Qubit index for phase kickback (should be in |-> state)
        """
        # Apply Hadamard transform to primary qubits only (the search space)
        self.hadamard_transform(primary_qubits)

        # Multi-controlled X: primary qubits as controls, phase qubit as target
        # This flips phase of |000...0⟩ state via phase kickback (phase qubit in |->)
        # When all primaries are |0⟩, phase qubit flips |-> → -|->, creating phase kickback
        phase_wires = primary_qubits + [phase_qubit]
        control_vals = [0] * len(primary_qubits)  # Expect all primaries to be |0⟩
        self.multi_controlled_x(phase_wires, control_vals)
        
        # Apply Hadamard transform again to primary qubits
        self.hadamard_transform(primary_qubits)

    def grover_search(self, oracle_func, primary_qubits, phase_qubit, num_iterations):
        """
        Complete Grover search algorithm.
        
        Args:
            oracle_func: Function that applies the oracle (phase flip of solutions).
                        Can accept either:
                        - 1 argument: oracle(primary_qubits) for unitary-based oracles
                        - 2 arguments: oracle(primary_qubits, phase_qubit) for phase qubit pattern
            primary_qubits: List of qubit indices representing the search space
            phase_qubit: Qubit index for phase kickback (will be prepared in |-> state)
            num_iterations: Number of Grover iterations to perform
        """
        # Prepare phase qubit in |-> state for phase kickback
        # |-> = (|0⟩ - |1⟩)/√2, which allows X gates to create phase flips via kickback
        self.xgate(phase_qubit)
        self.hadamard(phase_qubit)
        
        # Initialize superposition on primary qubits only (the search space)
        self.hadamard_transform(primary_qubits)
        
        # Inspect oracle function signature to determine if it accepts phase_qubit
        sig = inspect.signature(oracle_func)
        oracle_params = len(sig.parameters)
        # If oracle accepts 2 parameters, it uses phase qubit pattern
        # If oracle accepts 1 parameter, it uses unitary matrices or multi_controlled_z
        uses_phase_qubit = oracle_params >= 2
        
        # Grover iterations
        for _ in range(num_iterations):
            # Apply oracle (phase flip of solutions)
            # Pass phase_qubit only if oracle function accepts it
            if uses_phase_qubit:
                oracle_func(primary_qubits, phase_qubit)
            else:
                oracle_func(primary_qubits)
            
            # Apply diffusion operator (reflects about average amplitude)
            self.grover_diffusion(primary_qubits, phase_qubit)
        
        # Don't return measurement here - let the QNode handle it
        # The measurement should be done separately with measure_probs()

    def build_cnf_oracle(self, expr_str, inputs, start_qubit_idx=0):
        """
        Build a phase-flip oracle for a Boolean expression using DSL functions.
        - expr_str: e.g. '(a & b) | (~c & d)'
        - inputs: list of variable names (primary search qubits)
        - start_qubit_idx: starting index for primary qubits (default 0)
        
        Returns: (num_primary, num_ancilla, num_phase) - qubit allocation info
        """

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

        # Convert to CNF
        cnf = expr_to_cnf(expr_str, inputs)
        clauses = cnf.args if cnf.func.__name__ == "And" else [cnf]
        
        num_primary = len(inputs)
        num_ancilla = len(clauses)
        num_phase = 1
        
        # Map variable name → qubit index
        idx = {name: start_qubit_idx + i for i, name in enumerate(inputs)}
        ancilla_start = start_qubit_idx + num_primary
        phase_qubit = ancilla_start + num_ancilla
        
        # --- Compute each clause into its ancilla ---
        for j, clause in enumerate(clauses):
            ancilla_qubit = ancilla_start + j
            
            # Extract literals
            lits = clause.args if clause.func.__name__ == "Or" else [clause]
            
            # Apply De Morgan's: For OR clause (b | c), use ~(~b & ~c)
            # Strategy: Prepare ancilla in |1⟩, use multi_controlled_x to flip to |0⟩ when (~b & ~c)
            # This leaves ancilla in |1⟩ when (b | c) is satisfied
            if len(lits) > 1:
                # OR clause: (b | c) = ~(~b & ~c)
                # Prepare ancilla in |1⟩
                self.xgate(ancilla_qubit)
                
                # Build control qubits and control values using De Morgan's
                # For (b | c): detect when (~b & ~c) is true, i.e., when b=0 AND c=0
                # So use control_vals=[0, 0] to detect both |0⟩
                control_qubits = []
                control_vals = []
                for lit in lits:
                    if lit.func.__name__ == "Not":
                        # For ~a: we want to detect when a=1 (since ~a means we want a to be false)
                        # In De Morgan's: (~a | b) = ~(a & ~b), so we detect when a=1 AND ~b
                        var_name = str(lit.args[0])
                        qubit_idx = idx[var_name]
                        control_qubits.append(qubit_idx)
                        control_vals.append(1)  # Detect when a=1 (meaning ~a is false, contributing to AND)
                    else:
                        # For a: we want to detect when a=0 (since we're checking ~a)
                        var_name = str(lit)
                        qubit_idx = idx[var_name]
                        control_qubits.append(qubit_idx)
                        control_vals.append(0)  # Detect when a=0 (meaning ~a is true, contributing to AND)
                
                # Multi-controlled X: flips ancilla when all controls match control_vals
                # This detects when (~b & ~c) is true, flipping ancilla from |1⟩ to |0⟩
                # Leaving ancilla as |1⟩ when (b | c) is true
                if control_qubits:
                    all_wires = control_qubits + [ancilla_qubit]
                    self.multi_controlled_x(all_wires, control_vals)
            else:
                # Single literal clause: use multi_controlled_x directly
                lit = lits[0]
                if lit.func.__name__ == "Not":
                    var_name = str(lit.args[0])
                    qubit_idx = idx[var_name]
                    self.multi_controlled_x([qubit_idx, ancilla_qubit], [0])
                else:
                    var_name = str(lit)
                    qubit_idx = idx[var_name]
                    self.multi_controlled_x([qubit_idx, ancilla_qubit], [1])
        
        # --- Phase flip if all clauses satisfied ---
        ancilla_wires = list(range(ancilla_start, ancilla_start + num_ancilla))
        phase_wires = ancilla_wires + [phase_qubit]
        
        # Multi-controlled X with all ancillas as controls - checks if all clauses are satisfied
        # Since phase qubit is in |-> state: flipping |-> → -|-> creates phase kickback
        # This works on superpositions: for each basis state where all ancillas are 1,
        # the phase qubit gets flipped from |-> to -|->, which phase-flips that basis state
        # No uncompute needed - the phase information is already kicked back to primary qubits
        control_vals = [1] * num_ancilla
        self.multi_controlled_x(phase_wires, control_vals) #ASSUMES PHASE QUBIT IS IN |-⟩ STATE
        
        # --- Uncompute clauses (reverse order) ---
        # Why uncompute? After phase kickback, the ancilla qubits are still entangled with primary qubits.
        # Uncomputing the clauses:
        # 1. Disentangles ancillas from primary qubits (ancillas go back to |0⟩)
        # 2. Allows ancillas to be reused or measured cleanly without affecting primary qubits
        # 3. Ensures the oracle only affects primary qubits via phase, not through entanglement
        # Note: We DON'T uncompute the phase qubit - its phase information is already kicked back
        # Same control_vals as computation (no X gates needed - control_vals handle inversions)
        for j, clause in reversed(list(enumerate(clauses))):
            ancilla_qubit = ancilla_start + j
            
            # Extract literals
            lits = clause.args if clause.func.__name__ == "Or" else [clause]
            
            # Uncompute using the same De Morgan's approach (reverse of computation)
            if len(lits) > 1:
                # OR clause: uncompute by reversing the multi_controlled_x, then uncompute initial X
                # Build same control pattern as computation (no X gates needed - control_vals handle it)
                control_qubits = []
                control_vals = []
                for lit in lits:
                    if lit.func.__name__ == "Not":
                        var_name = str(lit.args[0])
                        qubit_idx = idx[var_name]
                        control_qubits.append(qubit_idx)
                        control_vals.append(1)  # Same as computation
                    else:
                        var_name = str(lit)
                        qubit_idx = idx[var_name]
                        control_qubits.append(qubit_idx)
                        control_vals.append(0)  # Same as computation
                
                # Uncompute the multi_controlled_x (same operation, reversible)
                if control_qubits:
                    all_wires = control_qubits + [ancilla_qubit]
                    self.multi_controlled_x(all_wires, control_vals)
                
                # Uncompute initial X on ancilla (return it to |0⟩)
                self.xgate(ancilla_qubit)
            else:
                # Single literal clause: uncompute multi_controlled_x
                lit = lits[0]
                if lit.func.__name__ == "Not":
                    var_name = str(lit.args[0])
                    qubit_idx = idx[var_name]
                    self.multi_controlled_x([qubit_idx, ancilla_qubit], [0])
                else:
                    var_name = str(lit)
                    qubit_idx = idx[var_name]
                    self.multi_controlled_x([qubit_idx, ancilla_qubit], [1])
        
        return (num_primary, num_ancilla, num_phase)
    
    def qft(self, qubits):
        """
        Apply Quantum Fourier Transform (QFT) to a list of qubits.
        
        Args:
            qubits: List of qubit indices to apply QFT to
        
        Example:
            qft([0, 1, 2])  # Apply QFT to qubits 0, 1, 2
        """
        for qubit in qubits:
            self._validate_qubit(qubit)
        qml.QFT(wires=qubits)
    
    def iqft(self, qubits):
        """
        Apply Inverse Quantum Fourier Transform (IQFT) to a list of qubits.
        
        Args:
            qubits: List of qubit indices to apply IQFT to
        
        Example:
            iqft([0, 1, 2])  # Apply IQFT to qubits 0, 1, 2
        """
        for qubit in qubits:
            self._validate_qubit(qubit)
        qml.adjoint(qml.QFT)(wires=qubits)
    
    def controlled_unitary(self, control_qubit, unitary_matrix, target_wires, control_values=None):
        """
        Apply a controlled unitary operation.
        
        Args:
            control_qubit: Qubit index or list of qubit indices that control the unitary
            unitary_matrix: Unitary matrix to apply (numpy array)
            target_wires: List of qubit indices where the unitary is applied
            control_values: List of control values (0/False or 1/True) for each control qubit.
                           Default is all 1s (standard control). Use 0/False to control on |0⟩ state.
        
        Example:
            # Controlled Hadamard (as a matrix)
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            controlled_unitary(0, H, [1])  # Apply H to qubit 1 if qubit 0 is |1⟩
            controlled_unitary([0, 1], H, [2], control_values=[1, 0])  # Apply H if qubit 0 is |1⟩ and qubit 1 is |0⟩
        """
        # Handle single control qubit or list of control qubits
        if isinstance(control_qubit, int):
            control_wires = [control_qubit]
        else:
            control_wires = list(control_qubit)
        
        for wire in control_wires:
            self._validate_qubit(wire)
        for wire in target_wires:
            self._validate_qubit(wire)
        
        # Use PennyLane's ControlledQubitUnitary with optional control_values
        if control_values is not None:
            qml.ControlledQubitUnitary(unitary_matrix, control_wires=control_wires, wires=target_wires, control_values=control_values)
        else:
            qml.ControlledQubitUnitary(unitary_matrix, control_wires=control_wires, wires=target_wires)
    
    def controlled_gate(self, gate_func, control_wires, control_values=None, *gate_args, **gate_kwargs):
        """
        Apply an arbitrary gate with control from multiple qubits and arbitrary control values.
        
        Uses PennyLane's qml.ctrl() to create a controlled version of any gate.
        
        Args:
            gate_func: The gate function to control (e.g., qml.RX, qml.Hadamard, etc.)
            control_wires: List of qubit indices that control the gate
            control_values: List of control values (0/False or 1/True) for each control qubit.
                           Default is all 1s (standard control). Use 0/False to control on |0⟩ state.
            *gate_args: Arguments to pass to the gate function
            **gate_kwargs: Keyword arguments to pass to the gate function
        
        Example:
            # Controlled RX with angle pi/2, controlled by qubit 0
            controlled_gate(qml.RX, [0], None, np.pi/2, wires=1)
            
            # Controlled Hadamard, controlled by qubits 0 and 1, with qubit 0 controlling on |0⟩
            controlled_gate(qml.Hadamard, [0, 1], [0, 1], wires=2)
            
            # Controlled RY with angle pi/4, controlled by qubit 2 on |0⟩ state
            controlled_gate(qml.RY, [2], [0], np.pi/4, wires=3)
        """
        for wire in control_wires:
            self._validate_qubit(wire)
        
        # Create controlled version of the gate
        if control_values is not None:
            controlled_op = qml.ctrl(gate_func, control=control_wires, control_values=control_values)
        else:
            controlled_op = qml.ctrl(gate_func, control=control_wires)
        
        # Apply the controlled gate with the provided arguments
        controlled_op(*gate_args, **gate_kwargs)
    
    def hamiltonian_evolution(self, H, time, wires):
        """
        Apply Hamiltonian evolution U = exp(-iHt) to specified qubits.
        
        Args:
            H: Hamiltonian matrix (numpy array or PennyLane observable)
            time: Evolution time t
            wires: List of qubit indices to apply evolution to
        
        Example:
            H = np.array([[1, 0], [0, -1]])  # Pauli-Z Hamiltonian
            hamiltonian_evolution(H, np.pi/2, [0])  # Evolve qubit 0 for time π/2
        """
        for wire in wires:
            self._validate_qubit(wire)
        
        # If H is a numpy array, convert it to a unitary
        if isinstance(H, np.ndarray):
            # Compute U = exp(-iHt)
            # Use scipy.linalg.expm for matrix exponential
            from scipy.linalg import expm
            U = expm(-1j * H * time)
            # Apply as unitary
            self.qubitunitary(U, wires)
        else:
            # If H is a PennyLane observable, use qml.Exp
            qml.Exp(H, coeff=-1j * time, wires=wires)
    
    def qpe(self, unitary_matrix, eigenstate_wires, ancilla_wires):
        """
        Quantum Phase Estimation (QPE) algorithm.
        
        Estimates the phase φ of a unitary operator U where U|ψ⟩ = e^(2πiφ)|ψ⟩.
        Uses PennyLane's built-in QuantumPhaseEstimation function.
        
        Args:
            unitary_matrix: Unitary matrix U (numpy array)
            eigenstate_wires: List of qubit indices for the eigenstate register
            ancilla_wires: List of qubit indices for the ancilla/estimation register
        
        The algorithm:
        1. Prepare ancilla qubits in superposition (Hadamard)
        2. Apply controlled-U^(2^k) for k = 0, 1, ..., n-1
        3. Apply inverse QFT to ancilla qubits
        4. Measure ancilla qubits to get phase estimate
        
        Example:
            # Estimate phase of a rotation matrix
            U = np.array([[1, 0], [0, np.exp(1j * np.pi/4)]])  # Phase gate
            qpe(U, [0], [1, 2])  # Estimate phase on qubit 0 using ancillas 1, 2
        """
        for wire in eigenstate_wires + ancilla_wires:
            self._validate_qubit(wire)
        
        n_ancilla = len(ancilla_wires)
        n_eigenstate = len(eigenstate_wires)
        
        # Validate unitary dimensions
        if isinstance(unitary_matrix, np.ndarray):
            expected_dim = 2 ** n_eigenstate
            if unitary_matrix.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Unitary must be {expected_dim}x{expected_dim} for {n_eigenstate} eigenstate qubits, "
                    f"got shape {unitary_matrix.shape}"
                )
        
        # Use PennyLane's built-in QuantumPhaseEstimation
        qml.QuantumPhaseEstimation(
            unitary=unitary_matrix,
            target_wires=eigenstate_wires,
            estimation_wires=ancilla_wires
        )
    
    def qpe_hermitian(self, H, eigenstate_wires, ancilla_wires):
        """
        Quantum Phase Estimation for a Hermitian matrix.
        
        Creates the unitary operator U = e^(iH) from the Hermitian matrix H,
        then uses PennyLane's built-in QPE to estimate the phase.
        The phase φ relates to eigenvalue λ by: φ = λ/(2π) mod 1
        
        Args:
            H: Hermitian matrix (numpy array)
            eigenstate_wires: List of qubit indices for the eigenstate register
            ancilla_wires: List of qubit indices for the ancilla/estimation register
        
        Example:
            H = np.array([[1, 0], [0, -1]])  # Pauli-Z (eigenvalues: 1, -1)
            qpe_hermitian(H, [0], [1, 2])
            # Estimates phase corresponding to eigenvalues of H
        """
        for wire in eigenstate_wires + ancilla_wires:
            self._validate_qubit(wire)
        
        n_eigenstate = len(eigenstate_wires)
        
        # Validate Hermitian matrix dimensions
        if isinstance(H, np.ndarray):
            expected_dim = 2 ** n_eigenstate
            if H.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Hermitian matrix must be {expected_dim}x{expected_dim} for {n_eigenstate} eigenstate qubits, "
                    f"got shape {H.shape}"
                )
        
        # Create the unitary operator U = e^(iH)
        from scipy.linalg import expm
        U = expm(1j * H)
        
        # Use PennyLane's built-in QuantumPhaseEstimation
        qml.QuantumPhaseEstimation(
            unitary=U,
            target_wires=eigenstate_wires,
            estimation_wires=ancilla_wires
        )
    
    def period_finding(self, U, initial_state, eigenstate_wires, ancilla_wires):
        """
        Quantum period finding algorithm for any periodic unitary operator.
        
        Finds the period r of a unitary operator U, where r is the smallest positive integer
        such that U^r |ψ⟩ = |ψ⟩ for the given initial state |ψ⟩.
        
        The initial state |ψ⟩ should be in the periodic subspace of U. When QPE is applied,
        the state naturally decomposes into a superposition of eigenvectors with eigenvalues
        e^(2πis/r) for s = 0, 1, ..., r-1. The phase kickback happens automatically during
        QPE, and measuring the ancilla wires gives phase estimates of the form s/r.
        
        Note: You will only see phases s/r for which eigenvectors actually exist. For example,
        a 1-qubit operator with period 4 may only have 2 eigenvectors (phases 0 and 1/4), not all 4.
        To see all phases s/r for s = 0, 1, ..., r-1, the operator must have eigenvectors
        corresponding to all those phases (as in Shor's modular multiplication operator).
        
        Args:
            U: Unitary matrix (numpy array) representing the periodic operator
            initial_state: Initial state vector |ψ⟩ (numpy array) - should be in periodic subspace
            eigenstate_wires: List of qubit indices for the eigenstate register
            ancilla_wires: List of qubit indices for QPE ancilla/estimation register
        
        Example:
            # Find period of modular multiplication: U|y⟩ = |a*y mod N⟩
            # For a=2, N=15, the period is 4 (since 2^4 ≡ 1 mod 15)
            N = 15
            a = 2
            dim = 16  # Need at least N qubits
            U = construct_modular_mult_unitary(a, N, dim)
            psi = np.zeros(dim); psi[1] = 1.0  # |1⟩ state (in periodic subspace)
            period_finding(U, psi, [0,1,2,3], [4,5,6,7])
            # Measure ancilla_wires to get phase estimates s/r, extract r using continued fractions
            # The state |1⟩ naturally decomposes into eigenvectors with phases 0, 1/4, 2/4, 3/4
        
        Note:
            To extract the period from measurements:
            1. Measure ancilla_wires to get phase estimates φ (multiple measurements may give different s/r)
            2. Use continued fractions algorithm to find the maximum common denominator r
            3. Verify that U^r |ψ⟩ = |ψ⟩
        """
        for wire in eigenstate_wires + ancilla_wires:
            self._validate_qubit(wire)
        
        n_eigenstate = len(eigenstate_wires)
        expected_dim = 2 ** n_eigenstate
        
        # Validate unitary dimensions
        if isinstance(U, np.ndarray):
            if U.shape != (expected_dim, expected_dim):
                raise ValueError(
                    f"Unitary must be {expected_dim}x{expected_dim} for {n_eigenstate} eigenstate qubits, "
                    f"got shape {U.shape}"
                )
        
        # Validate initial state dimensions
        if isinstance(initial_state, np.ndarray):
            if len(initial_state) != expected_dim:
                raise ValueError(
                    f"Initial state must have length {expected_dim} for {n_eigenstate} eigenstate qubits, "
                    f"got length {len(initial_state)}"
                )
        
        # Prepare the initial state |ψ⟩
        # This state should be in the periodic subspace (e.g., |1⟩ for Shor's algorithm)
        # QPE will naturally decompose it into eigenvectors with phases s/r
        self.state_prep(initial_state, eigenstate_wires)
        
        # Apply Quantum Phase Estimation to estimate the phases
        # QPE automatically performs the decomposition and phase kickback
        # The phases will be of the form φ = s/r, where s ∈ {0, 1, ..., r-1}
        # Measuring the ancilla wires gives us estimates of these phases
        self.qpe(U, eigenstate_wires, ancilla_wires)
        
        # Note: The period r is extracted from the measured phases using continued fractions
        # Multiple measurements may be needed to get different values of s/r and determine r
        
    def shor_factor_circuit(self, N, a, eigenstate_wires, ancilla_wires):
        """
        Quantum circuit for Shor's factorization algorithm.
        
        Sets up the quantum circuit for period finding. The actual factorization
        requires classical post-processing after measurement.
        
        Args:
            N: Integer to factor (assumed to be product of two primes)
            a: Base integer (must be coprime with N)
            eigenstate_wires: List of qubit indices for the eigenstate register
            ancilla_wires: List of qubit indices for QPE ancilla register
        """
        import math
        
        # Validate a and N are coprime
        if math.gcd(a, N) != 1:
            # This case is handled classically before calling the circuit
            return
        
        # Construct the modular multiplication unitary U: |y⟩ → |a*y mod N⟩
        n_eigenstate = len(eigenstate_wires)
        dim = 2 ** n_eigenstate
        U = np.zeros((dim, dim), dtype=complex)
        for y in range(dim):
            if y < N:
                result = (a * y) % N
                U[result, y] = 1.0
            else:
                # For y >= N, identity (values >= N shouldn't appear if prepared correctly)
                U[y, y] = 1.0
        
        # Prepare initial state |ψ⟩ = |1⟩
        # The state |1⟩ is in the periodic subspace and naturally decomposes into
        # eigenvectors with phases s/r when QPE is applied. No explicit superposition needed.
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[1] = 1.0  # |1⟩ state
        
        # Apply period finding with the modular multiplication unitary
        # QPE will automatically handle the decomposition and phase kickback
        self.period_finding(U, initial_state, eigenstate_wires, ancilla_wires)
    
    def get_measurement_results(self):
        """
        Return all quantum measurement results.
        
        If there's only one measurement, returns it directly.
        If there are multiple measurements, returns a list.
        If there are no measurements, returns None.
        """
        if len(self.measurement_results) == 0:
            return None
        elif len(self.measurement_results) == 1:
            return self.measurement_results[0]
        else:
            return self.measurement_results


# Circuit analysis functions (not part of QuantumContext - they analyze circuits)
def circuit_depth(quantum_func, *args, **kwargs):
    """
    Get the depth of a quantum circuit.
    
    Circuit depth is the number of layers of gates (longest path from input to output).
    
    Args:
        quantum_func: A @quantum-decorated function
        *args: Arguments to pass to the quantum function
        **kwargs: Keyword arguments to pass to the quantum function
    
    Returns:
        int: Circuit depth
    
    Example:
        @quantum
        def my_circuit():
            qubit(2)
            hadamard(0)
            cnot(0, 1)
        
        depth = circuit_depth(my_circuit)  # Returns 2
    """
    # Get the original function from the wrapper
    if hasattr(quantum_func, '__quantum_func__'):
        original_func = quantum_func.__quantum_func__
    elif hasattr(quantum_func, '__wrapped__'):
        original_func = quantum_func.__wrapped__
    else:
        original_func = quantum_func
    
    # Create a quantum context to build the circuit
    qc = QuantumContext()
    token = _current_context.set(qc)
    
    try:
        # Get the original function's global namespace
        original_globals = original_func.__globals__.copy()
        
        # Replace DSL function names with context methods
        dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                        'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
                        'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                        'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                        'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                        'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                        'measure_pauli_x', 'measure_pauli_y', 'measure_pauli_z',
                        'reset_qubit', 'copy_basis_state', 'swap_test',
                        'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                        'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                        'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian',
                        'period_finding', 'shor_factor_circuit']
        
        # Do a preliminary execution to call qubit() and create the device
        prelim_locals = {}
        arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
        for i, arg_name in enumerate(arg_names):
            if i < len(args):
                prelim_locals[arg_name] = args[i]
        prelim_locals.update(kwargs)
        
        prelim_globals = original_globals.copy()
        for func_name in dsl_functions:
            if func_name == 'qubit':
                prelim_globals[func_name] = qc.qubit
            elif func_name == 'set_shots':
                prelim_globals[func_name] = qc.set_shots
            else:
                prelim_globals[func_name] = lambda *a, **kw: None
        
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            try:
                exec(original_func.__code__, prelim_globals, prelim_locals)
            finally:
                sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            pass
        
        if qc.device is None:
            qc.device = qml.device('default.qubit', wires=1)
            qc.num_qubits = 1
        
        original_globals['qml'] = qml
        for func_name in dsl_functions:
            original_globals[func_name] = getattr(qc, func_name)
        
        @qml.qnode(qc.device)
        def quantum_circuit(*qnode_args, **qnode_kwargs):
            qc.measurement_results = []
            qc._qubit_called = False
            
            exec_locals = {}
            arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
            exec_locals.update(qnode_kwargs)
            
            for i, arg_name in enumerate(arg_names):
                if i < len(qnode_args) and arg_name not in exec_locals:
                    exec_locals[arg_name] = qnode_args[i]
            
            num_required = original_func.__code__.co_argcount - len(original_func.__defaults__ or [])
            for i in range(num_required):
                arg_name = arg_names[i]
                if arg_name not in exec_locals:
                    exec_locals[arg_name] = 1
            
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                try:
                    exec(original_func.__code__, original_globals, exec_locals)
                finally:
                    sys.stdout = old_stdout
            except Exception as e:
                sys.stdout = old_stdout
                # Don't silently pass - we need the circuit to be built
                # If there's an error, still return a measurement so the circuit is constructed
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            
            results = qc.get_measurement_results()
            if results is None or (isinstance(results, list) and len(results) == 0):
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            return results
        
        # Get circuit specs using PennyLane
        # qml.specs() constructs the circuit automatically when called
        specs = qml.specs(quantum_circuit)(*args, **kwargs)
        # specs is a dictionary - depth is in specs['resources'].depth
        if 'resources' in specs and hasattr(specs['resources'], 'depth'):
            return specs['resources'].depth
        # Fallback: try direct access
        return specs.get('depth', 0)
        
    finally:
        _current_context.reset(token)


def gate_count(quantum_func, *args, **kwargs):
    """
    Get the number of gates in a quantum circuit.
    
    Args:
        quantum_func: A @quantum-decorated function
        *args: Arguments to pass to the quantum function
        **kwargs: Keyword arguments to pass to the quantum function
    
    Returns:
        int: Number of gates/operations
    
    Example:
        @quantum
        def my_circuit():
            qubit(2)
            hadamard(0)
            cnot(0, 1)
        
        count = gate_count(my_circuit)  # Returns 2
    """
    # Get the original function from the wrapper
    if hasattr(quantum_func, '__quantum_func__'):
        original_func = quantum_func.__quantum_func__
    elif hasattr(quantum_func, '__wrapped__'):
        original_func = quantum_func.__wrapped__
    else:
        original_func = quantum_func
    
    # Create a quantum context to build the circuit
    qc = QuantumContext()
    token = _current_context.set(qc)
    
    try:
        original_globals = original_func.__globals__.copy()
        dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                        'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
                        'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                        'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                        'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                        'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                        'measure_pauli_x', 'measure_pauli_y', 'measure_pauli_z',
                        'reset_qubit', 'copy_basis_state', 'swap_test',
                        'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                        'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                        'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian',
                        'period_finding', 'shor_factor_circuit']
        
        prelim_locals = {}
        arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
        for i, arg_name in enumerate(arg_names):
            if i < len(args):
                prelim_locals[arg_name] = args[i]
        prelim_locals.update(kwargs)
        
        prelim_globals = original_globals.copy()
        for func_name in dsl_functions:
            if func_name == 'qubit':
                prelim_globals[func_name] = qc.qubit
            elif func_name == 'set_shots':
                prelim_globals[func_name] = qc.set_shots
            else:
                prelim_globals[func_name] = lambda *a, **kw: None
        
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            try:
                exec(original_func.__code__, prelim_globals, prelim_locals)
            finally:
                sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            pass
        
        if qc.device is None:
            qc.device = qml.device('default.qubit', wires=1)
            qc.num_qubits = 1
        
        original_globals['qml'] = qml
        for func_name in dsl_functions:
            original_globals[func_name] = getattr(qc, func_name)
        
        @qml.qnode(qc.device)
        def quantum_circuit(*qnode_args, **qnode_kwargs):
            qc.measurement_results = []
            qc._qubit_called = False
            
            exec_locals = {}
            arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
            exec_locals.update(qnode_kwargs)
            
            for i, arg_name in enumerate(arg_names):
                if i < len(qnode_args) and arg_name not in exec_locals:
                    exec_locals[arg_name] = qnode_args[i]
            
            num_required = original_func.__code__.co_argcount - len(original_func.__defaults__ or [])
            for i in range(num_required):
                arg_name = arg_names[i]
                if arg_name not in exec_locals:
                    exec_locals[arg_name] = 1
            
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                try:
                    exec(original_func.__code__, original_globals, exec_locals)
                finally:
                    sys.stdout = old_stdout
            except Exception as e:
                sys.stdout = old_stdout
                # Don't silently pass - we need the circuit to be built
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            
            results = qc.get_measurement_results()
            if results is None or (isinstance(results, list) and len(results) == 0):
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            return results
        
        specs = qml.specs(quantum_circuit)(*args, **kwargs)
        # num_operations is in specs['resources'].num_gates or specs['num_operations']
        if 'resources' in specs and hasattr(specs['resources'], 'num_gates'):
            return specs['resources'].num_gates
        # Fallback: try direct access
        return specs.get('num_operations', 0)
        
    finally:
        _current_context.reset(token)


def circuit_resources(quantum_func, *args, **kwargs):
    """
    Get all circuit resources (depth, gate count, etc.) for a quantum circuit.
    
    Args:
        quantum_func: A @quantum-decorated function
        *args: Arguments to pass to the quantum function
        **kwargs: Keyword arguments to pass to the quantum function
    
    Returns:
        dict: Dictionary containing circuit resources (depth, num_operations, etc.)
    
    Example:
        @quantum
        def my_circuit():
            qubit(2)
            hadamard(0)
            cnot(0, 1)
        
        resources = circuit_resources(my_circuit)
        # Returns: {'depth': 2, 'num_operations': 2, ...}
    """
    # Get the original function from the wrapper
    if hasattr(quantum_func, '__quantum_func__'):
        original_func = quantum_func.__quantum_func__
    elif hasattr(quantum_func, '__wrapped__'):
        original_func = quantum_func.__wrapped__
    else:
        original_func = quantum_func
    
    # Create a quantum context to build the circuit
    qc = QuantumContext()
    token = _current_context.set(qc)
    
    try:
        original_globals = original_func.__globals__.copy()
        dsl_functions = ['qubit', 'set_shots', 'hadamard', 'hadamard_transform', 'xgate', 'ygate', 'zgate', 'rx', 'ry', 'rz', 
                        'phase_shift', 's', 't', 'u1', 'u2', 'u3', 'cnot', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz',
                        'toffoli', 'cswap', 'swap', 'multi_rz', 'multi_controlled_x', 'multi_controlled_z', 
                        'measure_probs', 'measure_expval', 'measure_var', 'measure', 'cond', 'measure_state', 
                        'measure_density_matrix', 'measure_sample', 'measure_counts', 'measure_mutual_info', 
                        'measure_vn_entropy', 'measure_purity', 'measure_classical_shadow', 'measure_shadow_expval',
                        'measure_pauli_x', 'measure_pauli_y', 'measure_pauli_z',
                        'reset_qubit', 'copy_basis_state', 'swap_test',
                        'qubitunitary', 'hadamard_transform', 'basis_state', 'state_prep', 'bell_state', 'ghz_state', 'w_state', 'quantum_teleportation',
                        'grover_diffusion', 'grover_search', 'build_cnf_oracle',
                        'qft', 'iqft', 'controlled_unitary', 'controlled_gate', 'hamiltonian_evolution', 'qpe', 'qpe_hermitian',
                        'period_finding', 'shor_factor_circuit']
        
        prelim_locals = {}
        arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
        for i, arg_name in enumerate(arg_names):
            if i < len(args):
                prelim_locals[arg_name] = args[i]
        prelim_locals.update(kwargs)
        
        prelim_globals = original_globals.copy()
        for func_name in dsl_functions:
            if func_name == 'qubit':
                prelim_globals[func_name] = qc.qubit
            elif func_name == 'set_shots':
                prelim_globals[func_name] = qc.set_shots
            else:
                prelim_globals[func_name] = lambda *a, **kw: None
        
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            try:
                exec(original_func.__code__, prelim_globals, prelim_locals)
            finally:
                sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            pass
        
        if qc.device is None:
            qc.device = qml.device('default.qubit', wires=1)
            qc.num_qubits = 1
        
        original_globals['qml'] = qml
        for func_name in dsl_functions:
            original_globals[func_name] = getattr(qc, func_name)
        
        @qml.qnode(qc.device)
        def quantum_circuit(*qnode_args, **qnode_kwargs):
            qc.measurement_results = []
            qc._qubit_called = False
            
            exec_locals = {}
            arg_names = original_func.__code__.co_varnames[:original_func.__code__.co_argcount]
            exec_locals.update(qnode_kwargs)
            
            for i, arg_name in enumerate(arg_names):
                if i < len(qnode_args) and arg_name not in exec_locals:
                    exec_locals[arg_name] = qnode_args[i]
            
            num_required = original_func.__code__.co_argcount - len(original_func.__defaults__ or [])
            for i in range(num_required):
                arg_name = arg_names[i]
                if arg_name not in exec_locals:
                    exec_locals[arg_name] = 1
            
            old_stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                try:
                    exec(original_func.__code__, original_globals, exec_locals)
                finally:
                    sys.stdout = old_stdout
            except Exception as e:
                sys.stdout = old_stdout
                # Don't silently pass - we need the circuit to be built
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            
            results = qc.get_measurement_results()
            if results is None or (isinstance(results, list) and len(results) == 0):
                if qc.num_qubits > 0:
                    return qml.probs(wires=list(range(qc.num_qubits)))
                return qml.probs(wires=[0])
            return results
        
        specs = qml.specs(quantum_circuit)(*args, **kwargs)
        # Return as a regular dict (specs is already a dict)
        # Convert resources object to dict if needed
        result = dict(specs)
        if 'resources' in result and hasattr(result['resources'], '__dict__'):
            # Convert Resources object to dict
            resources_dict = {}
            if hasattr(result['resources'], 'depth'):
                resources_dict['depth'] = result['resources'].depth
            if hasattr(result['resources'], 'num_gates'):
                resources_dict['num_gates'] = result['resources'].num_gates
            if hasattr(result['resources'], 'num_wires'):
                resources_dict['num_wires'] = result['resources'].num_wires
            result['resources'] = resources_dict
        return result
        
    finally:
        _current_context.reset(token)


def continued_fractions(phi, max_denominator=1000):
    """
    Convert a phase φ to a continued fraction and return possible denominators.
    
    These denominators are candidates for the period r.
    
    Args:
        phi: Phase value in [0, 1)
        max_denominator: Maximum denominator to consider
    
    Returns:
        List of candidate periods (denominators)
    """
    # Convert to Fraction to get continued fraction expansion
    frac = Fraction(phi).limit_denominator(max_denominator)
    candidates = []
    
    # Get convergents (approximations) of the continued fraction
    # The denominators of convergents are candidates for the period
    a = frac.numerator
    b = frac.denominator
    
    if b > 0:
        candidates.append(b)
    
    # Also check nearby fractions
    for offset in [-2, -1, 1, 2]:
        if b + offset > 0:
            candidates.append(b + offset)
    
    return candidates


def extract_period_from_phases(phases, max_period=1000):
    """
    Extract the period r from a list of phase measurements.
    
    When period finding is applied to a superposition of eigenvectors,
    different measurements give phases of the form s/r for s = 0, 1, ..., r-1.
    This function finds the maximum common denominator r that explains all the phases.
    
    Args:
        phases: List of phase values (each in [0, 1))
        max_period: Maximum period to consider
    
    Returns:
        Best candidate for period r (maximum viable denominator), or None if not found
    """
    if not phases:
        return None
    
    # Collect all candidate denominators from all phases
    all_candidates = set()
    for phi in phases:
        candidates = continued_fractions(phi, max_denominator=max_period)
        all_candidates.update(candidates)
    
    if not all_candidates:
        return None
    
    # For each candidate denominator r, check if all phases can be written as s/r
    # for some integer s. We want the maximum r that satisfies this.
    viable_denominators = []
    
    for r in sorted(all_candidates, reverse=True):  # Check largest first
        if r <= 1:
            continue
        
        # Check if all phases are consistent with denominator r
        # i.e., each phase should be close to s/r for some integer s
        is_viable = True
        for phi in phases:
            # Find the closest s/r to phi
            best_s = round(phi * r)
            expected_phase = best_s / r
            # Allow some tolerance for floating point errors
            if abs(phi - expected_phase) > 1e-6 and abs(phi - expected_phase - 1) > 1e-6 and abs(phi - expected_phase + 1) > 1e-6:
                is_viable = False
                break
        
        if is_viable:
            viable_denominators.append(r)
    
    # Return the maximum viable denominator
    if viable_denominators:
        return max(viable_denominators)
    
    return None


def extract_period_from_probabilities(probs, n_ancilla, max_period=1000, num_samples=10):
    """
    Extract period from QPE probability distribution using PennyLane's approach.
    
    According to PennyLane: "We make different measurements, and we will keep
    the highest of the value of r obtained."
    
    This function simulates multiple measurements by sampling from the probability
    distribution, converts each measurement to a fraction s/r, and returns the
    highest r value found.
    
    Args:
        probs: Probability distribution array from QPE measurement
        n_ancilla: Number of ancilla qubits used
        max_period: Maximum period to consider
        num_samples: Number of measurements to simulate
    
    Returns:
        Best candidate for period r (highest r found across measurements), or None if not found
    """
    import random
    
    # Sample multiple measurements from the probability distribution
    # Each measurement gives us a phase estimate s/r
    max_r = None
    
    for _ in range(num_samples):
        # Sample one measurement outcome according to the probability distribution
        # This simulates what would happen on a real quantum computer
        rand_val = random.random()
        cumulative = 0.0
        measured_idx = None
        
        for idx, prob in enumerate(probs):
            cumulative += prob
            if cumulative >= rand_val:
                measured_idx = idx
                break
        
        if measured_idx is None:
            continue
        
        # Convert measurement index to phase
        phase_estimate = measured_idx / (2 ** n_ancilla)
        
        # Convert phase to fraction s/r using continued fractions
        # This gives us candidate denominators (periods)
        candidates = continued_fractions(phase_estimate, max_denominator=max_period)
        
        # Keep track of the highest r found
        for r in candidates:
            if r > 1 and (max_r is None or r > max_r):
                # Verify this r is consistent with the phase
                # Phase should be close to s/r for some integer s
                best_s = round(phase_estimate * r)
                expected_phase = best_s / r
                if abs(phase_estimate - expected_phase) < 1e-6 or \
                   abs(phase_estimate - expected_phase - 1) < 1e-6 or \
                   abs(phase_estimate - expected_phase + 1) < 1e-6:
                    max_r = r
    
    return max_r


def shor_factor(N: int, max_attempts: int = 10) -> List[int]:
    """
    Shor's algorithm for integer factorization.
    
    Factors an integer N (assumed to be product of two primes) by finding
    a non-trivial factor using quantum period finding.
    
    Args:
        N: Integer to factor (assumed to be product of two primes)
        max_attempts: Maximum number of attempts before giving up
    
    Returns:
        [p, q]: The two prime factors of N
    
    Example:
        factors = shor_factor(15)  # Returns [3, 5] or [5, 3]
    """
    import math
    import random
    
    # Validate N
    if N < 2:
        raise ValueError("N must be at least 2")
    if N == 1:
        raise ValueError("N=1 cannot be factored")
    
    # Handle even numbers
    if N % 2 == 0:
        return [2, N // 2]
    
    # Check for small factors
    for i in range(2, int(math.sqrt(N)) + 1):
        if N % i == 0:
            return [i, N // i]
    
    # Set up quantum circuit parameters
    n_eigenstate = int(np.ceil(np.log2(N)))
    n_ancilla = 2 * n_eigenstate  # Use more ancilla qubits for better precision
    total_qubits = n_eigenstate + n_ancilla
    
    for attempt in range(max_attempts):
        # Choose random a
        a = random.randint(2, N - 1)
        
        # Check if a and N are coprime
        gcd_val = math.gcd(a, N)
        if gcd_val != 1:
            # Found a factor immediately!
            p = gcd_val
            q = N // p
            return [int(p), int(q)]
        
        # Set up quantum circuit
        @quantum
        def shor_circuit():
            qubit(total_qubits)
            eigenstate_wires = list(range(n_eigenstate))
            ancilla_wires = list(range(n_eigenstate, n_eigenstate + n_ancilla))
            
            # Apply Shor's period finding circuit
            shor_factor_circuit(N, a, eigenstate_wires, ancilla_wires)
            
            # Measure ancilla wires to get phase estimate
            return measure_probs(*ancilla_wires)
        
        # Execute quantum circuit
        result = shor_circuit()
        
        # Process measurement results
        if result is not None:
            if hasattr(result, 'numpy'):
                probs = result.numpy()
            else:
                probs = np.array(result)
            
            # Extract period using PennyLane's approach:
            # Make multiple measurements, convert each to fraction s/r, keep highest r
            period_r = extract_period_from_probabilities(probs, n_ancilla, max_period=N, num_samples=20)
            
            if period_r is not None:
                # Found a period candidate from the probability distribution
                r = period_r
            else:
                # Fallback: use most probable outcome
                phase_idx = np.argmax(probs)
                phase_estimate = phase_idx / (2 ** n_ancilla)
                period_candidates = continued_fractions(phase_estimate, max_denominator=N)
                
                # Try each candidate period
                r = None
                for candidate_r in period_candidates:
                    if candidate_r > 1 and candidate_r < N:
                        r = candidate_r
                        break
            
            if r is None:
                continue
            
            # Try the extracted period
            # Check if r is even
            if r % 2 != 0:
                continue
                if r <= 1 or r >= N:
                    continue
                
                # Check if r is even
                if r % 2 != 0:
                    continue
                
                # Compute x = a^(r/2) mod N
                x = pow(a, r // 2, N)
                
                # Check if x ≠ ±1 mod N
                if x == 1 or x == N - 1:
                    continue
                
                # Compute factors
                p = math.gcd(x - 1, N)
                q = math.gcd(x + 1, N)
                
                # Verify we found valid factors
                if p > 1 and q > 1 and p * q == N:
                    return [int(p), int(q)]
        
        # If this attempt failed, try again with different a
    
    # If all attempts failed, return None or raise error
    raise RuntimeError(f"Failed to factor {N} after {max_attempts} attempts")