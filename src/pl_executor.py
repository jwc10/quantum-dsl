# src/pl_executor.py

from typing import Dict, List, Tuple, Optional
import itertools
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from .tokens import Span
from .quantum_ast import Program, QubitDecl, Hadamard, HadamardTransform, XGate, CNot, Swap, MeasureProbs, PythonCode


class ExecError(Exception):
    def __init__(self, message: str, span: Optional[Span] = None):
        super().__init__(message)
        self.message = message
        self.span = span

    def __str__(self):
        if self.span is None:
            return self.message
        return f"{self.message} (at {self.span.start_line}:{self.span.start_col})"


class PennyLaneExecutor:
    def __init__(
        self,
        device_name: str = "default.qubit",
        device_kwargs: Optional[dict] = None,
    ):
        """
        device_name: PennyLane device string, e.g. "default.qubit", "braket.aws.qubit", "qiskit.ibmq", etc.
        device_kwargs: passed to qml.device, e.g. {"shots": 10_000, "wires": ...} (wires is managed internally)
        """
        self.device_name = device_name
        self.device_kwargs = dict(device_kwargs or {})
        self.dev: Optional[qml.Device] = None
        self.wire_order: List[str] = []
        self.measurements: List[Tuple[List[str], Dict[str, float]]] = []


    def _validate_and_assign_wires(self, program: Program) -> List[str]:
        """
        - Expands QubitDecl with counts into concrete names (qN, qN+1, ...)
        - Validates redeclarations and use-before-declare
        - Collects a stable wire ordering (declaration order)
        Returns the full ordered wire list.
        """
        declared: List[str] = []
        declared_set = set()

        def declare_names(names: List[str], span: Optional[Span]):
            for nm in names:
                if nm in declared_set:
                    raise ExecError(f"Qubit '{nm}' redeclared", span)
                declared.append(nm)
                declared_set.add(nm)

        for stmt in program.stmts:
            if isinstance(stmt, QubitDecl):
                count = getattr(stmt, "count", None)
                names = list(getattr(stmt, "names", []) or [])
                if count is not None and names:
                    if count != len(names):
                        raise ExecError(
                            f"Qubit declaration count ({count}) does not match provided names ({len(names)})",
                            stmt.span,
                        )
                    declare_names(names, stmt.span)
                elif count is not None:
                    # Create integer qubit names: 0, 1, 2, ...
                    start = len(declared)
                    gen = [str(start + i) for i in range(count)]
                    declare_names(gen, stmt.span)
                elif names:
                    declare_names(names, stmt.span)
                else:
                    raise ExecError("Qubit declaration missing both count and names", stmt.span)
            else:
                # Validate that any referenced qubits were declared earlier
                refs = []
                if isinstance(stmt, Hadamard):
                    refs = [stmt.target]
                elif isinstance(stmt, HadamardTransform):
                    refs = list(stmt.qubits)
                elif isinstance(stmt, XGate):
                    refs = [stmt.target]
                elif isinstance(stmt, CNot):
                    refs = [stmt.control, stmt.target]
                elif isinstance(stmt, Swap):
                    refs = [stmt.q1, stmt.q2]
                elif isinstance(stmt, MeasureProbs):
                    refs = list(stmt.qubits)
                # else unknown statements validated later

                for r in refs:
                    if r not in declared_set:
                        raise ExecError(f"Unknown qubit '{r}' (not declared yet)", getattr(stmt, "span", None))

        return declared

    def _ensure_device(self, wires: List[str]):
        if self.dev is not None:
            return
        # create a device with named wires
        kwargs = dict(self.device_kwargs)
        kwargs["wires"] = wires
        self.dev = qml.device(self.device_name, **kwargs)
        self.wire_order = list(wires)

    def _apply_stmt_in_qnode(self, stmt):
        # Map AST ops to PennyLane ops (ignore QubitDecl inside the circuit)
        if isinstance(stmt, Hadamard):
            qml.Hadamard(wires=stmt.target)
        elif isinstance(stmt, HadamardTransform):
            for q in stmt.qubits:
                qml.Hadamard(wires=q)
        elif isinstance(stmt, XGate):
            qml.PauliX(wires=stmt.target)
        elif isinstance(stmt, CNot):
            qml.CNOT(wires=[stmt.control, stmt.target])
        elif isinstance(stmt, Swap):
            qml.SWAP(wires=[stmt.q1, stmt.q2])
        elif isinstance(stmt, QubitDecl):
            # no-op in the circuit body; wires are allocated on the device
            return
        elif isinstance(stmt, MeasureProbs):
            # handled outside (as the return of the QNode)
            return
        elif isinstance(stmt, PythonCode):
            exec(stmt.code)
        else:
            raise ExecError(f"Unknown statement type: {type(stmt).__name__}", getattr(stmt, "span", None))

    def _build_qnode_for_prefix(self, stmts_prefix: List, measure_stmt: MeasureProbs):
        # Define the quantum function that applies ops up to the measurement point
        def circuit():
            for st in stmts_prefix:
                self._apply_stmt_in_qnode(st)

            # terminal measurement
            return qml.probs(wires=list(measure_stmt.qubits))

        # Bind to the existing device
        assert self.dev is not None, "Device must be initialized before building QNodes"

        # Optional: set PennyLaneExecutor(plot_circuit=True) to show circuit diagram when running
        if getattr(self, "plot_circuit", False):
            qml.draw_mpl(circuit)()
            plt.show()

        return qml.QNode(circuit, self.dev)

    def _probs_array_to_dict(self, probs: np.ndarray, num_bits: int) -> Dict[str, float]:
        # qml.probs ordering is binary counting over the given wires, e.g., "00, 01, 10, 11"
        keys = [format(i, f"0{num_bits}b") for i in range(2**num_bits)]
        out: Dict[str, float] = {}
        # cast to float for JSON-ability, guard tiny negatives
        for k, v in zip(keys, probs):
            val = float(v)
            if abs(val) < 1e-15:
                val = 0.0
            out[k] = val
        # Small renormalization if numeric drift
        total = sum(out.values())
        if total not in (0.0, 1.0):
            for k in out:
                out[k] = out[k] / total
        return out

    def execute(self, program: Program) -> List[Tuple[List[str], Dict[str, float]]]:
        """
        Returns a list of (measured_qubit_names_in_order, probs_dict).
        One entry per MeasureProbs statement in the program, evaluated on the prefix up to that point.
        """
        self.measurements = []

        # 1) Validate and collect wires, and expand any count-only declarations to names
        wire_list = self._validate_and_assign_wires(program)

        # 2) Create the PennyLane device for those wires
        self._ensure_device(wire_list)

        # 3) For each measurement point, run a QNode on the prefix up to that point
        stmts = program.stmts
        for i, st in enumerate(stmts):
            if isinstance(st, MeasureProbs):
                prefix = [x for x in stmts[:i] if not isinstance(x, QubitDecl)]
                qnode = self._build_qnode_for_prefix(prefix, st)
                probs_arr = qnode()
                # PennyLane may return an autograd tensor; convert to np array
                probs_arr = np.array(probs_arr, dtype=float)
                probs = self._probs_array_to_dict(probs_arr, len(st.qubits))
                self.measurements.append((list(st.qubits), probs))

        return self.measurements


# Convenience entry points if you want to wire in quickly
def run_program_with_pennylane(program: Program,
                               device_name: str = "default.qubit",
                               device_kwargs: Optional[dict] = None):
    ex = PennyLaneExecutor(device_name=device_name, device_kwargs=device_kwargs)
    return ex.execute(program)

#  Pass in source string (entire dsl program)
def run_source_with_pennylane(source: str,
                              device_name: str = "default.qubit",
                              device_kwargs: Optional[dict] = None):
    from .lexer import tokenize
    from .parser import Parser

    tokens = tokenize(source)
    parser = Parser(tokens)
    program = parser.parse_program()
    ex = PennyLaneExecutor(device_name=device_name, device_kwargs=device_kwargs)
    return ex.execute(program)