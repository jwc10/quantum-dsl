# File: src/ast.py
# -----------------------------------------------------------------------------
# Abstract Syntax Tree (AST) node definitions for the quantum DSL.
# The parser will construct instances of these nodes from the token stream,
# and the interpreter/executor will walk these nodes to perform actions.
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Optional
from src.tokens import Span


@dataclass(frozen=True)
class Node:
    # Every AST node carries a source span for diagnostics.
    span: Span


@dataclass(frozen=True)
class Stmt(Node):
    # Base class for all statements.
    pass


@dataclass(frozen=True)
class Program(Node):
    # A program is a list of statements.
    stmts: List[Stmt] = field(default_factory=list)


# qubit(...) can be used in two forms:
#   - qubit(3)                  -> allocate 3 unnamed qubits (count form)
#   - qubit(q0, q1, q2)         -> declare named qubits (names form)
# The parser should enforce that a single statement uses either count or names, not both.
@dataclass(frozen=True)
class QubitDecl(Stmt):
    count: Optional[int]  # If not None, allocate this many qubits.
    names: List[str]      # If non-empty, declare these named qubits.


@dataclass(frozen=True)
class Hadamard(Stmt):
    target: str  # qubit identifier


@dataclass(frozen=True)
class HadamardTransform(Stmt):
    qubits: List[str]


@dataclass(frozen=True)
class XGate(Stmt):
    target: str  # qubit identifier


@dataclass(frozen=True)
class CNot(Stmt):
    control: str  # control qubit identifier
    target: str   # target qubit identifier


@dataclass(frozen=True)
class Swap(Stmt):
    q1: str  # first qubit identifier
    q2: str  # second qubit identifier


# measure_probs(...) reports probabilities for specific qubits.
# If qubits is empty, the interpreter may choose a sensible default
# (e.g., all currently allocated qubits).
@dataclass(frozen=True)
class MeasureProbs(Stmt):
    qubits: List[str]


@dataclass(frozen=True)
class PythonCode(Stmt):
    code: str

__all__ = [
    "Node",
    "Stmt",
    "Program",
    "QubitDecl",
    "Hadamard",
    "HadamardTransform",
    "XGate",
    "CNot",
    "Swap",
    "MeasureProbs",
    "PythonCode",
]