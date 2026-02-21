# File: src/tokens.py
# -----------------------------------------------------------------------------
# This module defines the fundamental token structures used by the lexer and parser.
# If you're new to language tooling:
# - A "token" is a classified chunk of text (like an identifier, a number, or a keyword).
# - The lexer (tokenizer) converts raw source text into a list of tokens.
# - The parser then consumes those tokens to build an AST (abstract syntax tree).
#
# This file is intentionally simple and stable. Other parts of the system
# (lexer, parser, interpreter) import these definitions to agree on how to
# represent source code as tokens.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Structural tokens that help the parser break the input into statements.
    # NEWLINE separates statements in this DSL.
    NEWLINE = auto()
    # EOF is a sentinel token emitted at the end of the input stream.
    EOF = auto()

    # Identifiers (user-defined names like q0, my_qubit, etc.).
    IDENT = auto()

    # Keywords (reserved words that have special meaning in the DSL).
    # In this v0, we support a minimal set of quantum operations:
    # - qubit: declare/allocate one or more qubits
    # - hadamard: apply H gate to a qubit
    # - hadamard_transform: apply H gate to a list of qubits
    # - xgate: apply X/Pauli-X gate to a qubit
    # - cnot: apply a controlled-NOT gate to two qubits (control, target)
    # - swap: swap the state of two qubits
    # - measure_probs: compute or estimate probabilities for specified qubits
    # - kw_python: execute arbitrary Python code
    KW_QUBIT = auto()
    KW_HADAMARD = auto()
    KW_HADAMARD_TRANSFORM = auto()
    KW_XGATE = auto()
    KW_CNOT = auto()
    KW_SWAP = auto()
    KW_MEASURE_PROBS = auto()
    KW_PYTHON = auto()

    # Numeric literals.
    # We keep INT and FLOAT separate so the parser can validate argument types.
    INT = auto()
    FLOAT = auto()

    # Punctuation used for function-like calls and lists of arguments.
    LPAREN = auto()   # (
    RPAREN = auto()   # )
    COMMA = auto()    # ,

    # Python code
    PYTHON_CODE = auto()

    # Error token used by the lexer when it encounters an unexpected character.
    # This lets us continue scanning and report better diagnostics.
    LEX_ERROR = auto()



@dataclass(frozen=True)
class Span:
    # Span tracks the exact source location of a token to produce helpful errors.
    # All positions are 1-based (first line is 1, first column is 1), which is
    # the most human-friendly convention for diagnostics.
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def merge(self, other: "Span") -> "Span":
        """
        Create a span that covers from the start of this span to the end of another one.
        Useful for error messages that should underline a range of tokens.

        Because the dataclass is frozen (immutable), we return a new instance instead of mutating.
        """
        # Determine the earliest start position
        if (other.start_line < self.start_line) or (
            other.start_line == self.start_line and other.start_col < self.start_col
        ):
            start_line, start_col = other.start_line, other.start_col
        else:
            start_line, start_col = self.start_line, self.start_col

        # Determine the latest end position
        if (other.end_line > self.end_line) or (
            other.end_line == self.end_line and other.end_col > self.end_col
        ):
            end_line, end_col = other.end_line, other.end_col
        else:
            end_line, end_col = self.end_line, self.end_col

        return Span(start_line, start_col, end_line, end_col)


@dataclass(frozen=True)
class Token:
    # A token has:
    # - type: what kind of thing it is (identifier, keyword, etc.).
    # - lexeme: the exact text from the source (e.g., "qubit", "q0", "(", "123").
    # - span: where it appeared (for errors and tooling).
    type: TokenType
    lexeme: str
    span: Span

    def __repr__(self) -> str:
        # Helpful for debugging and for tests: shows type name, lexeme, and location.
        return (
            f"Token(type={self.type.name}, lexeme={self.lexeme!r}, "
            f"span=({self.span.start_line}:{self.span.start_col}-"
            f"{self.span.end_line}:{self.span.end_col}))"
        )


# Public API of this module (useful if someone does `from src.tokens import *`)
__all__ = ["TokenType", "Span", "Token"]