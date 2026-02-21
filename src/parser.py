# File: src/parser.py
# -----------------------------------------------------------------------------
# The parser consumes a token stream (produced by the lexer) and builds an AST.
#
# Design goals:
# - Small, predictable, single-pass parser with minimal lookahead (1 token).
# - Clear, actionable error messages with precise source spans for diagnostics.
# - Grammar-oriented structure and helper methods (_at/_advance/_match/_expect).
# - Explicit whitespace/newline handling to keep statement boundaries unambiguous.
#
# Grammar (v0):
#   program            := (stmt NEWLINE+)* EOF
#
#   stmt               := qubit_decl
#                       | hadamard_call
#                       | xgate_call
#                       | cnot_call
#                       | swap_call
#                       | measure_probs_call
#
#   qubit_decl         := "qubit" "(" ( INT | ident_list ) ")"
#   hadamard_call      := "hadamard" "(" (IDENT | INT) ")"
#   xgate_call         := "xgate" "(" (IDENT | INT) ")"
#   cnot_call          := "cnot" "(" (IDENT | INT) "," (IDENT | INT) ")"
#   swap_call          := "swap" "(" (IDENT | INT) "," (IDENT | INT) ")"
#   measure_probs_call := "measure_probs" "(" [ qubit_list ] ")"
#
#   qubit_list         := (IDENT | INT) ("," (IDENT | INT))*
#
# Whitespace/newlines:
# - NEWLINE tokens delimit statements at top-level (program rule).
# - NEWLINE tokens are allowed and ignored inside parenthesized argument lists
#   to support multi-line calls (e.g., cnot(1,   <newline> 2)).
#
# Tokens and case:
# - The lexer classifies keywords with dedicated TokenType values (KW_*).
# - Keyword matching is case-insensitive in the lexer; identifiers preserve case.
# - The token stream is guaranteed to end with a single EOF token (peek-safe).
#
# Error handling:
# - On the first syntax error, a ParseError is raised with a human-readable message
#   and the offending source span. Callers can catch ParseError to report nicely.
#
# AST node spans:
# - Statement nodes receive spans that cover the entire construct (e.g., from the
#   keyword to the closing parenthesis). We merge token spans as needed.
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional

from .tokens import TokenType, Token, Span
from .quantum_ast import (
    Program,
    QubitDecl,
    Hadamard,
    HadamardTransform,
    XGate,
    CNot,
    Swap,
    MeasureProbs,
    PythonCode,
    Stmt,
)
from .lexer import tokenize


@dataclass
class ParseError(Exception):
    """
    ParseError signals a syntactic problem. It carries:
    - message: explanation of what the parser expected or what went wrong
    - span: precise source location (start/end line/column) for diagnostics
    """
    message: str
    span: Span

    def __str__(self) -> str:
        # Format: message (at start_line:start_col-end_line:end_col)
        return (
            f"{self.message} "
            f"(at {self.span.start_line}:{self.span.start_col}-"
            f"{self.span.end_line}:{self.span.end_col})"
        )


class Parser:
    """
    A simple, hand-written, single-token lookahead parser.

    Cursor mechanics:
    - self.tokens: the full token list from the lexer (must end with EOF)
    - self.i: index of the current token; _at() peeks without consuming,
      _advance() consumes and moves the cursor forward.

    The parser is designed to be resilient to benign whitespace by explicitly
    skipping NEWLINE tokens inside parentheses (_skip_newlines) while strictly
    requiring NEWLINE between top-level statements in parse_program.
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    # ------------- Core cursor utilities -------------

    def _at(self) -> Token:
        """
        Return the current token without consuming it (peek).
        Assumes the lexer always appends an EOF token so peeking at the end
        is safe and returns EOF.
        """
        return self.tokens[self.i]

    def _is_at_end(self) -> bool:
        """
        True if the cursor is positioned at EOF; false otherwise.
        """
        return self._at().type == TokenType.EOF

    def _advance(self) -> Token:
        """
        Consume and return the current token, then move the cursor to the next token.
        If already at EOF, remains at EOF and returns EOF (idempotent at end).
        """
        tok = self._at()
        if not self._is_at_end():
            self.i += 1
        return tok

    def _match(self, *types: TokenType) -> Optional[Token]:
        """
        If the current token's type is among 'types', consume and return it.
        Otherwise, return None and do not advance. Useful for optional constructs.
        """
        if self._at().type in types:
            return self._advance()
        return None

    def _expect(self, ttype: TokenType, what: str) -> Token:
        """
        Require the current token to be of 'ttype', consuming and returning it.
        If not, raise a ParseError explaining what was expected ('what' text).
        """
        tok = self._at()
        if tok.type != ttype:
            # Error position is at the unexpected token
            raise ParseError(f"Expected {what}", tok.span)
        return self._advance()

    def _skip_newlines(self) -> None:
        """
        Consume zero or more NEWLINE tokens. This is used inside parentheses
        to allow multi-line argument lists. At top-level, parse_program enforces
        statement separation, so we do not unconditionally skip newlines there.
        """
        while self._match(TokenType.NEWLINE):
            # Intentionally empty: repeatedly consume all consecutive newlines.
            pass

    # ------------- High-level parse entry points -------------

    def parse_program(self) -> Program:
        """
        program := (stmt NEWLINE+)* EOF

        Implementation notes:
        - Leading blank lines are permitted (skip them).
        - Between statements, at least one NEWLINE is required.
        - After consuming a statement and its trailing newlines, continue until EOF.
        - If a lexer error token is encountered, surface it as a ParseError early.
        - The program span is the union from the first non-NEWLINE token to EOF.
        """
        # The initial span anchor (may be NEWLINE if the file starts with one).
        start_span = self._at().span
        stmts: List[Stmt] = []

        # Allow leading blank lines before the first statement.
        self._skip_newlines()

        # Main loop: parse statements until EOF.
        while not self._is_at_end():
            # If the lexer flagged an invalid character, treat it as a hard error now.
            if self._at().type == TokenType.LEX_ERROR:
                tok = self._advance()
                raise ParseError(f"Invalid character {tok.lexeme!r}", tok.span)

            # Parse a single top-level statement.
            stmts.append(self._parse_stmt())

            # If EOF follows immediately, we're done (no newline required at EOF).
            if self._is_at_end():
                break

            # Require at least one NEWLINE after a statement.
            if not self._match(TokenType.NEWLINE):
                # No newline where one is required: signal a clear error at the token that follows.
                tok = self._at()
                raise ParseError("Expected newline after statement", tok.span)

            # Permit multiple blank lines between statements.
            self._skip_newlines()

        # Construct a program-wide span (from start to EOF).
        end_span = self._at().span  # This is the EOF token's span.
        prog_span = start_span.merge(end_span)
        return Program(span=prog_span, stmts=stmts)

    # ------------- Statement dispatch -------------

    def _parse_stmt(self) -> Stmt:
        """
        Dispatch to the appropriate statement parser based on the current token type.
        Any token type that cannot start a statement is reported as an error.
        """
        tok = self._at()
        t = tok.type

        # Recognize each statement by its keyword token type.
        if t == TokenType.KW_QUBIT:
            return self._parse_qubit_decl()
        if t == TokenType.KW_HADAMARD:
            return self._parse_unary_gate(kind="hadamard")
        if t == TokenType.KW_XGATE:
            return self._parse_unary_gate(kind="xgate")
        if t == TokenType.KW_CNOT:
            return self._parse_binary_gate(kind="cnot")
        if t == TokenType.KW_SWAP:
            return self._parse_binary_gate(kind="swap")
        if t == TokenType.KW_HADAMARD_TRANSFORM:
            return self._parse_multi_qubit_gate(kind="hadamard_transform")
        if t == TokenType.KW_MEASURE_PROBS:
            return self._parse_measure_probs()
        if t == TokenType.KW_PYTHON:
            return self._parse_python_code()

        if t == TokenType.EOF:
            # Defensive: parse_stmt shouldn't be called at EOF, but provide a clear error if it is.
            raise ParseError("Unexpected end of input", tok.span)

        # Anything else at statement start is unexpected.
        raise ParseError(f"Unexpected token {tok.lexeme!r}", tok.span)

    # ------------- Specific constructs -------------

    def _parse_qubit_decl(self) -> QubitDecl:
        """
        qubit_decl := "qubit" "(" ( INT | ident_list ) ")"

        Two forms are supported:
        - Count form: qubit(3) allocates 3 unnamed qubits (positive integer required).
        - Names form: qubit(q0, q1, q2) declares one or more named qubits.

        Notes:
        - An empty argument list is an error (must provide either a count or names).
        - NEWLINEs are permitted inside parentheses (e.g., qubit(q0, <nl> q1)).
        - The AST node span covers from 'qubit' through the closing ')'.
        """
        kw = self._expect(TokenType.KW_QUBIT, "'qubit'")
        self._expect(TokenType.LPAREN, "'(' after 'qubit'")
        self._skip_newlines()

        count: Optional[int] = None
        names: List[str] = []

        # Disallow qubit() with no arguments.
        if self._match(TokenType.RPAREN):
            raise ParseError("qubit(...) requires a count or one or more names", kw.span)

        # Decide between the two forms by peeking the next token.
        if self._at().type == TokenType.INT:
            # Count form: a single positive integer.
            int_tok = self._advance()
            try:
                val = int(int_tok.lexeme)
            except ValueError:
                # Should not happen if lexer ensured valid INT lexemes, but be defensive.
                raise ParseError("Invalid integer literal for qubit count", int_tok.span)
            if val <= 0:
                raise ParseError("Qubit count must be a positive integer", int_tok.span)
            count = val

            # Close the argument list: no trailing commas are allowed in this grammar.
            self._skip_newlines()
            self._expect(TokenType.RPAREN, "')' after qubit count")

            # Span: from 'qubit' to the closing ')'. We can reuse the last consumed token for merging.
            span = kw.span.merge(self.tokens[self.i - 1].span)
            return QubitDecl(span=span, count=count, names=names)

        # Names form: parse a non-empty identifier list.
        first_ident = self._expect(TokenType.IDENT, "identifier in qubit(...)")
        names.append(first_ident.lexeme)
        last_span = first_ident.span  # Track last element for potential diagnostics (kept for clarity).

        # Continue parsing ", IDENT" sequences until no comma is found.
        while True:
            self._skip_newlines()
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
            ident = self._expect(TokenType.IDENT, "identifier after ','")
            names.append(ident.lexeme)
            last_span = ident.span  # Update last span (not strictly needed, but illustrative).

        # Expect closing parenthesis after the list.
        self._skip_newlines()
        rpar = self._expect(TokenType.RPAREN, "')' to close qubit(...)")

        span = kw.span.merge(rpar.span)
        return QubitDecl(span=span, count=None, names=names)

    def _parse_unary_gate(self, kind: str):
        """
        Parse one of the unary gate calls with a single (IDENT | INT) argument:
          hadamard_call := "hadamard" "(" (IDENT | INT) ")"
          xgate_call    := "xgate" "(" (IDENT | INT) ")"

        'kind' selects which gate to parse and which AST node to construct.
        """
        if kind == "hadamard":
            kw_type = TokenType.KW_HADAMARD
            node_ctor = Hadamard
        elif kind == "xgate":
            kw_type = TokenType.KW_XGATE
            node_ctor = XGate
        else:
            # Internal misuse: this function only supports the above kinds.
            raise AssertionError(f"Unknown unary gate kind: {kind}")
        # Expect the keyword, then a parenthesized single qubit identifier (name or number).
        kw = self._expect(kw_type, f"'{kind}'")
        self._expect(TokenType.LPAREN, "'(' after keyword")
        self._skip_newlines()
        # Accept either IDENT or INT as qubit identifier
        if self._at().type == TokenType.IDENT:
            ident = self._advance()
        elif self._at().type == TokenType.INT:
            ident = self._advance()
        else:
            raise ParseError("Expected qubit identifier (name or number)", self._at().span)
        self._expect(TokenType.RPAREN, "')' after qubit identifier")
        span = kw.span.merge(self._at().span)
        return node_ctor(span=span, target=ident.lexeme)


    def _parse_binary_gate(self, kind: str):
        """
        Parse one of the binary gate calls with two (IDENT | INT) arguments separated by a comma:
          cnot_call := "cnot" "(" (IDENT | INT) "," (IDENT | INT) ")"
          swap_call := "swap" "(" (IDENT | INT) "," (IDENT | INT) ")"

        'kind' selects which gate to parse and which AST node to construct.
        """
        if kind == "cnot":
            kw_type = TokenType.KW_CNOT
            node_ctor = CNot
            arg_names = ("control", "target")
        elif kind == "swap":
            kw_type = TokenType.KW_SWAP
            node_ctor = Swap
            arg_names = ("q1", "q2")
        else:
            # Internal misuse: only 'cnot' and 'swap' are valid here.
            raise AssertionError(f"Unknown binary gate kind: {kind}")

        # Expect: keyword '(' (IDENT | INT) ',' (IDENT | INT) ')'
        kw = self._expect(kw_type, f"'{kind}'")
        self._expect(TokenType.LPAREN, "'(' after keyword")
        self._skip_newlines()

        # Accept either IDENT or INT as qubit identifier
        if self._at().type == TokenType.IDENT:
            a1 = self._advance()
        elif self._at().type == TokenType.INT:
            a1 = self._advance()
        else:
            raise ParseError(f"Expected {arg_names[0]} qubit identifier (name or number)", self._at().span)
        self._skip_newlines()
        self._expect(TokenType.COMMA, "comma between qubit identifiers")
        self._skip_newlines()
        if self._at().type == TokenType.IDENT:
            a2 = self._advance()
        elif self._at().type == TokenType.INT:
            a2 = self._advance()
        else:
            raise ParseError(f"Expected {arg_names[1]} qubit identifier (name or number)", self._at().span)
        self._skip_newlines()

        rpar = self._expect(TokenType.RPAREN, "')' after arguments")

        # Node span: from the keyword to the closing ')'.
        span = kw.span.merge(rpar.span)

        # Construct the AST node with the appropriate field names.
        if kind == "cnot":
            return node_ctor(span=span, control=a1.lexeme, target=a2.lexeme)
        else:
            return node_ctor(span=span, q1=a1.lexeme, q2=a2.lexeme)

    def _parse_multi_qubit_gate(self, kind: str):
        """
        Parse one of the multi-qubit gate calls with a list of (IDENT | INT) arguments separated by a comma:
          multi_qubit_gate_call := "multi_qubit_gate" "(" [ qubit_list ] ")"
        """
        if kind == "hadamard_transform":
            kw_type = TokenType.KW_HADAMARD_TRANSFORM
            node_ctor = HadamardTransform
            arg_names = ("qubits",)
        else:
            raise AssertionError(f"Unknown multi-qubit gate kind: {kind}")
        kw = self._expect(kw_type, f"'{kind}'")
        self._expect(TokenType.LPAREN, "'(' after keyword")
        self._skip_newlines()
        qubits: List[str] = []
        if self._match(TokenType.RPAREN):
            span = kw.span.merge(self.tokens[self.i - 1].span)
            return node_ctor(span=span, qubits=qubits)
        if self._at().type == TokenType.IDENT:
            first_ident = self._advance()
        elif self._at().type == TokenType.INT:
            first_ident = self._advance()
        else:
            raise ParseError("Expected qubit identifier (name or number) or ')' for empty list", self._at().span)
        qubits.append(first_ident.lexeme)
        while True:
            self._skip_newlines()
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
            if self._at().type == TokenType.IDENT:
                ident = self._advance()
            elif self._at().type == TokenType.INT:
                ident = self._advance()
            else:
                raise ParseError("Expected qubit identifier (name or number) after ','", self._at().span)
            qubits.append(ident.lexeme)
        self._skip_newlines()
        rpar = self._expect(TokenType.RPAREN, "')' after arguments")
        span = kw.span.merge(rpar.span)
        return node_ctor(span=span, qubits=qubits)

    def _parse_measure_probs(self) -> MeasureProbs:
        """
        measure_probs_call := "measure_probs" "(" [ qubit_list ] ")"

        Semantics:
        - With an empty argument list (measure_probs()), the executor may interpret
          this as "measure all currently allocated qubits" or some sensible default.
        - With a non-empty list, only those qubits are included.

        Notes:
        - NEWLINEs are permitted inside the parentheses.
        - No trailing commas are allowed (strict qubit_list).
        - qubit_list := (IDENT | INT) ("," (IDENT | INT))*
        """
        kw = self._expect(TokenType.KW_MEASURE_PROBS, "'measure_probs'")
        self._expect(TokenType.LPAREN, "'(' after keyword")
        self._skip_newlines()

        qubits: List[str] = []

        # Allow empty parens as a valid form: measure_probs()
        if self._match(TokenType.RPAREN):
            # Span merges from keyword to the right parenthesis we just consumed.
            span = kw.span.merge(self.tokens[self.i - 1].span)
            return MeasureProbs(span=span, qubits=qubits)

        # Otherwise parse a standard identifier list: IDENT/INT (',' IDENT/INT)*
        if self._at().type == TokenType.IDENT:
            first_ident = self._advance()
        elif self._at().type == TokenType.INT:
            first_ident = self._advance()
        else:
            raise ParseError("Expected qubit identifier (name or number) or ')' for empty list", self._at().span)
        qubits.append(first_ident.lexeme)

        # Continue parsing comma-separated identifiers.
        while True:
            self._skip_newlines()
            if not self._match(TokenType.COMMA):
                break
            self._skip_newlines()
            if self._at().type == TokenType.IDENT:
                ident = self._advance()
            elif self._at().type == TokenType.INT:
                ident = self._advance()
            else:
                raise ParseError("Expected qubit identifier (name or number) after ','", self._at().span)
            qubits.append(ident.lexeme)

        self._skip_newlines()
        rpar = self._expect(TokenType.RPAREN, "')' to close measure_probs(...)")

        # Node span: from keyword to closing ')'.
        span = kw.span.merge(rpar.span)
        return MeasureProbs(span=span, qubits=qubits)

    def _parse_python_code(self) -> PythonCode:
        """
        python_code := "python" [python code]
        """
        kw = self._expect(TokenType.KW_PYTHON, "'python'")
        python_code = self._expect(TokenType.PYTHON_CODE, "python code")
        return PythonCode(span=kw.span.merge(python_code.span), code=python_code.lexeme)

# ------------- Public convenience APIs -------------

def parse_tokens(tokens: List[Token]) -> Program:
    """
    Parse a pre-tokenized input into an AST Program.
    Useful when you want to run the lexer separately (e.g., to inspect tokens).
    May raise ParseError on invalid syntax.
    """
    parser = Parser(tokens)
    return parser.parse_program()


def parse(source: str) -> Program:
    """
    Convenience function: tokenize the given source and parse it into an AST Program.
    May raise ParseError on syntax errors.
    """
    tokens = tokenize(source)
    return parse_tokens(tokens)


__all__ = ["ParseError", "Parser", "parse_tokens", "parse"]