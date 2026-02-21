# File: src/lexer.py
# -----------------------------------------------------------------------------
# The lexer (also called tokenizer) converts raw source text into a list of tokens.
# Each token has:
#   - a type (from TokenType, e.g., IDENT, INT, LPAREN),
#   - a lexeme (the exact string of characters from the source),
#   - a span (start/end line/column for error reporting).
#
# This DSL uses:
#   - Whitespace and comments (# ... end-of-line) are ignored, except that
#     newlines produce NEWLINE tokens to separate statements.
#   - Identifiers are [A-Za-z_][A-Za-z0-9_]* and may collide with keywords.
#     If the lexeme matches a keyword (case-insensitive in v0), we emit that keyword.
#   - Numbers: simple INT (digits) and FLOAT (digits '.' digits).
#     Scientific notation is intentionally NOT supported in v0 for simplicity.
#   - Punctuation: '(', ')', ','.
#   - Errors: any unknown character becomes a LEX_ERROR token (so we can keep scanning).
#
# End-of-input: we always append a final EOF token.
# -----------------------------------------------------------------------------

from typing import List
from src.tokens import TokenType, Token, Span


# A single source of truth for the keyword set. We lowercase the lexeme before
# checking to make keywords case-insensitive (user can write "Qubit" or "qubit").
_KEYWORDS = {
    "qubit": TokenType.KW_QUBIT,
    "hadamard": TokenType.KW_HADAMARD,
    "hadamard_transform": TokenType.KW_HADAMARD_TRANSFORM,
    "xgate": TokenType.KW_XGATE,
    "cnot": TokenType.KW_CNOT,
    "swap": TokenType.KW_SWAP,
    "measure_probs": TokenType.KW_MEASURE_PROBS,
    "python": TokenType.KW_PYTHON,
}


def _single_char_span(line: int, col: int) -> Span:
    """
    Convenience helper: create a Span that covers a single character
    at the given 1-based line/column.
    """
    return Span(line, col, line, col)


def tokenize(source: str) -> List[Token]:
    """
    Convert the entire source string into a flat list of tokens.

    Notes on newline handling:
    - We normalize both Unix '\n' and Windows '\r\n' into a single NEWLINE token
      whose lexeme is always "\n".
    - A stand-alone '\r' (classic Mac/new Windows oddity) is also treated as a newline.

    Returns:
        List[Token]: The tokens representing the source, ending with an EOF token.
    """
    tokens: List[Token] = []

    # Current 1-based line/column for diagnostics.
    line = 1
    col = 1

    # i is our index into the raw source string.
    i = 0
    n = len(source)

    # Main scan loop
    while i < n:
        ch = source[i]

        # 1) Skip spaces and tabs (but not newlines—newlines are significant in this DSL).
        if ch == ' ' or ch == '\t':
            i += 1
            col += 1
            continue

        # 2) Line comments start with '#' and run to the end of the line.
        if ch == '#':
            j = i + 1
            while j < n and source[j] not in '\r\n':
                j += 1
            # We consumed characters from i..j-1; update column accordingly.
            col += (j - i)
            i = j
            # Do NOT emit a token for the comment; we just skip it.
            continue

        # 3) Newline handling (Unix '\n')
        if ch == '\n':
            # Emit a NEWLINE token to separate statements.
            tokens.append(Token(TokenType.NEWLINE, "\n", _single_char_span(line, col)))
            # Advance positions: move to the start of the next line.
            i += 1
            line += 1
            col = 1
            continue

        # 4) Newline handling (Windows '\r\n' or bare '\r')
        if ch == '\r':
            i += 1
            if i < n and source[i] == '\n':
                # Treat '\r\n' as a single newline token with lexeme "\n".
                i += 1
                tokens.append(Token(TokenType.NEWLINE, "\n", _single_char_span(line, col)))
            else:
                # Bare '\r' also becomes a newline.
                tokens.append(Token(TokenType.NEWLINE, "\n", _single_char_span(line, col)))
            line += 1
            col = 1
            continue

        # 5) Punctuation: '(', ')', ','
        if ch == '(':
            tokens.append(Token(TokenType.LPAREN, "(", _single_char_span(line, col)))
            i += 1
            col += 1
            continue

        if ch == ')':
            tokens.append(Token(TokenType.RPAREN, ")", _single_char_span(line, col)))
            i += 1
            col += 1
            continue

        if ch == ',':
            tokens.append(Token(TokenType.COMMA, ",", _single_char_span(line, col)))
            i += 1
            col += 1
            continue

        # 6) Identifiers and keywords: [A-Za-z_][A-Za-z0-9_]*
        #    We check the lexeme against the keyword table (case-insensitive).
        if ch.isalpha() or ch == '_':
            start_line = line
            start_col = col
            j = i
            while j < n and (source[j].isalnum() or source[j] == '_'):
                j += 1

            lexeme = source[i:j]
            # Case-insensitive keyword matching: we standardize to lowercase for lookup.
            ttype = _KEYWORDS.get(lexeme.lower(), TokenType.IDENT) #so if the lexeme is not a keyword, it is an identifier

            # Compute width and ending column for the span.
            width = j - i
            end_col = start_col + width - 1

            tokens.append(Token(ttype, lexeme, Span(start_line, start_col, start_line, end_col)))
            
            # Advance scanner positions.
            i = j
            col += width

            if ttype == TokenType.KW_PYTHON:
                while j < n and source[j] != '{':
                    j += 1
                j += 1
                while j < n and source[j] == ' ':
                    j += 1
                k = j
                while k < n and source[k] != '}':
                    k += 1
                lexeme = source[j:k]
                ttype = TokenType.PYTHON_CODE
                width = k - j
                end_col = start_col + width - 1
                tokens.append(Token(ttype, lexeme, Span(start_line, start_col, start_line, end_col)))
                i = k + 1
                col += width
                continue
            continue

        # 7) Numbers: INT or FLOAT
        #    - INT: digits+
        #    - FLOAT: digits+ '.' digits+
        #    Note: This simple v0 does not support leading '.', trailing '.', or exponents.
        if ch.isdigit():
            start_line = line
            start_col = col
            j = i

            # Consume the integer part
            while j < n and source[j].isdigit():
                j += 1

            is_float = False
            # If we see a '.', and it's followed by a digit, we parse it as a float
            if j < n and source[j] == '.' and (j + 1) < n and source[j + 1].isdigit():
                is_float = True
                j += 1  # consume '.'
                while j < n and source[j].isdigit():
                    j += 1

            lexeme = source[i:j]
            ttype = TokenType.FLOAT if is_float else TokenType.INT

            width = j - i
            end_col = start_col + width - 1

            tokens.append(Token(ttype, lexeme, Span(start_line, start_col, start_line, end_col)))

            i = j
            col += width
            continue

        # 8) Anything else is a lexical error. We emit a LEX_ERROR token for it so that:
        #    - the parser can attempt to continue,
        #    - we can report accurate error positions later.
        tokens.append(Token(TokenType.LEX_ERROR, ch, _single_char_span(line, col)))
        i += 1
        col += 1

    # End-of-input sentinel so the parser knows it's done.
    tokens.append(Token(TokenType.EOF, "", Span(line, col, line, col)))
    return tokens


# Public API
__all__ = ["tokenize"]