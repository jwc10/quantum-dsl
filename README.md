# Quantum DSL

Two front ends for quantum circuits, both targeting PennyLane: a **text language** (lexer, recursive-descent parser, AST, tree-walking interpreter) and an **embedded DSL** in Python (`@quantum` decorator, same gate primitives, lazy device creation, shared context for nested calls). Built during quantum computing research. Includes Grover search, CNF SAT phase oracles, QFT, QPE, and period finding.

## What it does

- **Text interpreter pipeline:** Hand-written lexer and recursive-descent parser build an AST; a tree-walking interpreter runs it via PennyLane. Example input: `qubit(2)`, `hadamard(0)`, `cnot(0,1)`, `measure_probs(0,1)`.
- **Decorator-based approach:** Python functions decorated with `@quantum` call the same DSL primitives (`qubit`, `hadamard`, `cnot`, etc.) with normal control flow. Device is created lazily; nested `@quantum` calls reuse the same context.
- **Algorithms:** Grover search, CNF SAT oracles (SymPy for formulas, then phase oracle), QFT, QPE, period finding. Standard measurements: probs, expectation values, classical shadows.
- Tests and example circuits in `src/test_*.py`; language spec in `spec/`.

## Tech stack

- Python 3.8+
- PennyLane (simulation and execution)
- NumPy, SymPy (numerics and symbolic logic for oracles)
- Matplotlib (circuit diagrams)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run all commands below from the **project root** (the directory that contains `src/`) with the venv activated.

## Quick start

**Decorator:**

```python
from src.dsl_decorator import quantum, draw

@quantum
def bell_circuit():
    qubit(2)
    hadamard(0)
    cnot(0, 1)
    measure_probs(0, 1)

result = bell_circuit()
draw(bell_circuit)  # opens a circuit diagram (matplotlib)
```

**Text DSL:**

```python
from src.pl_executor import run_source_with_pennylane

source = """
qubit(2)
hadamard(0)
cnot(0, 1)
measure_probs(0, 1)
"""
result = run_source_with_pennylane(source)
```

## Project structure

| Path | Description |
|------|-------------|
| `src/dsl_decorator.py` | `@quantum` decorator, `QuantumContext`, gates, Grover/CNF SAT, QFT, QPE, measurements, `draw()` |
| `src/lexer.py` | Tokenizer for the text DSL |
| `src/parser.py` | Recursive-descent parser → AST |
| `src/tokens.py` | Token and span definitions |
| `src/quantum_ast.py` | AST node types |
| `src/pl_executor.py` | Tree-walking executor (AST → PennyLane QNodes) |
| `src/dsl_main.py` | Example entry point for the pipeline |
| `src/test_*.py` | Tests and example circuits |
| `spec/` | Language and token spec (v0) |

## License

MIT. See [LICENSE](LICENSE).
