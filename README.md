# GPU Poker CFR Solver

GPU-accelerated Counterfactual Regret Minimization for poker, implementing the matrix-based approach from [Kim 2024](https://arxiv.org/abs/2408.14778).

## Quick Start

```bash
# Install minimal dependencies (CPU-only, for development)
pip install -r requirements-minimal.txt

# Or full dependencies with GPU support
pip install -r requirements.txt

# Run architecture tests
pytest tests/test_architecture.py -v

# Run all tests
pytest tests/ -v
```

## Project Structure

```
gpu_poker_cfr/
├── games/      # Layer 1: Game definitions (Kuhn, Leduc, NLHE)
├── matrix/     # Layer 2: Sparse matrix builders
├── engine/     # Layer 3: GPU/CPU compute operations
├── solvers/    # Layer 4: CFR algorithm implementations
└── testing/    # Test utilities and golden outputs
```

## Architecture

The project follows a strict layered architecture enforced by `grimp`:

```
solvers  →  can import from engine, matrix, games
engine   →  can import from matrix, games
matrix   →  can import from games
games    →  no internal dependencies
```

## Development

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design specification.
See [WORKPLAN.md](WORKPLAN.md) for the implementation roadmap.
