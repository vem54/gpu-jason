# GPU CFR Poker Solver - Dream Architecture Specification

## Vision

Build a GPU-accelerated poker solver starting with Kuhn Poker, scaling to NLHE with:
- **Full CFR on flop** (exact solving)
- **Sampling on turn/river** (GPU Partial Vector CFR)

---

## Core Insight (from Kim 2024 Paper)

CFR can be reformulated as **dense/sparse matrix operations**, eliminating recursive tree traversal and enabling massive GPU parallelism:

```
Traditional CFR: O(|V|) sequential tree traversals per iteration
Matrix CFR:      O(D) matrix multiplications where D = tree depth
```

Key matrices (all precomputed, constant during solving):
- `G` - Game tree adjacency matrix (sparse)
- `L^(l)` - Level graphs for each depth (sparse)
- `M^(Q+,V)` - Maps nodes to infoset-action pairs (sparse)
- `M^(H+,Q+)` - Maps infosets to infoset-action pairs (sparse)
- `M^(V,I+)` - Maps nodes to acting players (dense)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION                             │
│  (Kuhn Poker → Leduc → NLHE solver entry points)            │
├─────────────────────────────────────────────────────────────┤
│                       SOLVERS                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Vanilla CFR │  │    CFR+     │  │ Partial Vector CFR   │ │
│  │  (Phase 1)  │  │  (Phase 2)  │  │     (Phase 3)        │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    MATRIX ENGINE                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  GPU Backend (CuPy)  ←──switch──→  CPU Backend (NumPy)  ││
│  │                                                          ││
│  │  - Sparse matrix ops (CSR format)                       ││
│  │  - Dense vector ops                                      ││
│  │  - Level-wise tree traversal                            ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    GAME ABSTRACTION                          │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │  GameBuilder  │  │ MatrixBuilder │  │  InfosetCodec   │  │
│  │  (tree gen)   │  │ (G, L, M's)   │  │ (hash ↔ index)  │  │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      POKER GAMES                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────────────────────┐│
│  │   Kuhn    │  │   Leduc   │  │          NLHE             ││
│  │ (Phase 1) │  │ (Phase 2) │  │  (Phase 3: abstracted)    ││
│  └───────────┘  └───────────┘  └───────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure (grimp-enforced)

```
gpu_poker_cfr/
├── __init__.py
├── games/                    # LAYER 1: Game definitions (no upward deps)
│   ├── __init__.py
│   ├── base.py              # Abstract game interface
│   ├── kuhn.py              # Kuhn poker implementation
│   ├── leduc.py             # Leduc poker (Phase 2)
│   └── nlhe.py              # NLHE abstracted (Phase 3)
│
├── matrix/                   # LAYER 2: Matrix building (depends on: games)
│   ├── __init__.py
│   ├── builder.py           # Builds G, L, M matrices from game tree
│   ├── sparse.py            # CSR sparse matrix utilities
│   └── codec.py             # Infoset hash ↔ matrix index mapping
│
├── engine/                   # LAYER 3: Compute engine (depends on: matrix)
│   ├── __init__.py
│   ├── backend.py           # Backend switcher (CuPy/NumPy)
│   ├── ops.py               # Matrix operations (traversal, regret matching)
│   └── kernels.py           # Custom CUDA kernels if needed
│
├── solvers/                  # LAYER 4: CFR algorithms (depends on: engine)
│   ├── __init__.py
│   ├── vanilla.py           # Vanilla CFR
│   ├── plus.py              # CFR+ (Phase 2)
│   └── partial.py           # Partial Vector CFR (Phase 3)
│
├── testing/                  # Test infrastructure (can depend on all)
│   ├── __init__.py
│   ├── golden.py            # Golden output management
│   ├── equivalence.py       # Numerical equivalence checks
│   └── snapshots/           # Stored golden outputs
│
└── cli.py                    # Entry point
```

### Dependency Rules (enforced by grimp)

```python
# grimp contract - NO VIOLATIONS ALLOWED
LAYERS = [
    'gpu_poker_cfr.games',      # Layer 1 - lowest
    'gpu_poker_cfr.matrix',     # Layer 2
    'gpu_poker_cfr.engine',     # Layer 3
    'gpu_poker_cfr.solvers',    # Layer 4 - highest
]

# Each layer can only import from layers below it
# games: no internal imports
# matrix: can import from games
# engine: can import from matrix, games
# solvers: can import from engine, matrix, games
```

---

## Data Structures

### Game Tree Representation

```python
@dataclass
class GameTree:
    """Immutable game tree structure."""
    num_nodes: int              # |V|
    num_terminals: int          # |T|
    num_infosets: int           # |H+| (player infosets only)
    num_actions: int            # |A|
    num_infoset_actions: int    # |Q+| (infoset-action pairs)
    max_depth: int              # D
    num_players: int            # |I+| (excluding chance)

    # Terminal utilities: shape (num_terminals, num_players)
    terminal_utilities: ndarray

    # Node depths: shape (num_nodes,)
    node_depths: ndarray

    # Which nodes are terminals: shape (num_nodes,) bool
    is_terminal: ndarray
```

### Matrix Constants (precomputed once)

```python
@dataclass
class GameMatrices:
    """Sparse matrices representing game structure."""

    # Adjacency matrix G: (num_nodes, num_nodes) sparse
    G: csr_matrix

    # Level graphs L^(l): list of (num_nodes, num_nodes) sparse
    L: List[csr_matrix]  # length = max_depth

    # Node → infoset-action mapping: (num_infoset_actions, num_nodes) sparse
    M_QV: csr_matrix

    # Infoset → infoset-action mapping: (num_infosets, num_infoset_actions) sparse
    M_HQ: csr_matrix

    # Node → player mapping: (num_nodes, num_players) dense
    M_VI: ndarray

    # Chance probabilities: (num_nodes,) dense
    chance_probs: ndarray
```

### Solver State (mutable, updated each iteration)

```python
@dataclass
class CFRState:
    """Mutable state updated during CFR iterations."""

    # Current strategy: (num_infoset_actions,)
    sigma: ndarray

    # Average strategy: (num_infoset_actions,)
    sigma_avg: ndarray

    # Cumulative regrets: (num_infoset_actions,)
    regret_sum: ndarray

    # Cumulative reach probabilities: (num_infosets,)
    reach_sum: ndarray

    iteration: int
```

---

## CFR Iteration (Matrix Form)

Each iteration performs these matrix operations:

```python
def cfr_iteration(matrices: GameMatrices, state: CFRState, t: int):
    """Single CFR iteration using matrix operations."""

    # 1. Regret matching: compute strategy from regrets
    state.sigma = regret_match(state.regret_sum, matrices.M_HQ)

    # 2. Forward pass: compute reach probabilities (root → leaves)
    #    Uses level graphs L^(1), L^(2), ..., L^(D)
    s = compute_action_probs(state.sigma, matrices)  # (num_nodes,)
    pi = forward_reach(s, matrices.L, matrices.M_VI)  # (num_nodes, num_players)

    # 3. Backward pass: compute expected utilities (leaves → root)
    #    Uses level graphs in reverse L^(D), L^(D-1), ..., L^(1)
    U = backward_values(matrices.terminal_utilities, s, matrices.L)

    # 4. Compute counterfactual values and update regrets
    cf_reach = compute_cf_reach(pi, matrices)  # (num_infosets,)
    cf_values = compute_cf_values(U, pi, matrices)  # (num_infoset_actions,)

    instant_regret = compute_instant_regret(cf_values, cf_reach, matrices)
    state.regret_sum += instant_regret

    # 5. Update average strategy (weighted by reach)
    update_average_strategy(state, cf_reach, t)

    state.iteration = t
```

---

## Testing Strategy

### 1. Unit Tests (per module)
- `games/`: Tree generation correctness, node counts match expected
- `matrix/`: Matrix shapes, sparsity patterns, index mappings
- `engine/`: Individual ops produce expected outputs
- `solvers/`: Single iteration produces expected state changes

### 2. Golden Output Tests
```python
# Store known-good outputs for regression testing
GOLDEN_OUTPUTS = {
    'kuhn_1000_iter': {
        'exploitability': 0.0556,  # Known value
        'strategy_checksum': 'abc123...',
    }
}
```

### 3. Numerical Equivalence Tests
```python
def test_gpu_cpu_equivalence():
    """GPU and CPU backends must produce identical results."""
    game = KuhnPoker()

    cpu_solver = VanillaCFR(game, backend='numpy')
    gpu_solver = VanillaCFR(game, backend='cupy')

    cpu_solver.solve(iterations=100)
    gpu_solver.solve(iterations=100)

    np.testing.assert_allclose(
        cpu_solver.average_strategy,
        gpu_solver.average_strategy,
        rtol=1e-5
    )
```

### 4. Reference Validation (OpenSpiel)
```python
def test_against_openspiel():
    """Validate our exploitability matches OpenSpiel's."""
    our_strategy = solve_kuhn(iterations=10000)

    # Compare against OpenSpiel's known solution
    openspiel_exploitability = 0.0  # Nash equilibrium
    our_exploitability = compute_exploitability(our_strategy)

    assert our_exploitability < 0.001  # Should converge
```

---

## Phase Roadmap

### Phase 1: Kuhn Poker + Vanilla CFR (Current)
- [x] Architecture specification
- [ ] Project setup with grimp
- [ ] Kuhn Poker game tree generation
- [ ] Matrix builder
- [ ] GPU engine with CuPy
- [ ] Vanilla CFR solver
- [ ] Golden output tests
- [ ] OpenSpiel validation

### Phase 2: Leduc Poker + CFR+
- [ ] Leduc Poker game implementation
- [ ] CFR+ discounting schemes
- [ ] Performance benchmarking vs OpenSpiel
- [ ] Memory optimization for larger games

### Phase 3: NLHE + Partial Vector CFR
- [ ] Card abstraction (equity bucketing)
- [ ] Action abstraction (bet sizing)
- [ ] Full CFR on flop subgames
- [ ] Monte Carlo sampling on turn/river
- [ ] GPU Partial Vector CFR implementation

---

## Hardware Considerations

**Target GPU**: GTX 16-series (1650/1660)
- CUDA Capability: 7.5 (Turing)
- VRAM: 4-6 GB
- Consideration: Memory-bound, not compute-bound for poker

**Memory Budget for Kuhn**:
- ~58 nodes, ~16 infosets → trivial (KB)
- Good for validation, not performance testing

**Memory Budget for Leduc**:
- ~9,500 nodes, ~1,093 infosets → still small (MB)
- Good stepping stone

**Memory Budget for NLHE**:
- Millions of nodes → requires abstraction
- Phase 3 will need careful memory management

---

## Success Criteria

1. **Correctness**: Exploitability < 0.001 on Kuhn after 10k iterations
2. **GPU Acceleration**: Faster than OpenSpiel Python on Leduc+
3. **Architecture**: Zero grimp violations throughout development
4. **Testing**: 100% of golden tests passing, GPU/CPU equivalence verified
