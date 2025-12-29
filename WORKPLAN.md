# GPU CFR Poker Solver - Verifiable Work Chunks

Each chunk is small, testable, and builds on the previous.
**Rule**: Never proceed to chunk N+1 until chunk N passes all tests.

---

## Chunk 1: Project Skeleton + grimp

**Goal**: Empty project structure with architecture enforcement.

**Deliverables**:
```
gpu_poker_cfr/
├── __init__.py
├── games/__init__.py
├── matrix/__init__.py
├── engine/__init__.py
├── solvers/__init__.py
└── testing/__init__.py
```

**Test**: `grimp_contract.py` - fails if any layer imports from above it.

**Verification**:
```bash
python -m pytest tests/test_grimp_contract.py -v
# Must pass: 0 violations
```

---

## Chunk 2: Kuhn Poker Game Definition

**Goal**: Define Kuhn Poker rules, enumerate game tree.

**Deliverables**:
- `games/base.py` - Abstract `Game` interface
- `games/kuhn.py` - `KuhnPoker` class

**Test**: Game tree properties match known values.

**Verification**:
```python
def test_kuhn_tree_size():
    game = KuhnPoker()
    tree = game.build_tree()

    assert tree.num_nodes == 58
    assert tree.num_terminals == 30
    assert tree.num_infosets == 12  # 6 per player
    assert tree.max_depth == 4
```

**Golden Output**: `kuhn_tree.json` - serialized tree for visual inspection.

---

## Chunk 3: Matrix Builder (Adjacency + Level Graphs)

**Goal**: Build G and L matrices from game tree.

**Deliverables**:
- `matrix/builder.py` - `build_adjacency_matrix()`, `build_level_graphs()`
- `matrix/sparse.py` - CSR utilities

**Test**: Matrix shapes and sparsity match expected.

**Verification**:
```python
def test_kuhn_adjacency_matrix():
    game = KuhnPoker()
    tree = game.build_tree()
    G = build_adjacency_matrix(tree)

    assert G.shape == (58, 58)
    assert G.nnz == 57  # 57 edges (58 nodes - 1 root)

def test_kuhn_level_graphs():
    L = build_level_graphs(tree)

    assert len(L) == 4  # max_depth = 4
    # Sum of edges across all levels = total edges
    assert sum(l.nnz for l in L) == 57
```

**Golden Output**: Sparsity pattern visualization.

---

## Chunk 4: Matrix Builder (Mapping Matrices)

**Goal**: Build M^(Q+,V), M^(H+,Q+), M^(V,I+) matrices.

**Deliverables**:
- `matrix/builder.py` - `build_mapping_matrices()`
- `matrix/codec.py` - Infoset hash ↔ index codec

**Test**: Mappings are bijective, shapes correct.

**Verification**:
```python
def test_kuhn_mapping_matrices():
    matrices = build_game_matrices(tree)

    # M^(Q+,V): maps nodes to infoset-action pairs
    assert matrices.M_QV.shape == (NUM_INFOSET_ACTIONS, 58)

    # M^(H+,Q+): maps infosets to infoset-action pairs
    assert matrices.M_HQ.shape == (12, NUM_INFOSET_ACTIONS)

    # M^(V,I+): maps nodes to players
    assert matrices.M_VI.shape == (58, 2)

def test_codec_roundtrip():
    codec = InfosetCodec(tree)
    for infoset in tree.infosets:
        idx = codec.encode(infoset)
        assert codec.decode(idx) == infoset
```

**Golden Output**: `kuhn_matrices.npz` - all matrices saved.

---

## Chunk 5: GPU Engine Backend Switcher

**Goal**: Abstraction layer that works with both NumPy and CuPy.

**Deliverables**:
- `engine/backend.py` - `get_backend()`, `to_device()`, `to_host()`

**Test**: Same operations produce same results on both backends.

**Verification**:
```python
def test_backend_equivalence():
    np_backend = get_backend('numpy')
    cp_backend = get_backend('cupy')

    # Test sparse matrix-vector multiply
    A = random_sparse_matrix(100, 100)
    x = random_vector(100)

    np_result = np_backend.spmv(A, x)
    cp_result = cp_backend.spmv(to_device(A), to_device(x))

    np.testing.assert_allclose(np_result, to_host(cp_result), rtol=1e-6)
```

---

## Chunk 6: Forward Pass (Reach Probabilities)

**Goal**: Compute reach probabilities using level graphs.

**Deliverables**:
- `engine/ops.py` - `forward_reach()`

**Algorithm**:
```
For l = 1 to D:
    pi^(l) = (L^(l) ⊙ S) @ pi^(l-1) + pi^(l-1)
```

**Test**: Reach probabilities sum correctly, match hand calculation.

**Verification**:
```python
def test_forward_reach_kuhn():
    # Uniform strategy: all actions equally likely
    sigma = uniform_strategy(matrices)
    pi = forward_reach(sigma, matrices)

    # Root has reach 1.0 for both players
    assert pi[ROOT_NODE, 0] == 1.0
    assert pi[ROOT_NODE, 1] == 1.0

    # Terminal reaches sum to 1.0 (for each player path)
    # ... more specific checks
```

**Golden Output**: `kuhn_reach_uniform.npy` - reach probs under uniform strategy.

---

## Chunk 7: Backward Pass (Expected Utilities)

**Goal**: Compute expected utilities using reverse level traversal.

**Deliverables**:
- `engine/ops.py` - `backward_values()`

**Algorithm**:
```
For l = D down to 1:
    U^(l) = (L^(l) ⊙ S) @ U^(l+1) + U^(l+1)
```

**Test**: Utilities propagate correctly from terminals.

**Verification**:
```python
def test_backward_values_kuhn():
    sigma = uniform_strategy(matrices)
    U = backward_values(sigma, matrices)

    # Check a known terminal value propagates up
    # ... specific node checks

    # Game value at root should be ~0 (symmetric game)
    assert abs(U[ROOT_NODE, 0]) < 0.1
```

**Golden Output**: `kuhn_values_uniform.npy` - utilities under uniform strategy.

---

## Chunk 8: Regret Matching

**Goal**: Convert cumulative regrets to current strategy.

**Deliverables**:
- `engine/ops.py` - `regret_match()`

**Algorithm**:
```
For each infoset h:
    positive_regrets = max(0, regrets[h])
    if sum(positive_regrets) > 0:
        strategy[h] = positive_regrets / sum(positive_regrets)
    else:
        strategy[h] = uniform
```

**Test**: Strategies are valid probability distributions.

**Verification**:
```python
def test_regret_match():
    regrets = array([1.0, -2.0, 3.0])  # One infoset, 3 actions
    strategy = regret_match(regrets)

    assert strategy.sum() == approx(1.0)
    assert all(strategy >= 0)
    assert strategy == approx([0.25, 0.0, 0.75])  # [1,0,3]/4
```

---

## Chunk 9: Counterfactual Values + Regret Update

**Goal**: Compute instantaneous regrets from CFVs.

**Deliverables**:
- `engine/ops.py` - `compute_cf_values()`, `compute_instant_regret()`

**Test**: Regret update matches hand calculation.

**Verification**:
```python
def test_instant_regret_kuhn():
    # After 1 iteration from uniform, check specific regrets
    # ... hand-calculated expected values
```

**Golden Output**: `kuhn_regrets_iter1.npy` - regrets after iteration 1.

---

## Chunk 10: Vanilla CFR Solver (Full Integration)

**Goal**: Complete CFR solver that runs iterations.

**Deliverables**:
- `solvers/vanilla.py` - `VanillaCFR` class

**Test**: Exploitability decreases, converges to Nash.

**Verification**:
```python
def test_vanilla_cfr_convergence():
    solver = VanillaCFR(KuhnPoker(), backend='cupy')
    solver.solve(iterations=10000)

    # Kuhn Nash equilibrium exploitability is 0
    expl = compute_exploitability(solver.average_strategy)
    assert expl < 0.001

def test_vanilla_cfr_known_strategy():
    solver = VanillaCFR(KuhnPoker(), backend='cupy')
    solver.solve(iterations=10000)

    # Check specific Nash equilibrium probabilities
    # Player 1 with Jack should bet ~1/3 of time as bluff
    # ... more specific strategy checks
```

**Golden Output**: `kuhn_nash_strategy.json` - converged strategy.

---

## Chunk 11: OpenSpiel Validation

**Goal**: Verify our solution matches OpenSpiel's CFR.

**Deliverables**:
- `testing/openspiel_compare.py`

**Test**: Strategy and exploitability match within tolerance.

**Verification**:
```python
def test_matches_openspiel():
    import pyspiel

    # Our solver
    our_solver = VanillaCFR(KuhnPoker())
    our_solver.solve(iterations=1000)

    # OpenSpiel solver
    game = pyspiel.load_game("kuhn_poker")
    os_solver = pyspiel.CFRSolver(game)
    for _ in range(1000):
        os_solver.evaluate_and_update_policy()

    # Compare
    # ... detailed comparison
```

---

## Chunk Summary

| Chunk | Focus | Key Test |
|-------|-------|----------|
| 1 | Project + grimp | Zero import violations |
| 2 | Kuhn game tree | 58 nodes, 30 terminals |
| 3 | G, L matrices | Correct shapes, sparsity |
| 4 | M matrices | Bijective mappings |
| 5 | Backend switcher | GPU/CPU equivalence |
| 6 | Forward pass | Reach prob correctness |
| 7 | Backward pass | Utility propagation |
| 8 | Regret matching | Valid distributions |
| 9 | CFV + regret | Hand calculation match |
| 10 | Full CFR | Convergence to Nash |
| 11 | OpenSpiel | External validation |

---

## Golden Outputs Registry

All golden outputs stored in `testing/snapshots/`:

```
snapshots/
├── kuhn_tree.json              # Chunk 2
├── kuhn_matrices.npz           # Chunk 4
├── kuhn_reach_uniform.npy      # Chunk 6
├── kuhn_values_uniform.npy     # Chunk 7
├── kuhn_regrets_iter1.npy      # Chunk 9
└── kuhn_nash_strategy.json     # Chunk 10
```

Each golden output has:
1. The data file
2. A generator script (to recreate if needed)
3. A hash for integrity checking
