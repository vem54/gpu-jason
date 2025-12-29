# Project State - December 29, 2025 (Session 4)

## Environment
- **Branch:** master
- **Last commit:** `8becd25` - Initial commit: GPU poker CFR solver
- **Uncommitted changes:** No
- **Working directory:** `C:\Users\jsnbr\Desktop\gpu-jason`
- **GitHub repo:** https://github.com/vem54/gpu-jason
- **Key dependencies:** CuPy (GPU), NumPy, Python 3.x

## What Works (Verified)

| Component | Status | Verification Command | Last Verified |
|-----------|--------|---------------------|---------------|
| GPU River Solver | ✅ Working | `python -c "from gpu_poker_cfr.solvers.gpu_river_solver import GPURiverSolver; print('OK')"` | Session 2 |
| GPU Turn Solver Infrastructure | ✅ Working | `python gpu_turn_solver.py` (runs 1000 iter) | Session 3 |
| CPU Reference | ✅ Working | `python cpu_turn_cfr_reference.py` | Session 3 |
| CPU-GPU Regret Match | ✅ Working | `python debug_river_regret_scale.py` | Session 3 |
| GitHub Integration | ✅ Working | `git remote -v` shows origin | Session 4 |

## What's Broken

| Component | Status | Symptom | Error (verbatim) |
|-----------|--------|---------|------------------|
| Turn Strategy Convergence | ❌ Broken | Converges to ~99% Check instead of ~60% Check | No error - wrong equilibrium |

## In Progress (Partial)

| Component | % Done | What Remains | Blocked By |
|-----------|--------|--------------|------------|
| Turn Solver Validation | ~80% | Find why both CPU and GPU converge to wrong equilibrium | Strategy convergence blocker |

## Current Approach

### Strategy
GPU CFR+ solver with one CUDA thread per deal, looping over 44 valid river cards internally. Chance nodes average EVs with `river_weight = 1/N`. Regrets and strategies accumulate WITHOUT river_weight at river nodes. CPU reference implementation mirrors this for comparison.

### Key Assumptions
- CFR+ with linear averaging is the correct algorithm — if WASM uses different variant (DCFR, LCFR), results may legitimately differ
- River_weight should NOT be applied to regret/strategy accumulation — if wrong, would need to re-examine CFR theory for chance nodes
- Infosets are correctly defined (one per hand per decision node) — verified, unlikely to be wrong

### Implementation Details
- Tree has 24 nodes: turn decisions, chance nodes, river decisions, terminals
- Ranges: OOP (AA, KK, AK, AQ, AJ, AT), IP (AA, KK, JJ, 55, AK, AQ, JT)
- Board: Ks Qd 7h 2c (turn)
- Pot: 100, Stacks: 200, All-in only betting
- GPU kernel uses atomicAdd for regret/strategy accumulation

## Files Modified This Session

| File | Change Type | Description |
|------|-------------|-------------|
| `.gitignore` | Created (Prev Session) | Standard Python/CUDA gitignore |
| GitHub repo | Created | https://github.com/vem54/gpu-jason |

## Files Modified Previous Session (Session 3)

| File | Change Type | Description |
|------|-------------|-------------|
| `gpu_poker_cfr/solvers/gpu_turn_solver.py` | Modified | Removed `river_weight` from river regret update (lines 312-318) and strategy accumulation (lines 279-282) |
| `cpu_turn_cfr_reference.py` | Modified | Added `_compute_all_strategies()` method to fix mid-iteration strategy change bug |
| `debug_river_regret_scale.py` | Created | Compares CPU vs GPU regret values after 1 iteration |
| `debug_cpu_evs.py` | Created | Traces CPU EV computation with detailed logging |
| `debug_regret_per_deal.py` | Created | Analyzes per-deal regret contributions for IP JJ |

## Dead Ends (DO NOT RETRY)

### ❌ 1. Removing river_weight from regret update ONLY
- **What we tried:** Only removed river_weight from regret, not strategy accumulation
- **Why it failed:** Strategy and regrets must be scaled consistently; partial fix made things worse
- **Evidence:** Results went from ~97% Check to 99%+ Check
- **Why it's not worth retrying:** We now remove river_weight from BOTH, which is consistent

### ❌ 2. CPU reference as ground truth (without fix)
- **What we tried:** Comparing GPU to original CPU reference
- **Why it failed:** CPU reference had bug - `get_strategy()` called during chance node loop, strategy changed mid-iteration
- **Evidence:** CPU and GPU had OPPOSITE regret signs (CPU: Check better, GPU: All-in better)
- **Why it's not worth retrying:** Bug is fixed, CPU now matches GPU

### ❌ 3. Looking for EV bugs in GPU kernel
- **What we tried:** Created debug kernels to trace EV values at all nodes
- **Why it failed:** EVs are computed correctly; chance node averaging is correct
- **Evidence:** Debug output showed correct EV values
- **Why it's not worth retrying:** EV computation is verified correct

### ❌ 4. Double-weighting hypothesis
- **What we tried:** Checked if child EVs already contained reach weighting
- **Why it failed:** EVs are conditional values, not reach-weighted
- **Evidence:** Code inspection confirmed EVs are not pre-weighted
- **Why it's not worth retrying:** Not the issue

## Key Debug Data

### Regret Comparison After 1 Iteration (AFTER FIX)
```
Turn Node 1 (IP JJ):
  CPU: Check=0.00, All-in=344.74
  GPU: Check=0.00, All-in=344.74  ✓ MATCH

River Node 7 (OOP AA):
  CPU: Check=0, All-in=13946.875
  GPU: Check=0, All-in=13946.875  ✓ MATCH
```

### Final Strategy (1000 iterations) - BOTH CPU AND GPU
```
Node 0 (OOP turn): Check 97.0%, All-in 3.0%
Node 1 (IP after OOP check): Check 98.9%, All-in 1.1%
Node 4 (OOP facing all-in): Fold 55.3%, Call 44.7%
```

### WASM/Pio Expected
```
Node 0 (OOP turn): Check 91.8%, All-in 8.2%
Node 1 (IP after OOP check): Check ~60%, All-in ~40%
```

## Rollback Instructions
If next session breaks things worse:
1. `git checkout 8becd25` — last known good state (initial commit)
2. All code is functional, just converges to wrong equilibrium
