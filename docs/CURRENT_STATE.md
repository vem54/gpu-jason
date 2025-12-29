# Project State - December 29, 2025 (Session 3)

## What Works

### GPU River Solver (Fully Functional)
- **File:** `gpu_poker_cfr/solvers/gpu_river_solver.py`
- ~10,000 iter/s performance
- Validated against reference Python CFR

### GPU Turn Solver Infrastructure
- **File:** `gpu_poker_cfr/solvers/gpu_turn_solver.py`
- ~1,280 iter/s performance
- Handles CHANCE nodes (river card dealing)
- Tree building, infoset mapping, hand evaluation all correct
- CPU and GPU now produce **identical regrets** after fixes

### CPU Reference Implementation (Fixed This Session)
- **File:** `cpu_turn_cfr_reference.py`
- Now correctly computes strategies ONCE at iteration start
- Previously had bug where strategy changed mid-iteration during chance node loop

## What's Broken

### WASM/Pio Strategy Mismatch (STILL UNSOLVED)
**Our solver results (1000 iterations):**
- Node 0 (OOP turn): Check 97.0%, All-in 3.0%
- Node 1 (IP after OOP check): Check 98.9%, All-in 1.1%
- Node 4 (OOP facing all-in): Fold 55.3%, Call 44.7%

**WASM/Pio expected:**
- Node 0 (OOP turn): Check 91.8%, All-in 8.2%
- Node 1 (IP after OOP check): Check ~60%, All-in ~40%

**Status:** Regrets now match between CPU and GPU, but strategy still converges to wrong equilibrium.

## Current Approach
- CFR+ with linear averaging
- GPU kernel: one thread per deal, loops over valid rivers internally
- Chance nodes average EVs using `river_weight = 1/num_valid_rivers`
- River regrets/strategy: accumulate WITHOUT river_weight (fixed this session)
- Turn regrets/strategy: use averaged EVs from chance nodes (unchanged)

## Files Modified This Session

| File | Change |
|------|--------|
| `gpu_poker_cfr/solvers/gpu_turn_solver.py` | **Lines 312-318:** Removed `river_weight` from river regret update. **Lines 279-282:** Removed `river_weight` from river strategy accumulation. |
| `cpu_turn_cfr_reference.py` | Added `_compute_all_strategies()` method. Modified `iterate()` to compute strategies once at iteration start, clear after iteration. |
| `debug_river_regret_scale.py` | Created - compares CPU vs GPU regret values after 1 iteration |
| `debug_cpu_evs.py` | Created - traces CPU EV computation with detailed logging |
| `debug_regret_per_deal.py` | Created - analyzes per-deal regret contributions for IP JJ |

## Dead Ends (DO NOT RETRY THESE)

### 1. Removing river_weight from regret update ONLY (Session 2)
**Tried:** Only removed river_weight from regret, not strategy accumulation
**Result:** Made results WORSE (99%+ Check)
**Why it failed:** Strategy and regrets must be scaled consistently

### 2. CPU reference as ground truth (THIS SESSION)
**Tried:** Comparing GPU to original CPU reference
**Result:** CPU and GPU had OPPOSITE regret signs initially
**Why it failed:** CPU reference had bug - strategy changed mid-iteration during chance node loop because `get_strategy()` was called for each river and regrets were updated between calls

### 3. Looking for EV bugs in GPU kernel (Session 2)
**Tried:** Created debug kernels to trace EV values
**Result:** EVs are computed correctly, chance node averaging is correct
**Why it failed:** The bug was in regret accumulation scale, not EV computation

### 4. Double-weighting hypothesis (Session 2)
**Tried:** Checked if child EVs already contained reach weighting
**Result:** EVs are conditional values, not reach-weighted
**Why it failed:** Not the issue

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

### Debug Script Output: Per-Deal Analysis for IP JJ
```
Total deals with IP JJ: 234
Total Check regret: -4714.8 (floored to 0)
Total All-in regret: +4714.8
Expected optimal: All-in

When OOP plays optimally:
  Check EV: 30.6
  All-in EV: 43.5
  Optimal for IP JJ: All-in
```

### Problem Symptoms
- At iteration 1: All-in regret is positive (344.74), so IP should play All-in
- After many iterations: Check regret dominates, solver converges to 99% Check
- This happens even though All-in should be better based on manual EV calculation
