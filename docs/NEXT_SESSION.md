# Next Session Instructions

## Starting Prompt

Copy this exactly to start the next session:

---
Read these files in order before doing anything:
1. /docs/CURRENT_STATE.md
2. /docs/DECISIONS.md
3. /docs/BLOCKERS.md

Then confirm you understand:
- What's working: GPU and CPU solvers produce identical regrets
- What's broken: Both converge to ~99% Check instead of expected ~60% Check
- Key finding: CPU reference had bug (strategy changing mid-iteration), now fixed
- Key finding: GPU had 44x regret scale issue at river nodes, now fixed

After confirming, continue with: **Run fixed CPU reference for 1000 iterations and compare to GPU - verify they produce the same final strategy. If they match, the bug is in our CFR logic, not GPU-specific.**
---

## Priority Tasks (in order)

1. **Verify CPU and GPU produce same final strategy**
   - Run `cpu_turn_cfr_reference.py` for 1000 iterations
   - Run `gpu_turn_solver.py` for 1000 iterations
   - Compare Node 1 strategies - they should match now
   - If they match and both are wrong, bug is in CFR logic

2. **Debug CFR logic if both converge wrong**
   - Trace regret evolution for IP JJ at Node 1 across iterations
   - Identify at what iteration Check regret starts dominating
   - Check if OOP's adaptation at Node 4 causes this

3. **Compare to WASM variant**
   - Research what CFR variant WASM uses (vanilla, DCFR, LCFR, etc.)
   - Check if different discounting could explain the difference

## Context the Next Session Needs

### The Core Mystery
- At iteration 1: IP JJ All-in regret = +344.74, Check regret = 0 → All-in is better
- After 1000 iterations: Check dominates at 99% → somehow Check became better
- Manual EV calculation shows: All-in EV = 43.5, Check EV = 30.6 → All-in SHOULD be better
- WASM/Pio produce ~60/40 split, not 99/1

### Key Debug Scripts Created This Session
| Script | Purpose |
|--------|---------|
| `debug_river_regret_scale.py` | Compares CPU vs GPU regrets after 1 iteration - NOW THEY MATCH |
| `debug_cpu_evs.py` | Traces CPU EV computation with full logging |
| `debug_regret_per_deal.py` | Shows per-deal regret contributions for IP JJ |

### Specific Code Changes Made This Session
1. `gpu_poker_cfr/solvers/gpu_turn_solver.py` lines 312-318: Changed from `regret * opp_reach * river_weight` to `regret * opp_reach`
2. `gpu_poker_cfr/solvers/gpu_turn_solver.py` lines 279-282: Changed from `s * own_reach * river_weight * iteration_weight` to `s * own_reach * iteration_weight`
3. `cpu_turn_cfr_reference.py`: Added `_compute_all_strategies()` method called at start of each iteration

## Warnings

1. **DO NOT revert the river_weight removal** - This was a confirmed bug fix, not the cause of wrong equilibrium
2. **DO NOT assume CPU reference is correct without verification** - We fixed one bug, there may be others
3. **The tree structure is correct** - Tree has 24 nodes, verified multiple times
4. **EVs are computed correctly** - Verified with debug kernels, chance averaging is correct
5. **Regrets after 1 iteration now match** - Focus on WHY convergence is wrong, not on per-iteration values

## Key Files

| File | Purpose |
|------|---------|
| `gpu_poker_cfr/solvers/gpu_turn_solver.py` | GPU turn solver - MODIFIED THIS SESSION |
| `cpu_turn_cfr_reference.py` | CPU reference - MODIFIED THIS SESSION (strategy caching) |
| `debug_river_regret_scale.py` | Verifies CPU=GPU regrets - CREATED THIS SESSION |
| `gpu_poker_cfr/games/turn_toy_game.py` | Turn game tree builder |
