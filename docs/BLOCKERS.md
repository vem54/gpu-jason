# Current Blockers

## Blocker 1: Strategy Converges to Wrong Equilibrium (CRITICAL)

**Symptom:** GPU Turn Solver converges to ~99% Check at Node 1 (IP after OOP check), but WASM/Pio show ~60% Check / ~40% All-in.

**Our results (1000 iterations):**
```
Node 0 (OOP turn): Check 97.0%, All-in 3.0%
Node 1 (IP after OOP check): Check 98.9%, All-in 1.1%
Node 4 (OOP facing all-in): Fold 55.3%, Call 44.7%
```

**WASM/Pio expected:**
```
Node 0 (OOP turn): Check 91.8%, All-in 8.2%
Node 1 (IP after OOP check): Check ~60%, All-in ~40%
```

**Attempted fixes:**
1. **Fixed CPU reference bug** (strategy changing mid-iteration) - CPU and GPU now match, but both converge wrong
2. **Removed river_weight from regret update** - Now CPU and GPU regrets are identical after 1 iteration
3. **Removed river_weight from strategy accumulation** - Consistent with regret fix
4. **Verified EVs are correct** - Debug kernels confirmed correct EV values at all nodes
5. **Verified infoset mapping** - Each hand has unique infoset at each node

**Key observations:**
- At iteration 1, All-in regret for IP JJ = +344.74 (All-in is better)
- After many iterations, Check regret dominates and solver converges to 99% Check
- Manual EV calculation shows All-in EV = 43.5, Check EV = 30.6 for IP JJ (All-in should be better)
- OOP adapts their strategy at Node 4, but this shouldn't flip the equilibrium this much

**Hypotheses:**
1. **Regret matching across multiple rivers** - Something about how regrets accumulate across 44 rivers per deal might be wrong
2. **Counterfactual value calculation** - The reach probability weighting might be incorrect for chance node children
3. **Different CFR variant** - WASM might use a different CFR variant (e.g., DCFR, LCFR) with different discounting

**Next things to try (ordered by likelihood):**
1. **Compare CPU reference to WASM** - Run fixed CPU reference for 1000 iterations, compare to WASM
2. **Check if CPU reference also converges wrong** - If so, bug is in the CFR logic, not GPU implementation
3. **Try external sampling CFR** - Sample one river per iteration instead of averaging all
4. **Compare regret evolution** - Trace regret values iteration-by-iteration for specific hands to see when/why Check regret grows
5. **Check WASM CFR variant** - Determine if WASM uses different discounting or weighting

---

## Resolved Blockers

### [RESOLVED] CPU Reference Had Bug
**Symptom:** CPU and GPU had opposite regret signs
**Cause:** `get_strategy()` was called during chance node loop, so strategy changed mid-iteration as regrets updated
**Resolution:** Added `_compute_all_strategies()` called once at iteration start

### [RESOLVED] River Regrets Scaled Differently
**Symptom:** GPU river regrets were 44x smaller than CPU
**Cause:** GPU multiplied river regrets by `river_weight = 1/44`
**Resolution:** Removed river_weight from regret and strategy accumulation at river nodes

### [RESOLVED] GPU Turn Solver Not Implemented
**Resolution:** Created `gpu_poker_cfr/solvers/gpu_turn_solver.py`

### [RESOLVED] Turn Tree Infinite Loop
**Cause:** When stacks=0 after turn all-in + call, river nodes had "All-in" action with amount 0
**Resolution:** Only allow "Check" action when player's stack is 0
