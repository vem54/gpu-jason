# Current Blockers

> **Priority Order:** Blockers listed from most to least critical

## 🔴 Blocker 1: Strategy Converges to Wrong Equilibrium

**Severity:** Critical — Solver produces incorrect Nash equilibrium, making it useless for real poker analysis

**Symptom:**
Both CPU and GPU Turn CFR solvers converge to ~99% Check at Node 1 (IP after OOP check), but WASM/Pio show ~60% Check / ~40% All-in. The solvers match each other perfectly, but both are wrong.

**Expected behavior:**
Node 1 (IP after OOP check) should converge to approximately 60% Check / 40% All-in to match WASM/Pio ground truth.

**Reproduction steps:**
1. Run `python gpu_turn_solver.py` (or `python cpu_turn_cfr_reference.py`)
2. Wait for 1000 iterations to complete
3. Observe Node 1 strategy output
4. Compare to WASM-postflop with same setup

**Verification command:**
```bash
# Run this to confirm blocker still exists:
python -c "
from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
game = make_turn_toy_game()
solver = GPUTurnSolver(game)
solver.solve(iterations=1000)
strat = solver.get_aggregate_strategy(1)
print(f'Node 1: {strat}')
print(f'Check pct: {strat.get(\"Check\", 0)*100:.1f}%')
"
# Expected output when FIXED:
# Node 1: {'Check': ~0.60, 'All-in': ~0.40}
# Check pct: ~60.0%
```

**Our results (verbatim):**
```
Node 0 (OOP turn): Check 97.0%, All-in 3.0%
Node 1 (IP after OOP check): Check 98.9%, All-in 1.1%
Node 4 (OOP facing all-in): Fold 55.3%, Call 44.7%
```

**WASM/Pio expected (verbatim):**
```
Node 0 (OOP turn): Check 91.8%, All-in 8.2%
Node 1 (IP after OOP check): Check ~60%, All-in ~40%
```

**Root cause hypothesis (ranked by likelihood):**

1. **Different CFR variant** — WASM may use DCFR (Discounted CFR) or LCFR (Linear CFR) with different regret discounting
   - Evidence FOR: Our CFR+ implementation is standard, WASM is closed-source
   - Evidence AGAINST: 60% vs 99% is a HUGE difference, usually variants give similar equilibria
   - Test: Research what CFR variant WASM uses; try implementing DCFR

2. **Chance sampling vs full tree** — We traverse ALL 44 rivers per iteration; WASM might sample one river
   - Evidence FOR: External sampling CFR is common for efficiency
   - Evidence AGAINST: Full traversal should converge to same equilibrium, just slower
   - Test: Implement external sampling (sample 1 river per iteration) and compare

3. **Regret/strategy weighting at chance nodes** — Something subtle about how regrets accumulate across chance outcomes
   - Evidence FOR: This is a multi-street game with chance nodes, easy to get wrong
   - Evidence AGAINST: We verified CPU=GPU match, so our logic is internally consistent
   - Test: Compare regret evolution iteration-by-iteration to see when/why Check dominates

4. **Linear averaging weights** — Our strategy averaging uses weight=t (iteration number), maybe WASM uses different weights
   - Evidence FOR: Different averaging can affect convergence rate and final strategy
   - Evidence AGAINST: Shouldn't change equilibrium significantly
   - Test: Try constant weighting (weight=1) and compare

**Attempted fixes:**

| What we tried | Why it didn't work | Learned |
|---------------|-------------------|---------|
| Fixed CPU reference bug | Both CPU and GPU now match, but both wrong | Bug was real, but not the cause of wrong equilibrium |
| Removed river_weight from regret update | Regrets now match CPU exactly | This was correct fix for GPU-CPU mismatch |
| Removed river_weight from strategy accumulation | Consistent with regret fix | Necessary for correctness |
| Verified EVs are correct | EVs are correct | Problem is not in EV computation |
| Verified infoset mapping | Infosets are correct | Problem is not in infoset definition |

**Next things to try (in order):**

1. [ ] **Research WASM CFR variant** — Determine what algorithm WASM-postflop actually uses
   - Why: If WASM uses DCFR, our CFR+ results may be fundamentally different
   - Effort: Small (web research)

2. [ ] **Trace regret evolution** — Log IP JJ regrets at iterations 1, 10, 100, 500, 1000
   - Why: See exactly when and why Check regret starts dominating
   - Effort: Small (add logging)

3. [ ] **Try external sampling** — Sample 1 river per iteration instead of averaging all 44
   - Why: Different sampling can affect convergence behavior
   - Effort: Medium (modify CFR loop)

4. [ ] **Try DCFR** — Implement discounted CFR with α=1.5, β=0, γ=2 (standard DCFR params)
   - Why: DCFR is commonly used in modern solvers
   - Effort: Medium (modify regret update)

5. [ ] **Verify game tree against WASM** — Double-check pot sizes, stack sizes, payoffs exactly match
   - Why: Subtle differences in setup could explain different equilibrium
   - Effort: Small (manual verification)

**External resources:**
- WASM-postflop: https://wasm-postflop.pages.dev/
- CFR variants paper: https://arxiv.org/abs/1407.5042 (DCFR)
- CFR+ paper: https://arxiv.org/abs/1407.5042

---

# Resolved Blockers (for reference)

## ✅ CPU Reference Had Bug — Resolved Dec 29, 2025

**What fixed it:** Added `_compute_all_strategies()` method called once at iteration start in `cpu_turn_cfr_reference.py`

**Why it worked:** The bug was that `get_strategy()` was being called inside the chance node loop for each river card. Since regrets were updated after each river, the strategy would change mid-iteration. Computing all strategies ONCE at the start ensures consistent strategy throughout the iteration.

---

## ✅ River Regrets Scaled Differently (GPU 44x smaller) — Resolved Dec 29, 2025

**What fixed it:** Removed `river_weight` multiplication from regret update (lines 312-318) and strategy accumulation (lines 279-282) in `gpu_turn_solver.py`

**Why it worked:** In CFR, regrets and strategies accumulate across ALL situations. The `river_weight = 1/44` is only for computing expected values at chance nodes (averaging over outcomes), not for scaling individual regret contributions. CPU accumulates raw regrets, GPU should match.

---

## ✅ GPU Turn Solver Not Implemented — Resolved Dec 29, 2025

**What fixed it:** Created `gpu_poker_cfr/solvers/gpu_turn_solver.py`

**Why it worked:** Straightforward implementation following GPU river solver pattern.

---

## ✅ Turn Tree Infinite Loop — Resolved Dec 29, 2025

**What fixed it:** Modified tree builder to only allow "Check" action when player's stack is 0

**Why it worked:** When stacks=0 after turn all-in + call, river nodes incorrectly had "All-in" action with amount 0, causing infinite recursion. Restricting to "Check" when broke prevents this.
