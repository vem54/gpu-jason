# Next Session Instructions

## Pre-Flight Checklist
Before doing ANYTHING:
- [ ] Read `/docs/CURRENT_STATE.md`
- [ ] Read `/docs/DECISIONS.md` (at least Session 3 and 4 entries)
- [ ] Read `/docs/BLOCKERS.md`
- [ ] Run verification commands to confirm baseline

## Starting Prompt

Copy this exactly to start the next session:

---

Read these files in order before doing anything:
1. `docs/CURRENT_STATE.md`
2. `docs/DECISIONS.md`
3. `docs/BLOCKERS.md`

Then run these verification commands:
```bash
# Verify environment
git status && git log -1 --oneline

# Verify GPU solver runs
python -c "from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver; print('GPU solver imports OK')"

# Reproduce current blocker (shows ~99% Check instead of ~60%)
python -c "
from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game
game = make_turn_toy_game()
solver = GPUTurnSolver(game)
solver.solve(iterations=1000)
strat = solver.get_aggregate_strategy(1)
print(f'Node 1 (IP after OOP check): {strat}')
"
```

Confirm you understand:
- What's working: GPU and CPU solvers produce identical regrets and strategies
- What's broken: Both converge to ~99% Check instead of expected ~60% Check (per WASM/Pio)
- Current approach: CFR+ with linear averaging, full tree traversal over all 44 rivers
- What we already tried: Fixed CPU reference bug, removed river_weight from regret/strategy - these were real bugs but didn't fix the equilibrium

After confirming, continue with: **Research what CFR variant WASM-postflop uses. If it's DCFR (Discounted CFR), implement DCFR and compare results.**

---

## Priority Tasks (in order)

### P0: Determine why solver converges to wrong equilibrium
- **Goal:** Make Node 1 converge to ~60% Check / ~40% All-in (matching WASM/Pio)
- **Definition of done:** Running solver for 1000 iterations produces strategy within 5% of WASM
- **Suggested approach:**
  1. Research WASM CFR variant (check their docs/source)
  2. Trace regret evolution for IP JJ to see when Check starts dominating
  3. Try DCFR if WASM uses it
- **Estimated scope:** Medium - may require algorithm change

### P1: Add regret evolution logging
- **Goal:** Understand how regrets change over iterations
- **Definition of done:** Script that outputs IP JJ regrets at iterations 1, 10, 100, 500, 1000
- **Suggested approach:** Modify CPU reference to log after specific iterations
- **Estimated scope:** Small

### P2: Implement DCFR variant
- **Goal:** Have DCFR as alternative to CFR+
- **Definition of done:** Can run solver with `variant='dcfr'` and get different results
- **Suggested approach:** Standard DCFR params: α=1.5, β=0, γ=2
- **Estimated scope:** Medium

## Context Not Captured Elsewhere

- **User observation:** "By the way can you explain to me why you need to do CPU solving? Like Piosolver and Wasm already run on CPU" - Valid point. CPU reference was useful for debugging GPU, but WASM/Pio are the real ground truth.
- **GitHub repo:** https://github.com/vem54/gpu-jason (public)
- **WASM-postflop URL:** https://wasm-postflop.pages.dev/

## Warnings & Landmines

| Area | Warning | Why |
|------|---------|-----|
| `river_weight` | DO NOT add back to regret/strategy accumulation | Was confirmed bug, removing it made CPU=GPU match |
| CPU reference | Always call `_compute_all_strategies()` at iteration start | Otherwise strategy changes mid-iteration |
| Debugging | Use existing debug scripts in project root | `debug_river_regret_scale.py`, `debug_cpu_evs.py`, `debug_regret_per_deal.py` |

## If You Get Stuck

1. Check WASM-postflop source/docs for CFR variant info
2. Compare regret evolution between our solver and theoretical expectation
3. Try external sampling (sample 1 river instead of averaging all 44)
4. Verify game tree setup exactly matches WASM (pot, stacks, payoffs)

## Session Success Criteria

This session is successful if:
- [ ] Root cause of wrong equilibrium is identified OR
- [ ] DCFR variant is implemented and produces different results OR
- [ ] Regret evolution analysis reveals when/why Check dominates
- [ ] Handoff docs updated for next session

## Key Files

| File | Purpose |
|------|---------|
| `gpu_poker_cfr/solvers/gpu_turn_solver.py` | Main GPU solver - the production code |
| `cpu_turn_cfr_reference.py` | CPU reference for debugging |
| `gpu_poker_cfr/games/turn_toy_game.py` | Game tree and rules |
| `debug_river_regret_scale.py` | Compares CPU vs GPU regrets |
| `debug_cpu_evs.py` | Traces EV computation |
| `debug_regret_per_deal.py` | Per-deal regret analysis |
