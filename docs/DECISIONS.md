# Project Decisions Log

## December 29, 2025

### Decision: Skip CPU Turn Solver, Go Straight to GPU
**Context:** Plan originally called for CPU turn solver first for "correctness verification"
**Options considered:**
1. CPU turn solver first, then port to GPU
2. Direct GPU implementation

**Chosen:** Direct GPU implementation
**Rationale:** User explicitly stopped CPU solver work - "This is completely meaningless. We're trying to run GPU here." We already validated the GPU approach on river, no need to repeat for turn.

---

### Decision: Turn Game with Check/All-in Only
**Context:** Planning turn solver scaling, needed to decide bet sizing
**Options considered:**
1. Full bet sizing (33%, 50%, 67%, all-in)
2. Check/All-in only (simplest tree)

**Chosen:** Check/All-in only
**Rationale:** User chose simplest tree first for validation. Can add bet sizes later.

---

### Decision: Same Ranges for Turn as River
**Context:** Planning turn solver, needed to decide ranges
**Options considered:**
1. New ranges for turn
2. Same ranges as river toy game

**Chosen:** Same ranges:
- OOP: AA, KK, AK, AQ, AJ, AT
- IP: AA, KK, JJ, 55, AK, AQ, JT

**Rationale:** Consistency with river for easier validation. User explicitly chose this.

---

### Decision: WASM-postflop for Turn Validation
**Context:** Need external reference to validate turn solver
**Options considered:**
1. WASM-postflop (manual verification)
2. Extend Python CFR reference
3. Degenerate case testing

**Chosen:** WASM-postflop
**Rationale:** User chose this. Manual entry required but provides independent reference.

---

### Decision: 6-Card Evaluation via Best-5-of-6
**Context:** Need to evaluate hands on turn (2 hole + 4 board = 6 cards)
**Options considered:**
1. Evaluate all C(6,5)=6 subsets, take best
2. Pad with dummy card, use existing 7-card

**Chosen:** Option 1 - evaluate all 6 subsets
**Rationale:** Correct and clean. Performance acceptable for turn all-in cases.

---

## December 29, 2025 (Session 3)

### Decision: Fix CPU Reference Strategy Bug
**Context:** CPU and GPU had opposite regret signs - CPU said Check better, GPU said All-in better for IP JJ
**Options considered:**
1. Debug GPU assuming CPU is correct
2. Investigate CPU reference for bugs

**Chosen:** Option 2 - Investigated CPU reference
**Rationale:** Found bug where `get_strategy()` was called during chance node loop, so strategy changed mid-iteration as regrets were updated. Fixed by computing all strategies once at iteration start.

---

### Decision: Remove river_weight from River Regret/Strategy Accumulation
**Context:** River regrets in GPU were 44x smaller than CPU after CPU fix
**Options considered:**
1. Multiply CPU regrets by 1/44 to match GPU
2. Remove river_weight from GPU regrets

**Chosen:** Option 2 - Remove river_weight from GPU
**Rationale:** In vanilla CFR, regrets and strategies accumulate across all situations. The river_weight (1/44) is only for averaging EVs at chance nodes, not for scaling regrets. CPU accumulates without weighting, GPU should match.

---

## Earlier Sessions (Summarized)

### Decision: CFR+ with Linear Averaging
**Chosen:** CFR+ (regret floor at 0) with linear strategy averaging (weight = iteration number t)
**Rationale:** Standard approach, proven to converge faster than vanilla CFR

### Decision: Per-Infoset Storage (Not Per-Deal)
**Chosen:** Store regrets/strategies in arrays of shape (num_infosets, max_actions)
**Rationale:** CFR requires aggregating regrets across all deals reaching same infoset. Per-deal storage was a bug.

### Decision: Deal-Weighted Strategy Aggregation
**Chosen:** Weight each hand's strategy by number of valid opponent hands (deals) when computing aggregate frequencies
**Rationale:** Accounts for card removal effects correctly. Fixed discrepancy between our solver and WASM.
