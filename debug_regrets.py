"""Debug regrets for IP at different nodes."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Run some iterations
solver.solve(100)

# Get regrets for first IP hand at action_id=1 vs action_id=6
hand = solver.ip_hands[0]  # First IP hand
hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
print(f"Checking IP hand: {hand_str}")

# Get infoset indices
hand_idx = solver.ip_hand_to_idx[hand]
key1 = (1, hand_idx, 1)
key6 = (1, hand_idx, 6)
infoset1 = solver.infoset_key_to_idx[key1]
infoset6 = solver.infoset_key_to_idx[key6]

print(f"  Infoset at action_id=1: {infoset1}")
print(f"  Infoset at action_id=6: {infoset6}")

# Reshape cumulative regret
cum_regret = solver._cumulative_regret.reshape((solver.num_infosets, 3, solver.num_deals))
cum_strat = solver._cumulative_strategy.reshape((solver.num_infosets, 3, solver.num_deals))

# Get regrets for this hand
regret1 = cp.asnumpy(cum_regret[infoset1, :, :].sum(axis=1))
regret6 = cp.asnumpy(cum_regret[infoset6, :, :].sum(axis=1))

print(f"\nCumulative regret at action_id=1 (Check/All-in): {regret1}")
print(f"Cumulative regret at action_id=6 (Fold/Call): {regret6}")

strat1 = cp.asnumpy(cum_strat[infoset1, :, :].sum(axis=1))
strat6 = cp.asnumpy(cum_strat[infoset6, :, :].sum(axis=1))

print(f"\nCumulative strategy at action_id=1: {strat1}")
print(f"Cumulative strategy at action_id=6: {strat6}")

# Normalize
total1 = strat1.sum()
total6 = strat6.sum()
if total1 > 0:
    print(f"\nNormalized strategy at action_id=1: {strat1 / total1}")
if total6 > 0:
    print(f"Normalized strategy at action_id=6: {strat6 / total6}")
