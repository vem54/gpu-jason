"""Debug matching deals for a specific IP hand."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Run iterations
solver.solve(100)

# Check IP hand JhTs (a straight - should always call/bet)
hand = None
for h in solver.ip_hands:
    if card_name(h[0]) + card_name(h[1]) in ['JhTs', 'TsJh']:
        hand = h
        break

if hand is None:
    # Try to find JT
    for h in solver.ip_hands:
        s = card_name(h[0]) + card_name(h[1])
        if 'J' in s and 'T' in s:
            hand = h
            print(f"Found JT hand: {s}")
            break

if hand is None:
    hand = solver.ip_hands[2]  # Just pick one

hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
print(f"Checking IP hand: {hand_str}")
print(f"Hand cards: {hand}")

# Find matching deals
matching_deals = []
for d in range(solver.num_deals):
    oop_hand, ip_hand = game.get_deal(d)
    ip_hand = tuple(sorted(ip_hand, reverse=True))
    if ip_hand == hand:
        matching_deals.append(d)

print(f"Number of matching deals: {len(matching_deals)}")
print(f"Matching deal indices: {matching_deals[:10]}...")

# Get infoset indices
hand_idx = solver.ip_hand_to_idx[hand]
key1 = (1, hand_idx, 1)
key6 = (1, hand_idx, 6)
infoset1 = solver.infoset_key_to_idx[key1]
infoset6 = solver.infoset_key_to_idx[key6]

print(f"\nInfoset at action_id=1: {infoset1}")
print(f"Infoset at action_id=6: {infoset6}")

# Check cumulative regret/strategy for matching deals only
cum_regret = solver._cumulative_regret.reshape((solver.num_infosets, 3, solver.num_deals))
cum_strat = solver._cumulative_strategy.reshape((solver.num_infosets, 3, solver.num_deals))

matching_array = np.array(matching_deals)

# Sum over matching deals
regret1_matching = cp.asnumpy(cum_regret[infoset1, :, matching_array].sum(axis=0))
regret6_matching = cp.asnumpy(cum_regret[infoset6, :, matching_array].sum(axis=0))

print(f"\nCumulative regret at action_id=1 (matching only): {regret1_matching}")
print(f"Cumulative regret at action_id=6 (matching only): {regret6_matching}")

strat1_matching = cp.asnumpy(cum_strat[infoset1, :, matching_array].sum(axis=0))
strat6_matching = cp.asnumpy(cum_strat[infoset6, :, matching_array].sum(axis=0))

print(f"\nCumulative strategy at action_id=1 (matching only): {strat1_matching}")
print(f"Cumulative strategy at action_id=6 (matching only): {strat6_matching}")

# Show what opponents we face
print(f"\nOpponents faced with hand {hand_str}:")
for d in matching_deals[:5]:
    oop_hand, ip_hand = game.get_deal(d)
    oop_str = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"
    print(f"  OOP: {oop_str}")
