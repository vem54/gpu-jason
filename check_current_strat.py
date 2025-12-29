"""Check if current strategy (before averaging) is 100% check for OOP."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Run iterations
solver.solve(1000)

# Get the FINAL current strategy (after last iteration)
current_strat = solver._strategy.reshape((solver.num_infosets, 3, solver.num_deals))

print("CURRENT strategy (from regret matching) for OOP at root:\n")
print("Hand         | Check    | All-in")
print("-" * 40)

for hand in solver.oop_hands:
    hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
    hand_idx = solver.oop_hand_to_idx[hand]
    key = (0, hand_idx, 0)  # OOP at root
    h = solver.infoset_key_to_idx[key]

    # Find matching deals
    matching_deals = []
    for d in range(solver.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        oop_hand = tuple(sorted(oop_hand, reverse=True))
        if oop_hand == hand:
            matching_deals.append(d)

    matching_array = cp.array(matching_deals)

    # Current strategy (from this iteration)
    cur_s = cp.asnumpy(current_strat[h, :, matching_array].mean(axis=0))

    print(f"{hand_str:12s} | {cur_s[0]*100:6.1f}%  | {cur_s[1]*100:6.1f}%")
