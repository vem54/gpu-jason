"""Compare current strategy (from regrets) vs average strategy."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Run iterations
solver.solve(10000)

print("Comparing CURRENT strategy vs AVERAGE strategy for OOP at root:\n")

# Get cumulative regret and strategy
cum_regret = solver._cumulative_regret.reshape((solver.num_infosets, 3, solver.num_deals))
cum_strat = solver._cumulative_strategy.reshape((solver.num_infosets, 3, solver.num_deals))
current_strat = solver._strategy.reshape((solver.num_infosets, 3, solver.num_deals))

print("Hand         | Cur Check | Cur Bet | Avg Check | Avg Bet")
print("-" * 60)

for hand in solver.oop_hands[:10]:
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

    # Average strategy (cumulative)
    avg_s_raw = cp.asnumpy(cum_strat[h, :, matching_array].sum(axis=0))
    total = avg_s_raw.sum()
    if total > 0:
        avg_s = avg_s_raw / total
    else:
        avg_s = np.array([0.5, 0.5, 0])

    print(f"{hand_str:12s} | {cur_s[0]*100:7.1f}%  | {cur_s[1]*100:7.1f}% | {avg_s[0]*100:7.1f}%  | {avg_s[1]*100:7.1f}%")

# Check regret values
print("\n\nCumulative regret for first few hands:")
for hand in solver.oop_hands[:5]:
    hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
    hand_idx = solver.oop_hand_to_idx[hand]
    key = (0, hand_idx, 0)
    h = solver.infoset_key_to_idx[key]

    matching_deals = []
    for d in range(solver.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        oop_hand = tuple(sorted(oop_hand, reverse=True))
        if oop_hand == hand:
            matching_deals.append(d)

    matching_array = cp.array(matching_deals)
    regrets = cp.asnumpy(cum_regret[h, :, matching_array].sum(axis=0))
    print(f"{hand_str}: Check={regrets[0]:.2f}, Bet={regrets[1]:.2f}")
