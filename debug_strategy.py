"""Debug the strategy output issue."""

import numpy as np
import cupy as cp
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR


def main():
    game = make_aa_kk_vs_ak_aq_game()
    solver = GPUCustomRiverCFR(game)

    print(f"Num deals: {solver.num_deals}")
    print(f"Num infosets: {solver.num_infosets}")
    print(f"OOP hands: {len(solver.oop_hands)}")
    print(f"IP hands: {len(solver.ip_hands)}")

    # Run a few iterations
    solver.solve(1000)

    # Debug: look at first OOP hand at root
    hand = solver.oop_hands[0]
    hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
    print(f"\nDebugging hand: {hand_str} = {hand}")

    # Get infoset index
    hand_idx = solver.oop_hand_to_idx.get(hand, -1)
    print(f"Hand idx: {hand_idx}")

    key = (0, hand_idx, 0)  # player=0, hand_idx, action_id=0 (root)
    h = solver.infoset_key_to_idx.get(key, -1)
    print(f"Infoset key: {key}")
    print(f"Infoset index h: {h}")

    # Find matching deals
    matching_deals = []
    for d in range(solver.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        oop_hand = tuple(sorted(oop_hand, reverse=True))

        if oop_hand == hand:
            matching_deals.append(d)

    print(f"Matching deals for {hand_str}: {matching_deals} (count: {len(matching_deals)})")

    # Look at cumulative strategy shape
    cum_strat = solver._cumulative_strategy.reshape((solver.num_infosets, 3, solver.num_deals))
    print(f"\nCumulative strategy shape: {cum_strat.shape}")

    # Get values for this infoset
    infoset_strat = cp.asnumpy(cum_strat[h, :, :])
    print(f"Infoset {h} strategy shape: {infoset_strat.shape}")
    print(f"Infoset {h} strategy sum over deals: {infoset_strat.sum(axis=1)}")
    print(f"Infoset {h} strategy total sum: {infoset_strat.sum()}")

    # Get values for matching deals only
    matching_array = cp.array(matching_deals)
    strat_matching = cp.asnumpy(cum_strat[h, :, matching_array])
    print(f"\nMatching deals strategy shape: {strat_matching.shape}")
    print(f"Matching deals strategy values:\n{strat_matching}")
    print(f"Sum per action: {strat_matching.sum(axis=1)}")
    print(f"Total sum: {strat_matching.sum()}")

    # Look at regret values
    cum_regret = solver._cumulative_regret.reshape((solver.num_infosets, 3, solver.num_deals))
    regret_values = cp.asnumpy(cum_regret[h, :, :])
    print(f"\nCumulative regret sum over deals: {regret_values.sum(axis=1)}")

    # Now look at what print_oop_strategy does
    print("\n" + "=" * 50)
    print("Testing get_strategy_for_hand:")
    strat = solver.get_strategy_for_hand(0, hand, action_id=0)
    print(f"Result: {strat}")


if __name__ == "__main__":
    main()
