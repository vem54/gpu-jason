"""
Run the custom toy game:
- Board: As Ks Qs Th Jh
- OOP: AA, KK
- IP: AK, AQ
- Pot: 100, Stack: 100
- Actions: Check, All-in, Fold, Call
"""

import time
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR


def main():
    print("=" * 70)
    print("Custom River Toy Game Solver")
    print("=" * 70)

    # Create the game
    game = make_aa_kk_vs_ak_aq_game()

    print("\nGame Setup:")
    print(f"  Board: {' '.join(card_name(c) for c in game.board)}")
    print(f"  Pot: {game.pot_size}")
    print(f"  Stack: {game.stack_size}")
    print(f"  OOP Range: AA, KK ({len(game.oop_range)} combos)")
    print(f"  IP Range: AK, AQ ({len(game.ip_range)} combos)")
    print(f"  Total Deals: {game.num_deals}")

    print("\nValid Deals (first 10):")
    for i in range(min(10, game.num_deals)):
        print(f"  {game.describe_deal(i)}")

    # Create solver
    print("\nInitializing GPU solver...")
    solver = GPUCustomRiverCFR(game)
    print(f"  Nodes: {solver.num_nodes}")
    print(f"  Infosets: {solver.num_infosets}")

    # Solve
    iterations = 50000
    print(f"\nRunning {iterations} iterations...")
    start = time.time()
    solver.solve(iterations)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s ({iterations/elapsed:.1f} iter/s)")

    # Print strategy
    print("\n")
    solver.print_oop_strategy()

    # Also print IP strategy
    print("\n" + "=" * 60)
    print("IP Strategy (In Position Player)")
    print("=" * 60)

    # Node 1: After OOP Check - IP to act
    print("\n--- After OOP Check (IP to act) ---")
    print("Hand         | Check    | All-in")
    print("-" * 40)

    for hand in solver.ip_hands:
        hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
        strat = solver.get_strategy_for_hand(1, hand, action_id=1)
        check = strat.get('Check', 0) * 100
        allin = strat.get('All-in', 0) * 100
        print(f"{hand_str:12s} | {check:6.1f}%  | {allin:6.1f}%")

    # Node 6: After OOP All-in - IP to act
    print("\n--- After OOP All-in (IP to act) ---")
    print("Hand         | Fold     | Call")
    print("-" * 40)

    for hand in solver.ip_hands:
        hand_str = f"{card_name(hand[0])}{card_name(hand[1])}"
        strat = solver.get_strategy_for_hand(1, hand, action_id=6)
        fold = strat.get('Fold', 0) * 100
        call = strat.get('Call', 0) * 100
        print(f"{hand_str:12s} | {fold:6.1f}%  | {call:6.1f}%")


if __name__ == "__main__":
    main()
