"""
Debug script to check EV scale mismatch hypothesis.

The hypothesis: Chance node EVs are SUMMED over ~44 rivers instead of AVERAGED,
causing Check path to have ~44x higher EV than All-in path.

Expected if bug exists:
- Node 1 Check child EV: ~1000+ (summed over 44 rivers)
- Node 1 All-in child EV: ~100 (single fold pot value)

Expected if code is correct:
- Node 1 Check child EV: ~25-50 (averaged per river)
- Node 1 All-in child EV: Similar scale
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import make_turn_toy_game, build_turn_tree, print_tree_structure
from gpu_poker_cfr.games.turn_toy_game import NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
from gpu_poker_cfr.games.hand_eval import evaluate_7cards


def debug_single_deal_ev():
    """
    Manually compute EVs for a single deal to check scale.
    """
    print("=" * 60)
    print("DEBUG: EV Scale Check for Single Deal")
    print("=" * 60)

    game = make_turn_toy_game()
    nodes, child_indices = build_turn_tree(game)

    print("\nTree structure:")
    print_tree_structure(nodes, child_indices)

    # Find Node 1 (IP turn after OOP check)
    # Node 0 is OOP turn root
    # Node 1 should be IP turn after OOP check (first child of Node 0)
    node_0_children = child_indices[0]
    print(f"\nNode 0 children: {node_0_children}")
    print(f"  Check child: Node {node_0_children[0]}")
    print(f"  All-in child: Node {node_0_children[1]}")

    node_1_idx = node_0_children[0]  # IP after OOP check
    node_1 = nodes[node_1_idx]
    node_1_children = child_indices[node_1_idx]
    print(f"\nNode 1 (IP turn after OOP check):")
    print(f"  Type: {node_1['type']} (2=DECISION)")
    print(f"  Player: {node_1['player']} (1=IP)")
    print(f"  Actions: {node_1['actions']}")
    print(f"  Children: {node_1_children}")

    # Get child nodes
    check_child_idx = node_1_children[0]
    allin_child_idx = node_1_children[1]

    check_child = nodes[check_child_idx]
    allin_child = nodes[allin_child_idx]

    print(f"\n  Check child (Node {check_child_idx}): type={check_child['type']} (3=CHANCE)")
    print(f"  All-in child (Node {allin_child_idx}): type={allin_child['type']} (2=DECISION)")

    # Now manually compute EVs for a specific deal
    # Use deal 0 (first valid deal)
    deal_idx = 0
    oop_hand, ip_hand = game.get_deal(deal_idx)

    from gpu_poker_cfr.games.cards import card_name
    print(f"\n--- Analyzing Deal {deal_idx} ---")
    print(f"OOP hand: {card_name(oop_hand[0])} {card_name(oop_hand[1])}")
    print(f"IP hand: {card_name(ip_hand[0])} {card_name(ip_hand[1])}")
    print(f"Board: {' '.join(card_name(c) for c in game.board)}")

    # Get valid rivers
    valid_rivers = game.get_valid_rivers(deal_idx)
    print(f"Valid rivers: {len(valid_rivers)}")

    # Compute EV for each path at Node 1

    # === PATH 1: IP Checks at Node 1 ===
    # Goes to Chance node, then river betting
    # For simplicity, assume both players check on river -> showdown

    print("\n=== PATH 1: IP Checks ===")
    print("(Assuming check-check on river -> showdown)")

    check_evs = []
    for river in valid_rivers:
        # 7-card hand values
        board_with_river = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)

        v0 = evaluate_7cards(h0)
        v1 = evaluate_7cards(h1)

        # Showdown EV for IP (player 1) at pot=100, invested_oop=50, invested_ip=50
        if v1 > v0:
            ev_ip = 50  # IP wins OOP's investment
        elif v0 > v1:
            ev_ip = -50  # IP loses their investment
        else:
            ev_ip = 0  # Chop

        check_evs.append(ev_ip)

    sum_check_ev = sum(check_evs)
    avg_check_ev = sum_check_ev / len(check_evs)
    print(f"Sum of check EVs: {sum_check_ev:.2f}")
    print(f"Avg of check EVs: {avg_check_ev:.2f}")
    print(f"Num rivers: {len(check_evs)}")

    # === PATH 2: IP All-in at Node 1 ===
    # OOP faces all-in: Fold or Call
    # If OOP folds: IP wins pot (50)
    # If OOP calls: Showdown after river

    print("\n=== PATH 2: IP All-in ===")

    # Subpath A: OOP Folds
    fold_ev_ip = 50  # IP wins OOP's 50 invested
    print(f"OOP Folds: IP EV = {fold_ev_ip}")

    # Subpath B: OOP Calls
    call_evs = []
    for river in valid_rivers:
        board_with_river = list(game.board) + [river]
        h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
        h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)

        v0 = evaluate_7cards(h0)
        v1 = evaluate_7cards(h1)

        # All-in showdown: pot = 100 + 100 + 100 = 300, each invested 150
        # invested_oop = 50 + 100 = 150, invested_ip = 50 + 100 = 150
        if v1 > v0:
            ev_ip = 150  # IP wins OOP's investment
        elif v0 > v1:
            ev_ip = -150  # IP loses their investment
        else:
            ev_ip = 0  # Chop

        call_evs.append(ev_ip)

    sum_call_ev = sum(call_evs)
    avg_call_ev = sum_call_ev / len(call_evs)
    print(f"OOP Calls:")
    print(f"  Sum of call EVs: {sum_call_ev:.2f}")
    print(f"  Avg of call EVs: {avg_call_ev:.2f}")

    # If OOP mixes (say 50/50)
    mixed_ev = 0.5 * fold_ev_ip + 0.5 * avg_call_ev
    print(f"  Mixed (50/50): {mixed_ev:.2f}")

    print("\n=== SCALE COMPARISON ===")
    print(f"If correctly averaged:")
    print(f"  Check path EV: {avg_check_ev:.2f}")
    print(f"  All-in path EV (OOP folds): {fold_ev_ip}")
    print(f"  All-in path EV (OOP calls): {avg_call_ev:.2f}")
    print(f"  Scale: Similar order of magnitude")

    print(f"\nIf Check path is SUMMED (bug):")
    print(f"  Check path EV: {sum_check_ev:.2f}")
    print(f"  All-in path EV (OOP folds): {fold_ev_ip}")
    print(f"  Scale: Check is ~{sum_check_ev/fold_ev_ip:.0f}x larger!")


def check_kernel_accumulation():
    """
    Check if the kernel correctly averages chance node EVs.
    """
    print("\n" + "=" * 60)
    print("CHECKING KERNEL CHANCE NODE LOGIC")
    print("=" * 60)

    # Read the kernel source and look for the accumulation
    kernel_file = "gpu_poker_cfr/solvers/gpu_turn_solver.py"

    with open(kernel_file, 'r') as f:
        content = f.read()

    # Find the chance EV accumulation section
    if "chance_ev0[c] += ev0[child] * river_weight" in content:
        print("FOUND: Chance EV accumulation with river_weight (looks correct)")
    else:
        print("WARNING: Could not find expected accumulation pattern")

    # Check for river_weight definition
    if "float river_weight = 1.0f / num_valid_rivers" in content:
        print("FOUND: river_weight = 1/num_valid_rivers (correct averaging)")
    else:
        print("WARNING: Could not find river_weight definition")

    # Check where the chance EVs are used
    if "ev0[i] = chance_ev0[c]" in content:
        print("FOUND: Chance node EV assignment (direct copy)")
        print("  This is correct IF chance_ev was properly averaged during accumulation")


if __name__ == '__main__':
    debug_single_deal_ev()
    check_kernel_accumulation()
