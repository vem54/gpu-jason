"""Debug node-to-infoset mapping."""

import cupy as cp
import numpy as np
from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.games.cards import card_name
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

# Check node_h_p1 mapping for node 1 vs node 6
node_h_p1 = cp.asnumpy(solver.node_h_p1)

print("Checking node_h_p1 for first few deals:")
print(f"Node 1 (IP after OOP check) and Node 6 (IP after OOP all-in)")
print()

# Check for deals 0-5
for d in range(min(6, solver.num_deals)):
    oop_hand, ip_hand = game.get_deal(d)
    oop_str = f"{card_name(oop_hand[0])}{card_name(oop_hand[1])}"
    ip_str = f"{card_name(ip_hand[0])}{card_name(ip_hand[1])}"

    h_node1 = node_h_p1[1, d]  # IP's infoset at node 1
    h_node6 = node_h_p1[6, d]  # IP's infoset at node 6

    print(f"Deal {d}: OOP={oop_str}, IP={ip_str}")
    print(f"  Node 1 infoset: {h_node1}")
    print(f"  Node 6 infoset: {h_node6}")

# Check the difference between the two sets of infosets
print()
print("Are node 1 and node 6 using the same infoset indices?")
infosets_node1 = set(node_h_p1[1, :])
infosets_node6 = set(node_h_p1[6, :])
print(f"Unique infosets at node 1: {sorted(infosets_node1)[:10]}...")
print(f"Unique infosets at node 6: {sorted(infosets_node6)[:10]}...")
print(f"Overlap: {infosets_node1 & infosets_node6}")
