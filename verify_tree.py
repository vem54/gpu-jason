"""Verify game tree structure."""

from gpu_poker_cfr.games.custom_river_toy import make_aa_kk_vs_ak_aq_game
from gpu_poker_cfr.solvers.gpu_custom_river import GPUCustomRiverCFR

game = make_aa_kk_vs_ak_aq_game()
solver = GPUCustomRiverCFR(game)

print("Game Setup:")
print(f"  Pot: {game.pot_size}")
print(f"  Stack: {game.stack_size}")
print()

print("Game Tree Structure:")
print("-" * 60)
for i, node in enumerate(solver.nodes):
    node_type = ["FOLD", "SHOWDOWN", "DECISION"][node['type']]
    player = node.get('player', '-')
    pot = node.get('pot', '-')
    actions = node.get('actions', [])
    fold_player = node.get('fold_player', '-')

    print(f"Node {i}: {node_type}")
    print(f"  pot={pot}, player={player}")
    if actions:
        print(f"  actions={actions}")
    if fold_player != '-':
        print(f"  fold_player={fold_player}")

    # Show children
    children = solver.node_child_idx[i]
    valid_children = [c for c in children if c >= 0]
    if valid_children:
        print(f"  children={valid_children}")
    print()

print("\nAction Path Examples:")
print("Check -> Check -> Showdown: Pot=100, each player wins/loses 50")
print("Check -> All-in -> Fold: OOP loses 50, IP wins 50")
print("Check -> All-in -> Call: Pot=300, winner gets 150")
print("All-in -> Fold: IP loses 50, OOP wins 50")
print("All-in -> Call: Pot=300, winner gets 150")
