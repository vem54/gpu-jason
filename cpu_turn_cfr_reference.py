"""
CPU Reference Implementation of Turn CFR.

This is a clean, simple implementation to validate against the GPU solver.
We'll run a few iterations and compare regrets/strategies.
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import (
    make_turn_toy_game, build_turn_tree,
    NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
)
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


class CPUTurnCFR:
    """Simple CPU reference implementation of turn CFR."""

    def __init__(self, game):
        self.game = game
        self.nodes, self.node_children = build_turn_tree(game)
        self.num_nodes = len(self.nodes)
        self.max_actions = 2

        # Build infosets
        self._build_infosets()

        # Precompute hand values
        self._precompute_hand_values()

        # Regrets and strategies
        self.cumulative_regret = {}  # (player, hand, node) -> [regrets]
        self.cumulative_strategy = {}

        self.iterations = 0

    def _build_infosets(self):
        """Build hand to index mappings."""
        self.oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.oop_range))
        self.ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.ip_range))

        self.oop_hand_to_idx = {h: i for i, h in enumerate(self.oop_hands)}
        self.ip_hand_to_idx = {h: i for i, h in enumerate(self.ip_hands)}

    def _precompute_hand_values(self):
        """Precompute 7-card hand values for all (deal, river) combinations."""
        self.hand_values = {}  # (deal_idx, river) -> (p0_val, p1_val)

        for deal_idx in range(self.game.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            valid_rivers = self.game.get_valid_rivers(deal_idx)

            for river in valid_rivers:
                board_with_river = list(self.game.board) + [river]
                h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
                h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)

                self.hand_values[(deal_idx, river)] = (evaluate_7cards(h0), evaluate_7cards(h1))

    def get_strategy(self, player, hand, node_idx):
        """Get current strategy from regrets."""
        # Use cached strategy if available (computed at start of iteration)
        key = (player, hand, node_idx)
        if hasattr(self, '_iteration_strategy') and key in self._iteration_strategy:
            return self._iteration_strategy[key]

        node = self.nodes[node_idx]
        n_act = len(node['actions'])

        regrets = self.cumulative_regret.get(key, [0.0] * n_act)
        pos_regrets = [max(0, r) for r in regrets]
        total = sum(pos_regrets)

        if total > 0:
            return [r / total for r in pos_regrets]
        else:
            return [1.0 / n_act] * n_act

    def _compute_all_strategies(self):
        """Compute strategies for all infosets at the start of iteration."""
        self._iteration_strategy = {}

        for node_idx, node in enumerate(self.nodes):
            if node['type'] != NODE_DECISION:
                continue

            player = node['player']
            hands = self.oop_hands if player == 0 else self.ip_hands
            n_act = len(node['actions'])

            for hand in hands:
                key = (player, hand, node_idx)
                regrets = self.cumulative_regret.get(key, [0.0] * n_act)
                pos_regrets = [max(0, r) for r in regrets]
                total = sum(pos_regrets)

                if total > 0:
                    self._iteration_strategy[key] = [r / total for r in pos_regrets]
                else:
                    self._iteration_strategy[key] = [1.0 / n_act] * n_act

    def cfr_iteration(self, deal_idx, node_idx, reach0, reach1, river=None):
        """
        Run one CFR iteration for a single deal.

        Returns: (ev0, ev1) - expected values for each player
        """
        node = self.nodes[node_idx]
        oop_hand, ip_hand = self.game.get_deal(deal_idx)
        oop_hand = tuple(sorted(oop_hand, reverse=True))
        ip_hand = tuple(sorted(ip_hand, reverse=True))

        # Terminal nodes
        if node['type'] == NODE_FOLD:
            fp = node['fold_player']
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']
            if fp == 0:
                return (-inv_oop, inv_oop)
            else:
                return (inv_ip, -inv_ip)

        if node['type'] == NODE_SHOWDOWN:
            if river is None:
                raise ValueError("Showdown without river!")

            p0_val, p1_val = self.hand_values[(deal_idx, river)]
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']

            if p0_val > p1_val:
                return (inv_ip, -inv_ip)
            elif p1_val > p0_val:
                return (-inv_oop, inv_oop)
            else:
                return (0.0, 0.0)

        # Chance node
        if node['type'] == NODE_CHANCE:
            valid_rivers = self.game.get_valid_rivers(deal_idx)
            n_rivers = len(valid_rivers)
            river_weight = 1.0 / n_rivers

            ev0_sum = 0.0
            ev1_sum = 0.0

            for r in valid_rivers:
                child_idx = self.node_children[node_idx][0]
                child_ev0, child_ev1 = self.cfr_iteration(
                    deal_idx, child_idx, reach0, reach1, river=r
                )
                ev0_sum += child_ev0 * river_weight
                ev1_sum += child_ev1 * river_weight

            return (ev0_sum, ev1_sum)

        # Decision node
        player = node['player']
        hand = oop_hand if player == 0 else ip_hand
        n_act = len(node['actions'])

        strategy = self.get_strategy(player, hand, node_idx)

        # Compute child EVs
        child_evs = []
        for a in range(n_act):
            child_idx = self.node_children[node_idx][a]
            if child_idx < 0:
                child_evs.append((0.0, 0.0))
                continue

            if player == 0:
                new_reach0 = reach0 * strategy[a]
                new_reach1 = reach1
            else:
                new_reach0 = reach0
                new_reach1 = reach1 * strategy[a]

            child_ev0, child_ev1 = self.cfr_iteration(
                deal_idx, child_idx, new_reach0, new_reach1, river=river
            )
            child_evs.append((child_ev0, child_ev1))

        # Compute node EV
        node_ev0 = sum(strategy[a] * child_evs[a][0] for a in range(n_act))
        node_ev1 = sum(strategy[a] * child_evs[a][1] for a in range(n_act))

        # Update regrets
        key = (player, hand, node_idx)
        if key not in self.cumulative_regret:
            self.cumulative_regret[key] = [0.0] * n_act
        if key not in self.cumulative_strategy:
            self.cumulative_strategy[key] = [0.0] * n_act

        opp_reach = reach1 if player == 0 else reach0
        own_reach = reach0 if player == 0 else reach1
        t = self.iterations + 1

        for a in range(n_act):
            if player == 0:
                regret = (child_evs[a][0] - node_ev0) * opp_reach
            else:
                regret = (child_evs[a][1] - node_ev1) * opp_reach

            self.cumulative_regret[key][a] += regret
            # CFR+ floor
            self.cumulative_regret[key][a] = max(0, self.cumulative_regret[key][a])

            # Accumulate strategy
            self.cumulative_strategy[key][a] += strategy[a] * own_reach * t

        return (node_ev0, node_ev1)

    def iterate(self, n=1):
        """Run n iterations of CFR."""
        for _ in range(n):
            # Compute strategies for all infosets ONCE at start of iteration
            self._compute_all_strategies()

            for deal_idx in range(self.game.num_deals):
                self.cfr_iteration(deal_idx, 0, 1.0, 1.0)
            self.iterations += 1

            # Clear cached strategies after iteration
            self._iteration_strategy = {}

    def get_average_strategy(self, player, hand, node_idx):
        """Get average strategy."""
        key = (player, hand, node_idx)
        node = self.nodes[node_idx]
        n_act = len(node['actions'])

        strat_sum = self.cumulative_strategy.get(key, [0.0] * n_act)
        total = sum(strat_sum)

        if total > 0:
            return [s / total for s in strat_sum]
        else:
            return [1.0 / n_act] * n_act

    def get_aggregate_strategy(self, node_idx):
        """Get deal-weighted aggregate strategy at a node."""
        node = self.nodes[node_idx]
        if node['type'] != NODE_DECISION:
            return {}

        player = node['player']
        hands = self.oop_hands if player == 0 else self.ip_hands
        actions = node['actions']
        n_act = len(actions)

        # Count deals per hand
        hand_deals = {}
        for d in range(self.game.num_deals):
            oop_hand, ip_hand = self.game.get_deal(d)
            hand = tuple(sorted(oop_hand if player == 0 else ip_hand, reverse=True))
            hand_deals[hand] = hand_deals.get(hand, 0) + 1

        totals = [0.0] * n_act
        total_deals = 0

        for hand in hands:
            deals = hand_deals.get(hand, 0)
            if deals == 0:
                continue

            strat = self.get_average_strategy(player, hand, node_idx)
            for a in range(n_act):
                totals[a] += strat[a] * deals
            total_deals += deals

        if total_deals > 0:
            return {actions[a]: totals[a] / total_deals for a in range(n_act)}
        return {}


def compare_with_gpu():
    """Compare CPU and GPU solver results."""
    print("=" * 60)
    print("CPU vs GPU Turn CFR Comparison")
    print("=" * 60)

    # Create game
    game = make_turn_toy_game()

    # Run CPU solver
    print("\nRunning CPU CFR (100 iterations)...")
    cpu_solver = CPUTurnCFR(game)
    cpu_solver.iterate(100)

    print("\nCPU Results (100 iterations):")
    for node_idx in [0, 1, 2]:
        node = cpu_solver.nodes[node_idx]
        if node['type'] == NODE_DECISION:
            player = 'OOP' if node['player'] == 0 else 'IP'
            street = node.get('street', 'unknown')
            agg = cpu_solver.get_aggregate_strategy(node_idx)
            print(f"  Node {node_idx} ({player} {street}): {agg}")

    # Run GPU solver
    print("\nRunning GPU CFR (100 iterations)...")
    from gpu_poker_cfr.solvers.gpu_turn_solver import GPUTurnSolver
    gpu_solver = GPUTurnSolver(game)
    gpu_solver.solve(iterations=100)

    print("\nGPU Results (100 iterations):")
    for node_idx in [0, 1, 2]:
        agg = gpu_solver.get_aggregate_strategy(node_idx)
        if agg:
            node = gpu_solver.nodes[node_idx]
            player = 'OOP' if node['player'] == 0 else 'IP'
            street = node.get('street', 'unknown')
            print(f"  Node {node_idx} ({player} {street}): {agg}")


def debug_single_deal_cfr():
    """Debug CFR for a single deal to check EV scale."""
    print("\n" + "=" * 60)
    print("DEBUG: Single Deal CFR Trace")
    print("=" * 60)

    game = make_turn_toy_game()
    solver = CPUTurnCFR(game)

    # Get first deal
    deal_idx = 0
    oop_hand, ip_hand = game.get_deal(deal_idx)
    print(f"\nDeal {deal_idx}:")
    print(f"  OOP: {card_name(oop_hand[0])} {card_name(oop_hand[1])}")
    print(f"  IP: {card_name(ip_hand[0])} {card_name(ip_hand[1])}")

    # Run one iteration and trace
    print("\n--- Iteration 1 ---")
    ev0, ev1 = solver.cfr_iteration(deal_idx, 0, 1.0, 1.0)
    print(f"Root EV: OOP={ev0:.2f}, IP={ev1:.2f}")

    # Print regrets at key nodes
    oop_hand = tuple(sorted(oop_hand, reverse=True))
    ip_hand = tuple(sorted(ip_hand, reverse=True))

    # Node 1: IP turn after OOP check
    key = (1, ip_hand, 1)
    if key in solver.cumulative_regret:
        print(f"\nNode 1 (IP turn after OOP check) regrets for {card_name(ip_hand[0])}{card_name(ip_hand[1])}:")
        print(f"  Check: {solver.cumulative_regret[key][0]:.2f}")
        print(f"  All-in: {solver.cumulative_regret[key][1]:.2f}")


if __name__ == '__main__':
    debug_single_deal_cfr()
    print("\n")
    compare_with_gpu()
