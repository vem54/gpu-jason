"""
Debug: Trace CPU EV computation at Node 1 for IP JJ.
"""

import numpy as np
from gpu_poker_cfr.games.turn_toy_game import (
    make_turn_toy_game, build_turn_tree,
    NODE_FOLD, NODE_SHOWDOWN, NODE_DECISION, NODE_CHANCE
)
from gpu_poker_cfr.games.hand_eval import evaluate_7cards
from gpu_poker_cfr.games.cards import card_name


class DebugCPUTurnCFR:
    """CPU CFR with detailed tracing."""

    def __init__(self, game):
        self.game = game
        self.nodes, self.node_children = build_turn_tree(game)
        self.num_nodes = len(self.nodes)
        self.max_actions = 2

        self._build_infosets()
        self._precompute_hand_values()

        self.cumulative_regret = {}
        self.cumulative_strategy = {}
        self.iterations = 0
        self.trace = False
        self.trace_deal = None

    def _build_infosets(self):
        self.oop_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.oop_range))
        self.ip_hands = list(set(tuple(sorted(h, reverse=True)) for h in self.game.ip_range))
        self.oop_hand_to_idx = {h: i for i, h in enumerate(self.oop_hands)}
        self.ip_hand_to_idx = {h: i for i, h in enumerate(self.ip_hands)}

    def _precompute_hand_values(self):
        self.hand_values = {}
        for deal_idx in range(self.game.num_deals):
            oop_hand, ip_hand = self.game.get_deal(deal_idx)
            valid_rivers = self.game.get_valid_rivers(deal_idx)
            for river in valid_rivers:
                board_with_river = list(self.game.board) + [river]
                h0 = np.array(list(oop_hand) + board_with_river, dtype=np.int32)
                h1 = np.array(list(ip_hand) + board_with_river, dtype=np.int32)
                self.hand_values[(deal_idx, river)] = (evaluate_7cards(h0), evaluate_7cards(h1))

    def get_strategy(self, player, hand, node_idx):
        key = (player, hand, node_idx)
        node = self.nodes[node_idx]
        n_act = len(node['actions'])
        regrets = self.cumulative_regret.get(key, [0.0] * n_act)
        pos_regrets = [max(0, r) for r in regrets]
        total = sum(pos_regrets)
        if total > 0:
            return [r / total for r in pos_regrets]
        else:
            return [1.0 / n_act] * n_act

    def cfr_iteration(self, deal_idx, node_idx, reach0, reach1, river=None, depth=0):
        node = self.nodes[node_idx]
        oop_hand, ip_hand = self.game.get_deal(deal_idx)
        oop_hand = tuple(sorted(oop_hand, reverse=True))
        ip_hand = tuple(sorted(ip_hand, reverse=True))

        indent = "  " * depth

        if node['type'] == NODE_FOLD:
            fp = node['fold_player']
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']
            if fp == 0:
                ev = (-inv_oop, inv_oop)
            else:
                ev = (inv_ip, -inv_ip)
            if self.trace and deal_idx == self.trace_deal:
                print(f"{indent}Node {node_idx} FOLD: ev0={ev[0]:.2f}, ev1={ev[1]:.2f}")
            return ev

        if node['type'] == NODE_SHOWDOWN:
            if river is None:
                raise ValueError("Showdown without river!")
            p0_val, p1_val = self.hand_values[(deal_idx, river)]
            inv_oop = node['invested_oop']
            inv_ip = node['invested_ip']
            if p0_val > p1_val:
                ev = (inv_ip, -inv_ip)
            elif p1_val > p0_val:
                ev = (-inv_oop, inv_oop)
            else:
                ev = (0.0, 0.0)
            if self.trace and deal_idx == self.trace_deal:
                print(f"{indent}Node {node_idx} SHOWDOWN (river={card_name(river)}): ev0={ev[0]:.2f}, ev1={ev[1]:.2f}")
            return ev

        if node['type'] == NODE_CHANCE:
            valid_rivers = self.game.get_valid_rivers(deal_idx)
            n_rivers = len(valid_rivers)
            river_weight = 1.0 / n_rivers

            ev0_sum = 0.0
            ev1_sum = 0.0

            if self.trace and deal_idx == self.trace_deal:
                print(f"{indent}Node {node_idx} CHANCE: {n_rivers} rivers")

            for r in valid_rivers:
                child_idx = self.node_children[node_idx][0]
                child_ev0, child_ev1 = self.cfr_iteration(
                    deal_idx, child_idx, reach0, reach1, river=r, depth=depth+1
                )
                ev0_sum += child_ev0 * river_weight
                ev1_sum += child_ev1 * river_weight

            if self.trace and deal_idx == self.trace_deal:
                print(f"{indent}  -> averaged: ev0={ev0_sum:.2f}, ev1={ev1_sum:.2f}")
            return (ev0_sum, ev1_sum)

        # Decision node
        player = node['player']
        hand = oop_hand if player == 0 else ip_hand
        n_act = len(node['actions'])
        strategy = self.get_strategy(player, hand, node_idx)

        if self.trace and deal_idx == self.trace_deal:
            player_name = "OOP" if player == 0 else "IP"
            print(f"{indent}Node {node_idx} DECISION ({player_name}, {card_name(hand[0])}{card_name(hand[1])}): strategy={strategy}")

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
                deal_idx, child_idx, new_reach0, new_reach1, river=river, depth=depth+1
            )
            child_evs.append((child_ev0, child_ev1))

            if self.trace and deal_idx == self.trace_deal:
                action = node['actions'][a]
                print(f"{indent}  Action '{action}' -> ev0={child_ev0:.2f}, ev1={child_ev1:.2f}")

        node_ev0 = sum(strategy[a] * child_evs[a][0] for a in range(n_act))
        node_ev1 = sum(strategy[a] * child_evs[a][1] for a in range(n_act))

        if self.trace and deal_idx == self.trace_deal:
            print(f"{indent}  Node EV: ev0={node_ev0:.2f}, ev1={node_ev1:.2f}")

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

            old_regret = self.cumulative_regret[key][a]
            self.cumulative_regret[key][a] += regret
            self.cumulative_regret[key][a] = max(0, self.cumulative_regret[key][a])

            if self.trace and deal_idx == self.trace_deal:
                action = node['actions'][a]
                print(f"{indent}  Regret[{action}]: {regret:.2f} (total: {old_regret:.2f} -> {self.cumulative_regret[key][a]:.2f})")

            self.cumulative_strategy[key][a] += strategy[a] * own_reach * t

        return (node_ev0, node_ev1)

    def iterate_with_trace(self, deal_idx):
        """Run one iteration for a specific deal with tracing."""
        self.trace = True
        self.trace_deal = deal_idx
        print(f"\n=== Tracing deal {deal_idx} ===")
        oop_hand, ip_hand = self.game.get_deal(deal_idx)
        print(f"OOP: {card_name(oop_hand[0])}{card_name(oop_hand[1])}")
        print(f"IP: {card_name(ip_hand[0])}{card_name(ip_hand[1])}")
        print()

        ev0, ev1 = self.cfr_iteration(deal_idx, 0, 1.0, 1.0)
        print(f"\nRoot EV: OOP={ev0:.2f}, IP={ev1:.2f}")

        self.trace = False


def main():
    game = make_turn_toy_game()
    solver = DebugCPUTurnCFR(game)

    # Find a deal with IP JJ
    jj_deal = None
    for d in range(game.num_deals):
        oop_hand, ip_hand = game.get_deal(d)
        if ip_hand[0] // 4 == 9 and ip_hand[1] // 4 == 9:
            jj_deal = d
            break

    if jj_deal:
        solver.iterate_with_trace(jj_deal)

        # Print Node 1 regrets
        ip_hand = tuple(sorted(game.get_deal(jj_deal)[1], reverse=True))
        key = (1, ip_hand, 1)
        print(f"\n=== Node 1 regrets for IP {card_name(ip_hand[0])}{card_name(ip_hand[1])} ===")
        print(f"Check: {solver.cumulative_regret.get(key, [0,0])[0]:.2f}")
        print(f"All-in: {solver.cumulative_regret.get(key, [0,0])[1]:.2f}")


if __name__ == '__main__':
    main()
