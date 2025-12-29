"""Compare our GPU solver against the Python poker solver."""

import sys
sys.path.insert(0, 'python_poker_solver')

from treys import Card, Evaluator
from poker.hand import Range
from copy import deepcopy
import numpy as np

# Our solver
from gpu_poker_cfr.games.river_game import make_test_river_game
from gpu_poker_cfr.solvers.gpu_river_solver import GPURiverSolver

###############################################################################
# Setup the python_poker_solver with our game configuration
###############################################################################

evaluator = Evaluator()

# Board: As Ks Qs Th 5c
BOARD = [
    Card.new('As'),
    Card.new('Ks'),
    Card.new('Qs'),
    Card.new('Th'),
    Card.new('5c'),
]

def build_range_from_str(range_str):
    """Build range list from string."""
    combos = []
    for combo in Range(range_str).combos:
        c1_str = str(combo.first)
        c2_str = str(combo.second)

        def suit_to_char(s):
            if s == chr(9824): return 's'
            if s == chr(9829): return 'h'
            if s == chr(9830): return 'd'
            if s == chr(9827): return 'c'
            return s

        c1 = c1_str[0] + suit_to_char(c1_str[1])
        c2 = c2_str[0] + suit_to_char(c2_str[1])
        combos.append((Card.new(c1), Card.new(c2)))
    return combos

# OOP: AA, KK, AK, AQ, AJ, AT (all combos, not just offsuit)
OOP_RANGE = build_range_from_str('AA KK AK AQ AJ AT')
# IP: AA, KK, JJ, 55, AK, AQ, JT (all combos)
IP_RANGE = build_range_from_str('AA KK JJ 55 AK AQ JT')

# Filter out combos that use board cards
def filter_range(combos, board):
    board_set = set(board)
    filtered = []
    for c1, c2 in combos:
        if c1 not in board_set and c2 not in board_set:
            filtered.append((c1, c2))
    return filtered

OOP_RANGE = filter_range(OOP_RANGE, BOARD)
IP_RANGE = filter_range(IP_RANGE, BOARD)

print("Board: As Ks Qs Th 5c")
print(f"OOP Range: {len(OOP_RANGE)} combos")
print(f"IP Range: {len(IP_RANGE)} combos")

###############################################################################
# Simple CFR for river check/all-in game
###############################################################################

class SimpleRiverCFR:
    """Simple CFR for river with check/all-in only."""

    def __init__(self, oop_range, ip_range, board, pot=100, stack=100):
        self.oop_range = oop_range
        self.ip_range = ip_range
        self.board = board
        self.pot = pot
        self.stack = stack

        # Build valid deals
        self.deals = []
        for oop_hand in oop_range:
            oop_set = set(oop_hand)
            for ip_hand in ip_range:
                ip_set = set(ip_hand)
                if not (oop_set & ip_set):  # No overlapping cards
                    self.deals.append((oop_hand, ip_hand))

        print(f"Valid deals: {len(self.deals)}")

        # Infosets: (player, hand_tuple, action_history)
        self.regrets = {}
        self.strategy_sum = {}

    def get_hand_value(self, hand):
        """Evaluate 7-card hand (lower is better in treys)."""
        return evaluator.evaluate(self.board, list(hand))

    def get_strategy(self, key, n_actions):
        """Get current strategy via regret matching."""
        if key not in self.regrets:
            self.regrets[key] = np.zeros(n_actions)
            self.strategy_sum[key] = np.zeros(n_actions)

        regrets = self.regrets[key]
        pos_regrets = np.maximum(regrets, 0)
        total = pos_regrets.sum()

        if total > 0:
            return pos_regrets / total
        else:
            return np.ones(n_actions) / n_actions

    def cfr(self, deal, history, reach_oop, reach_ip):
        """Run CFR for a single deal."""
        oop_hand, ip_hand = deal

        # Terminal states
        if history == 'xc':  # Check-check showdown
            oop_val = self.get_hand_value(oop_hand)
            ip_val = self.get_hand_value(ip_hand)
            half_pot = self.pot / 2
            if oop_val < ip_val:  # Lower is better
                return half_pot, -half_pot
            elif ip_val < oop_val:
                return -half_pot, half_pot
            else:
                return 0, 0

        if history == 'xaf':  # Check-allin-fold (OOP folds)
            return -self.pot/2, self.pot/2

        if history == 'xac':  # Check-allin-call showdown
            oop_val = self.get_hand_value(oop_hand)
            ip_val = self.get_hand_value(ip_hand)
            total_pot = self.pot + 2 * self.stack
            half_pot = total_pot / 2
            if oop_val < ip_val:
                return half_pot, -half_pot
            elif ip_val < oop_val:
                return -half_pot, half_pot
            else:
                return 0, 0

        if history == 'af':  # Allin-fold (IP folds)
            return self.pot/2, -self.pot/2

        if history == 'ac':  # Allin-call showdown
            oop_val = self.get_hand_value(oop_hand)
            ip_val = self.get_hand_value(ip_hand)
            total_pot = self.pot + 2 * self.stack
            half_pot = total_pot / 2
            if oop_val < ip_val:
                return half_pot, -half_pot
            elif ip_val < oop_val:
                return -half_pot, half_pot
            else:
                return 0, 0

        # Decision nodes
        if history == '':  # OOP to act at root
            key = (0, oop_hand, history)
            strat = self.get_strategy(key, 2)  # Check, All-in

            # Recurse
            ev_check = self.cfr(deal, 'x', reach_oop * strat[0], reach_ip)
            ev_allin = self.cfr(deal, 'a', reach_oop * strat[1], reach_ip)

            node_ev = (strat[0] * ev_check[0] + strat[1] * ev_allin[0],
                       strat[0] * ev_check[1] + strat[1] * ev_allin[1])

            # Update regrets (weighted by opponent reach)
            self.regrets[key][0] += reach_ip * (ev_check[0] - node_ev[0])
            self.regrets[key][1] += reach_ip * (ev_allin[0] - node_ev[0])

            # Accumulate strategy
            self.strategy_sum[key] += reach_oop * strat

            return node_ev

        if history == 'x':  # IP to act after check
            key = (1, ip_hand, history)
            strat = self.get_strategy(key, 2)  # Check, All-in

            ev_check = self.cfr(deal, 'xc', reach_oop, reach_ip * strat[0])
            ev_allin = self.cfr(deal, 'xa', reach_oop, reach_ip * strat[1])

            node_ev = (strat[0] * ev_check[0] + strat[1] * ev_allin[0],
                       strat[0] * ev_check[1] + strat[1] * ev_allin[1])

            self.regrets[key][0] += reach_oop * (ev_check[1] - node_ev[1])
            self.regrets[key][1] += reach_oop * (ev_allin[1] - node_ev[1])

            self.strategy_sum[key] += reach_ip * strat

            return node_ev

        if history == 'xa':  # OOP facing all-in
            key = (0, oop_hand, history)
            strat = self.get_strategy(key, 2)  # Fold, Call

            ev_fold = self.cfr(deal, 'xaf', reach_oop * strat[0], reach_ip)
            ev_call = self.cfr(deal, 'xac', reach_oop * strat[1], reach_ip)

            node_ev = (strat[0] * ev_fold[0] + strat[1] * ev_call[0],
                       strat[0] * ev_fold[1] + strat[1] * ev_call[1])

            self.regrets[key][0] += reach_ip * (ev_fold[0] - node_ev[0])
            self.regrets[key][1] += reach_ip * (ev_call[0] - node_ev[0])

            self.strategy_sum[key] += reach_oop * strat

            return node_ev

        if history == 'a':  # IP facing all-in
            key = (1, ip_hand, history)
            strat = self.get_strategy(key, 2)  # Fold, Call

            ev_fold = self.cfr(deal, 'af', reach_oop, reach_ip * strat[0])
            ev_call = self.cfr(deal, 'ac', reach_oop, reach_ip * strat[1])

            node_ev = (strat[0] * ev_fold[0] + strat[1] * ev_call[0],
                       strat[0] * ev_fold[1] + strat[1] * ev_call[1])

            self.regrets[key][0] += reach_oop * (ev_fold[1] - node_ev[1])
            self.regrets[key][1] += reach_oop * (ev_call[1] - node_ev[1])

            self.strategy_sum[key] += reach_ip * strat

            return node_ev

        raise ValueError(f"Unknown history: {history}")

    def train(self, iterations):
        for t in range(iterations):
            for deal in self.deals:
                self.cfr(deal, '', 1.0, 1.0)

            # CFR+ floor
            for key in self.regrets:
                self.regrets[key] = np.maximum(self.regrets[key], 0)

    def get_aggregate_strategy(self, player, history):
        """Get deal-weighted aggregate strategy."""
        totals = np.zeros(2)
        total_weight = 0

        for deal in self.deals:
            oop_hand, ip_hand = deal
            hand = oop_hand if player == 0 else ip_hand
            key = (player, hand, history)

            if key in self.strategy_sum:
                strat_sum = self.strategy_sum[key]
                total = strat_sum.sum()
                if total > 0:
                    strat = strat_sum / total
                else:
                    strat = np.array([0.5, 0.5])
                totals += strat
                total_weight += 1

        if total_weight > 0:
            return totals / total_weight
        return np.array([0.5, 0.5])


###############################################################################
# Run comparison
###############################################################################

print("\n" + "="*60)
print("Running Simple CFR (reference implementation)")
print("="*60)

ref_solver = SimpleRiverCFR(OOP_RANGE, IP_RANGE, BOARD)
ref_solver.train(1000)

print("\nReference CFR Results:")
oop_root = ref_solver.get_aggregate_strategy(0, '')
print(f"  OOP at Root: Check={oop_root[0]*100:.1f}%, All-in={oop_root[1]*100:.1f}%")

ip_after_check = ref_solver.get_aggregate_strategy(1, 'x')
print(f"  IP after Check: Check={ip_after_check[0]*100:.1f}%, All-in={ip_after_check[1]*100:.1f}%")

print("\n" + "="*60)
print("Running Our GPU Solver")
print("="*60)

game = make_test_river_game(bet_sizes=[], raise_sizes=[], all_in=True)
our_solver = GPURiverSolver(game)
our_solver.solve(1000)

oop_agg = our_solver.get_aggregate_strategy(0)
print(f"\nOur GPU Solver Results:")
print(f"  OOP at Root: {oop_agg}")

ip_agg = our_solver.get_aggregate_strategy(1)
print(f"  IP after Check: {ip_agg}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"OOP Check:   Reference={oop_root[0]*100:.1f}%  Ours={oop_agg.get('Check',0)*100:.1f}%")
print(f"IP All-in:   Reference={ip_after_check[1]*100:.1f}%  Ours={ip_agg.get('All-in',0)*100:.1f}%")
