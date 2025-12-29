"""
Regression test for Kuhn Poker Nash equilibrium convergence.

This test verifies that Vanilla CFR converges to a Nash equilibrium
for Kuhn Poker. Note: Kuhn Poker has infinitely many Nash equilibria,
parameterized by different values of α. CFR converges to ONE of them,
not necessarily the textbook "canonical" one (α = 1/3).

Key Nash equilibrium INVARIANTS that must hold:
1. Game value = -1/18 ≈ -0.0556 for P1 (same for all Nash equilibria)
2. Zero-sum: P1 value + P2 value = 0
3. Low exploitability (< 0.01 after sufficient iterations)
4. P1 structural constraint: K: bet ≈ 3 × J: bet (the "3α" relationship)
5. Dominated strategies not played:
   - J:b (P2 facing bet with J): always fold
   - K:b (P2 facing bet with K): always call
   - K:c (P2 after check with K): always bet
   - K:c,b (P1 facing check-bet with K): always call
   - J:c,b (P1 facing check-bet with J): always fold

Run with: pytest tests/test_kuhn_nash_regression.py -v -s
"""

import pytest
import numpy as np

from gpu_poker_cfr.games.kuhn import KuhnPoker
from gpu_poker_cfr.solvers.vanilla import VanillaCFR


# Game value at Nash equilibrium (same for all Nash equilibria)
NASH_GAME_VALUE = -1.0 / 18.0  # ≈ -0.0556 for P1

# Dominated strategies that should NEVER be played
DOMINATED_STRATEGIES = {
    # (infoset, action, expected_prob, is_upper_bound)
    # Upper bound = True means strategy should be < expected
    # Upper bound = False means strategy should be > expected
    "J:b": ("c", 0.05, True),       # J facing bet: never call (fold)
    "K:b": ("c", 0.95, False),      # K facing bet: always call
    "K:c": ("b", 0.95, False),      # K after check: always bet
    "K:c,b": ("c", 0.95, False),    # K facing check-bet: always call
    "J:c,b": ("c", 0.05, True),     # J facing check-bet: never call
}


def get_strategy_dict(solver, infoset_name):
    """Get strategy as dict {action_name: probability} for named infoset."""
    for h_idx in range(solver.matrices.num_infosets):
        name = solver.get_infoset_name(h_idx)
        if name == infoset_name:
            strategy = solver.get_strategy_for_infoset(h_idx)
            infoset = solver._tree.infosets[h_idx]
            return {infoset.actions[i].name: strategy[i] for i in range(len(infoset.actions))}
    return None


class TestKuhnNashRegression:
    """Regression tests for Kuhn Nash equilibrium convergence."""

    @pytest.fixture(scope="class")
    def converged_solver(self):
        """Run CFR for enough iterations to converge."""
        solver = VanillaCFR(KuhnPoker(), backend='numpy')
        # 100k iterations should be enough for convergence
        solver.solve(iterations=100000)
        return solver

    def test_game_value(self, converged_solver):
        """Game value should be -1/18 for P1 (same for all Nash equilibria)."""
        solver = converged_solver

        # Get expected value at root
        avg_strategy = solver.average_strategy
        avg_strategy_backend = solver.backend.dense_to_backend(avg_strategy)

        from gpu_poker_cfr.engine.ops import forward_reach_simple, backward_values_simple

        reach = forward_reach_simple(avg_strategy_backend, solver.matrices, solver.backend)
        values = backward_values_simple(avg_strategy_backend, reach, solver.matrices, solver.backend)
        values_np = solver.backend.asnumpy(values)

        p1_value = values_np[0, 0]
        p2_value = values_np[0, 1]

        # Check zero-sum
        assert abs(p1_value + p2_value) < 0.001, f"Not zero-sum: P1={p1_value}, P2={p2_value}"

        # Check game value (same for ALL Nash equilibria)
        assert abs(p1_value - NASH_GAME_VALUE) < 0.01, \
            f"Game value {p1_value} != Nash value {NASH_GAME_VALUE}"

    def test_exploitability(self, converged_solver):
        """Exploitability should be near zero after convergence."""
        solver = converged_solver
        expl = solver.exploitability()

        # After 100k iterations, exploitability should be < 0.01
        assert expl < 0.01, f"Exploitability {expl} too high after 100k iterations"

    def test_p1_alpha_relationship(self, converged_solver):
        """P1 K: bet should be approximately 3 × P1 J: bet (the 3α rule)."""
        solver = converged_solver

        p1_j_bet = get_strategy_dict(solver, "J:")["b"]
        p1_k_bet = get_strategy_dict(solver, "K:")["b"]

        # K bet should be approximately 3 × J bet
        # (For α = 1/3, this is capped at 1.0, but CFR finds different α)
        expected_ratio = 3.0
        actual_ratio = p1_k_bet / p1_j_bet if p1_j_bet > 0.01 else float('inf')

        assert 2.5 < actual_ratio < 3.5, \
            f"P1 K/J ratio {actual_ratio:.2f} not close to 3.0 (K={p1_k_bet:.4f}, J={p1_j_bet:.4f})"

    @pytest.mark.parametrize("infoset,action,threshold,is_upper", [
        # Dominated strategies - should be near 0 or 1
        ("J:b", "c", 0.05, True),       # J facing bet: never call
        ("K:b", "c", 0.95, False),      # K facing bet: always call
        ("K:c", "b", 0.95, False),      # K after check: always bet
        ("K:c,b", "c", 0.95, False),    # K facing check-bet: always call
        ("J:c,b", "c", 0.05, True),     # J facing check-bet: never call
    ])
    def test_dominated_strategy(self, converged_solver, infoset, action, threshold, is_upper):
        """Dominated strategies should converge to their pure values."""
        solver = converged_solver
        strategy = get_strategy_dict(solver, infoset)

        assert strategy is not None, f"Infoset {infoset} not found"
        assert action in strategy, f"Action {action} not in {infoset}"

        actual = strategy[action]

        if is_upper:
            assert actual < threshold, \
                f"{infoset} {action}: got {actual:.4f}, expected < {threshold}"
        else:
            assert actual > threshold, \
                f"{infoset} {action}: got {actual:.4f}, expected > {threshold}"

    def test_p2_queen_facing_bet_calls(self, converged_solver):
        """P2 with Queen facing bet should call some fraction (mixing)."""
        solver = converged_solver
        strategy = get_strategy_dict(solver, "Q:b")

        call_freq = strategy["c"]

        # Q:b should mix - call probability should be between 0.1 and 0.9
        assert 0.1 < call_freq < 0.9, \
            f"P2 Q:b call frequency {call_freq} not in mixing range [0.1, 0.9]"

    def test_p1_bluff_frequency_reasonable(self, converged_solver):
        """P1 with Jack should bluff a reasonable amount (mixing)."""
        solver = converged_solver
        strategy = get_strategy_dict(solver, "J:")

        bet_freq = strategy["b"]

        # J: should mix - bluff probability should be between 0.1 and 0.5
        assert 0.1 < bet_freq < 0.5, \
            f"P1 J: bluff frequency {bet_freq} not in reasonable range [0.1, 0.5]"


@pytest.mark.slow
def test_kuhn_nash_full():
    """Full Nash convergence test with detailed output."""
    print("\n" + "="*60)
    print("Kuhn Poker Nash Equilibrium Regression Test")
    print("="*60)

    solver = VanillaCFR(KuhnPoker(), backend='numpy')

    # Track convergence
    checkpoints = [1000, 10000, 50000, 100000]

    for target in checkpoints:
        solver.iterate(target - solver.iterations)
        expl = solver.exploitability()

        p1_j_bet = get_strategy_dict(solver, "J:")["b"]
        p1_k_bet = get_strategy_dict(solver, "K:")["b"]
        ratio = p1_k_bet / p1_j_bet if p1_j_bet > 0.01 else float('inf')

        print(f"\nIteration {target:6d}: expl={expl:.6f}, P1 J:bet={p1_j_bet:.4f}, K:bet={p1_k_bet:.4f}, K/J={ratio:.2f}")

    print("\n" + "-"*60)
    print("Final strategies (CFR converges to ONE of many valid Nash equilibria):")
    print("-"*60)

    for infoset in ["J:", "Q:", "K:", "J:b", "Q:b", "K:b", "J:c", "Q:c", "K:c", "J:c,b", "Q:c,b", "K:c,b"]:
        strategy = get_strategy_dict(solver, infoset)
        if strategy:
            strat_str = ", ".join(f"{a}={p:.4f}" for a, p in strategy.items())
            print(f"  {infoset:8s}: {strat_str}")

    # Compute game value
    from gpu_poker_cfr.engine.ops import forward_reach_simple, backward_values_simple
    avg_strategy = solver.average_strategy
    avg_strategy_backend = solver.backend.dense_to_backend(avg_strategy)
    reach = forward_reach_simple(avg_strategy_backend, solver.matrices, solver.backend)
    values = backward_values_simple(avg_strategy_backend, reach, solver.matrices, solver.backend)
    values_np = solver.backend.asnumpy(values)
    game_value = values_np[0, 0]

    print("\n" + "-"*60)
    print(f"Final exploitability: {solver.exploitability():.6f}")
    print(f"Game value (P1): {game_value:.6f}")
    print(f"Expected Nash game value: {NASH_GAME_VALUE:.6f}")
    print(f"P1 K/J ratio: {get_strategy_dict(solver, 'K:')['b'] / get_strategy_dict(solver, 'J:')['b']:.2f} (expected: 3.0)")
    print("="*60)

    # Verify key properties
    assert solver.exploitability() < 0.01, "Exploitability too high"
    assert abs(game_value - NASH_GAME_VALUE) < 0.01, f"Game value {game_value} != expected {NASH_GAME_VALUE}"


if __name__ == "__main__":
    test_kuhn_nash_full()
