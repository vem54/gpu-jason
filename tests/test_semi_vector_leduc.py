"""
Tests for Semi-Vector MCCFR on Leduc Poker.

Run with: pytest tests/test_semi_vector_leduc.py -v
"""

import pytest
import numpy as np

from gpu_poker_cfr.games.leduc import LeducPoker, JACK, QUEEN, KING, CARD_NAMES
from gpu_poker_cfr.solvers.semi_vector_leduc import (
    SemiVectorLeducMCCFR,
    card_to_rank,
    card_to_name,
    get_all_private_deals,
    get_remaining_cards,
    validate_private_deal,
    hand_rank,
    LEDUC_NUM_CARDS,
    LEDUC_NUM_PRIVATE_DEALS,
)
from gpu_poker_cfr.solvers.vanilla import VanillaCFR


class TestLeducConstants:
    """Step 1: Test Leduc constants and hand indexing."""

    def test_num_cards(self):
        """Should have 6 cards in deck."""
        assert LEDUC_NUM_CARDS == 6

    def test_num_private_deals(self):
        """Should have 30 private deals (6 * 5)."""
        assert LEDUC_NUM_PRIVATE_DEALS == 30
        assert len(get_all_private_deals()) == 30

    def test_card_to_rank(self):
        """Card indices should map to correct ranks."""
        # Cards 0,1 are Jacks
        assert card_to_rank(0) == JACK
        assert card_to_rank(1) == JACK
        # Cards 2,3 are Queens
        assert card_to_rank(2) == QUEEN
        assert card_to_rank(3) == QUEEN
        # Cards 4,5 are Kings
        assert card_to_rank(4) == KING
        assert card_to_rank(5) == KING

    def test_card_to_name(self):
        """Card indices should have correct names."""
        assert card_to_name(0) == "J0"
        assert card_to_name(1) == "J1"
        assert card_to_name(2) == "Q0"
        assert card_to_name(3) == "Q1"
        assert card_to_name(4) == "K0"
        assert card_to_name(5) == "K1"

    def test_no_duplicate_deals(self):
        """No deal should have same card for both players."""
        for hero, villain in get_all_private_deals():
            assert hero != villain

    def test_remaining_cards(self):
        """Should have 4 remaining cards after dealing."""
        for hero, villain in get_all_private_deals():
            remaining = get_remaining_cards(hero, villain)
            assert len(remaining) == 4
            assert hero not in remaining
            assert villain not in remaining

    def test_validate_deal_valid(self):
        """validate_private_deal accepts valid deals."""
        for hero, villain in get_all_private_deals():
            validate_private_deal(hero, villain)  # Should not raise

    def test_validate_deal_invalid(self):
        """validate_private_deal rejects invalid deals."""
        with pytest.raises(AssertionError):
            validate_private_deal(0, 0)  # Same card
        with pytest.raises(AssertionError):
            validate_private_deal(-1, 0)  # Invalid card
        with pytest.raises(AssertionError):
            validate_private_deal(0, 6)  # Invalid card


class TestHandRanking:
    """Test Leduc hand ranking."""

    def test_pair_beats_high_card(self):
        """Pair should beat high card regardless of rank."""
        # J pair vs K high card
        assert hand_rank(0, 0) > hand_rank(4, 2)  # JJ pair vs K with Q community
        assert hand_rank(0, 1) > hand_rank(4, 2)  # JJ pair vs K high

    def test_higher_pair_beats_lower_pair(self):
        """Higher pair should beat lower pair."""
        # K pair vs J pair (same community card)
        # K0 with K community vs J0 with J community
        assert hand_rank(4, 4) > hand_rank(0, 0)  # KK > JJ
        assert hand_rank(2, 2) > hand_rank(0, 0)  # QQ > JJ
        assert hand_rank(4, 4) > hand_rank(2, 2)  # KK > QQ

    def test_higher_card_wins_without_pair(self):
        """Higher card wins when no pair."""
        # K vs J, neither has pair
        assert hand_rank(4, 2) > hand_rank(0, 2)  # K vs J, Q community


class TestSemiVectorLeducBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Solver should initialize without error."""
        solver = SemiVectorLeducMCCFR(LeducPoker())
        assert solver.iterations == 0
        assert solver.num_private_deals == 30
        assert solver.max_actions == 3

    def test_single_iteration(self):
        """Should complete a single iteration without error."""
        solver = SemiVectorLeducMCCFR(LeducPoker())
        solver.iterate(1)
        assert solver.iterations == 1

    def test_multiple_iterations(self):
        """Should complete multiple iterations."""
        solver = SemiVectorLeducMCCFR(LeducPoker())
        solver.iterate(10)
        assert solver.iterations == 10

    def test_strategy_valid_distribution(self):
        """Strategy should be valid probability distribution."""
        solver = SemiVectorLeducMCCFR(LeducPoker())
        solver.solve(iterations=50)

        avg_strategy = solver.average_strategy

        for h_idx in range(solver.num_infosets):
            num_actions = solver._infoset_num_actions[h_idx]
            for deal_idx in range(solver.num_private_deals):
                probs = avg_strategy[h_idx, :num_actions, deal_idx]
                assert np.all(probs >= 0), f"Negative probability at h={h_idx}, deal={deal_idx}"
                assert np.isclose(probs.sum(), 1.0, atol=0.01), \
                    f"Probabilities don't sum to 1: {probs.sum()}"


class TestLeducConvergence:
    """Test convergence properties."""

    def test_regret_bounded_growth(self):
        """Cumulative regrets should not grow unboundedly."""
        solver = SemiVectorLeducMCCFR(LeducPoker())

        max_regrets = []
        for _ in range(5):
            solver.iterate(20)
            max_regret = np.max(np.abs(solver._cumulative_regret))
            max_regrets.append(max_regret)

        # Later regrets should not be exponentially larger
        growth_ratio = max_regrets[-1] / max(max_regrets[0], 0.001)
        assert growth_ratio < 100, f"Regret growing too fast: {max_regrets}"

    def test_strategy_changes_over_iterations(self):
        """Strategy should evolve from uniform to specialized."""
        solver = SemiVectorLeducMCCFR(LeducPoker())

        # Initial strategy should be uniform
        initial_strat = solver._get_all_strategies()

        solver.solve(iterations=100)

        # After iterations, strategy should differ from uniform
        final_strat = solver._get_all_strategies()

        # Check at least some infosets have non-uniform strategy
        max_diff = 0
        for h_idx in range(solver.num_infosets):
            num_actions = solver._infoset_num_actions[h_idx]
            uniform = 1.0 / num_actions

            for deal_idx in range(solver.num_private_deals):
                diff = np.max(np.abs(final_strat[h_idx, :num_actions, deal_idx] - uniform))
                max_diff = max(max_diff, diff)

        assert max_diff > 0.1, "Strategy should deviate from uniform after training"


class TestBoardEnumeration:
    """Test full board enumeration (no sampling)."""

    def test_all_boards_visited(self):
        """All community cards should be considered."""
        solver = SemiVectorLeducMCCFR(LeducPoker(), sample_boards=False)

        # Run one iteration and check regrets are updated
        solver.iterate(1)

        # Should have non-zero regrets for round 2 infosets
        round2_regrets = 0
        for h_idx in range(solver.num_infosets):
            if solver._infoset_round[h_idx] == 2:
                round2_regrets += np.abs(solver._cumulative_regret[h_idx]).sum()

        assert round2_regrets > 0, "Round 2 infosets should have regrets"


class TestBoardSampling:
    """Test board sampling mode."""

    def test_sampling_mode_runs(self):
        """Sampling mode should run without error."""
        solver = SemiVectorLeducMCCFR(
            LeducPoker(),
            sample_boards=True,
            num_board_samples=2
        )
        solver.solve(iterations=50)
        assert solver.iterations == 50

    def test_sampling_converges(self):
        """Sampling should still converge (with more variance)."""
        solver = SemiVectorLeducMCCFR(
            LeducPoker(),
            sample_boards=True,
            num_board_samples=2
        )
        solver.solve(iterations=500)

        expl = solver.exploitability()
        # Higher threshold due to sampling variance
        assert expl < 1.0, f"Sampling exploitability too high: {expl}"


def test_leduc_smoke():
    """Quick smoke test."""
    solver = SemiVectorLeducMCCFR(LeducPoker())
    solver.solve(iterations=50)

    assert solver.iterations == 50
    print(f"\nLeduc smoke test:")
    print(f"  Iterations: {solver.iterations}")
    print(f"  Infosets: {solver.num_infosets}")
    print(f"  Private deals: {solver.num_private_deals}")
    print(f"  Exploitability: {solver.exploitability():.4f}")


if __name__ == "__main__":
    test_leduc_smoke()
