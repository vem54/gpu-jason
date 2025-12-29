"""
CFR+ (Counterfactual Regret Minimization Plus) Solver.

CFR+ improves on vanilla CFR with two key changes:
1. Regret flooring: Cumulative regrets are floored at 0 after each iteration
2. Linear weighting: Average strategy is weighted by iteration number t

These changes provide ~10x faster convergence and are used in Libratus/Pluribus.

Reference: Tammelin et al., "Solving Large Imperfect Information Games Using CFR+"
"""

import numpy as np
from typing import Literal

from gpu_poker_cfr.games.base import Game
from gpu_poker_cfr.matrix.builder import build_game_matrices, GameMatrices
from gpu_poker_cfr.engine.backend import get_backend, Backend
from gpu_poker_cfr.engine.ops import (
    forward_reach_simple,
    backward_values_simple,
    compute_counterfactual_values,
    compute_infoset_cf_values,
    compute_instant_regret,
    regret_match,
    REACH_P1,
    REACH_P2,
    REACH_CHANCE,
)


class CFRPlus:
    """
    CFR+ solver using matrix operations.

    Key differences from Vanilla CFR:
    1. Regrets are floored at 0 after each iteration (not just during regret matching)
    2. Average strategy is weighted by iteration number t

    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """

    def __init__(
        self,
        game: Game,
        backend: Literal['numpy', 'cupy'] = 'numpy'
    ):
        """
        Initialize the CFR+ solver.

        Args:
            game: Game to solve
            backend: 'numpy' for CPU or 'cupy' for GPU
        """
        self.game = game
        self.backend = get_backend(backend)

        # Build game tree and matrices
        tree = game.build_tree()
        self.matrices = build_game_matrices(tree)

        # Store tree for infoset-player mapping
        self._tree = tree

        # Build infoset to player mapping
        self._infoset_player = self._build_infoset_player_mapping()

        # Initialize accumulators
        # In CFR+, cumulative regrets are floored at 0
        self._cumulative_regret = self.backend.zeros(self.matrices.num_infoset_actions)
        self._cumulative_strategy = self.backend.zeros(self.matrices.num_infoset_actions)

        # Iteration counter (starts at 1 for weighting)
        self.iterations = 0

    def _build_infoset_player_mapping(self) -> np.ndarray:
        """Build mapping from infoset index to player index."""
        infoset_player = np.zeros(self.matrices.num_infosets, dtype=np.int32)
        for i, infoset in enumerate(self._tree.infosets):
            infoset_player[i] = infoset.player.value
        return infoset_player

    def iterate(self, num_iterations: int = 1) -> None:
        """
        Run CFR+ iterations.

        Args:
            num_iterations: Number of iterations to run
        """
        for _ in range(num_iterations):
            self._single_iteration()
            self.iterations += 1

    def _single_iteration(self) -> None:
        """Run a single CFR+ iteration."""
        # Current iteration number (1-indexed for weighting)
        t = self.iterations + 1

        # Get current strategy from regret matching
        strategy = regret_match(self._cumulative_regret, self.matrices, self.backend)

        # Compute reach probabilities (shape: num_nodes x 3)
        reach = forward_reach_simple(strategy, self.matrices, self.backend)

        # For each player, compute regrets
        for player in range(2):
            cf_values = compute_counterfactual_values(
                strategy, reach, self.matrices, self.backend, player
            )
            infoset_cf = compute_infoset_cf_values(cf_values, strategy, self.matrices, self.backend)
            instant_regret = compute_instant_regret(cf_values, infoset_cf, self.matrices, self.backend)

            # Only update regrets for this player's infosets
            instant_regret_np = self.backend.asnumpy(instant_regret)
            cumulative_np = self.backend.asnumpy(self._cumulative_regret)

            for h_idx in range(self.matrices.num_infosets):
                if self._infoset_player[h_idx] == player:
                    start = self.matrices.infoset_action_offsets[h_idx]
                    end = self.matrices.infoset_action_offsets[h_idx + 1]

                    # CFR+ key change #1: Add regret then floor at 0
                    cumulative_np[start:end] = np.maximum(
                        cumulative_np[start:end] + instant_regret_np[start:end],
                        0.0
                    )

            self._cumulative_regret = self.backend.dense_to_backend(cumulative_np)

        # Update cumulative strategy with linear weighting
        # CFR+ key change #2: Weight by iteration number t
        strategy_np = self.backend.asnumpy(strategy)
        reach_np = self.backend.asnumpy(reach)
        cumulative_strategy_np = self.backend.asnumpy(self._cumulative_strategy)

        for h_idx in range(self.matrices.num_infosets):
            player = self._infoset_player[h_idx]
            infoset = self._tree.infosets[h_idx]

            # Get player's reach column
            player_reach_col = REACH_P1 if player == 0 else REACH_P2

            # Compute average reach for this player at this infoset
            infoset_reach = 0.0
            count = 0
            for node_id in infoset.node_ids:
                infoset_reach += reach_np[node_id, player_reach_col]
                count += 1
            if count > 0:
                infoset_reach /= count

            start = self.matrices.infoset_action_offsets[h_idx]
            end = self.matrices.infoset_action_offsets[h_idx + 1]

            # CFR+ key change #2: Weight by iteration number t
            cumulative_strategy_np[start:end] += t * infoset_reach * strategy_np[start:end]

        self._cumulative_strategy = self.backend.dense_to_backend(cumulative_strategy_np)

    def solve(self, iterations: int = 1000) -> None:
        """
        Solve the game by running CFR+ iterations.

        Args:
            iterations: Number of iterations to run
        """
        self.iterate(iterations)

    @property
    def current_strategy(self) -> np.ndarray:
        """Get current strategy from regret matching."""
        strategy = regret_match(self._cumulative_regret, self.matrices, self.backend)
        return self.backend.asnumpy(strategy)

    @property
    def average_strategy(self) -> np.ndarray:
        """Get average strategy (converges to Nash equilibrium)."""
        cumulative = self.backend.asnumpy(self._cumulative_strategy)
        avg_strategy = np.zeros_like(cumulative)

        for h_idx in range(self.matrices.num_infosets):
            start = self.matrices.infoset_action_offsets[h_idx]
            end = self.matrices.infoset_action_offsets[h_idx + 1]

            total = cumulative[start:end].sum()
            if total > 0:
                avg_strategy[start:end] = cumulative[start:end] / total
            else:
                # Uniform if no strategy accumulated
                num_actions = end - start
                avg_strategy[start:end] = 1.0 / num_actions

        return avg_strategy

    def get_strategy_for_infoset(self, infoset_idx: int) -> np.ndarray:
        """Get average strategy for a specific infoset."""
        start = self.matrices.infoset_action_offsets[infoset_idx]
        end = self.matrices.infoset_action_offsets[infoset_idx + 1]
        return self.average_strategy[start:end]

    def get_infoset_name(self, infoset_idx: int) -> str:
        """Get human-readable name for an infoset."""
        return self._tree.infosets[infoset_idx].key

    def print_strategy(self) -> None:
        """Print the average strategy for all infosets."""
        avg_strategy = self.average_strategy

        print(f"\nCFR+ Average Strategy after {self.iterations} iterations:")
        print("-" * 50)

        for h_idx in range(self.matrices.num_infosets):
            infoset = self._tree.infosets[h_idx]
            player = self._infoset_player[h_idx]

            start = self.matrices.infoset_action_offsets[h_idx]
            end = self.matrices.infoset_action_offsets[h_idx + 1]

            probs = avg_strategy[start:end]
            action_strs = []
            for a_idx, action in enumerate(infoset.actions):
                action_strs.append(f"{action.name}={probs[a_idx]:.3f}")

            print(f"P{player+1} [{infoset.key}]: {', '.join(action_strs)}")

    def exploitability(self) -> float:
        """
        Compute exploitability of the average strategy.

        Exploitability measures how far the strategy is from Nash equilibrium.
        Returns the sum of best response values for both players.
        """
        avg_strategy = self.average_strategy
        avg_strategy_backend = self.backend.dense_to_backend(avg_strategy)

        total_exploitability = 0.0

        for player in range(2):
            br_value = self._best_response_value(avg_strategy_backend, player)
            total_exploitability += br_value

        return total_exploitability

    def _best_response_value(self, opponent_strategy: np.ndarray, player: int) -> float:
        """
        Compute the best response value for a player.

        This is the expected value when the player plays optimally against
        the opponent's fixed strategy.
        """
        reach = forward_reach_simple(opponent_strategy, self.matrices, self.backend)
        values = backward_values_simple(opponent_strategy, reach, self.matrices, self.backend)

        values_np = self.backend.asnumpy(values)
        reach_np = self.backend.asnumpy(reach)

        opp_col = REACH_P2 if player == 0 else REACH_P1

        br_strategy = np.copy(self.backend.asnumpy(opponent_strategy))

        # For player's infosets, choose the action with highest CF value
        for h_idx in range(self.matrices.num_infosets):
            if self._infoset_player[h_idx] == player:
                start = self.matrices.infoset_action_offsets[h_idx]
                end = self.matrices.infoset_action_offsets[h_idx + 1]

                infoset = self._tree.infosets[h_idx]
                action_values = []

                for a_idx, action in enumerate(infoset.actions):
                    q_idx = start + a_idx
                    M_QV_row = self.matrices.M_QV.getrow(q_idx)

                    action_value = 0.0
                    for node_id in M_QV_row.indices:
                        parent_id = self.matrices.parent_ids[node_id]
                        if parent_id >= 0:
                            opp_reach = reach_np[parent_id, opp_col]
                            chance_reach = reach_np[parent_id, REACH_CHANCE]
                            pi_minus_i = opp_reach * chance_reach
                            action_value += pi_minus_i * values_np[node_id, player]
                    action_values.append(action_value)

                best_action = np.argmax(action_values)
                br_strategy[start:end] = 0.0
                br_strategy[start + best_action] = 1.0

        br_strategy_backend = self.backend.dense_to_backend(br_strategy)
        br_reach = forward_reach_simple(br_strategy_backend, self.matrices, self.backend)
        br_values = backward_values_simple(br_strategy_backend, br_reach, self.matrices, self.backend)

        br_values_np = self.backend.asnumpy(br_values)
        return br_values_np[0, player]
