"""
Vectorized CFR+ Solver.

This implementation removes Python loops over infosets, enabling efficient GPU execution.
All per-infoset operations are replaced with vectorized array/matrix operations.
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


class CFRPlusVectorized:
    """
    Vectorized CFR+ solver - no Python loops over infosets.

    Key optimizations over non-vectorized CFR+:
    1. Player masks for vectorized regret updates
    2. Sparse matrix operations for infoset reach computation
    3. All operations use backend arrays (GPU-friendly)
    """

    def __init__(
        self,
        game: Game,
        backend: Literal['numpy', 'cupy'] = 'numpy'
    ):
        """
        Initialize the vectorized CFR+ solver.

        Args:
            game: Game to solve
            backend: 'numpy' for CPU or 'cupy' for GPU
        """
        self.game = game
        self.backend = get_backend(backend)

        # Build game tree and matrices
        tree = game.build_tree()
        self.matrices = build_game_matrices(tree)

        # Store tree for reporting
        self._tree = tree

        # Convert matrices to backend format
        self._setup_backend_arrays()

        # Initialize accumulators (on backend)
        self._cumulative_regret = self.backend.zeros(self.matrices.num_infoset_actions)
        self._cumulative_strategy = self.backend.zeros(self.matrices.num_infoset_actions)

        # Iteration counter
        self.iterations = 0

    def _setup_backend_arrays(self):
        """Convert numpy arrays and scipy sparse matrices to backend format."""
        # Player masks for infoset-actions (boolean masks for each player)
        self._player_masks = [
            self.backend.dense_to_backend(
                (self.matrices.infoset_action_player == p).astype(np.float32)
            )
            for p in range(2)
        ]

        # Infoset player mapping expanded to infoset-actions
        self._infoset_action_player = self.backend.dense_to_backend(
            self.matrices.infoset_action_player.astype(np.float32)
        )

        # M_HV sparse matrix for computing infoset reaches
        self._M_HV = self.backend.sparse_to_backend(self.matrices.M_HV.astype(np.float32))

        # Infoset node counts for averaging
        self._infoset_node_counts = self.backend.dense_to_backend(
            self.matrices.infoset_node_counts.astype(np.float32)
        )

        # Infoset player array
        self._infoset_player = self.backend.dense_to_backend(
            self.matrices.infoset_player.astype(np.float32)
        )

        # Infoset action to infoset mapping
        self._infoset_action_to_infoset = self.matrices.infoset_action_to_infoset

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
        """Run a single vectorized CFR+ iteration."""
        xp = self.backend.xp
        t = self.iterations + 1

        # Get current strategy from regret matching
        strategy = regret_match(self._cumulative_regret, self.matrices, self.backend)

        # Compute reach probabilities
        reach = forward_reach_simple(strategy, self.matrices, self.backend)

        # Compute regrets for both players and update with masks
        for player in range(2):
            cf_values = compute_counterfactual_values(
                strategy, reach, self.matrices, self.backend, player
            )
            infoset_cf = compute_infoset_cf_values(cf_values, strategy, self.matrices, self.backend)
            instant_regret = compute_instant_regret(cf_values, infoset_cf, self.matrices, self.backend)

            # Vectorized update: only update this player's infoset-actions
            mask = self._player_masks[player]

            # CFR+ regret update: cumulative = max(cumulative + instant_regret, 0)
            # Only apply to this player's infoset-actions
            new_regret = self._cumulative_regret + instant_regret * mask
            self._cumulative_regret = xp.maximum(new_regret, 0.0)

        # Vectorized average strategy update
        # Compute infoset reach for each player using sparse matrix multiplication

        # Get player reach columns
        reach_p1 = reach[:, REACH_P1]  # (num_nodes,)
        reach_p2 = reach[:, REACH_P2]  # (num_nodes,)

        # Sum reaches over nodes in each infoset: M_HV @ reach_p
        # Shape: (num_infosets,)
        infoset_reach_p1 = self._M_HV @ reach_p1
        infoset_reach_p2 = self._M_HV @ reach_p2

        # Average by dividing by node count (avoid div by zero)
        node_counts = xp.maximum(self._infoset_node_counts, 1.0)
        infoset_reach_p1 = infoset_reach_p1 / node_counts
        infoset_reach_p2 = infoset_reach_p2 / node_counts

        # Select correct reach based on which player owns each infoset
        # infoset_reach[h] = reach_p1[h] if player[h] == 0 else reach_p2[h]
        infoset_reach = xp.where(
            self._infoset_player == 0,
            infoset_reach_p1,
            infoset_reach_p2
        )

        # Expand infoset reach to infoset-actions
        # Each infoset-action gets the reach of its parent infoset
        infoset_action_reach = infoset_reach[self._infoset_action_to_infoset]

        # CFR+ average strategy update: weighted by iteration t
        self._cumulative_strategy = (
            self._cumulative_strategy +
            t * infoset_action_reach * strategy
        )

    def solve(self, iterations: int = 1000) -> None:
        """Solve the game by running CFR+ iterations."""
        self.iterate(iterations)

    @property
    def current_strategy(self) -> np.ndarray:
        """Get current strategy from regret matching."""
        strategy = regret_match(self._cumulative_regret, self.matrices, self.backend)
        return self.backend.asnumpy(strategy)

    @property
    def average_strategy(self) -> np.ndarray:
        """Get average strategy (converges to Nash equilibrium)."""
        xp = self.backend.xp
        cumulative = self._cumulative_strategy

        # Compute sum per infoset using M_HQ
        M_HQ = self.backend.sparse_to_backend(self.matrices.M_HQ.astype(np.float32))
        infoset_totals = M_HQ @ cumulative  # (num_infosets,)

        # Expand back to infoset-actions
        totals_per_action = infoset_totals[self._infoset_action_to_infoset]

        # Normalize, with uniform fallback for zero totals
        safe_totals = xp.maximum(totals_per_action, 1e-10)
        avg_strategy = cumulative / safe_totals

        # Fix zero-total infosets to uniform
        avg_strategy_np = self.backend.asnumpy(avg_strategy)
        totals_np = self.backend.asnumpy(infoset_totals)

        for h_idx in range(self.matrices.num_infosets):
            if totals_np[h_idx] < 1e-10:
                start = self.matrices.infoset_action_offsets[h_idx]
                end = self.matrices.infoset_action_offsets[h_idx + 1]
                num_actions = end - start
                avg_strategy_np[start:end] = 1.0 / num_actions

        return avg_strategy_np

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

        print(f"\nCFR+ Vectorized Strategy after {self.iterations} iterations:")
        print("-" * 50)

        for h_idx in range(self.matrices.num_infosets):
            infoset = self._tree.infosets[h_idx]
            player = self.matrices.infoset_player[h_idx]

            start = self.matrices.infoset_action_offsets[h_idx]
            end = self.matrices.infoset_action_offsets[h_idx + 1]

            probs = avg_strategy[start:end]
            action_strs = [f"{a.name}={probs[i]:.3f}" for i, a in enumerate(infoset.actions)]

            print(f"P{player+1} [{infoset.key}]: {', '.join(action_strs)}")

    def exploitability(self) -> float:
        """Compute exploitability of the average strategy."""
        avg_strategy = self.average_strategy
        avg_strategy_backend = self.backend.dense_to_backend(avg_strategy)

        total_exploitability = 0.0

        for player in range(2):
            br_value = self._best_response_value(avg_strategy_backend, player)
            total_exploitability += br_value

        return total_exploitability

    def _best_response_value(self, opponent_strategy, player: int) -> float:
        """Compute best response value for a player."""
        reach = forward_reach_simple(opponent_strategy, self.matrices, self.backend)
        values = backward_values_simple(opponent_strategy, reach, self.matrices, self.backend)

        values_np = self.backend.asnumpy(values)
        reach_np = self.backend.asnumpy(reach)

        opp_col = REACH_P2 if player == 0 else REACH_P1

        br_strategy = np.copy(self.backend.asnumpy(opponent_strategy))

        for h_idx in range(self.matrices.num_infosets):
            if self.matrices.infoset_player[h_idx] == player:
                start = self.matrices.infoset_action_offsets[h_idx]
                end = self.matrices.infoset_action_offsets[h_idx + 1]

                infoset = self._tree.infosets[h_idx]
                action_values = []

                for a_idx in range(len(infoset.actions)):
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
