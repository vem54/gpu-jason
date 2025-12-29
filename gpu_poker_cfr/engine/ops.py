"""
Core CFR operations using matrix formulation.

Following Kim 2024 paper's matrix-based approach:
- Forward pass: reach probabilities via level graphs
- Backward pass: expected utilities via reverse traversal
- Regret matching: convert regrets to strategies
- Counterfactual values: compute instantaneous regrets

All operations work on both NumPy and CuPy backends.

Reach matrix semantics (shape: num_nodes x 3):
    REACH_P1 (col 0): P1's realization probability to reach node
                      = product of P1's action probs on path (excludes chance, excludes P2)
    REACH_P2 (col 1): P2's realization probability to reach node
                      = product of P2's action probs on path (excludes chance, excludes P1)
    REACH_CHANCE (col 2): Chance realization probability to reach node
                          = product of chance probs on path (e.g., card deal prob)

For counterfactual values of player i:
    π_{-i}(node) = reach[node, opponent] * reach[node, REACH_CHANCE]
    where opponent = 1 - player
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_poker_cfr.matrix.builder import GameMatrices
    from gpu_poker_cfr.engine.backend import Backend

# Reach matrix column indices (explicit constants to avoid ambiguity)
REACH_P1 = 0       # Player 1's realization probability
REACH_P2 = 1       # Player 2's realization probability
REACH_CHANCE = 2   # Chance realization probability
NUM_REACH_COLS = 3 # Total columns in reach matrix


def forward_reach(
    strategy: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Compute reach probabilities for all nodes using forward level traversal.

    Uses the recurrence:
        pi^(l) = (L^(l) ⊙ S)^T @ pi^(l-1) + pi^(l-1)

    Where S[i,j] = strategy probability of the action leading from i to j.

    Args:
        strategy: Strategy array of shape (num_infoset_actions,)
            strategy[q] = probability of taking action a at infoset h (where q=(h,a))
        matrices: GameMatrices containing level graphs and mappings
        backend: Backend for array operations

    Returns:
        reach: Array of shape (num_nodes, num_players)
            reach[v, i] = probability player i plays to reach node v
    """
    num_nodes = matrices.num_nodes
    num_players = matrices.num_players

    # Initialize reach probabilities
    # reach[v, i] = reach probability for player i at node v
    reach = backend.ones((num_nodes, num_players))

    # Build action probability vector for each node
    # action_probs[v] = probability of the action that led to node v
    action_probs = _build_action_probs(strategy, matrices, backend)

    # Forward pass through levels
    for level_idx, L_l in enumerate(matrices.L):
        # Convert to backend sparse format
        L_l_backend = backend.sparse_to_backend(L_l)

        # For each player, multiply reach by action probability if that player acted
        for player_idx in range(num_players):
            # Get nodes where this player acted (to reach child)
            player_mask = matrices.M_VI[:, player_idx]

            # Compute reach contribution: parent_reach * action_prob
            # Only apply action_prob where this player acted
            weighted_probs = backend.where(
                backend.dense_to_backend(player_mask) > 0,
                action_probs,
                backend.ones(num_nodes)
            )

            # Propagate: child_reach = parent_reach * action_prob
            # Using L^T since L[parent, child] = 1
            parent_reach = reach[:, player_idx]

            # For edges in this level: child gets parent's reach * action prob
            # L_l^T @ (parent_reach * weighted_probs) gives contribution from parents
            # But we need to be careful: the action prob is at the child, not parent

            # Actually simpler: for each edge (p,c) in level l,
            # reach[c, i] = reach[p, i] * action_probs[c] (if player i acted at p)
            # = reach[p, i] (if player i didn't act at p)

            # Let's compute this differently:
            # child_reach = L_l^T @ parent_reach (inherits parent reach)
            # Then multiply by action prob where this player acted

            parent_contrib = backend.spmv(L_l_backend.T, parent_reach)

            # Where there's a parent contribution, apply player's action prob
            # Only if this player acted
            for node_id in range(num_nodes):
                if matrices.depths[node_id] == level_idx + 1:  # This node is at current level
                    parent_id = matrices.parent_ids[node_id]
                    if parent_id >= 0:
                        parent_player_acted = player_mask[node_id] > 0
                        if parent_player_acted:
                            reach[node_id, player_idx] = reach[parent_id, player_idx] * action_probs[node_id]
                        else:
                            reach[node_id, player_idx] = reach[parent_id, player_idx]

    return reach


def forward_reach_simple(
    strategy: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Simple (non-vectorized) forward reach computation for verification.

    Computes reach probabilities for each player and chance at each node.

    Args:
        strategy: Strategy array of shape (num_infoset_actions,)
        matrices: GameMatrices
        backend: Backend

    Returns:
        reach: Array of shape (num_nodes, 3) with columns:
               - REACH_P1 (0): P1's realization prob (product of P1's action probs only)
               - REACH_P2 (1): P2's realization prob (product of P2's action probs only)
               - REACH_CHANCE (2): Chance realization prob (product of chance probs)
    """
    num_nodes = matrices.num_nodes
    num_players = matrices.num_players
    assert num_players == 2, "This implementation assumes 2 players"

    # Get raw numpy arrays for simple iteration
    parent_ids = backend.asnumpy(backend.dense_to_backend(np.array(matrices.parent_ids)))

    # Reach matrix: 3 columns (P1, P2, chance)
    reach = np.ones((num_nodes, NUM_REACH_COLS), dtype=np.float32)
    action_probs = backend.asnumpy(_build_action_probs(strategy, matrices, backend))

    # Process nodes in order (parents before children due to tree structure)
    for node_id in range(1, num_nodes):  # Skip root
        parent_id = int(parent_ids[node_id])
        chance_prob = matrices.chance_probs[node_id]

        # P1 reach: only multiply if P1 acted to reach this node
        if matrices.M_VI[node_id, REACH_P1] > 0:
            reach[node_id, REACH_P1] = reach[parent_id, REACH_P1] * action_probs[node_id]
        else:
            reach[node_id, REACH_P1] = reach[parent_id, REACH_P1]

        # P2 reach: only multiply if P2 acted to reach this node
        if matrices.M_VI[node_id, REACH_P2] > 0:
            reach[node_id, REACH_P2] = reach[parent_id, REACH_P2] * action_probs[node_id]
        else:
            reach[node_id, REACH_P2] = reach[parent_id, REACH_P2]

        # Chance reach: multiply by chance prob if this is a chance outcome
        if chance_prob < 1.0:
            reach[node_id, REACH_CHANCE] = reach[parent_id, REACH_CHANCE] * chance_prob
        else:
            reach[node_id, REACH_CHANCE] = reach[parent_id, REACH_CHANCE]

    # Verify shape invariant
    assert reach.shape == (num_nodes, NUM_REACH_COLS), \
        f"Reach shape mismatch: {reach.shape} != ({num_nodes}, {NUM_REACH_COLS})"

    return backend.dense_to_backend(reach)


def backward_values(
    strategy: np.ndarray,
    reach: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Compute expected values via backward level traversal.

    Args:
        strategy: Strategy array of shape (num_infoset_actions,)
        reach: Reach probabilities of shape (num_nodes, num_players)
        matrices: GameMatrices
        backend: Backend

    Returns:
        values: Array of shape (num_nodes, num_players)
            values[v, i] = expected utility for player i at node v
    """
    num_nodes = matrices.num_nodes
    num_players = matrices.num_players

    values = backend.zeros((num_nodes, num_players))

    # Initialize terminal values
    terminal_utils = backend.dense_to_backend(matrices.terminal_utilities)
    for term_idx, node_idx in enumerate(matrices.terminal_node_indices):
        values[node_idx, :] = terminal_utils[term_idx, :]

    # Get action probabilities
    action_probs = _build_action_probs(strategy, matrices, backend)

    # Backward pass through levels (from max_depth-1 down to 0)
    # For internal nodes, value = sum over children of (action_prob * child_value)
    # But only the acting player's action prob matters

    return backward_values_simple(strategy, reach, matrices, backend)


def backward_values_simple(
    strategy: np.ndarray,
    reach: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Vectorized backward value computation using level graphs.

    Uses sparse matrix operations to propagate values from leaves to root,
    processing one level at a time.

    Args:
        strategy: Strategy array of shape (num_infoset_actions,)
        reach: Reach probabilities (unused in basic version, needed for CF values)
        matrices: GameMatrices
        backend: Backend

    Returns:
        values: Array of shape (num_nodes, num_players)
    """
    num_nodes = matrices.num_nodes
    num_players = matrices.num_players

    values = np.zeros((num_nodes, num_players), dtype=np.float32)

    # Initialize terminal values
    values[matrices.terminal_node_indices, :] = matrices.terminal_utilities

    # Build action probabilities for all nodes (used as edge weights)
    action_probs = backend.asnumpy(_build_action_probs(strategy, matrices, backend))
    chance_probs = matrices.chance_probs

    # Combined edge weights: action_probs for player nodes, chance_probs for chance nodes
    # At each node, the transition probability is either action_prob or chance_prob
    edge_weights = np.where(action_probs > 0, action_probs, chance_probs)

    # Process levels from deepest to root using level graphs L[l]
    # L[l] has edges from depth l-1 to depth l
    # To propagate values up, we need L[l].T (transpose)
    for l in range(matrices.max_depth, 0, -1):
        L_l = matrices.L[l - 1]  # L[l] is at index l-1

        if L_l.nnz == 0:
            continue

        # For each edge (parent, child) in L_l, propagate:
        # values[parent] += edge_weight[child] * values[child]

        # Create weighted values
        weighted_values = edge_weights[:, None] * values

        # Propagate up: L_l.T @ weighted_values adds child contributions to parents
        # L_l.T is (num_nodes, num_nodes) but sparse
        parent_contributions = L_l.T @ weighted_values

        # Add contributions to parent values
        values += parent_contributions

    return backend.dense_to_backend(values)


def compute_counterfactual_values(
    strategy: np.ndarray,
    reach: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend',
    player: int
) -> np.ndarray:
    """
    Compute counterfactual values for a player (vectorized).

    CFV(h, a) = Σ_{v∈h} π_{-i}(v) × v_i(v, a)

    Where π_{-i}(v) = opponent_reach(v) × chance_reach(v)
    This is the probability that the opponent and chance play to reach v.

    IMPORTANT: We use reach at the PARENT node (the infoset node where player acts),
    NOT at the child node. The child node's reach would incorrectly include the
    acting player's action probability.

    Args:
        strategy: Current strategy
        reach: Reach probabilities of shape (num_nodes, 3)
               - reach[:, REACH_P1]: P1 reach
               - reach[:, REACH_P2]: P2 reach
               - reach[:, REACH_CHANCE]: chance reach
        matrices: GameMatrices
        backend: Backend
        player: Player index (0 for P1, 1 for P2)

    Returns:
        cf_values: Array of shape (num_infoset_actions,)
            Counterfactual value for each infoset-action pair
    """
    xp = backend.xp
    assert player in (0, 1), f"Invalid player: {player}"

    # Opponent column index
    opp_col = REACH_P2 if player == 0 else REACH_P1

    # Get expected values
    values = backward_values_simple(strategy, reach, matrices, backend)

    # Verify reach shape
    assert reach.shape[1] == NUM_REACH_COLS, \
        f"Reach must have {NUM_REACH_COLS} columns, got {reach.shape[1]}"

    # Compute pi_minus_i at parent node for each node
    # pi_minus_i[v] = reach[parent[v], opp_col] * reach[parent[v], REACH_CHANCE]
    parent_ids = matrices.parent_ids  # (num_nodes,)

    # Handle root node (parent_id = -1) by clamping to 0
    safe_parent_ids = np.maximum(parent_ids, 0)

    # Get reach values using numpy indexing (vectorized)
    reach_np = backend.asnumpy(reach)
    opp_reach_at_parent = reach_np[safe_parent_ids, opp_col]
    chance_reach_at_parent = reach_np[safe_parent_ids, REACH_CHANCE]

    # Zero out nodes with no parent (root)
    opp_reach_at_parent[parent_ids < 0] = 0.0
    chance_reach_at_parent[parent_ids < 0] = 0.0

    pi_minus_i = opp_reach_at_parent * chance_reach_at_parent

    # Weighted values: pi_minus_i * values[:, player]
    values_np = backend.asnumpy(values)
    weighted_values = pi_minus_i * values_np[:, player]

    # cf_values = M_QV @ weighted_values (sparse matrix-vector multiply)
    cf_values = matrices.M_QV @ weighted_values

    return backend.dense_to_backend(cf_values.astype(np.float32))


def compute_infoset_cf_values(
    action_cf_values: np.ndarray,
    strategy: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Compute counterfactual value for each infoset (sum over actions weighted by strategy).

    CFV[h] = sum_a strategy[h,a] * CFV[h,a]

    Args:
        action_cf_values: CF values for each infoset-action pair
        strategy: Current strategy
        matrices: GameMatrices
        backend: Backend

    Returns:
        infoset_cf_values: Array of shape (num_infosets,)
    """
    num_infosets = matrices.num_infosets
    infoset_cf_values = np.zeros(num_infosets, dtype=np.float32)

    action_cf_values = backend.asnumpy(action_cf_values)
    strategy = backend.asnumpy(strategy)

    for h_idx in range(num_infosets):
        start = matrices.infoset_action_offsets[h_idx]
        end = matrices.infoset_action_offsets[h_idx + 1]

        cf_val = 0.0
        for q_idx in range(start, end):
            cf_val += strategy[q_idx] * action_cf_values[q_idx]
        infoset_cf_values[h_idx] = cf_val

    return backend.dense_to_backend(infoset_cf_values)


def compute_instant_regret(
    action_cf_values: np.ndarray,
    infoset_cf_values: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Compute instantaneous regret for each infoset-action pair.

    regret[h,a] = CFV[h,a] - CFV[h]

    Args:
        action_cf_values: CF values for each infoset-action
        infoset_cf_values: CF values for each infoset
        matrices: GameMatrices
        backend: Backend

    Returns:
        instant_regret: Array of shape (num_infoset_actions,)
    """
    num_infoset_actions = matrices.num_infoset_actions
    instant_regret = np.zeros(num_infoset_actions, dtype=np.float32)

    action_cf_values = backend.asnumpy(action_cf_values)
    infoset_cf_values = backend.asnumpy(infoset_cf_values)

    for h_idx in range(matrices.num_infosets):
        start = matrices.infoset_action_offsets[h_idx]
        end = matrices.infoset_action_offsets[h_idx + 1]

        h_cf_value = infoset_cf_values[h_idx]
        for q_idx in range(start, end):
            instant_regret[q_idx] = action_cf_values[q_idx] - h_cf_value

    return backend.dense_to_backend(instant_regret)


def regret_match(
    cumulative_regret: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Convert cumulative regrets to strategy via regret matching.

    For each infoset h:
        positive_regrets = max(0, regrets[h])
        if sum(positive_regrets) > 0:
            strategy[h] = positive_regrets / sum(positive_regrets)
        else:
            strategy[h] = uniform over actions

    Args:
        cumulative_regret: Array of shape (num_infoset_actions,)
        matrices: GameMatrices
        backend: Backend

    Returns:
        strategy: Array of shape (num_infoset_actions,) - valid probability distribution per infoset
    """
    num_infoset_actions = matrices.num_infoset_actions
    strategy = np.zeros(num_infoset_actions, dtype=np.float32)

    cumulative_regret = backend.asnumpy(cumulative_regret)

    for h_idx in range(matrices.num_infosets):
        start = matrices.infoset_action_offsets[h_idx]
        end = matrices.infoset_action_offsets[h_idx + 1]
        num_actions = end - start

        # Get positive regrets
        regrets = cumulative_regret[start:end]
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)

        if regret_sum > 0:
            strategy[start:end] = positive_regrets / regret_sum
        else:
            # Uniform strategy
            strategy[start:end] = 1.0 / num_actions

    return backend.dense_to_backend(strategy)


def uniform_strategy(
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Create uniform strategy (equal probability for all actions at each infoset).

    Args:
        matrices: GameMatrices
        backend: Backend

    Returns:
        strategy: Array of shape (num_infoset_actions,)
    """
    num_infoset_actions = matrices.num_infoset_actions
    strategy = np.zeros(num_infoset_actions, dtype=np.float32)

    for h_idx in range(matrices.num_infosets):
        start = matrices.infoset_action_offsets[h_idx]
        end = matrices.infoset_action_offsets[h_idx + 1]
        num_actions = end - start
        strategy[start:end] = 1.0 / num_actions

    return backend.dense_to_backend(strategy)


# =============================================================================
# Invariant checks for debugging CFR
# =============================================================================

def check_regret_invariant(
    strategy: np.ndarray,
    instant_regret: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend',
    tolerance: float = 1e-6
) -> bool:
    """
    Check CFR invariant: sum_a sigma[I,a] * instant_regret[I,a] ≈ 0 for each infoset.

    This must hold because:
    - instant_regret[I,a] = CFV[I,a] - CFV[I]
    - CFV[I] = sum_a sigma[I,a] * CFV[I,a]
    - Therefore: sum_a sigma[I,a] * (CFV[I,a] - CFV[I]) = CFV[I] - CFV[I] = 0

    Args:
        strategy: Current strategy
        instant_regret: Instantaneous regrets
        matrices: GameMatrices
        backend: Backend
        tolerance: Tolerance for floating point comparison

    Returns:
        True if invariant holds, raises AssertionError otherwise
    """
    strategy = backend.asnumpy(strategy)
    instant_regret = backend.asnumpy(instant_regret)

    for h_idx in range(matrices.num_infosets):
        start = matrices.infoset_action_offsets[h_idx]
        end = matrices.infoset_action_offsets[h_idx + 1]

        sigma_regret_sum = np.sum(strategy[start:end] * instant_regret[start:end])

        assert abs(sigma_regret_sum) < tolerance, \
            f"Regret invariant violated at infoset {h_idx}: " \
            f"sum(sigma * regret) = {sigma_regret_sum:.9f}, tolerance = {tolerance}"

    return True


def check_zero_sum_invariant(
    values: np.ndarray,
    backend: 'Backend',
    tolerance: float = 1e-6
) -> bool:
    """
    Check zero-sum invariant: EV_p1 + EV_p2 ≈ 0 at root.

    For a zero-sum game, the expected values for both players should sum to zero.

    Args:
        values: Node values of shape (num_nodes, 2)
        backend: Backend
        tolerance: Tolerance for floating point comparison

    Returns:
        True if invariant holds, raises AssertionError otherwise
    """
    values = backend.asnumpy(values)

    root_sum = values[0, 0] + values[0, 1]

    assert abs(root_sum) < tolerance, \
        f"Zero-sum invariant violated at root: " \
        f"EV_P1={values[0, 0]:.6f}, EV_P2={values[0, 1]:.6f}, sum={root_sum:.6f}"

    return True


def _build_action_probs(
    strategy: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Build action probability vector for each node.

    action_probs[v] = probability of the action that led to node v

    For chance nodes, this is the chance probability.
    For player nodes, this is the strategy probability.
    For root, this is 1.0.

    Args:
        strategy: Strategy array of shape (num_infoset_actions,)
        matrices: GameMatrices
        backend: Backend

    Returns:
        action_probs: Array of shape (num_nodes,)
    """
    num_nodes = matrices.num_nodes
    action_probs = np.ones(num_nodes, dtype=np.float32)

    strategy = backend.asnumpy(strategy)

    # Fill in from M_QV mapping (player actions)
    M_QV = matrices.M_QV

    for q_idx in range(matrices.num_infoset_actions):
        row = M_QV.getrow(q_idx)
        for node_id in row.indices:
            action_probs[node_id] = strategy[q_idx]

    # Fill in chance probabilities
    action_probs = action_probs * matrices.chance_probs

    return backend.dense_to_backend(action_probs)


# Vectorized versions for GPU acceleration (to be implemented)
# These will use sparse matrix operations instead of loops


def forward_reach_vectorized(
    strategy: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Vectorized forward reach computation using sparse matrix operations.

    TODO: Implement using level graph matrix multiplications.
    Currently falls back to simple version.
    """
    return forward_reach_simple(strategy, matrices, backend)


def backward_values_vectorized(
    strategy: np.ndarray,
    reach: np.ndarray,
    matrices: 'GameMatrices',
    backend: 'Backend'
) -> np.ndarray:
    """
    Vectorized backward value computation using sparse matrix operations.

    TODO: Implement using level graph matrix multiplications.
    Currently falls back to simple version.
    """
    return backward_values_simple(strategy, reach, matrices, backend)
