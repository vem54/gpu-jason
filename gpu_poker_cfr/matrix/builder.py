"""
Matrix builder for converting game trees to sparse matrix representation.

Following Kim 2024 paper notation:
- G: Adjacency matrix (parent → child edges)
- L^(l): Level graph for depth l (edges from depth l-1 to depth l)
- M^(Q+,V): Maps nodes to infoset-action pairs
- M^(H+,Q+): Maps infosets to infoset-action pairs
- M^(V,I+): Maps nodes to acting players
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

from gpu_poker_cfr.games.base import GameTree, Player
from .sparse import csr_from_edges, csr_from_weighted_edges, sparsity


@dataclass
class GameMatrices:
    """All matrices representing the game structure."""

    # Adjacency matrix G: (num_nodes, num_nodes)
    # G[i,j] = 1 if node i is parent of node j
    G: sparse.csr_matrix

    # Level graphs L^(l): list of sparse matrices
    # L[l][i,j] = 1 if node i (at depth l-1) is parent of node j (at depth l)
    L: List[sparse.csr_matrix]

    # Node → infoset-action mapping: (num_infoset_actions, num_nodes)
    # M_QV[q, v] = 1 if node v results from taking action a at infoset h (where q = (h,a))
    M_QV: sparse.csr_matrix

    # Infoset → infoset-action mapping: (num_infosets, num_infoset_actions)
    # M_HQ[h, q] = 1 if q = (h, a) for some action a
    M_HQ: sparse.csr_matrix

    # Node → player mapping: (num_nodes, num_players) dense
    # M_VI[v, i] = 1 if player i acts at node v's parent
    M_VI: np.ndarray

    # Chance probabilities at each node: (num_nodes,)
    chance_probs: np.ndarray

    # Terminal utilities: (num_terminals, num_players)
    terminal_utilities: np.ndarray

    # Index mappings
    terminal_node_indices: np.ndarray  # Maps terminal index → node id
    node_to_terminal_idx: np.ndarray   # Maps node id → terminal index (-1 if not terminal)

    # Tree structure arrays
    parent_ids: np.ndarray  # parent_ids[v] = parent of node v (-1 for root)
    depths: np.ndarray      # depths[v] = depth of node v (0 for root)

    # Infoset-action pair info
    num_infoset_actions: int
    infoset_action_to_infoset: np.ndarray  # Maps Q+ index → H+ index
    infoset_action_offsets: np.ndarray     # Start index of each infoset's actions in Q+

    # Vectorization support arrays
    infoset_action_player: np.ndarray   # Maps Q+ index → player (0 or 1)
    infoset_player: np.ndarray          # Maps H+ index → player (0 or 1)

    # Node → Infoset mapping: (num_infosets, num_nodes)
    # M_HV[h, v] = 1 if node v belongs to infoset h
    M_HV: sparse.csr_matrix

    # Number of nodes per infoset (for averaging)
    infoset_node_counts: np.ndarray  # (num_infosets,)

    # Game tree reference
    num_nodes: int
    num_terminals: int
    num_infosets: int
    num_players: int
    max_depth: int


def build_adjacency_matrix(tree: GameTree) -> sparse.csr_matrix:
    """
    Build adjacency matrix G from game tree.

    G[parent, child] = 1 for all parent-child edges.

    Args:
        tree: GameTree with parent_ids array

    Returns:
        Sparse CSR matrix of shape (num_nodes, num_nodes)
    """
    edges = []
    for node_id, parent_id in enumerate(tree.parent_ids):
        if parent_id >= 0:  # Skip root (parent_id = -1)
            edges.append((parent_id, node_id))

    return csr_from_edges(edges, (tree.num_nodes, tree.num_nodes))


def build_level_graphs(tree: GameTree) -> List[sparse.csr_matrix]:
    """
    Build level graphs L^(1), L^(2), ..., L^(D) from game tree.

    L^(l)[parent, child] = 1 if parent is at depth l-1 and child is at depth l.

    Args:
        tree: GameTree with depths and parent_ids arrays

    Returns:
        List of sparse CSR matrices, one per level (index 0 = level 1)
    """
    level_graphs = []

    for level in range(1, tree.max_depth + 1):
        edges = []
        for node_id, (parent_id, depth) in enumerate(zip(tree.parent_ids, tree.depths)):
            if depth == level and parent_id >= 0:
                edges.append((parent_id, node_id))

        L_l = csr_from_edges(edges, (tree.num_nodes, tree.num_nodes))
        level_graphs.append(L_l)

    return level_graphs


def build_infoset_action_mapping(tree: GameTree) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build infoset-action pair mappings.

    Returns:
        M_QV: (num_infoset_actions, num_nodes) - maps nodes to infoset-action pairs
        M_HQ: (num_infosets, num_infoset_actions) - maps infosets to their action pairs
        infoset_action_to_infoset: array mapping Q+ index to H+ index
        infoset_action_offsets: start index for each infoset's actions
    """
    # First, compute total number of infoset-action pairs
    # and build the infoset → action offset mapping
    num_infoset_actions = 0
    infoset_action_offsets = np.zeros(tree.num_infosets + 1, dtype=np.int32)

    for i, infoset in enumerate(tree.infosets):
        infoset_action_offsets[i] = num_infoset_actions
        num_infoset_actions += len(infoset.actions)
    infoset_action_offsets[tree.num_infosets] = num_infoset_actions

    # Build infoset_action_to_infoset mapping
    infoset_action_to_infoset = np.zeros(num_infoset_actions, dtype=np.int32)
    for i, infoset in enumerate(tree.infosets):
        start = infoset_action_offsets[i]
        end = infoset_action_offsets[i + 1]
        infoset_action_to_infoset[start:end] = i

    # Build M_HQ: (num_infosets, num_infoset_actions)
    # M_HQ[h, q] = 1 if q belongs to infoset h
    m_hq_edges = []
    for i, infoset in enumerate(tree.infosets):
        start = infoset_action_offsets[i]
        for a_idx in range(len(infoset.actions)):
            q_idx = start + a_idx
            m_hq_edges.append((i, q_idx))

    M_HQ = csr_from_edges(m_hq_edges, (tree.num_infosets, num_infoset_actions))

    # Build M_QV: (num_infoset_actions, num_nodes)
    # M_QV[q, v] = 1 if node v results from action a at infoset h (where q = (h,a))
    # This means: v's parent is in infoset h, and action a leads to v
    m_qv_edges = []

    for node in tree.nodes:
        if node.parent_id is None or node.parent_id < 0:
            continue

        parent_node = tree.nodes[node.parent_id]

        # Skip if parent is not a player decision node
        if parent_node.infoset_id is None:
            continue

        # Get the infoset of the parent
        infoset_id = parent_node.infoset_id
        infoset = tree.infosets[infoset_id]

        # Get the action that led to this node
        if node.action_from_parent is None:
            continue

        action_id = node.action_from_parent.id

        # Find which action index this is within the infoset
        action_idx = None
        for idx, action in enumerate(infoset.actions):
            if action.id == action_id:
                action_idx = idx
                break

        if action_idx is not None:
            q_idx = infoset_action_offsets[infoset_id] + action_idx
            m_qv_edges.append((q_idx, node.id))

    M_QV = csr_from_edges(m_qv_edges, (num_infoset_actions, tree.num_nodes))

    return M_QV, M_HQ, infoset_action_to_infoset, infoset_action_offsets


def build_player_mapping(tree: GameTree) -> np.ndarray:
    """
    Build node → player mapping M^(V, I+).

    M_VI[v, i] = 1 if player i acted to reach node v (i.e., player i is at v's parent).

    Args:
        tree: GameTree

    Returns:
        Dense array of shape (num_nodes, num_players)
    """
    M_VI = np.zeros((tree.num_nodes, tree.num_players), dtype=np.float32)

    for node in tree.nodes:
        if node.parent_id is None or node.parent_id < 0:
            continue

        parent_node = tree.nodes[node.parent_id]

        # Check if parent is a player (not chance, not terminal)
        if parent_node.player in (Player.PLAYER_1, Player.PLAYER_2):
            player_idx = parent_node.player.value  # 0 or 1
            M_VI[node.id, player_idx] = 1.0

    return M_VI


def build_vectorization_arrays(tree: GameTree, infoset_action_offsets: np.ndarray, num_infoset_actions: int) -> Tuple[np.ndarray, np.ndarray, sparse.csr_matrix, np.ndarray]:
    """
    Build arrays needed for vectorized CFR operations.

    Returns:
        infoset_action_player: Maps Q+ index → player (0 or 1)
        infoset_player: Maps H+ index → player (0 or 1)
        M_HV: (num_infosets, num_nodes) sparse matrix, M_HV[h, v] = 1 if v in infoset h
        infoset_node_counts: Number of nodes per infoset
    """
    # Build infoset player mapping
    infoset_player = np.zeros(tree.num_infosets, dtype=np.int32)
    for i, infoset in enumerate(tree.infosets):
        infoset_player[i] = infoset.player.value

    # Build infoset-action player mapping (expand infoset_player to all actions)
    infoset_action_player = np.zeros(num_infoset_actions, dtype=np.int32)
    for i, infoset in enumerate(tree.infosets):
        start = infoset_action_offsets[i]
        end = infoset_action_offsets[i + 1]
        infoset_action_player[start:end] = infoset.player.value

    # Build M_HV: (num_infosets, num_nodes) - node to infoset membership
    m_hv_edges = []
    infoset_node_counts = np.zeros(tree.num_infosets, dtype=np.int32)

    for i, infoset in enumerate(tree.infosets):
        for node_id in infoset.node_ids:
            m_hv_edges.append((i, node_id))
            infoset_node_counts[i] += 1

    M_HV = csr_from_edges(m_hv_edges, (tree.num_infosets, tree.num_nodes))

    return infoset_action_player, infoset_player, M_HV, infoset_node_counts


def build_game_matrices(tree: GameTree) -> GameMatrices:
    """
    Build all game matrices from a game tree.

    Args:
        tree: GameTree from a Game.build_tree() call

    Returns:
        GameMatrices containing all sparse/dense matrices for CFR
    """
    # Build adjacency and level graphs
    G = build_adjacency_matrix(tree)
    L = build_level_graphs(tree)

    # Build infoset-action mappings
    M_QV, M_HQ, infoset_action_to_infoset, infoset_action_offsets = build_infoset_action_mapping(tree)

    # Build player mapping
    M_VI = build_player_mapping(tree)

    # Build vectorization arrays
    num_infoset_actions = M_QV.shape[0]
    infoset_action_player, infoset_player, M_HV, infoset_node_counts = build_vectorization_arrays(
        tree, infoset_action_offsets, num_infoset_actions
    )

    # Terminal node mappings
    node_to_terminal_idx = np.full(tree.num_nodes, -1, dtype=np.int32)
    for term_idx, node_idx in enumerate(tree.terminal_indices):
        node_to_terminal_idx[node_idx] = term_idx

    return GameMatrices(
        G=G,
        L=L,
        M_QV=M_QV,
        M_HQ=M_HQ,
        M_VI=M_VI,
        chance_probs=tree.chance_probs,
        terminal_utilities=tree.terminal_utilities,
        terminal_node_indices=tree.terminal_indices,
        node_to_terminal_idx=node_to_terminal_idx,
        parent_ids=tree.parent_ids,
        depths=tree.depths,
        num_infoset_actions=num_infoset_actions,
        infoset_action_to_infoset=infoset_action_to_infoset,
        infoset_action_offsets=infoset_action_offsets,
        infoset_action_player=infoset_action_player,
        infoset_player=infoset_player,
        M_HV=M_HV,
        infoset_node_counts=infoset_node_counts,
        num_nodes=tree.num_nodes,
        num_terminals=tree.num_terminals,
        num_infosets=tree.num_infosets,
        num_players=tree.num_players,
        max_depth=tree.max_depth
    )


def print_matrix_stats(matrices: GameMatrices):
    """Print statistics about the game matrices."""
    print(f"Game Matrix Statistics:")
    print(f"  Nodes: {matrices.num_nodes}")
    print(f"  Terminals: {matrices.num_terminals}")
    print(f"  Infosets: {matrices.num_infosets}")
    print(f"  Infoset-actions (Q+): {matrices.num_infoset_actions}")
    print(f"  Max depth: {matrices.max_depth}")
    print(f"\nMatrix shapes and sparsity:")
    print(f"  G: {matrices.G.shape}, {sparsity(matrices.G):.2f}% sparse, {matrices.G.nnz} nnz")
    for i, L_l in enumerate(matrices.L):
        print(f"  L[{i+1}]: {L_l.shape}, {sparsity(L_l):.2f}% sparse, {L_l.nnz} nnz")
    print(f"  M_QV: {matrices.M_QV.shape}, {sparsity(matrices.M_QV):.2f}% sparse, {matrices.M_QV.nnz} nnz")
    print(f"  M_HQ: {matrices.M_HQ.shape}, {sparsity(matrices.M_HQ):.2f}% sparse, {matrices.M_HQ.nnz} nnz")
    print(f"  M_VI: {matrices.M_VI.shape} (dense)")
