"""
Abstract base classes for game definitions.

This module defines the interface that all games must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Dict, Optional, FrozenSet
import numpy as np


class Player(IntEnum):
    """Player identifiers."""
    CHANCE = -1
    PLAYER_1 = 0
    PLAYER_2 = 1


@dataclass(frozen=True)
class Action:
    """An action that can be taken at a decision node."""
    id: int
    name: str


@dataclass
class Node:
    """A node in the game tree."""
    id: int
    parent_id: Optional[int]
    player: Player
    infoset_id: Optional[int]  # None for chance/terminal nodes
    actions: Tuple[Action, ...]  # Empty for terminal nodes
    action_from_parent: Optional[Action]
    depth: int
    is_terminal: bool
    utility: Optional[Tuple[float, ...]]  # Payoffs for each player if terminal
    chance_prob: float  # Probability if this node resulted from chance action


@dataclass
class Infoset:
    """An information set (nodes indistinguishable to a player)."""
    id: int
    player: Player
    actions: Tuple[Action, ...]
    node_ids: List[int]  # Nodes belonging to this infoset
    key: str  # Human-readable key for debugging


@dataclass
class GameTree:
    """Complete game tree structure."""
    nodes: List[Node]
    infosets: List[Infoset]

    # Counts
    num_nodes: int
    num_terminals: int
    num_decision_nodes: int
    num_infosets: int
    num_players: int
    max_depth: int

    # Index mappings
    terminal_indices: np.ndarray  # Indices of terminal nodes
    decision_indices: np.ndarray  # Indices of decision nodes

    # Utilities matrix: (num_terminals, num_players)
    terminal_utilities: np.ndarray

    # Parent relationships: parent_id for each node (-1 for root)
    parent_ids: np.ndarray

    # Depths: depth for each node
    depths: np.ndarray

    # Player at each node (-1 for chance, 0/1 for players, -2 for terminal)
    players: np.ndarray

    # Infoset id for each node (-1 if not applicable)
    node_to_infoset: np.ndarray

    # Action from parent for each node (-1 for root)
    action_from_parent: np.ndarray

    # Number of actions at each infoset
    infoset_num_actions: np.ndarray

    # Chance probabilities for each node (1.0 if not chance outcome)
    chance_probs: np.ndarray


class Game(ABC):
    """Abstract base class for extensive-form games."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the game."""
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players (excluding chance)."""
        pass

    @abstractmethod
    def build_tree(self) -> GameTree:
        """Build and return the complete game tree."""
        pass

    @abstractmethod
    def infoset_key(self, player: Player, history: Tuple) -> str:
        """
        Return a string key identifying the information set.

        Two histories belong to the same infoset iff they have the same key.
        """
        pass
