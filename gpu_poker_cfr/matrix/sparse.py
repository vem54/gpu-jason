"""
Sparse matrix utilities.

All sparse matrices use CSR (Compressed Sparse Row) format for efficient
row-based operations and GPU compatibility.
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple


def csr_from_edges(
    edges: List[Tuple[int, int]],
    shape: Tuple[int, int],
    dtype=np.float32
) -> sparse.csr_matrix:
    """
    Create a CSR sparse matrix from a list of edges.

    Args:
        edges: List of (row, col) tuples representing non-zero entries
        shape: (num_rows, num_cols) of the output matrix
        dtype: Data type for values (default float32)

    Returns:
        CSR sparse matrix with 1.0 at each edge position
    """
    if not edges:
        return sparse.csr_matrix(shape, dtype=dtype)

    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=dtype)
    return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=dtype)


def csr_from_weighted_edges(
    edges: List[Tuple[int, int, float]],
    shape: Tuple[int, int],
    dtype=np.float32
) -> sparse.csr_matrix:
    """
    Create a CSR sparse matrix from weighted edges.

    Args:
        edges: List of (row, col, value) tuples
        shape: (num_rows, num_cols)
        dtype: Data type

    Returns:
        CSR sparse matrix
    """
    if not edges:
        return sparse.csr_matrix(shape, dtype=dtype)

    rows, cols, data = zip(*edges)
    return sparse.csr_matrix((data, (rows, cols)), shape=shape, dtype=dtype)


def sparsity(matrix: sparse.spmatrix) -> float:
    """
    Calculate sparsity percentage of a matrix.

    Returns:
        Percentage of zero entries (0-100)
    """
    total = matrix.shape[0] * matrix.shape[1]
    if total == 0:
        return 100.0
    nonzero = matrix.nnz
    return 100.0 * (1.0 - nonzero / total)


def validate_csr(matrix: sparse.csr_matrix, name: str = "matrix"):
    """Validate CSR matrix properties."""
    assert sparse.isspmatrix_csr(matrix), f"{name} must be CSR format"
    assert not np.any(np.isnan(matrix.data)), f"{name} contains NaN"
    assert not np.any(np.isinf(matrix.data)), f"{name} contains Inf"
