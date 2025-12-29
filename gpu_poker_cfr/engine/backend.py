"""
Backend abstraction for GPU/CPU computation.

Provides a unified interface for array operations that works with both:
- NumPy/SciPy (CPU)
- CuPy/cuSPARSE (GPU)

Usage:
    backend = get_backend('cupy')  # or 'numpy'
    x = backend.array([1, 2, 3])
    y = backend.zeros(10)
"""

from typing import Union, Optional, Literal
from dataclasses import dataclass
import numpy as np
from scipy import sparse as sp_sparse

# Try to import CuPy and verify CUDA is fully functional
CUPY_AVAILABLE = False
CUPY_ERROR_MSG = None
cp = None
cp_sparse = None

def _ensure_cuda_in_path():
    """Ensure CUDA bin directory is in PATH (Windows fix)."""
    import os
    import sys
    if sys.platform == 'win32':
        cuda_path = os.environ.get('CUDA_PATH', '')
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, 'bin')
            if cuda_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = cuda_bin + ';' + os.environ.get('PATH', '')

def _test_cupy_functional():
    """Test that CuPy and CUDA are fully functional."""
    _ensure_cuda_in_path()
    import cupy as _cp
    # Test basic array
    arr = _cp.array([1.0, 2.0])
    # Test computation (this triggers NVRTC compilation)
    _ = _cp.maximum(arr, 0)
    # Test sparse (this triggers cusparse)
    import cupyx.scipy.sparse as _cp_sparse
    from scipy import sparse as _sp_sparse
    # Create scipy sparse first, then convert (more reliable)
    scipy_mat = _sp_sparse.csr_matrix([[1.0, 0], [0, 1.0]], dtype=np.float32)
    _ = _cp_sparse.csr_matrix(scipy_mat)
    return _cp, _cp_sparse

try:
    cp, cp_sparse = _test_cupy_functional()
    CUPY_AVAILABLE = True
except ImportError as e:
    CUPY_ERROR_MSG = f"CuPy not installed: {e}"
except Exception as e:
    # CuPy installed but CUDA not working (missing DLLs, no GPU, etc.)
    CUPY_ERROR_MSG = f"CuPy/CUDA not functional: {e}"
    cp = None
    cp_sparse = None


BackendType = Literal['numpy', 'cupy']


@dataclass
class Backend:
    """
    Array backend abstraction.

    Provides consistent interface for array operations across NumPy and CuPy.
    """
    name: BackendType
    xp: any  # numpy or cupy module
    sparse: any  # scipy.sparse or cupyx.scipy.sparse

    def array(self, data, dtype=np.float32):
        """Create array from data."""
        return self.xp.array(data, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        """Create zero-filled array."""
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        """Create ones-filled array."""
        return self.xp.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=np.float32):
        """Create array filled with value."""
        return self.xp.full(shape, fill_value, dtype=dtype)

    def copy(self, arr):
        """Copy array."""
        return arr.copy()

    def asnumpy(self, arr):
        """Convert to numpy array (for results)."""
        if self.name == 'cupy':
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def maximum(self, arr, val):
        """Element-wise maximum."""
        return self.xp.maximum(arr, val)

    def sum(self, arr, axis=None):
        """Sum array."""
        return self.xp.sum(arr, axis=axis)

    def where(self, condition, x, y):
        """Conditional selection."""
        return self.xp.where(condition, x, y)

    def csr_matrix(self, data, shape=None):
        """
        Create or convert to CSR sparse matrix.

        Args:
            data: Can be dense array, tuple (data, indices, indptr), or existing sparse
            shape: Shape if creating from tuple
        """
        if isinstance(data, tuple):
            return self.sparse.csr_matrix(data, shape=shape)
        return self.sparse.csr_matrix(data)

    def sparse_to_backend(self, scipy_sparse: sp_sparse.csr_matrix):
        """
        Convert scipy sparse matrix to backend sparse format.

        Args:
            scipy_sparse: SciPy CSR sparse matrix

        Returns:
            Backend-compatible sparse matrix
        """
        if self.name == 'numpy':
            return scipy_sparse
        else:
            # CuPy sparse from SciPy sparse
            return cp_sparse.csr_matrix(scipy_sparse)

    def dense_to_backend(self, numpy_array: np.ndarray):
        """
        Convert numpy array to backend array.

        Args:
            numpy_array: NumPy array

        Returns:
            Backend-compatible array
        """
        if self.name == 'numpy':
            return numpy_array
        else:
            return cp.asarray(numpy_array)

    def spmv(self, A, x):
        """
        Sparse matrix-vector multiply: y = A @ x

        Args:
            A: Sparse CSR matrix
            x: Dense vector

        Returns:
            Dense vector y
        """
        return A @ x

    def spmm(self, A, B):
        """
        Sparse matrix-matrix multiply: C = A @ B

        Both A and B can be sparse or dense.
        """
        return A @ B

    def hadamard_spmv(self, A, s, x):
        """
        Hadamard product with sparse matrix then matrix-vector multiply.

        Computes: (A ⊙ S) @ x where S is broadcast from vector s.

        This is a key operation in CFR tree traversal.

        For efficiency, we compute this as: A @ (s * x) when A has the right structure.
        But for correctness, we do element-wise on the sparse matrix.

        Args:
            A: Sparse CSR matrix (parent → child edges)
            s: Dense vector of action probabilities at each node
            x: Dense vector/matrix to multiply

        Returns:
            Result of (A ⊙ broadcast(s)) @ x
        """
        # The paper's approach: A ⊙ S where S[i,j] = s[j]
        # This means each column j of A is scaled by s[j]

        # For CSR format, column scaling is efficient:
        # Create diagonal matrix from s and multiply: A @ diag(s) @ x = A @ (s * x)
        # Wait, that's not right. We need (A ⊙ S) where S broadcasts s to columns.

        # Actually S[i,j] = s[j] means we scale column j by s[j]
        # In matrix form: A ⊙ S = A @ diag(s) element-wise... no

        # Let me reconsider: S = (s_v')_{(v,v') ∈ V²}
        # So S[v, v'] = s[v'] - the child's action probability
        # (A ⊙ S)[i,j] = A[i,j] * s[j]

        # This is equivalent to: (A ⊙ S) @ x = A @ (diag(s) @ x) when x is vector
        # No wait, let me verify:
        # ((A ⊙ S) @ x)[i] = Σ_j A[i,j] * s[j] * x[j]
        # (A @ (s * x))[i] = Σ_j A[i,j] * (s[j] * x[j]) = same!

        # So for vectors: (A ⊙ S) @ x = A @ (s * x)
        if x.ndim == 1:
            return self.spmv(A, s * x)
        else:
            # For matrices, need to be more careful
            # ((A ⊙ S) @ X)[i,k] = Σ_j A[i,j] * s[j] * X[j,k]
            # This is still A @ (diag(s) @ X) = A @ (s[:, None] * X)
            return self.spmm(A, s[:, None] * x)


# Global backend cache
_backends = {}


def get_backend(name: BackendType = 'numpy') -> Backend:
    """
    Get or create a backend instance.

    Args:
        name: 'numpy' for CPU or 'cupy' for GPU

    Returns:
        Backend instance
    """
    if name in _backends:
        return _backends[name]

    if name == 'numpy':
        backend = Backend(
            name='numpy',
            xp=np,
            sparse=sp_sparse
        )
    elif name == 'cupy':
        if not CUPY_AVAILABLE:
            raise ImportError(
                f"CuPy GPU backend not available.\n"
                f"Reason: {CUPY_ERROR_MSG}\n"
                f"Install with: pip install cupy-cuda11x (for CUDA 11)\n"
                f"Or use backend='numpy' for CPU computation."
            )
        backend = Backend(
            name='cupy',
            xp=cp,
            sparse=cp_sparse
        )
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'numpy' or 'cupy'.")

    _backends[name] = backend
    return backend


def to_numpy(arr) -> np.ndarray:
    """Convert any array to numpy."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def is_cupy_available() -> bool:
    """Check if CuPy is available."""
    return CUPY_AVAILABLE
