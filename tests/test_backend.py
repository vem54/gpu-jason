"""
Tests for GPU/CPU backend abstraction.

Run with: pytest tests/test_backend.py -v
"""

import pytest
import numpy as np
from scipy import sparse as sp_sparse

from gpu_poker_cfr.engine.backend import (
    get_backend,
    to_numpy,
    is_cupy_available,
    Backend
)


class TestNumpyBackend:
    """Test NumPy (CPU) backend."""

    @pytest.fixture
    def backend(self):
        return get_backend('numpy')

    def test_backend_name(self, backend):
        assert backend.name == 'numpy'

    def test_array_creation(self, backend):
        arr = backend.array([1, 2, 3])
        assert arr.shape == (3,)
        assert arr.dtype == np.float32

    def test_zeros(self, backend):
        arr = backend.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert np.all(arr == 0)

    def test_ones(self, backend):
        arr = backend.ones(5)
        assert arr.shape == (5,)
        assert np.all(arr == 1)

    def test_maximum(self, backend):
        arr = backend.array([-1, 0, 1, 2])
        result = backend.maximum(arr, 0)
        expected = np.array([0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_where(self, backend):
        arr = backend.array([1, 2, 3, 4])
        result = backend.where(arr > 2, arr, backend.zeros(4))
        expected = np.array([0, 0, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_asnumpy_is_identity(self, backend):
        arr = backend.array([1, 2, 3])
        result = backend.asnumpy(arr)
        assert isinstance(result, np.ndarray)


class TestSparseOperations:
    """Test sparse matrix operations on NumPy backend."""

    @pytest.fixture
    def backend(self):
        return get_backend('numpy')

    def test_sparse_to_backend(self, backend):
        # Create scipy sparse
        scipy_csr = sp_sparse.csr_matrix(np.eye(3, dtype=np.float32))
        backend_csr = backend.sparse_to_backend(scipy_csr)
        assert sp_sparse.isspmatrix_csr(backend_csr)

    def test_spmv(self, backend):
        """Test sparse matrix-vector multiply."""
        # Identity matrix
        A = sp_sparse.csr_matrix(np.eye(3, dtype=np.float32))
        x = backend.array([1, 2, 3])
        y = backend.spmv(A, x)
        np.testing.assert_array_almost_equal(backend.asnumpy(y), [1, 2, 3])

    def test_spmv_with_real_matrix(self, backend):
        """Test spmv with non-trivial matrix."""
        # [[1, 2], [3, 4]]
        A = sp_sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        x = backend.array([1, 1])
        y = backend.spmv(A, x)
        np.testing.assert_array_almost_equal(backend.asnumpy(y), [3, 7])

    def test_hadamard_spmv_vector(self, backend):
        """Test Hadamard product sparse multiply with vector."""
        # A = [[1, 1], [0, 1]]
        A = sp_sparse.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.float32))
        s = backend.array([0.5, 0.5])  # action probs
        x = backend.array([2, 4])

        # (A ⊙ S) where S[i,j] = s[j]
        # A ⊙ S = [[0.5, 0.5], [0, 0.5]]
        # (A ⊙ S) @ x = [[0.5, 0.5], [0, 0.5]] @ [2, 4] = [3, 2]

        result = backend.hadamard_spmv(A, s, x)
        expected = np.array([3, 2], dtype=np.float32)
        np.testing.assert_array_almost_equal(backend.asnumpy(result), expected)


class TestBackendEquivalence:
    """Test that operations produce same results."""

    @pytest.fixture
    def numpy_backend(self):
        return get_backend('numpy')

    def test_operations_match_numpy(self, numpy_backend):
        """Verify backend operations match raw numpy."""
        arr = numpy_backend.array([1, -2, 3, -4])

        # maximum
        result = numpy_backend.maximum(arr, 0)
        expected = np.maximum(np.array([1, -2, 3, -4], dtype=np.float32), 0)
        np.testing.assert_array_equal(result, expected)

        # sum
        assert numpy_backend.sum(arr) == np.sum(np.array([1, -2, 3, -4], dtype=np.float32))


class TestCuPyBackend:
    """Test CuPy (GPU) backend - skipped if CuPy not available."""

    @pytest.fixture
    def backend(self):
        if not is_cupy_available():
            pytest.skip("CuPy not available")
        return get_backend('cupy')

    def test_backend_name(self, backend):
        assert backend.name == 'cupy'

    def test_array_creation(self, backend):
        arr = backend.array([1, 2, 3])
        assert arr.shape == (3,)
        # Verify it's on GPU
        import cupy as cp
        assert isinstance(arr, cp.ndarray)

    def test_asnumpy_transfers_to_cpu(self, backend):
        arr = backend.array([1, 2, 3])
        result = backend.asnumpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_sparse_to_backend(self, backend):
        scipy_csr = sp_sparse.csr_matrix(np.eye(3, dtype=np.float32))
        backend_csr = backend.sparse_to_backend(scipy_csr)
        # Should be cupyx sparse
        import cupyx.scipy.sparse as cp_sparse
        assert cp_sparse.isspmatrix_csr(backend_csr)

    def test_spmv(self, backend):
        scipy_csr = sp_sparse.csr_matrix(np.eye(3, dtype=np.float32))
        A = backend.sparse_to_backend(scipy_csr)
        x = backend.array([1, 2, 3])
        y = backend.spmv(A, x)
        np.testing.assert_array_almost_equal(backend.asnumpy(y), [1, 2, 3])


class TestGPUCPUEquivalence:
    """Test that GPU and CPU produce identical results."""

    def test_array_ops_equivalent(self):
        if not is_cupy_available():
            pytest.skip("CuPy not available for equivalence test")

        np_backend = get_backend('numpy')
        cp_backend = get_backend('cupy')

        data = [1, -2, 3, -4, 5]

        np_arr = np_backend.array(data)
        cp_arr = cp_backend.array(data)

        # Test maximum
        np_result = np_backend.maximum(np_arr, 0)
        cp_result = cp_backend.maximum(cp_arr, 0)
        np.testing.assert_array_almost_equal(
            np_backend.asnumpy(np_result),
            cp_backend.asnumpy(cp_result)
        )

        # Test sum
        assert np_backend.sum(np_arr) == pytest.approx(cp_backend.asnumpy(cp_backend.sum(cp_arr)))

    def test_sparse_ops_equivalent(self):
        if not is_cupy_available():
            pytest.skip("CuPy not available for equivalence test")

        np_backend = get_backend('numpy')
        cp_backend = get_backend('cupy')

        # Create test matrix and vector
        A_scipy = sp_sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
        x_np = np.array([1, 2], dtype=np.float32)

        A_np = np_backend.sparse_to_backend(A_scipy)
        A_cp = cp_backend.sparse_to_backend(A_scipy)

        x_np_arr = np_backend.dense_to_backend(x_np)
        x_cp_arr = cp_backend.dense_to_backend(x_np)

        # Test spmv
        y_np = np_backend.spmv(A_np, x_np_arr)
        y_cp = cp_backend.spmv(A_cp, x_cp_arr)

        np.testing.assert_array_almost_equal(
            np_backend.asnumpy(y_np),
            cp_backend.asnumpy(y_cp)
        )


def test_get_backend_caches():
    """Verify backends are cached."""
    b1 = get_backend('numpy')
    b2 = get_backend('numpy')
    assert b1 is b2


def test_invalid_backend_raises():
    """Invalid backend name should raise."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend('invalid')


def test_to_numpy_from_numpy():
    """to_numpy should handle numpy arrays."""
    arr = np.array([1, 2, 3])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)
