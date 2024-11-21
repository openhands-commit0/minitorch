from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numba import njit, prange
from .tensor_data import MAX_DIMS, broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps
if TYPE_CHECKING:
    from typing import Callable, Optional
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)

class FastOps(TensorOps):

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        def _map(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            tensor_map(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
            )
            return out
        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        def _zip(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            tensor_zip(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                b._tensor._storage,
                b._tensor._shape,
                b._tensor._strides,
            )
            return out
        return _zip

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float=0.0) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            tensor_reduce(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                dim,
            )
            return out
        return _reduce

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """
        # Setup shapes
        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        # Create output
        out = a.zeros(tuple(ls))

        # Call main function
        tensor_matrix_multiply(
            out._tensor._storage,
            out._tensor._shape,
            out._tensor._strides,
            a._tensor._storage,
            a._tensor._shape,
            a._tensor._strides,
            b._tensor._storage,
            b._tensor._shape,
            b._tensor._strides,
        )
        return out

def tensor_map(fn: Callable[[float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """
    @njit(parallel=True)
    def _map(out_storage: Storage,
             out_shape: Shape,
             out_strides: Strides,
             in_storage: Storage,
             in_shape: Shape,
             in_strides: Strides) -> None:
        # Check if the tensors are stride-aligned
        if np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape):
            for i in prange(len(out_storage)):
                out_storage[i] = fn(in_storage[i])
        else:
            # Create index buffers
            out_index = np.zeros(len(out_shape), np.int32)
            in_index = np.zeros(len(in_shape), np.int32)
            for i in prange(len(out_storage)):
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                in_position = index_to_position(in_index, in_strides)
                out_position = index_to_position(out_index, out_strides)
                out_storage[out_position] = fn(in_storage[in_position])
    return _map

def tensor_zip(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """
    @njit(parallel=True)
    def _zip(out_storage: Storage,
             out_shape: Shape,
             out_strides: Strides,
             a_storage: Storage,
             a_shape: Shape,
             a_strides: Strides,
             b_storage: Storage,
             b_shape: Shape,
             b_strides: Strides) -> None:
        # Check if the tensors are stride-aligned
        if (np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides) and
            np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape)):
            for i in prange(len(out_storage)):
                out_storage[i] = fn(a_storage[i], b_storage[i])
        else:
            # Create index buffers
            out_index = np.zeros(len(out_shape), np.int32)
            a_index = np.zeros(len(a_shape), np.int32)
            b_index = np.zeros(len(b_shape), np.int32)
            for i in prange(len(out_storage)):
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                a_position = index_to_position(a_index, a_strides)
                b_position = index_to_position(b_index, b_strides)
                out_position = index_to_position(out_index, out_strides)
                out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])
    return _zip

def tensor_reduce(fn: Callable[[float, float], float]) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """
    @njit(parallel=True)
    def _reduce(out_storage: Storage,
                out_shape: Shape,
                out_strides: Strides,
                in_storage: Storage,
                in_shape: Shape,
                in_strides: Strides,
                reduce_dim: int) -> None:
        # Create index buffers
        out_index = np.zeros(len(out_shape), np.int32)
        in_index = np.zeros(len(in_shape), np.int32)
        for i in prange(len(out_storage)):
            to_index(i, out_shape, out_index)
            # Setup initial
            in_index[:] = out_index[:]
            in_index[reduce_dim] = 0
            in_position = index_to_position(in_index, in_strides)
            reduced = in_storage[in_position]
            # Reduce over dimension
            for j in range(1, in_shape[reduce_dim]):
                in_index[reduce_dim] = j
                in_position = index_to_position(in_index, in_strides)
                reduced = fn(reduced, in_storage[in_position])
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = reduced
    return _reduce

def _tensor_matrix_multiply(out: Storage, out_shape: Shape, out_strides: Strides, a_storage: Storage, a_shape: Shape, a_strides: Strides, b_storage: Storage, b_shape: Shape, b_strides: Strides) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    # Get dimensions
    a_batch = a_shape[0] if len(a_shape) > 2 else 1
    b_batch = b_shape[0] if len(b_shape) > 2 else 1
    batch = max(a_batch, b_batch)
    m = a_shape[-2]
    n = b_shape[-1]
    p = a_shape[-1]

    # Main loop in parallel
    for b in prange(batch):
        for i in range(m):
            for j in range(n):
                # Compute output position
                out_pos = (
                    (b if len(out_shape) > 2 else 0) * out_strides[0] if len(out_shape) > 2 else 0
                ) + i * out_strides[-2] + j * out_strides[-1]

                # Initialize accumulator
                acc = 0.0

                # Inner loop - matrix multiply
                for k in range(p):
                    # Compute positions in a and b
                    a_pos = (
                        (b if len(a_shape) > 2 else 0) * a_strides[0] if len(a_shape) > 2 else 0
                    ) + i * a_strides[-2] + k * a_strides[-1]
                    b_pos = (
                        (b if len(b_shape) > 2 else 0) * b_strides[0] if len(b_shape) > 2 else 0
                    ) + k * b_strides[-2] + j * b_strides[-1]

                    # Multiply and accumulate
                    acc += a_storage[a_pos] * b_storage[b_pos]

                # Store result
                out[out_pos] = acc
tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)