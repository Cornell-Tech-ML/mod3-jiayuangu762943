from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange  # type: ignore
from numba import njit as _njit  # type: ignore

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """A decorator to apply Numba's `@njit` (no-Python mode just-in-time compilation)
    to the given function with additional default options.

    Args:
    ----
        fn (Fn): The function to be compiled by Numba's `njit`.
        **kwargs (Any): Optional keyword arguments to customize the Numba `njit` decorator behavior.
                        These may include options such as `nogil`, `fastmath`, etc.

    Returns:
    -------
        Fn: The same function wrapped with Numba's `@njit`, optimized with the specified arguments.

    Notes:
    -----
        - The `inline="always"` option is enforced by default for this wrapper,
          which hints to Numba that the function should be inlined during compilation.
        - Additional `kwargs` are passed directly to the Numba `njit` decorator.
        - Requires Numba to be installed and properly configured.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def mul_reduce(a: Tensor, dim: int) -> Tensor:  # noqa: D102
        return FastOps.reduce(operators.mul, start=1.0)(a, dim)  # type: ignore # noqa: F821

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        size = np.prod(out_shape)

        if np.array_equal(out_shape, in_shape) and np.array_equal(
            out_strides, in_strides
        ):
            # Stride-aligned, avoid indexing
            for i in prange(size):
                out[i] = fn(float(in_storage[i]))
        else:
            for i in prange(size):
                out_index = np.empty(len(out_shape), dtype=np.int32)
                in_index = np.empty(len(in_shape), dtype=np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                out_pos = index_to_position(out_index, out_strides)
                in_pos = index_to_position(in_index, in_strides)
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        size = np.prod(out_shape)  # Total elements in output
        # if (np.array_equal(out_shape, a_shape)
        #     and np.array_equal(a_shape, b_shape)
        #     and np.array_equal(out_strides, a_strides)
        #     and np.array_equal(a_strides, b_strides)
        # ):
        #     # Stride-aligned case
        #     for i in prange(size):
        #         out[i] = fn(float(a_storage[i]), float(b_storage[i]))
        # else:
        for i in prange(size):
            out_index = np.empty(len(out_shape), dtype=np.int32)
            a_index = np.empty(len(a_shape), dtype=np.int32)
            b_index = np.empty(len(b_shape), dtype=np.int32)

            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        out_size = len(out)
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)

        for i in range(out_size):
            # Get the index into 'out'
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            # Initialize total with 'start' value from 'out[out_pos]'
            total = out[out_pos]

            # Iterate over the reduction dimension
            for s in range(a_shape[reduce_dim]):
                # Build 'a_index' from 'out_index', varying 'reduce_dim'
                for dim in range(len(a_shape)):
                    if dim == reduce_dim:
                        a_index[dim] = s
                    else:
                        a_index[dim] = out_index[dim]
                a_pos = index_to_position(a_index, a_strides)
                total = fn(total, a_storage[a_pos])
            out[out_pos] = total

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
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
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    batch_size = int(out_shape[0])
    M = int(out_shape[1])
    N = int(out_shape[2])
    K = int(a_shape[2])  # a_shape[-1] == b_shape[-2]

    # Precompute batch strides (0 if batch size is 1 for broadcasting)
    out_batch_stride = int(out_strides[0])

    # Precompute other strides
    a_M_stride = int(a_strides[1])
    a_K_stride = int(a_strides[2])
    b_K_stride = int(b_strides[1])
    b_N_stride = int(b_strides[2])
    out_M_stride = int(out_strides[1])
    out_N_stride = int(out_strides[2])

    for batch in prange(batch_size):
        # Handle broadcasting over batch dimension
        a_batch_index = int(batch if a_shape[0] > 1 else 0)
        b_batch_index = int(batch if b_shape[0] > 1 else 0)

        a_batch_offset = a_batch_index * a_batch_stride
        b_batch_offset = b_batch_index * b_batch_stride
        out_batch_offset = batch * out_batch_stride

        for i in range(M):
            a_row_offset = a_batch_offset + i * a_M_stride
            out_row_offset = out_batch_offset + i * out_M_stride
            for j in range(N):
                total = 0.0
                for k in range(K):
                    a_pos = a_row_offset + k * a_K_stride
                    b_pos = b_batch_offset + k * b_K_stride + j * b_N_stride
                    total += a_storage[int(a_pos)] * b_storage[int(b_pos)]
                out_pos = out_row_offset + j * out_N_stride
                out[int(out_pos)] = total


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
