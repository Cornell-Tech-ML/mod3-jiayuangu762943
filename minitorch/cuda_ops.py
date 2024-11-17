# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    """Create a CUDA device function using Numba.

    Args:
    ----
        fn (Fn): The function to decorate as a CUDA device function.
        **kwargs (Any): Additional keyword arguments for the Numba decorator.

    Returns:
    -------
        Fn: The CUDA-decorated device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    """Create a CUDA kernel function using Numba.

    Args:
    ----
        fn (Callable): The function to decorate as a CUDA kernel.
        **kwargs (Any): Additional keyword arguments for the Numba decorator.

    Returns:
    -------
        FakeCUDAKernel: The CUDA-decorated kernel function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(  # noqa: D102
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:  # noqa: D102
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return
        to_index(i, out_shape, out_index)
        # Handle broadcasting
        broadcast_index(out_index, out_shape, in_shape, in_index)
        out_pos = index_to_position(out_index, out_strides)
        in_pos = index_to_position(in_index, in_strides)
        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return
        # Convert linear index to multidimensional index
        to_index(i, out_shape, out_index)
        # Handle broadcasting
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        out_pos = index_to_position(out_index, out_strides)
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])


    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """  # noqa: D301, D404
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    # TODO: Implement for Task 3.3.
    # Load data into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    # Reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Write result to global memory
    if pos == 0:
        out[bid] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Apply the `_sum_practice` kernel to reduce an input tensor along blocks of a fixed size.

    Args:
    ----
        a (Tensor): Input tensor.

    Returns:
    -------
        TensorData: Reduced tensor data with size equal to the number of blocks.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x

        # TODO: Implement for Task 3.3.
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x

        # Each block handles one output element
        if bid >= out_size:
            return

        # Get the index into out tensor
        to_index(bid, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)

        # Initialize total
        total = reduce_value

        # Calculate the size of the reduction dimension
        reduce_size = a_shape[reduce_dim]

        # Each thread processes multiple elements along the reduction dimension
        for s in range(tid, reduce_size, BLOCK_DIM):
            # Build a_index from out_index, varying reduce_dim
            for dim in range(len(a_shape)):
                if dim == reduce_dim:
                    a_index[dim] = s
                else:
                    a_index[dim] = out_index[dim] if a_shape[dim] > 1 else 0
            a_pos = index_to_position(a_index, a_strides)
            total = fn(total, a_storage[a_pos])

        # Store the partial result in shared memory
        cache[tid] = total
        cuda.syncthreads()

        # Reduce within the block
        stride = BLOCK_DIM // 2
        while stride > 0:
            if tid < stride:
                cache[tid] = fn(cache[tid], cache[tid + stride])
            cuda.syncthreads()
            stride //= 2

        # Write the result to the output
        if tid == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """  # noqa: D404
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Shared memory for tiles of A and B
    sA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    row = ty
    col = tx

    # Load data into shared memory
    if row < size and col < size:
        a_pos = row * size + col
        b_pos = row * size + col
        sA[row, col] = a[a_pos]
        sB[row, col] = b[b_pos]
    else:
        sA[row, col] = 0.0
        sB[row, col] = 0.0
    cuda.syncthreads()

    # Perform matrix multiplication
    temp = 0.0
    if row < size and col < size:
        for k in range(size):
            temp += sA[row, k] * sB[k, col]
        out[row * size + col] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication on two small fixed-size matrices using `_mm_practice`.

    Args:
    ----
        a (Tensor): The left-hand side matrix.
        b (Tensor): The right-hand side matrix.

    Returns:
    -------
        TensorData: Resulting matrix from the multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    M = out_shape[-2]
    N = out_shape[-1]
    K = a_shape[-1]

    # Initialize accumulator
    temp = 0.0

    # Loop over tiles
    for t in range((K + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load A tile into shared memory
        if i < M and (t * BLOCK_DIM + pj) < K:
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index[0] = batch if a_shape[0] > 1 else 0
            a_index[1] = i
            a_index[2] = t * BLOCK_DIM + pj
            a_pos = index_to_position(a_index, a_strides)
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        # Load B tile into shared memory
        if j < N and (t * BLOCK_DIM + pi) < K:
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index[0] = batch if b_shape[0] > 1 else 0
            b_index[1] = t * BLOCK_DIM + pi
            b_index[2] = j
            b_pos = index_to_position(b_index, b_strides)
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0
        cuda.syncthreads()

        # Compute partial product
        for k in range(BLOCK_DIM):
            temp += a_shared[pi, k] * b_shared[k, pj]
        cuda.syncthreads()

    # Write result to global memory
    if i < M and j < N:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_index[0] = batch if out_shape[0] > 1 else 0
        out_index[1] = i
        out_index[2] = j
        out_pos = index_to_position(out_index, out_strides)
        out[out_pos] = temp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
