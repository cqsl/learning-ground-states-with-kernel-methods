import jax
from functools import partial, wraps
from jax.tree_util import Partial

from netket.jax._vmap_chunked import _chunk_vmapped_function


def block_kernel_fn(f, block_size):
    # support different block size in x and y
    if hasattr(block_size, "__len__"):
        block_size_0, block_size_1 = block_size
    else:
        block_size_0 = block_size_1 = block_size

    # @partial(jax.jit, inline=True)
    def kernel_fn_blocked(f, x, y):
        f1 = lambda x, y: f(x, y).T
        f2 = lambda x, y: _chunk_vmapped_function(f1, chunk_size=block_size_1, argnums=1)(x, y).T
        return _chunk_vmapped_function(f2, chunk_size=block_size_0, argnums=0)(x, y)

    return Partial(kernel_fn_blocked, f)


def block_local_val_kernel(local_val_kernel, block_size=None):
    if block_size is None:
        return local_val_kernel
    return _chunk_vmapped_function(local_val_kernel, chunk_size=block_size, argnums=1)
