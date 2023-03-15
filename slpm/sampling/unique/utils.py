import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

# utilities to deterministically convert vectors of spin configurations [-1,1,1,1,-1,1,...]
# to integers, by interpreting them as bitstrings.
#
# Beware that when working with restricted hilbert spaces (e.g. total_sz=0) in netket
# netket returns DIFFERENT, incompatible integers than these functions here.

# Below are the utilities for finding representatives for each configuration using the minimum integer representation convention


@partial(jax.jit, inline=True, static_argnames=("N", "qubit", "dtype"))
def int2vec(x, N, qubit=False, dtype=None):
    """
    convert integers into vectors of spin configurations
    x: an array of integers encoding the spin configurations
        If N<=64 a single integer, so shape [:]
        If N>64 a vector of two integers for each configuration in x, so shape [:, 2]

    N: the number of spins
    qubit: set true if the x contains qubits encoded with 0,1 instead of -1,1
    dtype: output dtype, defaults to int64/uint64 for spins/qubits


    returns an array of (flatened) spin configs [[-1,1,1,...], [1,-1,-,1], ...]
    """
    assert x.dtype == jnp.uint64  # make sure to use the correct dtype
    if N <= 64:
        v = np.flip(2 ** np.arange(N, dtype=np.uint64))
        res = jax.lax.bitwise_and(jnp.expand_dims(x, -1), v[(jnp.newaxis,) * x.ndim]) != 0
        if not qubit:
            if dtype is None:
                dtype = jnp.int64
            return 2 * res.astype(dtype) - 1
        else:
            if dtype is None:
                dtype = jnp.uint64
            return res.astype(dtype)
    else:  # N>64
        n, r = divmod(N, 64)
        lengths = (64,) * n
        if r != 0:
            lengths = (r,) + lengths
        start = np.cumsum((0,) + lengths[:-1])
        end = np.cumsum(lengths)
        return jnp.concatenate([int2vec(i, N, qubit=qubit, dtype=dtype) for i, N in zip(jnp.moveaxis(x, -1, 0), lengths)], axis=-1)


@partial(jax.jit, inline=True, static_argnames=("qubit",))
def vec2int(x, qubit=False):
    """
    convert vectors of spin configurations into integers
    x: an array of (flatened) spin configs [[-1,1,1,...], [1,-1,-,1], ...]
    qubit: set true if the x contains qubits encoded with 0,1 instead of -1,1

    If the number of spins is <=64 returns a single int
    If the number of spins is >64 returns two integers for each configuration in x
    """
    N = x.shape[-1]
    if not qubit:
        x = ((x + 1) // 2).astype(jnp.uint64)
    if N <= 64:
        v = np.flip(2 ** np.arange(N, dtype=np.uint64))
        return x.dot(v)
    else:  # N>64
        n, r = divmod(N, 64)
        lengths = (64,) * n
        if r != 0:
            lengths = (r,) + lengths
        start = np.cumsum((0,) + lengths[:-1])
        end = np.cumsum(lengths)
        return np.moveaxis(jnp.array([vec2int(x[..., l:r], qubit=qubit) for l, r in zip(start, end)]), 0, -1)


@partial(jax.jit, inline=True, static_argnames="N")
@partial(jax.vmap, in_axes=(0, None))
def shiftall(x, N):
    # using shifts of the bin repr
    # N = x.shape[-1]
    mask = np.array(2**N - 1, dtype=np.uint64)
    # x = vec2int(x)
    assert x.dtype == jnp.uint64
    xs = []
    for i in range(N):
        xs.append(jnp.bitwise_and(mask, jnp.left_shift(x, i)) + jnp.right_shift(x, (N - i)))
    res = jnp.array(xs).min()
    # return int2vec(res, N), res
    return res


# g = nk.graph....
# use
# G = np.array(g.translation_group())
# G = np.array(g.space_group())
# etc
@partial(jax.jit, static_argnames=("spin_flip_symm", "ret_int"), inline=True)
def shiftall_mat(x, G, spin_flip_symm=False, ret_int=False):
    """
    find represenative for each configuration in x w.r.t a symmetry group, using the minimum integer convention
    x: an array of (flatened) spin configs [[-1,1,1,...], [1,-1,-,1], ...]
    G: an integer matrix of permutations
    spin_flip_symm: use also Z2 symmetry in addition to the permutations in G, taking the minimum integer of both x and -x

    get G from netket with:
    # g = nk.graph....
    # use
    # G = np.array(g.translation_group())
    # G = np.array(g.space_group())
    # etc

    if ret_int:
        return the integer representation
    else: (default)
        return the vector of spin configs

    """
    # TODO lexsort for multiple ints if N>64/32
    x_ = x.reshape(x.shape[0], -1)
    N = x.shape[-1]
    x__ = x_[:, G]
    if N <= 64:
        r = vec2int(x__).min(axis=1)
        if spin_flip_symm:
            r2 = vec2int(-x__).min(axis=1)
            r = jax.lax.min(r, r2)
        if ret_int:
            return r
        return int2vec(r, x_.shape[1]).reshape(x.shape)
    else:

        def lexmin(x):
            # TODO more efficient, don't need full sort
            ind = jnp.lexsort(np.moveaxis(x, -1, 0))[..., 0]
            return x.reshape((-1,) + x.shape[-2:])[np.arange(len(ind.ravel())), ind.ravel()].reshape(ind.shape + x.shape[-1:])

        r = lexmin(vec2int(x__))
        if spin_flip_symm:
            r2 = lexmin(vec2int(-x__))
            # TODO write a lexmin for a tuple
            r = lexmin(jnp.concatenate([r[..., jnp.newaxis, :], r2[..., jnp.newaxis, :]], axis=-2))
        if ret_int:
            return r
        return int2vec(r, x_.shape[1]).reshape(x.shape)
