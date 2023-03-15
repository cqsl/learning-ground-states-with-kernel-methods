import jax
import jax.numpy as jnp
import netket as nk

from functools import partial

# step part 4: normalize


@partial(jax.jit, inline=True)
def pseudo_renormalize_log(y, token):
    # normalization in logspace:
    # subtract the largest element
    # -> the exponential of the returned value is <= 1
    m, token = nk.utils.mpi.mpi_max_jax(jnp.max(y), token=token)
    return y - m, m, token


@partial(jax.jit, inline=True)
def pseudo_renormalize(y, token):
    # normalization in real space:
    # divide by the mean
    m, token = nk.utils.mpi.mpi_mean_jax(jnp.mean(y.real), token=token)
    return y / m, token
