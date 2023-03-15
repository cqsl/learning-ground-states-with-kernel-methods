import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial


@partial(jax.jit, inline=True)
def _ising_kernel_jax(x, edges, h, J):

    n_sites = x.shape[1]
    n_conn = n_sites + 1

    mels = jnp.zeros((x.shape[0], n_conn))  # TODO dtype?
    mels = mels.at[:, 1:].set(-h)
    mels = mels.at[:, 0].add(J * (x[:, edges[:, 0]] * x[:, edges[:, 1]]).sum(axis=-1))

    def _flip_lower_diag(x):
        _flip = jax.lax.neg
        cond = jnp.eye(*x.shape[-2:], k=-1, dtype=bool)
        cond = jax.lax.broadcast(cond, x.shape[:-2])
        return jax.lax.select(cond, _flip(x), x)

    x_prime = jax.lax.broadcast_in_dim(x, (x.shape[0], n_conn, n_sites), (0, 2))
    x_prime = _flip_lower_diag(x_prime)

    return x_prime, mels


def get_ising_kernel_jax(edges, h, J):
    return Partial(_ising_kernel_jax, edges=edges, h=h, J=J)
