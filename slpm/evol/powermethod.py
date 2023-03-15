import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial


# Apply the shifted hamiltonian
# Λ - H
# local val kernels for imag time evol and powermethod
# !!! these assume that the 1st col of the connected elements is the diagonal !!!
# which is the case for e.g. our ising and heisenberg operators
#
# also they compute the energy for free at the same time


@partial(jax.jit, inline=True, static_argnames=("return_logpsi_x", "sgn_fn", "return_eloc"))
def local_val_kernel_log_lambda(logpsi_fun, x, lambd, op_kernel, sgn_fn=None, return_logpsi_x=False, return_eloc=False):
    """
    apply one power method step in log-space: compute log[(Λ - H) exp(logψ)] for all configurations in x

    logpsi_fun: function returning the log-amplitues. needs to be wrapped in a jax.tree_util.Partial
    x: input conficurations
    lambd: the value of the diagonal shift Λ
    op_kernel: a function which returns the connected configurations and matrix elements of the operator H, given a config x
               !! The first config/mel returned is assumed to be the diagonal !!
    return_logpsi_x: also return logψ(x)
    return_eloc: return the local energies H Psi(x) / Psi(x) (~for free)
    sgn_fn (optional): a function which returns the sign of ψ(x) given x; useful for using positive real ψ if the sign is known a priori
    """
    assert isinstance(op_kernel, Partial)  # user needs to pass a Partial as op_kernel..
    xp, mels = op_kernel(x)
    logpsi_xp = logpsi_fun(xp.reshape((-1,) + x.shape[1:])).reshape(mels.shape)
    logpsi_x = logpsi_xp[:, 0:1]
    if sgn_fn is None:
        sgn_x = sgn_xp = 1
    else:
        sgn_xp = sgn_fn(xp.reshape((-1,) + x.shape[1:2])).reshape(mels.shape)
        sgn_x = sgn_xp[:, 0:1]
    eloc = jnp.sum(mels * sgn_x * sgn_xp * jnp.exp(logpsi_xp - logpsi_x), axis=1)
    mels_xp = -mels
    mels_xp = mels_xp.at[:, 0].add(lambd)
    res = jax.scipy.special.logsumexp(a=logpsi_xp + 0.0j, b=sgn_xp * mels_xp + 0.0j, axis=1).real
    if return_logpsi_x and return_eloc:
        return res, eloc, logpsi_x
    elif return_logpsi_x:
        return res, logpsi_x
    elif return_eloc:
        return res, eloc
    else:
        return res


@partial(jax.jit, inline=True, static_argnames=("return_psi_x", "sgn_fn", "return_eloc"))
def local_val_kernel_lambda(psi_fun, x, lambd, op_kernel, sgn_fn=None, return_psi_x=False, return_eloc=False):
    """
    apply one power method step in real space: compute (Λ - H) ψ for all configurations in x

    psi_fun: function returning the amplitues. needs to be wrapped in a jax.tree_util.Partial
    x: input conficurations
    lambd: the value of the diagonal shift Λ
    op_kernel: a function which returns the connected configurations and matrix elements of the operator H, given a config x
               !! The first config/mel returned is assumed to be the diagonal !!
    return_logpsi_x: also return logψ(x)
    return_eloc: return the local energies H Psi(x) / Psi(x) (~for free)
    sgn_fn (optional): a function which returns the sign of ψ(x) given x; useful for using positive real ψ if the sign is known a priori
    """
    assert isinstance(op_kernel, Partial)  # user needs to pass a Partial as op_kernel..
    xp, mels = op_kernel(x)
    psi_xp = psi_fun(xp.reshape((-1,) + x.shape[1:])).reshape(mels.shape)
    psi_x = psi_xp[:, 0:1]

    if sgn_fn is None:
        sgn_x = sgn_xp = 1
    else:
        sgn_xp = sgn_fn(xp.reshape((-1,) + x.shape[1:2])).reshape(mels.shape)
        sgn_x = sgn_xp[:, 0:1]

    eloc = jnp.sum(mels * sgn_x * sgn_xp * psi_xp / psi_x, axis=1)
    mels_xp = -mels
    mels_xp = mels_xp.at[:, 0].add(lambd)
    res = jnp.sum(psi_xp * sgn_xp * mels_xp, axis=1)
    if return_psi_x and return_eloc:
        return res, eloc, psi_x
    elif return_psi_x:
        return res, psi_x
    elif return_eloc:
        return res, eloc
    else:
        return res
