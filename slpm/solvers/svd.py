import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial
from .cholesky import add_diagonal_regularizer


@partial(jax.jit, inline=True)
def _solve_fn(v, sinv, uh, rhs):
    res = uh @ rhs
    # res = jnp.diag(sinv) @ res
    res = jnp.einsum("i, ...ij-> ij", sinv, res)
    res = v @ res
    return res


def get_svd_solve(A, diag_reg=None, cutoff=0):
    """
    do svd decomp and return solve func
    """
    if diag_reg is not None:
        A = add_diagonal_regularizer(A, diag_reg)
    u, s, vh = jnp.linalg.svd(A)
    uh = u.T.conj()
    v = vh.T.conj()
    sinv = (jnp.abs(s) > cutoff) * (1.0 / s)
    solver_stats = (s,)  # return the singular values so we can check them
    return Partial(_solve_fn, v, sinv, uh), solver_stats
