import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from functools import partial

from jax._src.scipy.linalg import _cholesky, _cho_solve
_cho_solve = partial(_cho_solve.__wrapped__, lower=False)

def add_diagonal_regularizer(A, diag_reg):
    return A.at[jnp.diag_indices_from(A)].add(diag_reg)


def get_cho_solve(A, diag_reg=None):
    """
    do cholesky decomp and return solve func
    """
    # lower = False
    if diag_reg is not None:
        A = add_diagonal_regularizer(A, diag_reg)
    C = _cholesky.__wrapped__(A, lower=False)
    return Partial(_cho_solve, C), None
