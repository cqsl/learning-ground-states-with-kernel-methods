import netket as nk
from netket.utils import mpi
import jax
import jax.numpy as jnp

import numpy as np

from functools import partial
from jax.tree_util import Partial

from slpm.sampling.unique.utils import *

# kernel ridge regression
# TODO could be extended to optionally do full GPR (covariance matrices take up a lot of memory...)


################################################################################
# PREDICTOR FUNTIONS
################################################################################

# if we jit here it gives
# AttributeError: 'CompiledFunction' object has no attribute '__code__'
# errors in some places, dunno
# @partial(jax.jit, inline=True)
def _predict_fn(kernel_fn, KXX_inv_Y, x_train, x_test):
    # kernel_fn is supposed to be a Partial (that way we can support changing the hyerparameters without rejitting)
    assert isinstance(kernel_fn, Partial)
    KxX = kernel_fn(x_test, x_train)
    μ = KxX @ KXX_inv_Y
    return μ


def _predict_fn_mpi(kernel_fn, KXX_inv_Y_ours, x_train_ours, x_test, token=None):
    # TODO check if it would be better to do the transpose (-> allreduce instead of sum)
    # kernel_fn is supposed to be a Partial (that way we can support changing the hyerparameters without rejitting)
    assert isinstance(kernel_fn, Partial)
    KxX = kernel_fn(x_test, x_train_ours)
    μ = KxX @ KXX_inv_Y_ours
    μ, token = nk.utils.mpi.mpi_sum_jax(μ, token=token)
    return μ, token


################################################################################
# SOLVING THE LSE | FIXED DATASET SIZE
################################################################################

# example for a solver:
# get_solve_fn = Partial(get_cho_solve, diag_reg=diag_reg)
@partial(jax.jit, inline=True)
def kernel_learn(kernel_fn, x_train, y_train, get_solve_fn):
    assert isinstance(kernel_fn, Partial)  # kernel_fn is supposed to be a Partial (that way we can support changing the hyerparameters without rejitting)

    KXX = kernel_fn(x_train, x_train)
    KXX_solve, solver_stats = get_solve_fn(KXX)

    assert y_train.ndim > 1  # user is supposed to add dummy dim if necessary; this way we can predict vector-valued stuff
    KXX_inv_Y = KXX_solve(y_train)
    predict_fn = Partial(_predict_fn, kernel_fn, KXX_inv_Y, x_train)
    return predict_fn, solver_stats


@partial(jax.jit, static_argnames=("maxiter", "solve_fn"))
def kernel_learn_iterative(kernel_fn, x_train, y_train, diag_reg, maxiter, solve_fn, token=None):
    # iterative (mostly for MPI)

    assert isinstance(kernel_fn, Partial)  # kernel_fn is supposed to be a Partial (that way we can support changing the hyerparameters without rejitting)

    # TODO add support for passing some x0

    x_train_ours = x_train
    y_train_ours = y_train
    if nk.utils.mpi.n_nodes > 1:
        x_train, token = nk.utils.mpi.mpi_allgather_jax(x_train, token=token)
        y_train, token = nk.utils.mpi.mpi_allgather_jax(y_train, token=token)
        x_train = x_train.reshape((-1,) + x_train.shape[2:])
        y_train = y_train.reshape((-1,) + y_train.shape[2:])

    # TODO special-case x_train_ours==x_train
    KXX = kernel_fn(x_train_ours, x_train)

    def _matvec(KXX, diag_reg, x, token):
        res = KXX @ x
        res, token = nk.utils.mpi.mpi_allgather_jax(res, token=token)
        res = res.reshape(x.shape)
        return res + diag_reg * x, token

    matvec = Partial(_matvec, KXX, diag_reg)

    # TODO eventually write a parallel cg which solves all at once and stops when all converged (and with mpi...)
    # for now we solve component for component via scan
    def _solve(token, y_train):
        x0 = jnp.zeros_like(y_train)
        KXX_inv_Y, solver_stats, token = solve_fn(matvec, y_train, x0=x0, maxiter=maxiter, token=token)
        return token, (KXX_inv_Y, solver_stats)

    assert y_train.ndim == 2  # user is supposed to add dummy dim if necessary; this way we can predict vector-valued stuff; for now we only support 1 extra dim

    token, (KXX_inv_Y, solver_stats) = jax.lax.scan(_solve, token, y_train.T)
    KXX_inv_Y = KXX_inv_Y.T

    N_per_rank = x_train_ours.shape[0]
    KXX_inv_Y_ours = KXX_inv_Y[mpi.rank * N_per_rank : (mpi.rank + 1) * N_per_rank]

    # we return both a predict fn which uses mpi and one without
    # where the user can just split the input manually which is probably better in most cases
    predict_fn = Partial(_predict_fn, kernel_fn, KXX_inv_Y, x_train)
    predict_fn_mpi = Partial(_predict_fn_mpi, kernel_fn, KXX_inv_Y_ours, x_train_ours)
    return predict_fn, predict_fn_mpi, solver_stats, token


################################################################################
# SOLVING THE LSE | DYNAMIC DATASET SIZE
################################################################################

# jit cache
# TODO apparently with newer versions of jax this is done internally
# even when jitting a Partial inside another function
@partial(jax.jit, inline=True)
def evalf(f, *args, **kwargs):
    return f(*args, **kwargs)


# padding helper function
@partial(jax.jit, static_argnames="npad")
def _pad0(x, npad):
    return jnp.pad(x, ((0, npad),) + ((0, 0),) * (x.ndim - 1), mode="constant", constant_values=0)


def kernel_learn_dynamic(kernel_fn, x_train, y_train, get_solve_fn):
    """
    Kernel Ridge Regression
    version supporting dynamic dataset sizes
    pads to the next power of 2 (so that only a log number of versions of the kernel needs to be compiled)
    Note that the solver is still compiled for every size
    """

    assert isinstance(kernel_fn, Partial)  # kernel_fn is supposed to be a Partial (that way we can support changing the hyerparameters without rejitting)

    # pad to the next power of 2
    n_train = len(x_train)
    n = n_train
    npad = 2 ** int(np.ceil(np.log2(n))) - n

    if npad > 0:
        x_train = _pad0(x_train, npad)

    KXX = evalf(kernel_fn, x_train, x_train)
    KXX_ = KXX[:n_train, :]
    del KXX
    KXX = KXX_[:, :n_train]
    del KXX_
    KXX_solve, solver_stats = evalf(get_solve_fn, KXX)
    del KXX

    assert y_train.ndim > 1  # user is supposed to add dummy dim if necessary; this way we can predict vector-valued stuff
    KXX_inv_Y = evalf(KXX_solve, y_train[:n_train])
    del KXX_solve
    if npad > 0:
        KXX_inv_Y = _pad0(KXX_inv_Y, npad)
    predict_fn = Partial(_predict_fn, kernel_fn, KXX_inv_Y, x_train)
    return Partial(evalf, predict_fn), solver_stats
