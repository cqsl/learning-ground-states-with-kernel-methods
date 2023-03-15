import jax
import jax.numpy as jnp
from functools import partial
from slpm.krr import *
from slpm.evol import *
from slpm.sampling import *
from slpm.normalization import *
from jax.tree_util import Partial
from slpm.utils import Compose
from slpm.utils.stats import statistics

from netket.jax._vmap_chunked import _chunk_vmapped_function as _chunk_vmapped_function

# internal version of jax unique which returns all the quantities we need which are not returned by the public version
from jax._src.numpy.setops import _unique

_uniq = jax.jit(_unique, static_argnames=("return_inverse", "return_counts", "return_true_size", "size", "axis"), inline=True)


# there is two versions of step
# the first one has a constant number of samples and is fully jitted
# the second one has changing number of samples, which are padded to the next power of 2; this one has only the individual components jitted to allow for changing shapes


def _real_log(x):
    return jax.lax.log(x + 0.0j).real


_remove_final_dummy_dim = lambda x: x[..., 0]


@partial(jax.jit, static_argnames=("chain_length", "N_samples_per_rank", "n_discard", "iterative", "sample_unique", "notbehind", "realspace", "ni_ratio", "maxiter"))
def step(
    kernel_fn,
    x_train,
    y_train,
    y_train_prev,
    sa,
    sampler_state,
    chain_length,
    N_samples_per_rank,
    n_discard,
    get_solve_fn,
    iterative_solve_fn,
    diag_reg,
    iterative,
    sample_unique,
    notbehind,
    realspace,
    op,
    symm_repr,
    x_test,
    ni_ratio,
    maxiter,
    block_size_logpsi,
    token,
):
    """
    Perform one approximate step of the power method
    version with a constant number of samples in the dataset (fully jitted)
    """
    # step 1: learn
    if realspace:
        y_train = jax.lax.exp(y_train)
    if iterative:
        # kernel_learn_iterative takes care of gathering all the samples across mpi
        _pred_fn, pred_fn_mpi, solver_stats, token = kernel_learn_iterative(kernel_fn, x_train, y_train[..., jnp.newaxis], diag_reg=diag_reg, solve_fn=iterative_solve_fn, token=token, maxiter=maxiter)
        # in the following we only use pred_fn, since the input (here the samples) is split across mpi ranks already (e.g. in the sampling)
        pred_fn = Compose(_remove_final_dummy_dim, _pred_fn)
    else:
        _pred_fn, solver_stats = kernel_learn(kernel_fn, x_train, y_train[..., jnp.newaxis], get_solve_fn)
        pred_fn = Compose(_remove_final_dummy_dim, _pred_fn)
    if realspace:
        pred_fn = Compose(_real_log, pred_fn)

    if notbehind:
        pred_fn_maybe_propagated = Partial(op, pred_fn)
    else:
        pred_fn_maybe_propagated = pred_fn

    # step 2: sample

    if sample_unique:
        x_train_new, counts, sampler_state, sampler_stats = do_sampling_unique(pred_fn_maybe_propagated, sa, sampler_state, N_samples_per_rank, n_discard, symm_repr, ni_ratio=ni_ratio)
    else:
        x_train_new, sampler_state = do_sampling(pred_fn_maybe_propagated, sa, sampler_state, chain_length, n_discard)
        sampler_stats = None
        counts = None

    # step 3: propagate (and get energy predicition for ~free)
    y_train_new, eloc, y_train_prev = op(pred_fn, x_train_new, return_eloc=True, return_logpsi_x=True)
    y_train_new, shift, token = pseudo_renormalize_log(y_train_new, token)

    if not sample_unique:
        # split eloc back into the chains to get the good stats
        # for the unique sampling we can't do it as we don't keep
        # track of the order and on which chain each sample came from
        eloc = eloc.reshape((sa.n_chains_per_rank, -1))
    stats, token = statistics(eloc, token=token)

    if x_test is not None:
        y_test = _chunk_vmapped_function(pred_fn, block_size_logpsi)(x_test)
    else:
        y_test = None

    return x_train_new, y_train_new, y_train_prev, sampler_state, eloc, stats, solver_stats, counts, sampler_stats, y_test, shift, token


# jit cache to avoid recompilation
# TODO apparently with newer versions of jax this is done internally
# even when jitting a Partial inside another function
@partial(jax.jit, inline=True)
def evalf(f, *args, **kwargs):
    return f(*args, **kwargs)


def step_dynamic(
    kernel_fn,
    x_train,
    y_train,
    y_train_prev,
    sa,
    sampler_state,
    chain_length,
    N_samples_per_rank,
    n_discard,
    get_solve_fn,
    iterative_solve_fn,
    diag_reg,
    iterative,
    sample_unique,
    notbehind,
    realspace,
    op,
    symm_repr,
    x_test,
    ni_ratio,
    maxiter,
    block_size_logpsi,
    token,
):
    """
    Perform one approximate step of the power method
    version with a dynamic number of samples in the dataset (only functions called are jitted)
    """
    # step 1: learn
    assert not iterative  # not implemented
    assert sample_unique  # only unique is implemented
    assert not realspace  # not implemented

    _pred_fn, solver_stats = kernel_learn_dynamic(kernel_fn, x_train, y_train[..., jnp.newaxis], get_solve_fn)
    pred_fn = Compose(_remove_final_dummy_dim, _pred_fn)

    # step 2: sample
    if notbehind:
        pred_fn_maybe_propagated = Partial(op, pred_fn)
    else:
        pred_fn_maybe_propagated = pred_fn

    x_train_new, sampler_state = do_sampling(pred_fn, sa, sampler_state, chain_length, n_discard)
    ix_train_new = evalf(symm_repr, x_train_new)  # reduce w.r.t symm

    n_iters = None
    n_total = len(ix_train_new)

    ix_train_new, inverse, counts, n_unique = _uniq(ix_train_new, 0, return_inverse=True, return_counts=True, return_true_size=True, size=n_total)

    sampler_stats = (n_unique, inverse, n_iters, n_total)

    ntot = n_unique
    # use padded one to avoid recompilation
    x_train_new = int2vec(ix_train_new, x_train.shape[-1])
    counts = counts[:ntot]

    # step 3: propagate (and get energy predicition for ~free)

    y_train_new, eloc, y_train_prev = op(pred_fn, x_train_new, return_eloc=True, return_logpsi_x=True)
    y_train_new, shift, token = pseudo_renormalize_log(y_train_new, token)
    y_train_new = y_train_new[:ntot]
    x_train_new = x_train_new[:ntot]
    eloc = eloc[:ntot]

    stats, token = statistics(eloc[inverse].reshape(-1, sa.n_chains_per_rank), token=token)

    if x_test is not None:
        y_test = _chunk_vmapped_function(pred_fn, block_size_logpsi)(x_test)
    else:
        y_test = None

    return x_train_new, y_train_new, y_train_prev, sampler_state, eloc, stats, solver_stats, counts, sampler_stats, y_test, shift, token
