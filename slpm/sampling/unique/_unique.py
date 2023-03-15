import jax
import jax.numpy as jnp
from functools import partial
import netket as nk
import numpy as np

from .utils import *
from ..sampling import _posterior_apply_fn
from jax._src.numpy.setops import _unique, _unique_sorted_mask

# functions to sample distinct configurations from a wavefunction, counting the number of occurences


def unique_with_weights(x, weights):
    """
    jax.numpy.unique but also returns the sum of weights of all the occurences of each unique element
    x: an array of elements
    w: an array with the weights of the elements
    """
    # TODO make padding optional?

    size = len(x)  # N+Ni
    # axis=0 so we already support multiple ints as keys (cols if you think of it as a matrix)
    x_new, index, inverse, counts, n_unique = _unique(x, 0, return_index=True, return_counts=True, return_inverse=True, return_true_size=True, size=size)
    maxcounts = counts.max()
    # TODO extract perm and mask from jax._src.numpy.lax_numpy._unique?
    aux, mask, perm = _unique_sorted_mask(x, 0)

    weights = weights[perm]
    start_idx = jnp.nonzero(mask, size=size + 1)[0][:-1]
    end_idx = start_idx + counts

    curr_idx = start_idx
    out_counts = jnp.zeros_like(weights)

    def body_fun(i, val):
        curr_idx = start_idx + i
        return val + weights[curr_idx] * (curr_idx < end_idx)

    return x_new, index, inverse, counts, n_unique, jax.lax.fori_loop(0, counts.max(), body_fun, out_counts)


# test:
# x = np.random.randint(1, 10, 16)
# weights = jnp.array(np.random.randint(1, 4, 16))
# for i in np.unique(x):
#     print(i, weights[jnp.where(x==i)].sum())

# print(unique_with_weights(x, weights)[-1])


# TODO we would actually need to keep track from which chain which samples came from for the stats


@partial(jax.jit, static_argnames=("ma", "N", "Ni", "red_fun"), inline=True)
def sample_unique(sa, ma, variables, sampler_state, N, Ni, maxiter, red_fun=lambda x: vec2int(x)):
    """
    monte-carlo sampling which only keeps unique samples, counting the number of repetitions
    until a predefined number of unique samples is reached
    uses a flax module as logpsi

    sa: a netket sampler
    ma: a flax module representing logpsi (duck typing allowed)
    variables: the variables of ma, representing e.g. parameters
    sampler_state: sampler state
    N: how many samples you want
    Ni: how many samples per iterative step step to add
    maxiter: how many iterations to do at most
    red_fun: function which returns a unique integer given a sample, for comparing different configurations; use it for symmetries

    how it works:
        - first generate N samples
        - then iteratively add Ni new samples
        - until there are >=N unique ones

    returns a tuple containing:
        - the unique samples
        - number of repetitions of each sample (w.r.t f)
        - final sampler state
        - Sampler stats: (number of unique samples found, number of iterations performed, number of samples sampled in total)
    """
    assert N <= 2**sa.hilbert.size

    # round up Ni to the next mutliple of n_chains
    chain_length_N = (N + (sa.n_chains - 1)) // sa.n_chains
    chain_length_Ni = (Ni + (sa.n_chains - 1)) // sa.n_chains
    # round up ni
    Ni = chain_length_Ni * sa.n_chains

    samp_0, sampler_state = sa.sample(ma, variables, state=sampler_state, chain_length=chain_length_N)
    i0 = red_fun(samp_0.reshape(-1, samp_0.shape[-1])[-N:])  # trash extra samples if N is not divisible by chain length
    x = jnp.zeros((N + Ni,) + i0.shape[1:], dtype=jnp.uint64)
    # fill x
    x = x.at[:N].set(i0)
    weights = jnp.zeros(N + Ni, dtype=x.dtype)
    weights = weights.at[:N].set(1)

    init_val = x, weights, sampler_state, jnp.zeros_like(x), jnp.zeros_like(weights), jnp.zeros(N + Ni, dtype=jnp.int64), jnp.array(0), jnp.array(0)

    def cond_fun(val):
        *_, n_unique, i = val
        return (n_unique < N) & (i < maxiter)

    def body_fun(val):
        x, weights, sampler_state, *_, i = val
        samp_i, sampler_state = sa.sample(ma, variables, state=sampler_state, chain_length=chain_length_Ni)
        x = x.at[-Ni:].set(red_fun(samp_i.reshape(-1, samp_i.shape[-1])))
        weights = weights.at[-Ni:].set(1)
        # x_new, index, n_unique = jax._src.numpy.lax_numpy._unique(x, 0, return_index=True, return_true_size=True, size=N+Ni)
        x_new, index, _, _, n_unique, weights_new = unique_with_weights(x, weights)
        return x_new, weights_new, sampler_state, x, weights, index, n_unique, i + 1

    out = jax.lax.while_loop(cond_fun, body_fun, init_val)

    x, weights, sampler_state, _, _, index, n_unique, n_iters = out
    # jnp.unique sorts unique states
    # but we want the order sampled, so revert that
    mask = np.arange(len(index)) < n_unique
    maxint = N + Ni + 1
    index = jnp.sort(jax.lax.select(mask, index, jnp.ones_like(index) * maxint))[:N]
    res = x[index]  # unique samples
    res2 = weights[index]  # how often each one was seen, useful for estimating observables
    n_total = weights.sum()
    # success = n_unique >=N
    return int2vec(res, samp_0.shape[-1], dtype=sa.dtype), res2, sampler_state, (n_unique, None, n_iters, n_total)


def do_sampling_unique(logpsi, sa, sampler_state, N, n_discard, f=lambda x: vec2int(x), maxiter=1000, ni_ratio=1):
    """
    monte-carlo sampling which only keeps unique samples, counting the number of repetitions
    until a predefined number of unique samples is reached

    logpsi: a jax.tree_util.Partial representing the log-amplitude of the wavefunction
    sa: a netket sampler
    sampler_state: sampler state
    N: how many samples you want
    n_discard: how many samples to discard at the beginning
    f: function which returns a unique integer given a sample, for comparing different configurations; use it for symmetries
    maxiter (optional): how many iterations to do at most; default is 1000
    ni_ratio: ratio of how many potential samples to add per iterative step, in terms of N; tune it to increase performance; default is 1

    how it works:
        - first generate N samples
        - then iteratively add Ni new samples
        - until there are >=N unique ones

    returns a tuple containing:
        - the unique samples
        - number of repetitions of each sample (w.r.t f)
        - final sampler state
        - Sampler stats: (number of unique samples found, number of iterations performed, number of samples sampled in total)
    """
    var = {"logpsi": logpsi, "params": ()}

    # discard
    _, sampler_state = sa.sample(_posterior_apply_fn, var, chain_length=n_discard, state=sampler_state)

    Ni = int(N * ni_ratio)

    return sample_unique(sa, _posterior_apply_fn, var, sampler_state, N, Ni, maxiter, f)


# test:
# hi = nk.hilbert.Spin(1/2, 12)
# sa = nk.sampler.MetropolisLocal(hi, n_chains=32)
# ma = nk.models.RBM()
# vs = nk.vqs.MCState(sa, ma)
# _ = vs.sample()
# sample_unique(sa, ma, vs.variables, vs.sampler_state, 512, 128, 1000)
