import jax
from functools import partial

# wrapper for the netket sampler
# we pass the (parameterised) function as a Partial in the variables
# that way we avoid re-jitting of _sample_chain
# (ideally netket would just accept a Partial directly....)


@jax.jit
def get_sampler_state(sa, key):
    """
    initialize a sampler state for do_sampling

    sa: a netket sampler
    key: a jax.random.PRNGKey
    """
    dummy_var = {"logpsi": lambda x: jnp.zeros(x.shape[0]), "params": ()}
    sampler_state = sa._init_state(None, None, key)
    return sampler_state


def _posterior_apply_fn(variables, x):
    logpsi = variables["logpsi"]
    return logpsi(x)


# @partial(jax.jit, static_argnames=("sa", "chain_length", "n_discard"), inline=True)
def do_sampling(logpsi, sa, sampler_state, chain_length, n_discard):
    """
    logpsi: a jax.tree_util.Partial for the log-amplitudes
    sa: a netket sampler
    sampler_state: sampler state initialized with get_sampler_state
    chain_length: how many samples
    n_discard: how many samples to discard at the beginning
    """
    var = {"logpsi": logpsi, "params": ()}
    x, sampler_state = sa.sample(_posterior_apply_fn, var, chain_length=chain_length + n_discard, state=sampler_state)
    x = x[n_discard:].reshape(-1, x.shape[-1])  # remove chain dim
    return x, sampler_state
