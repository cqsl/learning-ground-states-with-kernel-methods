import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, inline=True, static_argnames=("L", "out_dtype"))
def simple_kernel_fn(x, y, L, out_dtype, G, f):
    """
    simple symmetrized kernel

    x,y: vectors of inputs
    L: number of spins
    out_dtype: output dtype
    G: permutation matrix corresponding to the Permutation group
    f: nonlinearity; use an even function if you want Z2 symmetry; needs to be a jax.Partial so that we can change hyperparams of f without rejitting

    get G from netket with:
    # g = nk.graph....
    # use
    # G = np.array(g.translation_group())
    # G = np.array(g.space_group())
    # etc

    returns the kernel matrix for x and y

    """

    @partial(jax.vmap, in_axes=(0, None))  # x
    @partial(jax.vmap, in_axes=(None, 0))  # y
    def _kf(x, y):
        ys = y[G]
        i = (ys @ x).astype(out_dtype) / L
        res = f(i)
        res = res.mean()  # -> symmetry averaging
        return res

    return _kf(x, y)
