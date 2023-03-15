import netket as nk
from jax.tree_util import Partial
from functools import reduce


def _gof(f, g, args_f, kwargs_f, args_g, kwargs_g, *args, **kwargs):
    return f(*args_f, g(*args_g, *args, **kwargs_g, **kwargs), **kwargs_f)


def Compose(*funcs):
    """
    function composition which also works with jax.tree_util.Partial and has stable hash
    compose(f,g,h)(x) is equivalent to f(g(h(x)))
    """

    def _compose(f, g):
        if not isinstance(f, Partial):
            f = Partial(f)
        if not isinstance(g, Partial):
            g = Partial(g)
        return Partial(nk.jax.HashablePartial(_gof, f.func, g.func), f.args, f.keywords, g.args, g.keywords)

    return reduce(_compose, funcs)
