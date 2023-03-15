# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# adapted from jax since we want the iterations and error as well as mpi4jax support


import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax._src.scipy.sparse.linalg import _identity, _vdot_real_tree, _sub, _add, _mul
from jax import lax


# adapted from 
# https://github.com/google/jax/blob/a683186570404a05a2aea9f7e452400bb0299e57/jax/_src/scipy/sparse/linalg.py#L105
def _cg_solve(A, b, x0=None, *, maxiter=1000, tol=1e-5, atol=0.0, M=_identity, token=None):

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k, token = value
        rs = gamma if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k, token = value
        Ap, token = A(p, token)
        alpha = gamma / _vdot_real_tree(p, Ap)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1, token

    Ax0, token = A(x0, token)
    r0 = _sub(b, Ax0)
    p0 = z0 = M(r0)
    gamma0 = _vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0, 0, token)

    x_final, *_, token = lax.while_loop(cond_fun, body_fun, initial_value)
    return x_final, _, token


@partial(jax.jit, static_argnames=("maxiter",), inline=True)
def cg(A, b, x0=None, *, maxiter=1000, tol=1e-5, atol=0.0, M=_identity, token=None):
    # cg with mpi4jax support, returns solver stats
    return _cg_solve(A, b, x0, maxiter=maxiter, tol=tol, atol=atol, M=M, token=token)
