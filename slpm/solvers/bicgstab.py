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

from jax._src.scipy.sparse.linalg import _identity, _vdot_real_tree, _sub, _add, _mul, dtypes, _vdot_tree
from jax._src.lax import lax as lax_internal
from jax import lax



# adapted from
# https://github.com/google/jax/blob/a683186570404a05a2aea9f7e452400bb0299e57/jax/_src/scipy/sparse/linalg.py#L143
# TODO add support for MPI M

def _bicgstab_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity, token=None):

    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

    def cond_fun(value):
        x, r, *_, k, token = value
        rs = _vdot_real_tree(r, r)
        # the last condition checks breakdown
        return (rs > atol2) & (k < maxiter) & (k >= 0)

    def body_fun(value):
        x, r, rhat, alpha, omega, rho, p, q, k, token = value
        rho_ = _vdot_tree(rhat, r)
        beta = rho_ / rho * alpha / omega
        p_ = _add(r, _mul(beta, _sub(p, _mul(omega, q))))
        phat = M(p_)
        q_, token = A(phat, token=token)
        alpha_ = rho_ / _vdot_tree(rhat, q_)
        s = _sub(r, _mul(alpha_, q_))
        exit_early = _vdot_real_tree(s, s) < atol2
        shat = M(s)
        t, token = A(shat, token=token)
        omega_ = _vdot_tree(t, s) / _vdot_tree(t, t)  # make cases?
        x_ = jax.tree_map(partial(jnp.where, exit_early), _add(x, _mul(alpha_, phat)), _add(x, _add(_mul(alpha_, phat), _mul(omega_, shat))))
        r_ = jax.tree_map(partial(jnp.where, exit_early), s, _sub(s, _mul(omega_, t)))
        k_ = jnp.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
        k_ = jnp.where((rho_ == 0), -10, k_)
        return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_, token

    val, token = A(x0, token=token)
    r0 = _sub(b, val)
    rho0 = alpha0 = omega0 = lax_internal._convert_element_type(1, *dtypes._lattice_result_type(*jax.tree_leaves(b)))
    initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0, token)

    x_final, *_, token = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final, _, token


# one needs to pass a Partial for A
@partial(jax.jit, static_argnames=("maxiter",), inline=True)
def bicgstab(A, b, x0, *, tol=1e-5, atol=0.0, maxiter=1000, M=_identity, token=None):
    # bicgstab with mpi4jax support, returns solver stats
    return _bicgstab_solve(A, b, x0, maxiter=maxiter, tol=tol, atol=atol, M=M, token=token)
