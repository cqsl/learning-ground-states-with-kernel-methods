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
from jax._src.scipy.sparse.linalg import _normalize_matvec, _norm, _safe_normalize, _iterative_classical_gram_schmidt
from jax._src.scipy.sparse.linalg import _lstsq, _dot, _apply_givens_rotations, _rotate_vectors
from jax._src.lax import lax as lax_internal
from jax import lax


# adapted from 
# https://github.com/google/jax/blob/a683186570404a05a2aea9f7e452400bb0299e57/jax/_src/scipy/sparse/linalg.py#L460
def _gmres_incremental(A, b, x0, unit_residual, residual_norm, ptol, restart, M, token):
    # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf

    V = jax.tree_map(
        lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
        unit_residual,
    )
    dtype = jnp.result_type(*jax.tree_leaves(b))
    # use eye() to avoid constructing a singular matrix in case of early
    # termination
    R = jnp.eye(restart, restart + 1, dtype=dtype)

    givens = jnp.zeros((restart, 2), dtype=dtype)
    beta_vec = jnp.zeros((restart + 1), dtype=dtype)
    beta_vec = beta_vec.at[0].set(residual_norm)

    def loop_cond(carry):
        k, err, _, _, _, _, token = carry
        return jnp.logical_and(k < restart, err > ptol)

    def arnoldi_qr_step(carry):
        k, _, V, R, beta_vec, givens, token = carry
        V, H, _, token = _kth_arnoldi_iteration(k, A, M, V, R, token=token)
        R_row, givens = _apply_givens_rotations(H[k, :], givens, k)
        R = R.at[k, :].set(R_row)
        beta_vec = _rotate_vectors(beta_vec, k, *givens[k, :])
        err = abs(beta_vec[k + 1])
        return k + 1, err, V, R, beta_vec, givens, token

    carry = (0, residual_norm, V, R, beta_vec, givens, token)
    carry = lax.while_loop(loop_cond, arnoldi_qr_step, carry)
    k, residual_norm, V, R, beta_vec, _, token = carry

    y = jax.scipy.linalg.solve_triangular(R[:, :-1].T, beta_vec[:-1])
    dx = jax.tree_map(lambda X: _dot(X[..., :-1], y), V)

    x = _add(x0, dx)
    val, token = A(x, token=token)
    residual = M(_sub(b, val))
    unit_residual, residual_norm = _safe_normalize(residual)
    # TODO(shoyer): "Inner loop tolerance control" on ptol, like SciPy
    return x, unit_residual, residual_norm, k, token


def _kth_arnoldi_iteration(k, A, M, V, H, token):
    eps = jnp.finfo(jnp.result_type(*jax.tree_leaves(V))).eps

    v = jax.tree_map(lambda x: x[..., k], V)  # Gets V[:, k]
    val, token = A(v, token=token)
    v = M(val)
    _, v_norm_0 = _safe_normalize(v)
    v, h = _iterative_classical_gram_schmidt(V, v, v_norm_0, max_iterations=2)

    tol = eps * v_norm_0
    unit_v, v_norm_1 = _safe_normalize(v, thresh=tol)
    V = jax.tree_map(lambda X, y: X.at[..., k + 1].set(y), V, unit_v)

    h = h.at[k + 1].set(v_norm_1)
    H = H.at[k, :].set(h)
    breakdown = v_norm_1 == 0.0
    return V, H, breakdown, token


def _gmres_batched(A, b, x0, unit_residual, residual_norm, ptol, restart, M, token):
    del ptol  # unused
    # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
    V = jax.tree_map(
        lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, restart),)),
        unit_residual,
    )
    dtype, weak_type = dtypes._lattice_result_type(*jax.tree_leaves(b))
    H = lax_internal._convert_element_type(jnp.eye(restart, restart + 1, dtype=dtype), weak_type=weak_type)

    def loop_cond(carry):
        _, _, breakdown, k, token = carry
        return jnp.logical_and(k < restart, jnp.logical_not(breakdown))

    def arnoldi_process(carry):
        V, H, _, k, token = carry
        V, H, breakdown, token = _kth_arnoldi_iteration(k, A, M, V, H, token=token)
        return V, H, breakdown, k + 1, token

    carry = (V, H, False, 0, token)
    V, H, _, k, token = lax.while_loop(loop_cond, arnoldi_process, carry)

    beta_vec = jnp.zeros_like(H, shape=(restart + 1,)).at[0].set(residual_norm)
    y = _lstsq(H.T, beta_vec)
    dx = jax.tree_map(lambda X: _dot(X[..., :-1], y), V)

    x = _add(x0, dx)

    val, token = A(x, token=token)
    residual = M(_sub(b, val))
    unit_residual, residual_norm = _safe_normalize(residual)
    return x, unit_residual, residual_norm, k, token


def _gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func, token):
    val, token = A(x0, token=token)
    residual = M(_sub(b, val))
    unit_residual, residual_norm = _safe_normalize(residual)

    def cond_fun(value):
        _, k, _, residual_norm, _, token = value
        return jnp.logical_and(k < maxiter, residual_norm > atol)

    def body_fun(value):
        x, k, unit_residual, residual_norm, _, token = value
        x, unit_residual, residual_norm, k_, token = gmres_func(A, b, x, unit_residual, residual_norm, ptol, restart, M, token=token)
        return x, k + 1, unit_residual, residual_norm, k_, token

    initialization = (x0, 0, unit_residual, residual_norm, 0, token)
    x_final, *_, token = lax.while_loop(cond_fun, body_fun, initialization)
    return x_final, _, token


@partial(jax.jit, static_argnames=("maxiter", "restart", "solve_method"), inline=True)
def gmres(A, b, x0, *, tol=1e-5, atol=0.0, restart=20, maxiter=None, M=_identity, solve_method="batched", token=None):
    A = _normalize_matvec(A)
    M = _normalize_matvec(M)
    size = sum(bi.size for bi in jax.tree_leaves(b))

    if maxiter is None:
        maxiter = 10 * size  # copied from scipy
    restart = min(restart, size)

    b_norm = _norm(b)
    atol = jnp.maximum(tol * b_norm, atol)

    Mb = M(b)
    Mb_norm = _norm(Mb)
    ptol = Mb_norm * jnp.minimum(1.0, atol / b_norm)

    if solve_method == "incremental":
        gmres_func = _gmres_incremental
    elif solve_method == "batched":
        gmres_func = _gmres_batched
    else:
        raise ValueError(f"invalid solve_method {solve_method}, must be either 'incremental' or 'batched'")

    return _gmres_solve(A, b, x0, atol, ptol, restart, maxiter, M, gmres_func, token=token)
