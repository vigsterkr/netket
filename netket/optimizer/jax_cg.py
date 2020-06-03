from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes

import jax
import jax.numpy as jnp

from jax import jit
from jax.scipy.sparse.linalg import cg
from jax.tree_util import tree_flatten
from netket.vmc_common import jax_shape_for_update
from netket.utils import jit_if_singleproc, n_nodes


def _cg_converged(residual, reltol):
    return residual < reltol


def _cg_not_done(pars):
    A, b, x, u, r, c, residual, reltol, prev_residual, iter_n, maxiter = pars
    return not (iter_n >= maxiter or _cg_converged(residual, reltol))


def _cg_not_done2(residual, iter_n, reltol, maxiter):
    return not (iter_n >= maxiter or _cg_converged(residual, reltol))


def _cg_loop(pars):
    A, b, x, u, r, c, residual, reltol, prev_residual, iter_n, maxiter = pars

    beta = residual ** 2 / prev_residual
    u = r + beta * u

    c = A(u)
    alpha = residual ** 2 / jax.np.dot(u, c)

    x += alpha * u
    r -= alpha * c

    prev_residual = residual
    residual = jax.np.linalg.norm(r)

    iteration += 1

    return A, b, x, u, r, c, residual, reltol, prev_residual, iter_n, maxiter


def jax_cg(A, b, x0=None, tol=1e-6, maxiter=None):
    if x0 is None:
        x0 = jax.np.zeros(b.shape, b.dtype)
        init_zero = True
    else:
        init_zero = False

    if maxiter == None:
        maxiter = jax.np.array(1000)

    u = jax.np.zeros(x0.shape, x0.dtype)
    r = b
    c = jax.np.zeros(x0.shape, x0.dtype)

    if init_zero:
        residual = jax.np.linalg.norm(b)
        reltol = residual * tol
    else:
        c = A(x0)
        r = b - c
        residual = jax.np.linalg.norm(r)
        reltol = jax.np.linalg.norm(b) * tol

    iter_n = jax.np.zeros((), jax.np.int32)
    prev_residual = jax.np.ones((), b.dtype)

    pars = (A, b, x0, u, r, c, residual, reltol, prev_residual, iter_n, maxiter)
    pars_res = jax.lax.while_loop(_cg_not_done, _cg_loop, pars)
    _, _, x, _, _, _, residual, _, _, iter_n, _ = pars_res

    return (x, residual, iter_n)


def jax_cg(A, b, x0=None, tol=1e-6, maxiter=None):
    if x0 is None:
        x0 = jax.np.zeros(b.shape, b.dtype)
        init_zero = True
    else:
        init_zero = False

    if maxiter == None:
        maxiter = jax.np.array(1000)

    u = jax.np.zeros(x0.shape, x0.dtype)
    r = b
    c = jax.np.zeros(x0.shape, x0.dtype)

    if init_zero:
        residual = jax.np.linalg.norm(b)
        reltol = residual * tol
    else:
        c = A(x0)
        r = b - c
        residual = jax.np.linalg.norm(r)
        reltol = jax.np.linalg.norm(b) * tol

    iter_n = jax.np.zeros((), jax.np.int32)
    prev_residual = jax.np.ones((), b.dtype)

    # pars = (A, b, x0, u, r, c, residual, reltol, prev_residual, iter_n, maxiter)
    # pars_res = jax.lax.while_loop(_cg_done, _cg_loop, pars)
    x = x0
    iter_n = 0
    while _cg_not_done2(residual, iter_n, reltol, maxiter):
        # print("status ", iter_n, residual)
        beta = residual ** 2 / prev_residual
        u = r + beta * u

        c = A(u)
        alpha = residual ** 2 / jax.np.dot(u, c)

        x += alpha * u
        r -= alpha * c

        prev_residual = residual
        residual = jax.np.linalg.norm(r)

        iter_n += 1

    # _, _, x0, _, _, _, residual, _, _, iter_n, _ = pars_res

    return (x, residual, iter_n)
