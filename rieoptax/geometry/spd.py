from typing import Callable

from jax import grad, jit
from jax import numpy as jnp
from jax import vmap
from functools import partial
from rieoptax.geometry.base import RiemannianManifold


class SPDManifold(RiemannianManifold):
    def sqrt_neg_sqrt(self, spd):
        eigval, eigvec = jnp.linalg.eigh(spd[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @ eigvec.swapaxes(1, 2)
        return result

    def diff_pow(self, spd: jnp.array, sym: jnp.array, power_fun: Callable):
        e_val, e_vec = jnp.linalg.eigh(spd)
        pow_e_val = power_fun(e_val)
        deno = e_val[:, None] - e_val[None, :]
        nume = pow_e_val[:, None] - pow_e_val[None, :]
        same_sub = vmap(grad(power_fun))(e_val)[:, None]
        diff_pow_diag = jnp.where(deno != 0, nume / deno, same_sub)
        diag = (e_vec.T @ sym @ e_vec) * diff_pow_diag
        return e_vec @ diag @ e_vec.T

    @partial(jit, static_argnums=(0,))
    def diff_exp(self, base_point, tangent_vec):
        return self.diff_pow(base_point, tangent_vec, jnp.exp)

    @partial(jit, static_argnums=(0,))
    def diff_log(self, base_point, tangent_vec):
        return self.diff_pow(base_point, tangent_vec, jnp.log)


class SPDAffineInvariant(SPDManifold):
    def __init__(self, m):
        self.m = m
        super().__init__()

    def exp(self, base_point, tangent_vec):

        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tangent_vec @ powers[1])
        middle_exp = (jnp.exp(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        exp = powers[0] @ middle_exp @ powers[0]
        return exp

    def log(self, base_point, point):
        powers = self.sqrt_neg_sqrt(base_point)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ point @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        log = powers[0] @ middle_log @ powers[0]
        return log

    def dist(self, point_a, point_b):
        eigval = jnp.linalg.eigvals(jnp.linalg.inv(point_b) @ point_a)
        dist = jnp.linalg.norm(jnp.log(eigval))
        return dist

    def egrad_to_rgrad(self, egrad, base_point):
        return base_point @ egrad @ base_point.T
