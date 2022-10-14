from functools import partial
from typing import Callable

from jax import grad, jit
from jax import numpy as jnp
from jax import vmap

from rieoptax.geometry.base import RiemannianManifold


class SPDManifold(RiemannianManifold):
    def symmetrize(self, mat: Array) -> Array:
        return (mat + mat.T) / 2

    def trace_mat_prod(self, mat_a: Array, mat_b: Array) -> float:
        return jnp.einsum("ij,ij->", mat_a, mat_b)

    def logm(self, spd: Array) -> Array:
        eigval, eigvec = jnp.linalg.eigh(spd)
        return (jnp.log(e_val).reshape(1, -1) * e_vec) @ e_vec.T

    def expm(self, sym: Array) -> Array:
        e_val, e_vec = jnp.linalg.eigh(sym)
        return (jnp.exp(e_val).reshape(1, -1) * e_vec) @ e_vec.T

    def lyapunov(self, spd: Array, sym: Array) -> Array:
        e_val, e_vec = gs.linalg.eigh(spd)
        pair_sum = e_val[:, None] + e_val[None, :]
        rotated = e_vec.T @ sym @ e_vec
        sol = e_vec @ (rotated / pair_sum) @ e_vec.T
        return sol

    def sqrt_neg_sqrt(self, spd: Array) -> Array:
        eigval, eigvec = jnp.linalg.eigh(spd[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @ eigvec.swapaxes(1, 2)
        return result

    def diff_pow(self, spd: Array, sym: Array, power_fun: Callable) -> Array:
        e_val, e_vec = jnp.linalg.eigh(spd)
        pow_e_val = power_fun(e_val)
        deno = e_val[:, None] - e_val[None, :]
        nume = pow_e_val[:, None] - pow_e_val[None, :]
        same_sub = vmap(grad(power_fun))(e_val)[:, None]
        diff_pow_diag = jnp.where(deno != 0, nume / deno, sub)
        diag = (e_vec.T @ sym @ e_vec) * diff_pow_diag
        d_pow = e_vec @ diag @ e_vec.T
        return d_pow

    @partial(jit, static_argnums=(0,))
    def diff_expm(self, base_point: Array, tangent_vec: Array) -> Array:
        return self.diff_pow(base_point, tangent_vec, jnp.exp)

    @partial(jit, static_argnums=(0,))
    def diff_logm(self, base_point: Array, tangent_vec: Array) -> Array:
        return diff_pow(base_point, tangent_vec, jnp.log)



class SPDAffineInvariant(SPDManifold):
   
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

    
class SPDLogEuclidean(SPDManifold):
    
    def exp(self, base_point, tangent_vec):
        log_bp = self.diff_logm(base_point, tangent_vec)
        return self.expm(self.logm(base_point) + log_bp )
        
    def log(self, base_point, point):
        logm_bp = self.logm(base_point)
        logm_p = self.logm(point) 
        return self.diff_expm(logm_bp, logm_p - logm_bp)
        
    def pt(self, start_point, end_point, tangent_vec):
        logm_ep = self.logm(end_point)
        tv = self.diff_logm(start_point, tangent_vec)
        return self.diff_expm(logm_ep, tv)    
    
    def inp(self, base_point, tangent_vec_a, tannget_vec_b):
        de_a = self.diff_logm(base_point, tangent_vec_a)
        de_b = self.diff_logm(base_point, tangent_vec_b)
        return jnp.inner(de_a, de_b)
    
    def dist(self, point_a, point_b):
        diff = self.logm(point_a) - self.logm(point_b)
        return self.norm(diff)
    
    def norm(self, base_point, tangent_vec):
        norm = self.diff_logm(base_point, tangent_vec)
        return self.norm(diff)
    
