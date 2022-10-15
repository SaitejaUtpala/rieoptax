from functools import partial
from typing import Callable

from chex import Array

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
    def diff_expm(self, bpt: Array, tv: Array) -> Array:
        return self.diff_pow(bpt, tv, jnp.exp)

    @partial(jit, static_argnums=(0,))
    def diff_logm(self, bpt: Array, tv: Array) -> Array:
        return diff_pow(bpt, tv, jnp.log)



class SPDAffineInvariant(SPDManifold):
   
    def exp(self, bpt, tv):
        powers = self.sqrt_neg_sqrt(bpt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tv @ powers[1])
        m_exp = (jnp.exp(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        exp = powers[0] @ m_exp @ powers[0]
        return exp

    def log(self, bpt, pt):
        powers = self.sqrt_neg_sqrt(bpt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ pt @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        log = powers[0] @ middle_log @ powers[0]
        return log

    def inp(self, bpt, tv) : 

    def pt(self, bpt, tv_a, tv_b):


    def dist(self, pt_a, pt_b):
        eigval = jnp.linalg.eigvals(jnp.linalg.inv(pt_b) @ pt_a)
        dist = jnp.linalg.norm(jnp.log(eigval))
        return dist

    def egrad_to_rgrad(self, egrad, bpt):
        return bpt @ egrad @ bpt.T

    
class SPDLogEuclidean(SPDManifold):
    
    def exp(self, bpt, tv):
        log_bp = self.diff_logm(bpt, tv)
        return self.expm(self.logm(bpt) + log_bp )
        
    def log(self, bpt, pt):
        logm_bp = self.logm(bpt)
        logm_p = self.logm(pt) 
        return self.diff_expm(logm_bp, logm_p - logm_bp)
        
    def pt(self, start_point, end_point, tv):
        logm_ep = self.logm(end_point)
        tv = self.diff_logm(start_point, tv)
        return self.diff_expm(logm_ep, tv)    
    
    def inp(self, bpt, tangent_vec_a, tannget_vec_b):
        de_a = self.diff_logm(bpt, tangent_vec_a)
        de_b = self.diff_logm(bpt, tangent_vec_b)
        return jnp.inner(de_a, de_b)
    
    def dist(self, pt_a, pt_b):
        diff = self.logm(pt_a) - self.logm(pt_b)
        return self.norm(diff)
    
    def norm(self, bpt, tv):
        norm = self.diff_logm(bpt, tv)
        return self.norm(diff)
    
class SPDBuresWasserstein(SPDManifold):
    
    def exp(self, bpt, tv):
        lyp = self.lyapunov(bpt, tv)
        return bpt + tv + lyp @ bpt @ lyp
        
    def log(self, bpt, pt):
        powers = self.sqrt_neg_sqrt(bpt)
        pdt = self.sqrt(powers[0] @ pt @ powers[0])
        sqrt_product = powers[0] @ pdt @ powers[1]
        return sqrt_product + sqrt_product.T  - 2 * bpt
    
    def inp(self, bpt, tangent_vec_a, tangent_vec_b):
        lyp = self.lyapunov(bpt, tv)
        return 0.5 * self.trace_matprod(lyp, tv)
        
    def dist(self, pt_a, pt_b):
        sqrt_a =  self.sqrt(pt_a)
        prod = self.sqrt(sqrt_a @ pt_b @ sqrt_a)
        return jnp.trace(pt_a) + jnp.trace(pt_b)-2*jnp.trace(prod)
        
    def egrad_to_rgrad(self, bpt, egrad):
        return 4*self.symmetrize(egrad @ bpt)
        
class SPDEuclidean(SPDManifold):
    
    def inp(self, bpt, tangent_vec_a, tangent_vec_b):
        return self.trace_matprod(tangent_vec_a, tangent_vec_b)
    
    def exp(self, bpt, tv):
        return bpt + tv
   
    def log(self, bpt, pt) :
        return pt-bpt
        
    def egrad_to_rgrad(self, bpt, egrad):
        return egrad 
    
    def dist(self, pt_a, pt_b):
        return self.norm(pt_a - pt_b)
    
    def pt(self, start_point, end_point, tv):
        return tv
