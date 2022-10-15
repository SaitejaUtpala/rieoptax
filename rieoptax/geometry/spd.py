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

    def trace_matprod(self, mat_a: Array, mat_b: Array) -> float:
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
    def diff_expm(self, b_pt: Array, tv: Array) -> Array:
        return self.diff_pow(b_pt, tv, jnp.exp)

    @partial(jit, static_argnums=(0,))
    def diff_logm(self, b_pt: Array, tv: Array) -> Array:
        return diff_pow(b_pt, tv, jnp.log)


class SPDAffineInvariant(SPDManifold):
    def exp(self, b_pt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            b_pt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{b_pt}(tv).
        """
        powers = self.sqrt_neg_sqrt(b_pt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tv @ powers[1])
        m_exp = (jnp.exp(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        exp = powers[0] @ m_exp @ powers[0]
        return exp

    def log(self, b_pt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            b_pt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{b_pt}(pt).
        """
        powers = self.sqrt_neg_sqrt(b_pt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ pt @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        log = powers[0] @ middle_log @ powers[0]
        return log

    def inp(self, b_pt: Array, tv_a: Array, tv_b: Array) -> float:
        pass

    def pt(self, b_pt: Array, tv_a: Array, tv_b: Array) -> Array:
        pass

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        eigval = jnp.linalg.eigvals(jnp.linalg.inv(pt_b) @ pt_a)
        dist = jnp.linalg.norm(jnp.log(eigval))
        return dist

    def egrad_to_rgrad(self, egrad: Array, b_pt: Array) -> Array:
        return b_pt @ egrad @ b_pt.T


class SPDLogEuclidean(SPDManifold):
    def exp(self, b_pt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.
        
        Args:
            b_pt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{b_pt}(tv).
        """
        log_bp = self.diff_logm(b_pt, tv)
        return self.expm(self.logm(b_pt) + log_bp)

    def log(self, b_pt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            b_pt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{b_pt}(pt).
        """
        logm_bp = self.logm(b_pt)
        logm_p = self.logm(pt)
        return self.diff_expm(logm_bp, logm_p - logm_bp)

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point, a SPD matrix.
            e_pt: end point, a SPD matrix.
            tv: tangent vector at start point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        logm_ep = self.logm(e_pt)
        tv = self.diff_logm(s_pt, tv)
        return self.diff_expm(logm_ep, tv)

    def inp(self, b_pt, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            b_pt: base point, a SPD matrix.
            tv_a: tangent vector at base point, a Symmetric matrix.
            tv_b: tangent vector at base point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        de_a = self.diff_logm(b_pt, tv_a)
        de_b = self.diff_logm(b_pt, tv_b)
        return jnp.inner(de_a, de_b)

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        diff = self.logm(pt_a) - self.logm(pt_b)
        return self.norm(diff)

    def norm(self, b_pt: Array, tv: Array) -> float:
        norm = self.diff_logm(b_pt, tv)
        return self.norm(diff)


class SPDBuresWasserstein(SPDManifold):
    def exp(self, b_pt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.
        
        Args:
            b_pt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{b_pt}(tv).
        """
        lyp = self.lyapunov(b_pt, tv)
        return b_pt + tv + lyp @ b_pt @ lyp

    def log(self, b_pt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            b_pt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{b_pt}(pt).
        """
        powers = self.sqrt_neg_sqrt(b_pt)
        pdt = self.sqrt(powers[0] @ pt @ powers[0])
        sqrt_product = powers[0] @ pdt @ powers[1]
        return sqrt_product + sqrt_product.T - 2 * b_pt

    def inp(self, b_pt: Array, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            b_pt: base point, a SPD matrix.
            tv_a: tangent vector at base point, a Symmetric matrix.
            tv_b: tangent vector at base point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        lyp = self.lyapunov(b_pt, tv)
        return 0.5 * self.trace_matprod(lyp, tv)

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        sqrt_a = self.sqrt(pt_a)
        prod = self.sqrt(sqrt_a @ pt_b @ sqrt_a)
        return jnp.trace(pt_a) + jnp.trace(pt_b) - 2 * jnp.trace(prod)

    def egrad_to_rgrad(self, b_pt: Array, egrad: Array) -> float:
        return 4 * self.symmetrize(egrad @ b_pt)


class SPDEuclidean(SPDManifold):
    def inp(self, b_pt: Array, tv_a: Array, tv_b: Array) -> Array:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            b_pt: base point, a SPD matrix.
            tv_a: tangent vector at base point, a Symmetric matrix.
            tv_b: tangent vector at base point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        return self.trace_matprod(tv_a, tv_b)

    def exp(self, b_pt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.
        
        Args:
            b_pt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{b_pt}(tv).
        """
        return b_pt + tv

    def log(self, b_pt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            b_pt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{b_pt}(pt).
        """
        return pt - b_pt

    def egrad_to_rgrad(self, b_pt: Array, egrad: Array) -> Array:
        return egrad

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        return self.norm(pt_a - pt_b)

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        return tv
