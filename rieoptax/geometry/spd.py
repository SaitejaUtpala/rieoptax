from functools import partial
from typing import Callable

from chex import Array

from jax import grad, jit
from jax import numpy as jnp
from jax import vmap

from rieoptax.geometry.base import RiemannianManifold


class SPDManifold(RiemannianManifold):
    def __int__(m):
        self.m = m
        super().__int__()

    def symmetrize(self, mat: Array) -> Array:
        """Symmetrization of matrix.

        Args:
            mat: matrix.

        Returns:
            returns symmetrized version of mat.
        """
        return (mat + mat.T) / 2

    def trace_matprod(self, mat_a: Array, mat_b: Array) -> float:
        """Trace of Product of Two matrices.

        Args:
            mat_a: matrix.
            mat_b: matrix.

        Returns:
            returns Trace[mat_a. mat_b].
        """
        return jnp.einsum("ij,ij->", mat_a, mat_b)

    def powm(self, spd: Array, pow_fun: Callable) -> Array:
        """Matrix power of spd matrix.

        Args :
            spd : SPD Matrix.
            pow_fun : power function to be applied to eigen values.

        Retuns :
            returns matrix power of spd.
        """
        e_val, e_vec = jnp.linalg.eigh(spd)
        return (pow_fun(e_val).reshape(1, -1) * e_vec) @ e_vec.T

    def logm(self, spd: Array) -> Array:
        """Matrix Logarithm of spd matrix.

        Args:
            spd: SPD Matrix.

        Returns:
            returns matirx logarithm of spd.
        """
        return powm(spd, jnp.log)

    def expm(self, sym: Array) -> Array:
        """Matrix Exponential of Symmetric matrix.

        Args:
            sym: symmetric matrix.

        Returns:
            returns matrix exponential of sym.
        """
        return powm(spd, jnp.exp)

    def sqrtm(self, spd: Array) -> Array:
        sqrt = partial(jnp.power, x2=0.5)
        return powm(spd, sqrt)

    def neg_sqrtm(self, spd: Array) -> Array:
        sqrt = partial(jnp.power, x2=-0.5)
        return powm(spd, sqrt)

    def sqrtm_neg_sqrtm(self, spd: Array) -> Array:
        """Compute matrix square root and negative matrix square root.
        Note : This computes matrix square root and negative square root
        in one go by using just single eigen decomposition and hence faster
        than caller than 'sqrtm' and 'neg_sqrtm'.

        Args:
            spd: SPD matrix.

        Returns:
            returns result where result[0] contains square root and
            result[1] contain negative square root.
        """
        eigval, eigvec = jnp.linalg.eigh(spd[None])
        pow_eigval = jnp.stack([jnp.power(eigval, 0.5), jnp.power(eigval, -0.5)])
        result = (pow_eigval * eigvec) @ eigvec.swapaxes(1, 2)
        return result

    def diff_pow(self, spd: Array, sym: Array, power_fun: Callable) -> Array:
        """Differential of SPD matrix power.

        Args:
            spd: point on the manifold, SPD matrix.
            sym: tangent vector at base point, symmetric matrix.
            power_fun: power function,

        Returns:
            returns differential of matrix power at base point spd and
            evaluated at sym.
        """
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
        """Differential of matrix exponential.

        Args:
            spd: point on the manifold, SPD matrix.
            sym: tangent vector at base point, symmetric matrix.

        Returns:
            returns differential of matrix exponential at base point 'spd'
            and evaluated at 'sym'.
        """
        return self.diff_pow(bpt, tv, jnp.exp)

    @partial(jit, static_argnums=(0,))
    def diff_logm(self, bpt: Array, tv: Array) -> Array:
        """Differential of matrix logarithm.

        Args:
            spd: point on the manifold, SPD matrix.
            sym: tangent vector at base point, symmetric matrix.

        Returns:
            returns differential of matrix logarithm at base point 'spd'
            and evaluated at tangent vector 'sym'.
        """
        return diff_pow(bpt, tv, jnp.log)

    def lyapunov(self, spd: Array, sym: Array) -> Array:
        """Lyapunov Equation solver i.e., solve for  spd. X + X. spd = sym

        Args:
            spd: SPD matrix.
            sym: Symmetric matrix.

        Returns:
            returns solution to Lyapunov.
        """
        e_val, e_vec = gs.linalg.eigh(spd)
        pair_sum = e_val[:, None] + e_val[None, :]
        rotated = e_vec.T @ sym @ e_vec
        sol = e_vec @ (rotated / pair_sum) @ e_vec.T
        return sol

    def sqrtm_ABinv(self, spd_a, spd_b):
        """Compute (spd_a. (spd_b)^{-1})^{1/2}.
        Note : spd_a. (spd_b)^{-1} need not be SPD matrix.
        Hence one cannot use eigen decomposition to compute matrix square root,
        and has to use 'sqrtm' rather which is not supported in GPUs currently.
        Following routine employs clever manipulation in such a way square root
        is taken for a SPD matrix. This is used in Affine Invariant Metric.

        Args:
            spd_a: SPD matrix.
            spd_b: SPD matrix.

        Returns:
            returns (spd_a. (spd_b)^{-1})^{1/2}.
        """
        powers = self.sqrtm_neg_sqrtm(spd_b)
        ans = powers[0] @ self.sqrtm(powers[1] @ spd_a @ powers[1]) @ powers[1]
        return ans

    def sqrtm_AB(self, spd_a, spd_b):
        """Compute (spd_a. spd_b)^{1/2}.
        Note : spd_a. spd_b need not be SPD matrix.
        Hence one cannot use eigen decomposition to compute matrix square root,
        and has to use 'sqrtm' rather which is not supported in GPUs currently.
        Following routine employs clever manipulation in such a way square root
        is taken for a SPD matrix. This is used in Bures Wasserstein Metric.

        Args:
            spd_a: SPD matrix.
            spd_b: SPD matrix.

        Returns:
            returns (spd_a. spd_b)^{1/2}.
        """
        powers = self.sqrtm_neg_sqrtm(spd_b)
        ans = powers[0] @ self.sqrtm(powers[0] @ spd_a @ powers[0]) @ powers[1]
        return ans


class SPDAffineInvariant(SPDManifold):
    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        powers = self.sqrtm_neg_sqrtm(bpt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ tv @ powers[1])
        m_exp = (jnp.exp(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        exp = powers[0] @ m_exp @ powers[0]
        return exp

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        powers = self.sqrtm_neg_sqrtm(bpt)
        eigval, eigvec = jnp.linalg.eigh(powers[1] @ pt @ powers[1])
        middle_log = (jnp.log(eigval).reshape(1, -1) * eigvec) @ eigvec.T
        log = powers[0] @ middle_log @ powers[0]
        return log

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold, a SPD matrix.
            tv_a: tangent vector at bpt, a Symmetric matrix.
            tv_b: tangent vector at bpt, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        bpt_inv = jnp.linalg.inv(bpt)
        return self.trace_matprod(bpt_inv @ tv_a, bpt_inv @ tv_b)

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point, a SPD matrix.
            e_pt: end point, a SPD matrix.
            tv: tangent vector at start point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        out = self.sqrtm_ABinv(e_pt, s_pt)
        pt = out @ tv @ out.T
        return pt

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold, a SPD matrix.
            pt_b: point on the manifold, a SPD matrix.

        Returns:
            returns distance between pt_a, pt_b.
        """
        eigval = jnp.linalg.eigvals(jnp.linalg.inv(pt_b) @ pt_a)
        dist = jnp.linalg.norm(jnp.log(eigval))
        return dist

    def egrad_to_rgrad(self, egrad: Array, bpt: Array) -> Array:

        return bpt @ egrad @ bpt.T


class SPDLogEuclidean(SPDManifold):
    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        log_bp = self.diff_logm(bpt, tv)
        return self.expm(self.logm(bpt) + log_bp)

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        logm_bp = self.logm(bpt)
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

    def inp(self, bpt, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold, a SPD matrix.
            tv_a: tangent vector at bpt, a Symmetric matrix.
            tv_b: tangent vector at bpt, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        de_a = self.diff_logm(bpt, tv_a)
        de_b = self.diff_logm(bpt, tv_b)
        return jnp.inner(de_a, de_b)

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold, a SPD matrix.
            pt_b: point on the manifold, a SPD matrix.

        Returns:
            returns distance between pt_a, pt_b.
        """
        diff = self.logm(pt_a) - self.logm(pt_b)
        return self.norm(diff)

    def norm(self, bpt: Array, tv: Array) -> float:
        norm = self.diff_logm(bpt, tv)
        return self.norm(diff)


class SPDBuresWasserstein(SPDManifold):
    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        lyp = self.lyapunov(bpt, tv)
        return bpt + tv + lyp @ bpt @ lyp

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        sqrt_product = self.sqrtm_AB(pt, bpt)
        return sqrt_product + sqrt_product.T - 2 * bpt

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold, a SPD matrix.
            tv_a: tangent vector at bpt, a Symmetric matrix.
            tv_b: tangent vector at bpt, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        lyp = self.lyapunov(bpt, tv)
        return 0.5 * self.trace_matprod(lyp, tv)

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold, a SPD matrix.
            pt_b: point on the manifold, a SPD matrix.

        Returns:
            returns distance between pt_a, pt_b.
        """
        sqrt_a = self.sqrtm(pt_a)
        prod = self.sqrtm(sqrt_a @ pt_b @ sqrt_a)
        return jnp.trace(pt_a) + jnp.trace(pt_b) - 2 * jnp.trace(prod)

    def egrad_to_rgrad(self, bpt: Array, egrad: Array) -> float:
        return 4 * self.symmetrize(egrad @ bpt)


class SPDEuclidean(SPDManifold):
    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> Array:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold, a SPD matrix.
            tv_a: tangent vector at bpt, a Symmetric matrix.
            tv_b: tangent vector at bpt, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        return self.trace_matprod(tv_a, tv_b)

    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        return bpt + tv

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        return pt - bpt

    def egrad_to_rgrad(self, bpt: Array, egrad: Array) -> Array:
        """Euclidean gradient to Riemannian Gradient Convertor.

        Args:
            bpt: base_point, a SPD matrix.
            r: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        return egrad

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold, a SPD matrix.
            pt_b: point on the manifold, a SPD matrix.

        Returns:
            returns distance between pt_a, pt_b.
        """
        return self.norm(pt_a - pt_b)

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point, a SPD matrix.
            e_pt: end point, a SPD matrix.
            tv: tangent vector at start point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        return tv
