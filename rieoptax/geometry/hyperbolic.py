from abc import ABC, abstractmethod
from functools import partial
from math import sqrt
from typing import Any, Callable, Tuple

from chex import Array
from jax import jit
from jax import numpy as jnp
from jax import vmap

from rieoptax.core import straight_through_f
from rieoptax.geometry.base import RiemannianManifold

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any


class Hyperbolic(RiemannianManifold):
    def poincare_to_lorentz(self, pt: Array, curv: float) -> Array:
        """Poincare to Loretnz Isometric convertor.

        Args:
            pt: point on the PoincareBall manifold.
            curv: curvature of the manifold.

        Returns:
            returns loretnz representation of pt
        """
        return pt[1:] / (1 + sqrt(abs(curv)) * pt[0])

    def lorentz_to_poincare(self, pt: Array, curv: float) -> Array:
        """Loretnz to Poincare Isometric convertor.

        Args:
            pt: point on the LoretnzHyperboloid manifold.
            curv: curvature of the manifold.

        Returns:
            returns poincare representation of pt
        """
        pt_norm = jnp.linalg.norm(pt) ** 2
        denom = 1 - curv * pt_norm**2
        z = (1 + curv * pt_norm**2) / denom
        k = 2 * pt / denom
        return jnp.hstack([z, k])


class PoincareBall(Hyperbolic):
    def __init__(
        self,
        m: int,
        curv: float = -1.0,
        in_radii: float = 1e-12,
        out_radii: float = 1e-5,
    ):
        self.m = m
        self.curv = curv
        self.abs_sqrt_curv = sqrt(abs(self.curv))
        self.ref_point = jnp.zeros(m)
        self.in_radii = in_radii
        self.out_radii = out_radii

    def __str__(self) -> str:
        return f"hyperbolic@PoincareBall({self.m},{self.curv})"

    @classmethod
    def from_str(cls, m_str: str, curv_str: str):
        return cls(int(m_str), float(curv_str))

    def regularize(self, pt: Array) -> Array:
        def _regularize(pt):
            norm = jnp.linalg.norm(pt)
            in_pt = pt + jnp.maximum(norm, self.in_radii) - norm
            out_pt = in_pt / jnp.maximum(norm / (1 - self.out_radii), 1)
            return out_pt

        reg_pt = straight_through_f(_regularize)(pt)
        return reg_pt

    def mobius_add(self, pt_a: Array, pt_b: Array) -> Array:
        """Mobius addition operation.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns a new vector.
        """
        c = self.curv
        x = self.regularize(pt_a)
        y = self.regularize(pt_b)
        xy = jnp.inner(x, y)
        x2 = jnp.linalg.norm(pt_a) ** 2
        y2 = jnp.linalg.norm(pt_b) ** 2
        numerator = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denominator = 1 - 2 * c * xy + c**2 * y2 * x2
        ma = numerator / denominator
        return self.regularize(ma)

    def mobius_sub(self, pt_a: Array, pt_b: Array) -> Array:
        """Mobius subtraction operation.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns a new point on the manifold.
        """
        ms = self.mobius_add(pt_a, -1 * pt_b)
        return ms

    def gyra(self, pt_a: Array, pt_b: Array, vec: Array) -> Array:
        """Gyration operator.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.
            vec: any vector.

        Returns:
            returns gyr[pt_a, pt_b](vec) which is again any vector
        """
        c = self.curv
        x = self.regularize(pt_a)
        y = self.regularize(pt_b)
        z = vec
        xy = jnp.inner(x, y)
        xz = jnp.inner(x, z)
        yz = jnp.inner(y, z)
        x2 = jnp.linalg.norm(x) ** 2
        y2 = jnp.linalg.norm(y) ** 2

        nomin1 = ((-1 * c * xz * y2) - yz * (1 - 2 * c * xy)) * x
        nomin2 = (-1 * c * yz * x2 + xz) * y
        denom = 1 - 2 * xy + c**2 * x2 * y2
        res = z + 2 * c * ((nomin1 + nomin2) / denom)
        return res

    def cf(self, pt: Array) -> float:
        """Conformal factor.

        Args:
            pt: point on the manifold.

        Returns:
            returns conformal factor at pt
        """
        cf = 2 / (1 + self.curv * jnp.linalg.norm(pt) ** 2)
        return cf

    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point.
            tv: tangent_vec.

        Returns:
            returns Exp_{bpt}(tv).
        """
        t = sqrt(abs(self.curv)) * jnp.linalg.norm(tv)
        pt = jnp.tanh((t * self.cf(bpt)) / 2) * (tv / t)
        exp = self.regularize(self.mobius_add(bpt, pt))
        return exp

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point.
            pt: tangent_vec.

        Returns:
            returns Log_{bpt}(pt).
        """
        ma = self.mobius_add(-1 * bpt, pt)
        norm_ma = jnp.linalg.norm(ma)
        mul = (2 / (self.abs_sqrt_curv * self.cf(bpt))) * jnp.arctanh(
            self.abs_sqrt_curv * norm_ma
        )
        log = mul * (ma / norm_ma)
        return log

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> float:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold.
            tv_a: tangent vector at bpt.
            tv_b: tangent vector at bpt.

        Returns:
            returns <tv_a, tv_b>_{bpt}.
        """
        inp = (self.cf(bpt) ** 2) * jnp.inner(tv_a, tv_b)
        return inp

    def norm(self, bpt: Array, tv: Array) -> float:
        """Norm of tangent vector at a point on manifold.

        Args:
            bpt: point on the manifold.
            tv: tangent vector at bpt.

        Returns:
            returns ||tv||_{bpt}.
        """
        return jnp.sqrt(self.inp(bpt, tv, tv))

    def ptransp(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point.
            e_pt: end point.
            tv: tangent vector at start point.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        ptransp = self.gyra(e_pt, -1 * s_pt, tv) * (self.cf(s_pt) / self.cf(e_pt))
        return ptransp

    def dist(self, pt_a: Array, pt_b: Array) -> Array:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns distance between pt_a, pt_b.
        """
        t = (2 * self.curv * jnp.linalg.norm(pt_a - pt_b) ** 2) / (
            (1 + self.curv * jnp.linalg.norm(pt_a) ** 2)
            * (1 + self.curv * jnp.linalg.norm(pt_b) ** 2)
        )
        dist = jnp.arccosh(1 - t) / (sqrt(abs(self.curv)))
        return dist

    def mobius_matvec(self, mat: Array, vec: Array) -> Array:
        """Mobius matrix vector multiplication.

        Args:
            mat: arbitrary Matrix.
            vec: vector lying on Poincare ball.

        Returns:
            returns mobius version of mat @ vec which belongs to the poincare ball.
        """
        vec = self.regularize(vec)
        matvec = mat @ vec
        matvec_norm = jnp.linalg.norm(matvec)
        vec_norm = jnp.linalg.norm(vec)
        coeff = (1 / self.abs_sqrt_curv) * jnp.tanh(
            matvec_norm / vec_norm * jnp.arctanh(self.abs_sqrt_curv * vec_norm)
        )
        out = coeff * matvec / matvec_norm
        return self.regularize(out)

    def mobius_pw_prod(self, pt_a: Array, pt_b: Array) -> Array:
        """Mobius point wise product.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns mobius version of mat @ vec which belongs to the poincare ball.
        """
        # TODO : make it faster
        return self.mobius_matvec(jnp.diag(pt_a), pt_b)

    def mobius_f(self, f: Callable) -> Callable:
        """Generic mobius version of f: R^{m} -> R^{n}

        Args:
            f : any function.

        Returns:
            returns mobius version of f
        """

        def mobius_f(x: Array) -> Array:
            return self.exp(self.ref_point, f(self.log(self.ref_point, x)))

        return mobius_f

    def egrad_to_rgrad(self, bpt: Array, egrad: Array) -> Array:
        """Euclidean gradient to Riemannian gradient Convertor.

        Args:
            bpt: base_point.
            egrad: tangent_vec.

        Returns:
            returns Riemannian gradient.
        """
        return egrad / (self.cf(bpt) ** 2)

    def sdist_to_gyroplanes(self, bpt: Array, tv: Array, pt: Array) -> Array:
        """Signed distance to hypergyroplanes.

        Args:
            bpt: point on the manifold.
            tv: tangent vector at base point 'bpt'.
            pt: point on the manifold

        Returns:
            distance from 'pt' to hypergyroplane defined by 'bpt', 'tv'.
        """
        bpt = self.regularize(bpt)
        pt = self.regularize(pt)

        norm = jnp.linalg.norm(tv)
        add = self.mobius_add(-1 * bpt, pt)
        asc = self.abs_sqrt_curv
        dist_nomin = 2 * asc * jnp.inner(add, tv)
        dist_denom = (1 + self.curv * jnp.linalg.norm(add) ** 2) * norm
        sdist = jnp.arcsinh(dist_nomin / dist_denom) / asc
        return sdist


    def busemann(self, bpt: Array, ip: Array) -> float:
        """Buseman Projection of 'bpt' along geodesic from origin to ideal point 'ip' 

        Args:
            bpt: base point on the manifold.
            ip: Idea point (belongs to hypersphere).

        Returns:
            returns busemann coordinates of base point 'bpt' along 
            idea point 'ip'. 
        """
        return jnp.log(jnp.linalg.norm(bpt- ip)**2/(1- jnp.linalg.norm(bpt)**2))
         



class LorentzHyperboloid(Hyperbolic):
    def __init__(self, m, curv=-1):
        self.m = m
        self.curv = curv
        super().__init__()

    def lorentz_inp(self, x, y):
        lip = jnp.inner(x, y) - 2 * x[0] * y[0]
        return lip

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> Array:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold.
            tv_a: tangent vector at bpt.
            tv_b: tangent vector at bpt.

        Returns:
            returns <tv_a, tv_b>_{bpt}.
        """
        return self.lorentz_inp(tv_a, tv_b)

    def dist(self, pt_a: Array, pt_b: Array) -> Array:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns distance between pt_a, pt_b.
        """
        dist = jnp.arccosh(self.curv * self.lorentz_inner(pt_a, pt_b)) / (
            jnp.sqrt(self.curv)
        )
        return dist

    def exp(self, bpt: Array, tv: Array):
        """Riemannian Exponential map.

        Args:
            bpt: base_point.
            tv: tangent vector at base point 'bpt'.

        Returns:
            returns Exp_{bpt}(tv).
        """
        tv_ln = jnp.sqrt(self.lorentz_inner(tv, tv) * jnp.abs(self.curv))
        exp = jnp.cosh(tv_ln) * bpt + (jnp.sinh(tv_ln) / tv_ln) * tv
        return exp

    def log(self, bpt: Array, pt: Array):
        """Riemannian Logarithm map.

        Args:
            bpt: base point.
            pt: tangent vector at base point 'bpt'.

        Returns:
            returns Log_{bpt}(pt).
        """
        k_xy = self.curv * self.lorentz_inner(bpt, pt)
        arccosh_k_xy = jnp.arccosh(k_xy)
        log = (arccosh_k_xy / jnp.sinh(arccosh_k_xy)) * (pt - (k_xy * bpt))
        return log

    def ptrans(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point.
            e_pt: end point.
            tv: tangent vector at start point 's_pt'.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        k_yv = self.curv * self.lorentz_inp(e_pt, tv)
        k_xy = self.curv * self.lorentz_inp(s_pt, e_pt)
        pt = tv - (k_yv / k_xy)(s_pt + e_pt)
        return pt
