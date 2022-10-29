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
        in_radii : float = 1e-15,
        out_radii : float= 1e-8,
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
            mul = max(self.out_radii / norm, 1.0)
            pt = pt / self.norm(pt)

        reg_pt = straight_through_f(_regularize)(pt)
        return reg_pt

    def mobius_add(self, pt_a: Array, pt_b: Array) -> Array:
        """Mobius add operation

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns a new point on the manifold.
        """
        inp = jnp.dot(pt_a, pt_b)
        b_norm = jnp.linalg.norm(pt_b) ** 2
        a_norm = jnp.linalg.norm(pt_a) ** 2

        numerator = (1 - 2 * self.curv * inp - self.curv * b_norm) * pt_a + (
            1 + self.curv * a_norm
        ) * pt_b
        denominator = 1 - 2 * self.curv + self.curv**2 * b_norm * a_norm
        ma = numerator / denominator
        return ma

    def mobius_sub(self, pt_a: Array, pt_b: Array) -> Array:
        ms = self.mobius_add(pt_a, -1 * pt_b)
        return ms

    def gyration_op(self, pt_a: Array, pt_b: Array, vec: Array) -> Array:
        gb = self.mobius_add(pt_b, vec)
        ggb = self.mobius_add(pt_b, gb)
        gab = self.mobius_add(pt_a, pt_b)
        return -1 * self.mobius_add(gab, ggb)

    def cf(self, pt: Array) -> float:
        cp_norm = self.curv * jnp.linalg.norm(pt) ** 2
        cf = 2 / (1 + cp_norm)
        return cf

    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point.
            tv: tangent_vec.

        Returns:
            returns Exp_{bpt}(tv).
        """
        t = jnp.sqrt(jnp.abs(self.curv)) * jnp.linalg.norm(tv)
        pt = (jnp.tanh(t / 2 * self.cf(bpt)) / t) * t
        exp = self.mobius_add(bpt, pt)
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

    def metric(self, bpt, tv_a, tv_b):
        metric = self.cf(bpt) * jnp.inp(tv_a, tv_b)
        return metric

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point.
            e_pt: end point.
            tv: tangent vector at start point.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        pt = self.gyration_op(e_pt, -1 * s_pt, tv)
        return pt

    def dist(self, pt_a: Array, pt_b: Array) -> Array:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns distance between pt_a, pt_b.
        """
        t = (2 * self.curv * jnp.linalg.norm(pt_a - pt_b) ** 2) / (
            (1 + self.curv * jnp.inner(pt_a) ** 2)(1 + self.curv * jnp.inner(pt_b) ** 2)
        )
        dist = jnp.arccosh(1 - t) / (jnp.sqrt(jnp.abs(self.curv)))

    def mobius_matvec(self, mat: Array, vec: Array) -> Array:
        """Mobius matrix vector multiplication.

        Args:
            mat: arbitrary Matrix.
            vec: vector lying on Poincare ball.

        Returns:
            returns mobius version of mat @ vec which belongs to the poincare ball.
        """
        matvec = mat @ vec
        matvec_norm = jnp.linalg.norm(matvec)
        vec_norm = jnp.linalg.norm(vec)
        coeff = (1 / self.abs_sqrt_curv) * jnp.tanh(
            matvec_norm / vec_norm * jnp.arctanh(self.abs_sqrt_curv * vec_norm)
        )
        return coeff * matvec / matvec_norm

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


class LorentzHyperboloid(Hyperbolic):
    def __init__(self, m, curv=-1):
        self.m = m
        self.curv = curv
        super().__init__()

    def lorentz_inp(self, x, y):
        lip = jnp.inner(x, y) - 2 * x[0] * y[0]
        return lip

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> Array:
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

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
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
