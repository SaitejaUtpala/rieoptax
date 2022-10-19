from abc import ABC, abstractmethod
from functools import partial

from chex import Array
from jax import jit
from jax import numpy as jnp
from jax import vmap

from rieoptax.geometry.base import RiemannianManifold


class Hyperbolic(RiemannianManifold):
    pass


class PoincareBall(Hyperbolic):
    def __init__(self, dim, curv=-1):
        self.dim = dim
        self.curv = curv

    def mobius_addition(self, pt_a : Array, pt_b : Array)-> Array:
        """_summary_

        Args:
            pt_a (_type_): _description_
            pt_b (_type_): _description_

        Returns:
            _type_: _description_
        """
        inp = jnp.dot(pt_a, pt_b)
        b_norm = jnp.norm(pt_b) ** 2
        a_norm = jnp.norm(pt_a) ** 2

        numerator = (
            1 - 2 * self.curv * inp - self.curv * b_norm
        ) * pt_a + (1 + self.curv * a_norm) * pt_b
        denominator = 1 - 2 * self.curv + self.curv**2 * b_norm * a_norm
        ma = numerator / denominator
        return ma

    def mobius_subtraction(self, pt_a : Array, pt_b : Array)->Array:
        ms = self.mobius_addition(pt_a, -1 * pt_b)
        return ms

    def gyration_operator(self, pt_a: Array, pt_b: Array, vec : Array) -> Array:
        gb = self.mobius_addition(pt_b, vec)
        ggb = self.mobius_addition(pt_b, gb)
        gab = self.mobius_addition(pt_a, pt_b)
        return -1 * self.mobius_addition(gab, ggb)

    def conformal_factor(self, pt : Array) -> float:
        cp_norm = self.curv * jnp.norm(pt) ** 2
        cf = 2 / (1 + cp_norm)
        return cf

    def exp(self, tv : Array, bpt : Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        t = jnp.sqrt(jnp.abs(self.curv)) * jnp.norm(tv)
        pt = (jnp.tanh(t / 2 * self.conformal_factor(bpt)) / t) * t
        exp = self.mobius_addition(bpt, pt)
        return exp

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        ma = self.mobius_addition(-1 * bpt, pt)
        abs_sqrt_curv = jnp.sqrt(jnp.abs(self.curv))
        norm_ma = jnp.norm(ma)
        mul = (2 / (abs_sqrt_curv * self.conformal_factor(bpt))) * jnp.arctanh(
            abs_sqrt_curv * norm_ma
        )
        log = mul * (ma / norm_ma)
        return log

    def metric(self, bpt, tangent_vec_a, tangent_vec_b):
        metric = self.conformal_factor(bpt) * jnp.inp(
            tangent_vec_a, tangent_vec_b
        )
        return metric

    def parallel_transport(self, s_pt, e_pt, tv):
        """Parallel Transport.

        Args:
            s_pt: start poin.
            e_pt: end point.
            tv: tangent vector at start point.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        self.conformal_factor(s_pt)
        self.conformal_facotr(e_pt)
        pt = self.gyration_operator(e_pt, -1 * s_pt, tv)
        return pt

    def dist(self, pt_a, pt_b):
        t = (2 * self.curv * jnp.norm(pt_a - pt_b) ** 2) / (
            (1 + self.curv * jnp.inner(pt_a) ** 2)(
                1 + self.curv * jnp.inner(pt_b) ** 2
            )
        )
        dist = jnp.arccosh(1 - t) / (jnp.sqrt(jnp.abs(self.curv)))

    def tangent_gaussian(self, sigma):
        pass 


class LorentzHyperboloid(Hyperbolic):
    def __init__(self, m, curv=-1):
        self.m = m
        self.curv = curv
        super().__init__()

    def lorentz_inner(self, x, y):
        lip = jnp.inner(x, y) - 2* x[0] * y[0]
        return lip

    def inp(self, bpt, tangent_vec_a, tangent_vec_b):
        return self.lorentz_inner(tangent_vec_a, tangent_vec_b)

    def dist(self, pt_a, pt_b):
        dist = jnp.arccosh(self.curv * self.lorentz_inner(pt_a, pt_b)) / (
            jnp.sqrt(self.curv)
        )
        return dist
    
    def exp(self, bpt, tv):
        """Riemannian Exponential map.

        Args:
            bpt: base_point, a SPD matrix.
            tv: tangent_vec, a Symmetric matrix.

        Returns:
            returns Exp_{bpt}(tv).
        """
        tv_ln = jnp.sqrt(self.lorentz_inner(tv, tv)* jnp.abs(self.curv))
        exp = jnp.cosh(tv_ln) * bpt + (jnp.sinh(tv_ln) / tv_ln) * tv
        return exp

    def log(self, bpt, pt):
        """Riemannian Logarithm map.

        Args:
            bpt: base_point, a SPD matrix.
            pt: tangent_vec, a SPD matrix.

        Returns:
            returns Log_{bpt}(pt).
        """
        k_xy = self.curv * self.lorentz_inner(bpt, pt)
        arccosh_k_xy = jnp.arccosh(k_xy)
        log = (arccosh_k_xy / jnp.sinh(arccosh_k_xy) ) *(pt - (k_xy * bpt))
        return log

    def parallel_transport(self, s_pt, e_pt, tv):
        """Parallel Transport.

        Args:
            s_pt: start point, a SPD matrix.
            e_pt: end point, a SPD matrix.
            tv: tangent vector at start point, a Symmetric matrix.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        k_yv = self.curv * self.loretnz_inner(e_pt, tv)
        k_xy = self.curv * self.loretnz_inner(s_pt, e_pt)
        pt = tv - (k_yv / k_xy)(s_pt + e_pt)
        return pt

    def tangent_gaussian(self, sigma):
        pass  
