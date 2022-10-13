from rieoptax.geometry.base import RiemannianManifold

from jax import numpy as jnp


class HypersphereCanonicalMetric(RiemannianManifold):
    def __init__(self, m):
        self.m = m

    def random_point(self, key):
        x = jax.random.normal(key, shape=(self.m,))
        norm_x = x/jnp.linalg.norm(x)
        return norm_x 
        
    def exp(self, base_point, tangent_vec):
        norm = jnp.linalg.norm(tangent_vec)
        return base_point * jnp.cos(norm) + (tangent_vec/norm) * jnp.sin(norm)

    def log(self, base_point, point):
        coeff = self.dist(base_point, point)
        v = point.value-base_point.value
        proj = v - jnp.inner(v, base_point.value) * base_point.value
        log = coeff * (proj / jnp.linalg.norm(proj))
        return log

    def parallel_transport(self, start_point, end_point, tangent_vec):
        v = self.log(start_point, tangent_vec)
        v_norm = jnp.norm(v)
        inp = jnp.inner(v, tangent_vec)
        a = ((jnp.cos(v_norm) - 1) * inp) / v_norm
        b = (jnp.sin(v_norm) * inp) / v_norm
        pt = tangent_vec + a * tangent_vec - b * start_point
        return pt

    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        return jnp.inner(tangent_vec_a, tangent_vec_b)

    def dist(self, point_a, point_b):
        dist = jnp.arccos(jnp.clip(jnp.inner(point_a.value, point_b.value), -1, 1))
        return dist

    def tangent_gaussian(self, base_point, sigma):
        sample = jnp.random.normal(0, sigma, size=(self.m,1))
        zero_padded = jnp.hstack([jnp.zeros((1,)), sample])
        sample = self.parallel_transport(
            zero_padded, base_point=self.ref_point, end_point=base_point
        )
        return sample

    @staticmethod
    def egrad_to_rgrad(base_point, egrad):
        return egrad.value - jnp.inner(base_point.value, egrad.value) * base_point.value