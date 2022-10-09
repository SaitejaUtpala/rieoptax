from base import RiemannianManifold
from jax import numpy as jnp

class GrassmannCanonical(RiemannianManifold):
    def __init__(self, m, r):

        self.m = m
        self.r = r
        self.dim = m * r - r * r
        self.shape = (m, r)
        self.ref_point = jnp.eye(m, r)

    def exp(self, base_point, tangent_vec):
        u, s, vt = jnp.linalg.svd(tangent_vec, full_matrices=False)
        exp = (
            base_point @ (vt.T * jnp.cos(s).reshape(1, -1)) @ vt
            + (u * jnp.sin(s).reshape(1, -1)) @ vt
        )
        return exp

    def retr(self, base_point, tangent_vec):
        # print("a")
        u, _, vt = jnp.linalg.svd(base_point + tangent_vec, full_matrices=False)
        return u @ vt

    def log(self, base_point, point):
        ytx = point.T @ base_point
        At = point.T - ytx @ base_point.T
        Bt = jnp.linalg.solve(ytx, At)
        u, s, vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        log = (u * jnp.arctan(s).reshape(1, -1)) @ vt
        return log

    def norm(self, base_point, vec):
        return jnp.linalg.norm(vec, axis=(-1, -2))

    def dist(self, point_a, point_b):
        s = jnp.clip(jnp.linalg.svd(point_a.T @ point_b, compute_uv=False), a_max=1.0)
        dist = jnp.linalg.norm(jnp.arccos(s))
        return dist

    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        ip = jnp.tensordot(tangent_vec_a, tangent_vec_b, axes=2)
        return ip

    def parallel_transport(self, start_point, end_point, tangent_vec):
        direction = self.log(start_point, end_point)
        u, s, vt = jnp.linalg.svd(direction, full_matrices=False)
        ut_delta = u.T @ tangent_vec
        pt = (
            (
                start_point @ (vt.T * -1 * jnp.sin(s).reshape(1, -1))
                + (u * jnp.cos(s).reshape(1, -1))
            )
            @ (ut_delta)
            + tangent_vec
            - u @ (ut_delta)
        )
        return pt

    def egrad_to_rgrad(self, base_point, egrad):
        return self.projector(base_point, egrad)

    def projector(self, base_point, vec):
        return vec.value - base_point.value @ (base_point.value.T @ vec.value)

    def random(self):
        def tangent_gaussian(rng_key, base_point, mean=None, sigma=1.0):
            sample_at_ref = jnp.vstack(
                [
                    jnp.zeros((self.r, self.r)),
                    jnp.random.normal(
                        rng_key, mean=0, sigma=sigma, shape=(self.m - self.r, self.r)
                    ),
                ]
            )
            sample = self.parallel_transport(
                start_point=self.ref_point,
                end_point=base_point,
                tangent_vec=sample_at_ref,
            )
            return sample if mean is None else sample + mean
