from jax import numpy as jnp


class GrassmannCanonical():
    def __init__(self, m, r):

        self.m = m
        self.r = r
        self.dim = m * r - r * r
        self.shape = (m, r)
        self.ref_point = jnp.eye(m, r)

    def exp(self, b_pt, tv):
        u, s, vt = jnp.linalg.svd(tv, full_matrices=False)
        exp = (
            b_pt @ (vt.T * jnp.cos(s).reshape(1, -1)) @ vt
            + (u * jnp.sin(s).reshape(1, -1)) @ vt
        )
        return exp

    def retr(self, b_pt, tv):
        u, _, vt = jnp.linalg.svd(b_pt + tv, full_matrices=False)
        return u @ vt

    def log(self, b_pt, pt):
        ytx = pt.T @ b_pt
        At = pt.T - ytx @ b_pt.T
        Bt = jnp.linalg.solve(ytx, At)
        u, s, vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        log = (u * jnp.arctan(s).reshape(1, -1)) @ vt
        return log

    def norm(self, b_pt, vec):
        return jnp.linalg.norm(vec, axis=(-1, -2))

    def dist(self, pt_a, pt_b):
        s = jnp.clip(jnp.linalg.svd(pt_a.T @ pt_b, compute_uv=False), a_max=1.0)
        dist = jnp.linalg.norm(jnp.arccos(s))
        return dist

    def inner_product(self, b_pt, tangent_vec_a, tangent_vec_b):
        ip = jnp.tensordot(tangent_vec_a, tangent_vec_b, axes=2)
        return ip

    def parallel_transport(self, s_pt, e_pt, tv):
        direction = self.log(s_pt, e_pt)
        u, s, vt = jnp.linalg.svd(direction, full_matrices=False)
        ut_delta = u.T @ tv
        pt = (
            (
                s_pt @ (vt.T * -1 * jnp.sin(s).reshape(1, -1))
                + (u * jnp.cos(s).reshape(1, -1))
            )
            @ (ut_delta)
            + tv
            - u @ (ut_delta)
        )
        return pt

    def egrad_to_rgrad(self, b_pt, egrad):
        return self.projector(b_pt, egrad)

    def projector(self, b_pt, vec):
        return vec.value - b_pt.value @ (b_pt.value.T @ vec.value)

    def random(self):
        def tangent_gaussian(rng_key, b_pt, mean=None, sigma=1.0):
            sample_at_ref = jnp.vstack(
                [
                    jnp.zeros((self.r, self.r)),
                    jnp.random.normal(
                        rng_key, mean=0, sigma=sigma, shape=(self.m - self.r, self.r)
                    ),
                ]
            )
            sample = self.parallel_transport(
                s_pt=self.ref_point,
                e_pt=b_pt,
                tv=sample_at_ref,
            )
            return sample if mean is None else sample + mean
