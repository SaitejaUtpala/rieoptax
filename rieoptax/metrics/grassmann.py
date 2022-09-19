from base import RiemannianManifold
from jax import numpy as jnp


class GrassmannCanonicalMetric(RiemannianManifold):
    def __init__(self, m, r):

        self.m = m
        self.r = r
        self.dim = mr

    def exp(self, base_point, tangent_vec):
        u, s, vt = np.linalg.svd(tangent_vector, full_matrices=False)
        exp = (
            point @ (multitransp(vt) * np.cos(s).reshape(1, -1)) @ vt
            + (u * np.sin(s).reshape(1, -1)) @ vt
        )
        return exp

    def log(self, base_point, point):
        ytx = point_b.T @ point_a
        At = point_b.T - ytx @ point_a.T
        Bt = jnp.linalg.solve(ytx, At)
        u, s, vt = jnp.linalg.svd(Bt.T, full_matrices=False)
        log = (u * np.arctan(s).reshape(1, -1)) @ vt
        return log

    def dist(self, point_a, point_b):
        s = jnp.clip(jnp.linalg.svd(point_a.T @ point_b, compute_uv=False), a_max=1.0)
        dist = jnp.linalg.norm(jnp.arccos(s))
        return dist

    def inner_product(self, base_point, tangent_vec_a, tangnet_vec_b):
        ip = jnp.tensordot(tangent_vec_a, tangent_vec_b, axes=2)
        return ip

    def paralle_transport(self, start_point, end_point, tangent_vec):
        direction = self.space.log(start_point, end_point)
        u, s, vt = jnp.linalg.svd(direction, full_matrices=False)
        ut_delta = ut @ tangent_vec  # r @ m, m r
        pt = (
            (
                base_point @ (v * -1 * np.sin(s).reshape(1, -1))
                + (u * np.cos(s).reshape(1, -1))
            )
            @ (ut_delta)
            + tangent_vec
            - u @ (ut_delta)
        )
        return pt
