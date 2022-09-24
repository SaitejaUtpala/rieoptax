from base import RiemannianManifold


class HypersphereCanonicalMetric(RiemannianManifold):
    def __init__(self, m):
        self.m = m
        t = np.zeros(m + 1)
        t[0] = 1
        self.ref_point = t

    def exp(self, base_point, tangent_vec):
        norm = jnp.norm(tangent_vector)
        return point * jnp.cos(norm) + tangent_vec * jnp.sinc(norm / jnp.pi)

    def log(self, base_point, point):
        coeff = self.dist(base_point, point)
        v = point - base_point
        proj = v - jnp.inner(v, x) * x
        log = coeff * (v / jnp.norm(proj))
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
        return jnp.tensordot(tangent_vector_a, tangent_vector_b, axes=1)

    def dist(self, point_a, point_b):
        dist = jnp.clip(self.inner_product(point_a, point_a, point_b), -1, 1)
        return dist

    def tangent_gaussian(self, base_point, sigma):
        sample = jnp.random.normal(0, sigma, size=(self.m,))
        zero_padded = jnp.hstack([np.zeros((1,)), sample])
        sample = self.parallel_transport(
            zero_padded, base_point=self.ref_point, end_point=base_points
        )
        return sample
