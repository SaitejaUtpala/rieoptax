from base import RiemannianManifold


class HypersphereCanonicalMetric(RiemannianManifold):
    def __init__(self, m):
        self.m = m
        t = np.zeros(m + 1)
        t[0] = 1
        self.ref_point = t

    def exp(self, base_point, tangent_vec):
        pass

    def log(self, base_point, point):
        pass

    def parallel_transport(self, start_point, end_point, tangent_vec):
        pass

    def inner_product(self, base_point, tangent_vec_a, tangent_vec_b):
        pass

    def dist(self, point_a, point_b):
        pass

    def tangent_gaussian(self, base_point, sigma):
        sample = np.random.normal(0, sigma, size=(self.m,))
        zero_padded = np.hstack([np.zeros((1,)), sample])
        sample = self.parallel_transport(
            zero_padded, base_point=self.ref_point, end_point=base_points
        )
        return sample
