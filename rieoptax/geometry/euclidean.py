from chex import Array

from rieoptax.geometry.base import RiemannianManifold


class Euclidean(RiemannianManifold):

    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> Array:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold.
            tv_a: tangent vector at bpt.
            tv_b: tangent vector at bpt.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        return self.trace_matprod(tv_a, tv_b)

    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point.
            tv: tangent_vec.

        Returns:
            returns Exp_{bpt}(tv).
        """
        return bpt + tv

    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point.
            pt: tangent_vec.

        Returns:
            returns Log_{bpt}(pt).
        """
        return pt - bpt

    def egrad_to_rgrad(self, bpt: Array, egrad: Array) -> Array:
        """Euclidean gradient to Riemannian Gradient Convertor.

        Args:
            bpt: base_point.
            egrad: tangent_vec.

        Returns:
            returns Log_{bpt}(pt).
        """
        return egrad

    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns distance between pt_a, pt_b.
        """
        return self.norm(pt_a - pt_b)

    def pt(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point.
            e_pt: end point.
            tv: tangent vector at start point.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        return tv