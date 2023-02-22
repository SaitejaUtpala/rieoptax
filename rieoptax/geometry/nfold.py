
from jax import vmap, jit 
from functools import partial

from chex import Array


class NFoldManifold():
    def __init__(self, manifold, N):
        self.manifold = manifold
        self.N = N 

    def __str__(self):
        return f"nfold.NFoldManifold({str(self.manifold)}, {self.N})"

    @partial(jit, static_argnums=(0,))
    def inp(self, bpt: Array, tv_a: Array, tv_b: Array) -> Array:
        """Inner product between two tangent vectors at a point on manifold.

        Args:
            bpt: point on the manifold.
            tv_a: tangent vector at bpt.
            tv_b: tangent vector at bpt.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        inp = vmap(self.manifold.inp, in_axes=(0,0,0))(bpt, tv_a, tv_b)
        return inp.sum()
    
    @partial(jit, static_argnums=(0,))
    def exp(self, bpt: Array, tv: Array) -> Array:
        """Riemannian Exponential map.

        Args:
            bpt: base_point.
            tv: tangent_vec.

        Returns:
            returns Exp_{bpt}(tv).
        """
        exp = vmap(self.manifold.exp, in_axes=(0,0))(bpt, tv)
        return exp

    @partial(jit, static_argnums=(0,))
    def log(self, bpt: Array, pt: Array) -> Array:
        """Riemannian Logarithm map.

        Args:
            bpt: base_point.
            pt: tangent_vec.

        Returns:
            returns Log_{bpt}(pt).
        """
        log = vmap(self.manifold.log, in_axes=(0,0))(bpt, pt)
        return log
    
    @partial(jit, static_argnums=(0,))
    def egrad_to_rgrad(self, bpt: Array, egrad: Array) -> Array:
        """Euclidean gradient to Riemannian Gradient Convertor.

        Args:
            bpt: base_point.
            egrad: tangent_vec.

        Returns:
            returns Log_{bpt}(pt).
        """
        rgrad = vmap(self.manifold.egrad_to_rgrad, in_axes=(0,0))(bpt, egrad)
        return rgrad

    @partial(jit, static_argnums=(0,))
    def dist(self, pt_a: Array, pt_b: Array) -> float:
        """Distance between two points on the manifold induced by Riemannian metric.

        Args:
            pt_a: point on the manifold.
            pt_b: point on the manifold.

        Returns:
            returns distance between pt_a, pt_b.
        """
        dist = vmap(self.manifold.dist, in_axes=(0,0))(pt_a, pt_a)
        return dist.sum()

    @partial(jit, static_argnums=(0,))
    def ptrans(self, s_pt: Array, e_pt: Array, tv: Array) -> Array:
        """Parallel Transport.

        Args:
            s_pt: start point.
            e_pt: end point.
            tv: tangent vector at start point.

        Returns:
            returns PT_{s_pt ->e_pt}(tv).
        """
        ptrans = vmap(self.manifold.ptrans, in_axes=(0,0,0))(s_pt, e_pt, tv)
        return ptrans 
