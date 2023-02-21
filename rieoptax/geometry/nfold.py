
from jax import vmap, jit 
from functools import partial

from chex import Array


class NFoldManifold():
    def __init__(self, manifold, N):
        self.manifold = manifold
        self.N = N 

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



        
