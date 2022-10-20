
from absl.testing import absltest
from chex import assert_trees_all_close
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class TestPoincareBall(absltest.TestCase):
    manifold = PoincareBall(2)

    def test_mobius_add(self):
        pt_a = jnp.array([])
        pt_b = jnp.array([])
        exp = jnp.array([])
        assert_trees_all_close(self.manifold.mobius_add(pt_a, pt_b), exp)

    def test_mobius_sub(self):
        pt_a = jnp.array([])
        pt_b = jnp.array([])
        exp = jnp.array([])
        assert_trees_all_close(self.manifold.mobius_sub(pt_a, pt_b), exp)
