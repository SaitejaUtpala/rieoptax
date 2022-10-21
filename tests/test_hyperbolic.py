
from absl.testing import absltest
from chex import assert_trees_all_close
from jax import numpy as jnp

from rieoptax.geometry.hyperbolic import PoincareBall


class TestPoincareBall(absltest.TestCase):
    manifold = PoincareBall(2)

   