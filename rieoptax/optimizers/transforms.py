import functools
from typing import Any, Callable, NamedTuple, Optional

import jax
from rieoptax.core import EmptyState, ManifoldArray, RiemannianGradientTransformation

ScaleState = EmptyState()

class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: ManifoldArray 




class EmptyState(NamedTuple):
  """An empty state for the simplest stateless transformations."""

ScaleState = EmptyState

def scale(step_size) :
  """Scale updates by some fixed scalar `step_size`."""

  def init_fn(params):
    del params 
    return ScaleState()

  def update_fn(updates, state, params=None):
    del params
    updates = step_size*updates.value
    return updates, state

  return RiemannianGradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(learning_rate, flip_sign=True):
    m = -1 if flip_sign else 1
    return scale(m * learning_rate)



  