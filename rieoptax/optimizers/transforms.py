import functools
from typing import Any, Callable, NamedTuple, Optional

from core import EmptyState, ManifoldArray

ScaleState = EmptyState()

class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: ManifoldArray 

def scale(step_size):

    def init_fn(params):
        del params





  