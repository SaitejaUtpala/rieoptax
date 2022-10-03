import functools
from typing import Any, Callable, NamedTuple, Optional
from core import ManifoldArray

class TraceState(NamedTuple):
  """Holds an aggregation of past updates."""
  trace: ManifoldArray 
  