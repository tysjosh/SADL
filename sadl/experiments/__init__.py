"""RQ drivers.  Imports are lazy so ``python -m sadl.experiments.run`` is clean."""
from __future__ import annotations

from typing import Any

__all__ = ["load_correctness", "run_grid", "run_key", "run_one"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import run as _run

        return getattr(_run, name)
    raise AttributeError(name)
