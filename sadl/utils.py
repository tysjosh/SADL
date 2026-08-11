"""Shared utilities: seeding, devices, timing, JSON IO."""
from __future__ import annotations

import contextlib
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SADL_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SADL_RESULTS", ROOT / "results"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    env = os.environ.get("SADL_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Timer:
    """Wall-clock timer usable as a context manager."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        with contextlib.suppress(Exception):
            return torch.mps.current_allocated_memory() / 1e6
    return float("nan")


def _default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_default)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def source_fingerprint() -> str:
    """Short hash of the package source.

    Runs are cached by ``(dataset, method, seed, budget, tag)``, which does not
    change when the code does.  Recording the fingerprint lets a cached record be
    recognised as coming from a different version of the method, instead of being
    silently mixed into a fresh sweep.
    """
    import hashlib

    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def entropy_bits(p: torch.Tensor) -> torch.Tensor:
    """Binary entropy in bits for a probability tensor."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
