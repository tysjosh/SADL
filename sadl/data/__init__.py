"""Dataset registry."""
from __future__ import annotations

from pathlib import Path

from ..utils import DATA_DIR
from .base import (
    EnvBalancedLoader,
    EnvSplit,
    MultiEnvDataset,
    SyntheticGenerator,
    concat_splits,
    stability_pairs,
)
from .causal3dident import make_causal3dident_lite
from .colored_mnist import make_colored_mnist
from .dsprites import make_dsprites
from .real import REAL_SPECS, DataUnavailable, make_real

SYNTHETIC = ["dSprites", "Causal3DIdent-lite", "ColoredMNIST"]
REAL = list(REAL_SPECS)
ALL_DATASETS = SYNTHETIC + REAL


def get_dataset(name: str, data_dir: Path | None = None, **kwargs) -> MultiEnvDataset:
    data_dir = Path(data_dir or DATA_DIR)
    if name == "dSprites":
        return make_dsprites(data_dir, **kwargs)
    if name == "Causal3DIdent-lite":
        return make_causal3dident_lite(**kwargs)
    if name == "ColoredMNIST":
        return make_colored_mnist(data_dir, **kwargs)
    if name in REAL_SPECS:
        return make_real(name, data_dir, **kwargs)
    raise KeyError(f"unknown dataset {name!r}; known: {ALL_DATASETS}")


def env_specs_for(name: str, k: int) -> list[tuple[str, dict]]:
    """Training-environment specs for the |E| sweep of RQ4."""
    from .causal3dident import TRAIN_ENV_SPECS as C3
    from .causal3dident import _env_spec as c3_spec
    from .colored_mnist import _expand_envs
    from .dsprites import _env_spec as ds_spec

    if name == "ColoredMNIST":
        return _expand_envs(k)
    if name == "dSprites":
        ps = [0.90, 0.80, 0.85, 0.75, 0.95, 0.70, 0.88, 0.78]
        bgs = [
            [0.7, 0.2, 0.07, 0.03],
            [0.05, 0.15, 0.5, 0.3],
            [0.3, 0.3, 0.2, 0.2],
            [0.1, 0.4, 0.4, 0.1],
            [0.5, 0.1, 0.1, 0.3],
            [0.2, 0.2, 0.3, 0.3],
            [0.4, 0.4, 0.1, 0.1],
            [0.15, 0.35, 0.35, 0.15],
        ]
        texs = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.4, 0.4, 0.2], [0.2, 0.4, 0.4]] * 2
        gams = [[0.1, 0.8, 0.1], [0.45, 0.1, 0.45], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]] * 2
        return [(f"tint{ps[i]:.2f}", ds_spec(ps[i], bgs[i], texs[i], gams[i])) for i in range(k)]
    if name == "Causal3DIdent-lite":
        ps = [0.90, 0.80, 0.85, 0.75, 0.95, 0.70, 0.88, 0.78]
        lds = [
            [0.6, 0.2, 0.15, 0.05],
            [0.05, 0.15, 0.2, 0.6],
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.1, 0.4, 0.1],
            [0.1, 0.4, 0.1, 0.4],
            [0.3, 0.3, 0.2, 0.2],
            [0.2, 0.2, 0.3, 0.3],
            [0.45, 0.05, 0.05, 0.45],
        ]
        lhs = lds[::-1]
        texs = [[0.6, 0.3, 0.1], [0.1, 0.3, 0.6], [0.34, 0.33, 0.33], [0.2, 0.4, 0.4]] * 2
        return [(f"bg{ps[i]:.2f}", c3_spec(ps[i], lds[i], lhs[i], texs[i])) for i in range(k)]
    raise KeyError(f"no environment sweep defined for {name!r}")


__all__ = [
    "ALL_DATASETS",
    "DataUnavailable",
    "EnvBalancedLoader",
    "EnvSplit",
    "MultiEnvDataset",
    "REAL",
    "SYNTHETIC",
    "SyntheticGenerator",
    "concat_splits",
    "env_specs_for",
    "get_dataset",
    "stability_pairs",
]
