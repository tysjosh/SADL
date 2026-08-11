"""Loaders for the naturalistic benchmarks named in Section 6.2.

PACS, TerraIncognita (DomainBed) and Camelyon17, iWildCam (WILDS) are tens of
gigabytes and are never downloaded automatically.  Each loader reads a local
directory if it is present and otherwise raises ``DataUnavailable``, which the
runner records as ``status="data_unavailable"`` so the corresponding table cell
stays empty instead of being silently filled.

Layout expected under ``$SADL_DATA``:

    PACS/{art_painting,cartoon,photo,sketch}/<class>/*.jpg
    terra_incognita/{L100,L38,L43,L46}/<class>/*.jpg
    camelyon17/{hospital_0..4}/<class>/*.png
    iwildcam/{region_0..k}/<class>/*.jpg

Images are resized to ``res`` and loaded into memory, so use ``max_per_class``
to keep the footprint manageable.  A WILDS/DomainBed installation can be
converted into this layout with ``scripts/prepare_real_data.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EnvSplit, MultiEnvDataset, split_fit_val


class DataUnavailable(RuntimeError):
    pass


REAL_SPECS: dict[str, dict] = {
    "PACS": {
        "root": "PACS",
        "envs": ["art_painting", "cartoon", "photo", "sketch"],
        "test_envs": ["sketch"],
        "res": 64,
        "exts": (".jpg", ".png"),
    },
    "TerraIncognita": {
        "root": "terra_incognita",
        "envs": ["L100", "L38", "L43", "L46"],
        "test_envs": ["L100"],
        "res": 64,
        "exts": (".jpg", ".png"),
    },
    "Camelyon17": {
        "root": "camelyon17",
        "envs": ["hospital_0", "hospital_1", "hospital_2", "hospital_3", "hospital_4"],
        "test_envs": ["hospital_4"],
        "res": 64,
        "exts": (".png", ".jpg"),
    },
    "iWildCam": {
        "root": "iwildcam",
        "envs": None,  # discovered from directory listing
        "test_envs": None,  # last two discovered environments
        "res": 64,
        "exts": (".jpg", ".png"),
    },
}


def _load_env(dirpath: Path, classes: list[str], res: int, exts: tuple[str, ...], max_per_class: int, rng) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    xs, ys = [], []
    for ci, cname in enumerate(classes):
        files = sorted(
            p for p in (dirpath / cname).iterdir() if p.suffix.lower() in exts
        ) if (dirpath / cname).is_dir() else []
        if len(files) > max_per_class:
            files = [files[i] for i in rng.choice(len(files), max_per_class, replace=False)]
        for p in files:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB").resize((res, res)), dtype=np.float32) / 255.0
            xs.append(arr.transpose(2, 0, 1))
            ys.append(ci)
    if not xs:
        raise DataUnavailable(f"no images found under {dirpath}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def make_real(
    name: str,
    data_dir: Path,
    seed: int = 0,
    val_frac: float = 0.2,
    max_per_class: int = 400,
    **_: object,
) -> MultiEnvDataset:
    spec = REAL_SPECS[name]
    root = Path(data_dir) / spec["root"]
    if not root.is_dir():
        raise DataUnavailable(f"{name}: expected data at {root}")
    envs = spec["envs"] or sorted(p.name for p in root.iterdir() if p.is_dir())
    if not envs:
        raise DataUnavailable(f"{name}: no environment directories under {root}")
    test_envs = spec["test_envs"] or envs[-2:]
    classes = sorted({p.name for e in envs for p in (root / e).iterdir() if p.is_dir()})
    if not classes:
        raise DataUnavailable(f"{name}: no class directories under {root}")

    rng = np.random.default_rng(seed)
    splits: dict[str, EnvSplit] = {}
    for e in envs:
        x, y = _load_env(root / e, classes, spec["res"], spec["exts"], max_per_class, rng)
        fit, val = split_fit_val(len(x), val_frac, rng)
        z = np.zeros((len(x), 1), dtype=np.int64)
        splits[e] = EnvSplit(e, x, y, z, z.copy(), fit, val)

    return MultiEnvDataset(
        name=name,
        in_shape=(3, spec["res"], spec["res"]),
        n_classes=len(classes),
        train_envs=[splits[e] for e in envs if e not in test_envs],
        test_envs=[splits[e] for e in test_envs],
        inv_factor_names=[],
        env_factor_names=[],
        aux_task_names=[],
        fr_factor_names=[],
        fr_bins={},
        generator=None,
        notes=f"classes={classes}; test_envs={test_envs}; max_per_class={max_per_class}",
    )
