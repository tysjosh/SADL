"""ColoredMNIST, following the construction of Arjovsky et al. and DomainBed.

Label: ``y = 1[digit >= 5]`` flipped with probability 0.25.  Colour is
``y`` flipped with probability ``e``; the digit occupies one of two channels.
Training environments use ``e in {0.1, 0.2}`` where colour predicts the label
better (0.9 / 0.8) than shape does (0.75).  Held-out environments use
``e in {0.9, 0.5}``, inverting and then removing the shortcut.

The renderer is exposed so that stability pairs (same digit image, independently
resampled colour) can be produced for Equation 25.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .base import EnvSplit, MultiEnvDataset, SyntheticGenerator, split_fit_val
from .real import require_files

INV_NAMES = ["digit", "label_flip", "mnist_idx"]
ENV_NAMES = ["color"]
FR_NAMES = ["digit"]
FR_BINS = {"digit": 10}


def _read_idx(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(f">{ndim}I", fh.read(4 * ndim))
        buf = fh.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(shape)


class ColoredMNISTGenerator(SyntheticGenerator):
    inv_factor_names = INV_NAMES
    env_factor_names = ENV_NAMES
    in_shape = (2, 28, 28)
    n_classes = 2

    def __init__(self, data_dir: Path, label_noise: float = 0.25) -> None:
        raw = Path(data_dir) / "mnist" / "raw"
        require_files(
            [raw / f for f in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")],
            "ColoredMNIST",
        )
        imgs = np.concatenate(
            [_read_idx(raw / "train-images-idx3-ubyte"), _read_idx(raw / "t10k-images-idx3-ubyte")]
        )
        labs = np.concatenate(
            [_read_idx(raw / "train-labels-idx1-ubyte"), _read_idx(raw / "t10k-labels-idx1-ubyte")]
        )
        self.imgs = (imgs.astype(np.float32) / 255.0)
        self.digits = labs.astype(np.int64)
        self.label_noise = label_noise

    def sample_inv(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self.imgs), n)
        digit = self.digits[idx]
        flip = (rng.random(n) < self.label_noise).astype(np.int64)
        return np.stack([digit, flip, idx], 1).astype(np.int64)

    def label(self, z_inv: np.ndarray) -> np.ndarray:
        base = (z_inv[:, 0] >= 5).astype(np.int64)
        return np.abs(base - z_inv[:, 1]).astype(np.int64)

    def sample_env(self, z_inv: np.ndarray, env: dict, rng: np.random.Generator) -> np.ndarray:
        y = self.label(z_inv)
        flip = (rng.random(len(y)) < env["e"]).astype(np.int64)
        return np.abs(y - flip)[:, None].astype(np.int64)

    def render(self, z_inv: np.ndarray, z_env: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        digit_img = self.imgs[z_inv[:, 2]]
        x = np.zeros((len(z_inv), 2, 28, 28), dtype=np.float32)
        color = z_env[:, 0]
        x[np.arange(len(z_inv)), color] = digit_img
        return x


TRAIN_ENV_SPECS = [("e0.10", {"e": 0.1}), ("e0.20", {"e": 0.2})]
TEST_ENV_SPECS = [("e0.90", {"e": 0.9}), ("e0.50", {"e": 0.5})]


def make_colored_mnist(
    data_dir: Path,
    n_per_env: int = 6000,
    seed: int = 0,
    val_frac: float = 0.2,
    n_train_envs: int | None = None,
    env_specs: list[tuple[str, dict]] | None = None,
) -> MultiEnvDataset:
    gen = ColoredMNISTGenerator(data_dir)
    rng = np.random.default_rng(seed + 7717)
    specs = env_specs if env_specs is not None else TRAIN_ENV_SPECS
    if n_train_envs is not None:
        specs = _expand_envs(n_train_envs) if n_train_envs > len(specs) else specs[:n_train_envs]

    def build(spec_list: list[tuple[str, dict]], n: int) -> list[EnvSplit]:
        out = []
        for name, spec in spec_list:
            z_inv = gen.sample_inv(n, rng)
            z_env = gen.sample_env(z_inv, spec, rng)
            x = gen.render(z_inv, z_env, rng)
            y = gen.label(z_inv)
            fit, val = split_fit_val(n, val_frac, rng)
            out.append(EnvSplit(name, x, y, z_inv, z_env, fit, val))
        return out

    return MultiEnvDataset(
        name="ColoredMNIST",
        in_shape=(2, 28, 28),
        n_classes=2,
        train_envs=build(specs, n_per_env),
        test_envs=build(TEST_ENV_SPECS, n_per_env // 2),
        inv_factor_names=INV_NAMES,
        env_factor_names=ENV_NAMES,
        aux_task_names=["digit"],
        fr_factor_names=FR_NAMES,
        fr_bins=FR_BINS,
        generator=gen,
        train_env_specs=[s for _, s in specs],
    )


def _expand_envs(k: int) -> list[tuple[str, dict]]:
    """Evenly spaced shortcut strengths for the |E| sweep (RQ4)."""
    es = np.linspace(0.05, 0.25, k)
    return [(f"e{e:.2f}", {"e": float(e)}) for e in es]
