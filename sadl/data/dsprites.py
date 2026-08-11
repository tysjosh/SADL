"""dSprites with environment interventions on nuisance factors.

Invariant (content) factors: shape, scale, orientation, x-position, y-position.
Nuisance (style) factors: object tint, background luminance, background texture
amplitude, gamma/contrast.  Environments intervene only on the nuisance law; the
marginal law of the invariant factors is identical in every environment, as
required by Assumption 3.1.

The shortcut: object tint agrees with the (noisy) label with probability ``p_e``.
Training environments use high ``p_e``, the held-out environments use low ``p_e``,
so a tint-based predictor beats a shape-based predictor in-distribution and fails
out-of-distribution.  Label noise makes the shortcut strictly more predictive
than the invariant factor in the training environments, following the
ColoredMNIST construction of Arjovsky et al.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import EnvSplit, MultiEnvDataset, SyntheticGenerator, split_fit_val

# dSprites latent grid: (color=1, shape=3, scale=6, orientation=40, posX=32, posY=32)
_SHAPES, _SCALES, _ORIENTS, _POS = 3, 6, 40, 32
_ORIENT_SEL = np.arange(0, 40, 5)  # 8 orientations
_POS_SEL = np.array([4, 9, 14, 19, 24, 29])  # 6 positions per axis

TINT_PALETTE = np.array(
    [[0.90, 0.25, 0.20], [0.20, 0.80, 0.35], [0.30, 0.45, 0.95]], dtype=np.float32
)
BG_LEVELS = np.array([0.06, 0.20, 0.34, 0.48], dtype=np.float32)
TEX_LEVELS = np.array([0.0, 0.07, 0.14], dtype=np.float32)
GAMMA_LEVELS = np.array([0.75, 1.0, 1.35], dtype=np.float32)

INV_NAMES = ["shape", "scale", "orientation", "pos_x", "pos_y", "label_flip"]
ENV_NAMES = ["tint", "bg_level", "texture", "gamma"]
FR_NAMES = ["shape", "scale", "orientation", "pos_x", "pos_y"]
FR_BINS = {"shape": 3, "scale": 6, "orientation": 8, "pos_x": 6, "pos_y": 6}


def _flat_index(shape: np.ndarray, scale: np.ndarray, orient: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (
        shape * (_SCALES * _ORIENTS * _POS * _POS)
        + scale * (_ORIENTS * _POS * _POS)
        + orient * (_POS * _POS)
        + px * _POS
        + py
    )


def _build_cache(npz_path: Path, cache_path: Path) -> None:
    """Extract the sprite subset used here so the 3 GB full array is transient."""
    data = np.load(npz_path)
    imgs = data["imgs"]  # [737280, 64, 64] uint8
    sh, sc, orr, px, py = np.meshgrid(
        np.arange(_SHAPES),
        np.arange(_SCALES),
        _ORIENT_SEL,
        _POS_SEL,
        _POS_SEL,
        indexing="ij",
    )
    flat = _flat_index(sh.ravel(), sc.ravel(), orr.ravel(), px.ravel(), py.ravel())
    subset = imgs[flat].astype(np.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, imgs=subset, shape=sh.ravel(), scale=sc.ravel())
    del imgs, data


def _procedural_sprites() -> np.ndarray:
    """Offline fallback: render the same factor grid with analytic shapes.

    Used only when the official dSprites archive is unavailable.  The factor
    structure (shape / scale / orientation / position) is identical.
    """
    n = _SHAPES * _SCALES * len(_ORIENT_SEL) * len(_POS_SEL) ** 2
    out = np.zeros((n, 64, 64), dtype=np.uint8)
    g = (np.arange(64) + 0.5) / 64.0
    yy, xx = np.meshgrid(g, g, indexing="ij")
    i = 0
    for sh in range(_SHAPES):
        for sc in range(_SCALES):
            r = 0.06 + 0.028 * sc
            for orient in _ORIENT_SEL:
                th = orient / 40.0 * 2 * np.pi
                for cx in _POS_SEL / _POS:
                    for cy in _POS_SEL / _POS:
                        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
                        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
                        if sh == 0:  # square
                            m = np.maximum(np.abs(u), np.abs(v)) <= r
                        elif sh == 1:  # ellipse
                            m = (u / r) ** 2 + (v / (0.6 * r)) ** 2 <= 1.0
                        else:  # triangle / heart proxy
                            m = (v >= -r) & (v <= r - 2 * np.abs(u))
                        out[i] = m.astype(np.uint8)
                        i += 1
    return out


class DSpritesGenerator(SyntheticGenerator):
    inv_factor_names = INV_NAMES
    env_factor_names = ENV_NAMES
    in_shape = (3, 64, 64)
    n_classes = 3

    def __init__(self, data_dir: Path, label_noise: float = 0.15) -> None:
        cache = Path(data_dir) / "dsprites" / "dsprites_subset.npz"
        raw = Path(data_dir) / "dsprites" / "dsprites.npz"
        if not cache.exists():
            if raw.exists():
                _build_cache(raw, cache)
            else:
                imgs = _procedural_sprites()
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, imgs=imgs, procedural=np.array([1]))
        blob = np.load(cache)
        self.imgs = blob["imgs"].astype(np.float32)
        self.procedural = "procedural" in blob
        self.n_or, self.n_pos = len(_ORIENT_SEL), len(_POS_SEL)
        self.label_noise = label_noise

    # ---- generative model -------------------------------------------------
    def sample_inv(self, n: int, rng: np.random.Generator) -> np.ndarray:
        shape = rng.integers(0, _SHAPES, n)
        scale = rng.integers(0, _SCALES, n)
        orient = rng.integers(0, self.n_or, n)
        px = rng.integers(0, self.n_pos, n)
        py = rng.integers(0, self.n_pos, n)
        flip = (rng.random(n) < self.label_noise).astype(np.int64)
        return np.stack([shape, scale, orient, px, py, flip], 1).astype(np.int64)

    def label(self, z_inv: np.ndarray) -> np.ndarray:
        shape, flip = z_inv[:, 0], z_inv[:, 5]
        return np.where(flip == 1, (shape + 1) % _SHAPES, shape).astype(np.int64)

    def sample_env(self, z_inv: np.ndarray, env: dict, rng: np.random.Generator) -> np.ndarray:
        n = len(z_inv)
        y = self.label(z_inv)
        agree = rng.random(n) < env["p_tint"]
        tint = np.where(agree, y, (y + 1 + rng.integers(0, 2, n)) % _SHAPES)
        bg = rng.choice(len(BG_LEVELS), size=n, p=env["bg_probs"])
        tex = rng.choice(len(TEX_LEVELS), size=n, p=env["tex_probs"])
        gam = rng.choice(len(GAMMA_LEVELS), size=n, p=env["gamma_probs"])
        return np.stack([tint, bg, tex, gam], 1).astype(np.int64)

    def _sprite_index(self, z_inv: np.ndarray) -> np.ndarray:
        shape, scale, orient, px, py = (z_inv[:, i] for i in range(5))
        return (
            shape * (_SCALES * self.n_or * self.n_pos * self.n_pos)
            + scale * (self.n_or * self.n_pos * self.n_pos)
            + orient * (self.n_pos * self.n_pos)
            + px * self.n_pos
            + py
        )

    def render(self, z_inv: np.ndarray, z_env: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        m = self.imgs[self._sprite_index(z_inv)][:, None]  # [N,1,64,64]
        tint = TINT_PALETTE[z_env[:, 0]][:, :, None, None]
        bg = BG_LEVELS[z_env[:, 1]][:, None, None, None]
        tex = TEX_LEVELS[z_env[:, 2]][:, None, None, None]
        gamma = GAMMA_LEVELS[z_env[:, 3]][:, None, None, None]
        n = len(z_inv)
        coarse = rng.random((n, 1, 8, 8)).astype(np.float32)
        texture = np.repeat(np.repeat(coarse, 8, axis=2), 8, axis=3)
        base = m * tint + (1.0 - m) * (bg + tex * (texture - 0.5) * 2.0)
        x = np.clip(base, 1e-4, 1.0) ** gamma
        return np.clip(x, 0.0, 1.0).astype(np.float32)


def _env_spec(p_tint: float, bg: list[float], tex: list[float], gamma: list[float]) -> dict:
    return {
        "p_tint": p_tint,
        "bg_probs": np.array(bg, dtype=np.float64) / np.sum(bg),
        "tex_probs": np.array(tex, dtype=np.float64) / np.sum(tex),
        "gamma_probs": np.array(gamma, dtype=np.float64) / np.sum(gamma),
    }


# Environment specifications are fixed before any model is trained (Section 6.2).
TRAIN_ENV_SPECS = [
    ("tint0.90", _env_spec(0.90, [0.7, 0.2, 0.07, 0.03], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1])),
    ("tint0.80", _env_spec(0.80, [0.05, 0.15, 0.5, 0.3], [0.1, 0.3, 0.6], [0.45, 0.1, 0.45])),
]
TEST_ENV_SPECS = [
    ("tint0.10", _env_spec(0.10, [0.25, 0.25, 0.25, 0.25], [0.34, 0.33, 0.33], [0.34, 0.33, 0.33])),
    ("tint0.33", _env_spec(1.0 / 3.0, [0.4, 0.1, 0.1, 0.4], [0.2, 0.2, 0.6], [0.5, 0.0, 0.5])),
]


def make_dsprites(
    data_dir: Path,
    n_per_env: int = 6000,
    seed: int = 0,
    val_frac: float = 0.2,
    n_train_envs: int | None = None,
    env_specs: list[tuple[str, dict]] | None = None,
) -> MultiEnvDataset:
    gen = DSpritesGenerator(data_dir)
    rng = np.random.default_rng(seed)
    specs = env_specs if env_specs is not None else TRAIN_ENV_SPECS
    if n_train_envs is not None:
        specs = specs[:n_train_envs]

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
        name="dSprites",
        in_shape=(3, 64, 64),
        n_classes=3,
        train_envs=build(specs, n_per_env),
        test_envs=build(TEST_ENV_SPECS, n_per_env // 2),
        inv_factor_names=INV_NAMES,
        env_factor_names=ENV_NAMES,
        aux_task_names=["scale", "pos_x", "orientation"],
        fr_factor_names=FR_NAMES,
        fr_bins=FR_BINS,
        generator=gen,
        train_env_specs=[s for _, s in specs],
        notes="procedural fallback sprites" if gen.procedural else "official dSprites sprites",
    )
