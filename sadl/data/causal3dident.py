"""Causal3DIdent-lite: a procedural content/style benchmark.

The original Causal3DIdent renders Blender scenes and ships as a multi-gigabyte
archive.  This module reproduces its *structure* -- the content/style
decomposition of von Kugelgen et al. that the paper's Assumption 3.1 follows --
with an analytic renderer that runs on CPU in seconds:

  content (invariant):  object class, position x/y, size, rotation
  style   (nuisance):   spotlight direction, spotlight hue, background hue,
                        surface texture amplitude

Object class is the label.  Background hue is the shortcut: it agrees with the
noisy label with probability ``p_bg``, high in the training environments and low
in the held-out ones.  Shading depends on the spotlight direction, so class
information requires integrating shape rather than reading a single pixel.

This is a substitute, not the published dataset; results are not comparable to
numbers reported on the original Causal3DIdent.
"""
from __future__ import annotations

import numpy as np

from .base import EnvSplit, MultiEnvDataset, SyntheticGenerator, split_fit_val

RES = 64
N_CLASS = 3  # sphere, cube, cylinder
POS_BINS = 5
SIZE_BINS = 3
ROT_BINS = 4

HUES = np.array(
    [[0.85, 0.30, 0.25], [0.25, 0.75, 0.35], [0.30, 0.40, 0.90], [0.80, 0.75, 0.30]],
    dtype=np.float32,
)
LIGHT_DIRS = np.array(
    [[-0.7, -0.4], [0.7, -0.4], [-0.7, 0.4], [0.7, 0.4]], dtype=np.float32
)
TEX_LEVELS = np.array([0.0, 0.08, 0.16], dtype=np.float32)

INV_NAMES = ["obj_class", "pos_x", "pos_y", "size", "rotation", "label_flip"]
ENV_NAMES = ["light_dir", "light_hue", "bg_hue", "texture"]
FR_NAMES = ["obj_class", "pos_x", "pos_y", "size", "rotation"]
FR_BINS = {"obj_class": 3, "pos_x": 5, "pos_y": 5, "size": 3, "rotation": 4}


class Causal3DIdentLiteGenerator(SyntheticGenerator):
    inv_factor_names = INV_NAMES
    env_factor_names = ENV_NAMES
    in_shape = (3, RES, RES)
    n_classes = N_CLASS

    def __init__(self, label_noise: float = 0.15) -> None:
        self.label_noise = label_noise
        g = (np.arange(RES) + 0.5) / RES * 2.0 - 1.0
        self.vv, self.uu = np.meshgrid(g, g, indexing="ij")

    def sample_inv(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cls = rng.integers(0, N_CLASS, n)
        px = rng.integers(0, POS_BINS, n)
        py = rng.integers(0, POS_BINS, n)
        size = rng.integers(0, SIZE_BINS, n)
        rot = rng.integers(0, ROT_BINS, n)
        flip = (rng.random(n) < self.label_noise).astype(np.int64)
        return np.stack([cls, px, py, size, rot, flip], 1).astype(np.int64)

    def label(self, z_inv: np.ndarray) -> np.ndarray:
        cls, flip = z_inv[:, 0], z_inv[:, 5]
        return np.where(flip == 1, (cls + 1) % N_CLASS, cls).astype(np.int64)

    def sample_env(self, z_inv: np.ndarray, env: dict, rng: np.random.Generator) -> np.ndarray:
        n = len(z_inv)
        y = self.label(z_inv)
        agree = rng.random(n) < env["p_bg"]
        bg = np.where(agree, y, (y + 1 + rng.integers(0, 2, n)) % N_CLASS)
        light_dir = rng.choice(len(LIGHT_DIRS), size=n, p=env["light_dir_probs"])
        light_hue = rng.choice(len(HUES), size=n, p=env["light_hue_probs"])
        tex = rng.choice(len(TEX_LEVELS), size=n, p=env["tex_probs"])
        return np.stack([light_dir, light_hue, bg, tex], 1).astype(np.int64)

    def render(self, z_inv: np.ndarray, z_env: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(z_inv)
        out = np.empty((n, 3, RES, RES), dtype=np.float32)
        for s in range(0, n, 512):
            sl = slice(s, min(n, s + 512))
            out[sl] = self._render_block(z_inv[sl], z_env[sl], rng)
        return out

    def _render_block(self, zi: np.ndarray, ze: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(zi)
        cls = zi[:, 0]
        cx = ((zi[:, 1] + 0.5) / POS_BINS * 1.2 - 0.6)[:, None, None]
        cy = ((zi[:, 2] + 0.5) / POS_BINS * 1.2 - 0.6)[:, None, None]
        r = (0.20 + 0.07 * zi[:, 3])[:, None, None]
        th = (zi[:, 4] / ROT_BINS * (np.pi / 2))[:, None, None]

        u = self.uu[None] - cx
        v = self.vv[None] - cy
        ur = u * np.cos(th) + v * np.sin(th)
        vr = -u * np.sin(th) + v * np.cos(th)

        ld = LIGHT_DIRS[ze[:, 0]]
        lx, ly = ld[:, 0][:, None, None], ld[:, 1][:, None, None]
        lz = np.sqrt(np.clip(1.0 - lx**2 - ly**2, 0.0, 1.0))

        mask = np.zeros((n, RES, RES), dtype=np.float32)
        shade = np.full((n, RES, RES), 0.5, dtype=np.float32)

        # sphere: smooth Lambertian falloff
        is_s = cls == 0
        if is_s.any():
            d2 = (u[is_s] ** 2 + v[is_s] ** 2) / r[is_s] ** 2
            m = (d2 <= 1.0).astype(np.float32)
            nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
            nx = u[is_s] / r[is_s]
            ny = v[is_s] / r[is_s]
            sh = nx * lx[is_s] + ny * ly[is_s] + nz * lz[is_s]
            mask[is_s] = m
            shade[is_s] = np.clip(0.25 + 0.75 * sh, 0.05, 1.0)

        # cube: piecewise-constant facet shading
        is_c = cls == 1
        if is_c.any():
            m = (np.maximum(np.abs(ur[is_c]), np.abs(vr[is_c])) <= r[is_c] * 0.9).astype(np.float32)
            facet = 0.45 + 0.28 * np.sign(ur[is_c]) * lx[is_c] + 0.28 * np.sign(vr[is_c]) * ly[is_c]
            mask[is_c] = m
            shade[is_c] = np.clip(facet, 0.05, 1.0)

        # cylinder: curvature in one axis only
        is_y = cls == 2
        if is_y.any():
            w = r[is_y] * 0.62
            m = ((np.abs(ur[is_y]) <= w) & (np.abs(vr[is_y]) <= r[is_y])).astype(np.float32)
            nx = np.clip(ur[is_y] / w, -1.0, 1.0)
            nz = np.sqrt(np.clip(1.0 - nx**2, 0.0, 1.0))
            sh = nx * lx[is_y] + nz * lz[is_y]
            mask[is_y] = m
            shade[is_y] = np.clip(0.25 + 0.75 * sh, 0.05, 1.0)

        obj_col = HUES[ze[:, 1]][:, :, None, None] * 0.45 + 0.55  # spotlight tints object
        bg_col = HUES[ze[:, 2]][:, :, None, None] * 0.42 + 0.05
        tex = TEX_LEVELS[ze[:, 3]][:, None, None, None]
        coarse = rng.random((n, 1, 8, 8)).astype(np.float32)
        texture = np.repeat(np.repeat(coarse, RES // 8, axis=2), RES // 8, axis=3)

        m4 = mask[:, None]
        s4 = shade[:, None]
        img = m4 * (obj_col * s4) + (1 - m4) * (bg_col + tex * (texture - 0.5) * 2.0)
        return np.clip(img, 0.0, 1.0).astype(np.float32)


def _env_spec(p_bg: float, ld: list[float], lh: list[float], tex: list[float]) -> dict:
    return {
        "p_bg": p_bg,
        "light_dir_probs": np.array(ld, float) / np.sum(ld),
        "light_hue_probs": np.array(lh, float) / np.sum(lh),
        "tex_probs": np.array(tex, float) / np.sum(tex),
    }


TRAIN_ENV_SPECS = [
    ("bg0.90", _env_spec(0.90, [0.6, 0.2, 0.15, 0.05], [0.5, 0.2, 0.2, 0.1], [0.6, 0.3, 0.1])),
    ("bg0.80", _env_spec(0.80, [0.05, 0.15, 0.2, 0.6], [0.1, 0.2, 0.2, 0.5], [0.1, 0.3, 0.6])),
]
TEST_ENV_SPECS = [
    ("bg0.10", _env_spec(0.10, [0.25] * 4, [0.25] * 4, [0.34, 0.33, 0.33])),
    ("bg0.33", _env_spec(1.0 / 3.0, [0.4, 0.1, 0.1, 0.4], [0.4, 0.1, 0.1, 0.4], [0.2, 0.2, 0.6])),
]


def make_causal3dident_lite(
    n_per_env: int = 6000,
    seed: int = 0,
    val_frac: float = 0.2,
    n_train_envs: int | None = None,
    env_specs: list[tuple[str, dict]] | None = None,
    **_: object,
) -> MultiEnvDataset:
    gen = Causal3DIdentLiteGenerator()
    rng = np.random.default_rng(seed + 991)
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
        name="Causal3DIdent-lite",
        in_shape=(3, RES, RES),
        n_classes=N_CLASS,
        train_envs=build(specs, n_per_env),
        test_envs=build(TEST_ENV_SPECS, n_per_env // 2),
        inv_factor_names=INV_NAMES,
        env_factor_names=ENV_NAMES,
        aux_task_names=["size", "pos_y", "rotation"],
        fr_factor_names=FR_NAMES,
        fr_bins=FR_BINS,
        generator=gen,
        train_env_specs=[s for _, s in specs],
        notes="procedural substitute for Causal3DIdent (see module docstring)",
    )
