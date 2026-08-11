"""Multi-environment dataset containers and samplers.

Every dataset exposes the structural model of Assumption 3.1 of the paper:
``x = f(z_inv, z_env)`` with ``y = h(z_inv)``.  Synthetic datasets additionally
expose the renderer ``f`` so that paired samples sharing ``z_inv`` with
independently resampled ``z_env`` can be produced.  Those pairs are what the
empirical stability gap of Equation 25 is computed on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch


@dataclass
class EnvSplit:
    """One environment's data, already split into fit / validation indices."""

    env_name: str
    x: np.ndarray  # float32 [N, C, H, W], values in [0, 1]
    y: np.ndarray  # int64 [N]
    z_inv: np.ndarray  # int64 [N, m_inv] discrete codes of invariant factors
    z_env: np.ndarray  # int64 [N, m_env] discrete codes of nuisance factors
    fit_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    val_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, idx: np.ndarray) -> "EnvSplit":
        return EnvSplit(
            env_name=self.env_name,
            x=self.x[idx],
            y=self.y[idx],
            z_inv=self.z_inv[idx],
            z_env=self.z_env[idx],
            fit_idx=np.arange(len(idx)),
            val_idx=np.empty(0, dtype=np.int64),
        )

    def fit(self) -> "EnvSplit":
        return self.subset(self.fit_idx)

    def val(self) -> "EnvSplit":
        return self.subset(self.val_idx)


@dataclass
class MultiEnvDataset:
    name: str
    in_shape: tuple[int, int, int]
    n_classes: int
    train_envs: list[EnvSplit]
    test_envs: list[EnvSplit]
    inv_factor_names: list[str]
    env_factor_names: list[str]
    aux_task_names: list[str] = field(default_factory=list)
    # Invariant factors scored by the factor-recovery metric (Equation 24).
    # Latents that are not rendered into the image (e.g. label noise) are excluded.
    fr_factor_names: list[str] = field(default_factory=list)
    # Number of quantile bins used to discretize each scored factor.
    fr_bins: dict[str, int] = field(default_factory=dict)
    generator: "SyntheticGenerator | None" = None
    train_env_specs: list[dict] = field(default_factory=list)
    notes: str = ""

    @property
    def n_train_envs(self) -> int:
        return len(self.train_envs)

    def aux_targets(self, split: EnvSplit) -> dict[str, np.ndarray]:
        """Probe targets for composability (Definition 3.7).

        These are deterministic functions of the invariant latent that are never
        observed by any pretraining objective or by model selection.
        """
        out: dict[str, np.ndarray] = {}
        for k, name in enumerate(self.aux_task_names):
            out[name] = split.z_inv[:, self.inv_factor_names.index(name)]
        return out


class SyntheticGenerator:
    """Renderer interface for datasets with a known generative decomposition."""

    inv_factor_names: list[str]
    env_factor_names: list[str]
    in_shape: tuple[int, int, int]
    n_classes: int

    def sample_inv(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def sample_env(self, z_inv: np.ndarray, env: dict, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def render(self, z_inv: np.ndarray, z_env: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def label(self, z_inv: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def stability_pairs(
    gen: SyntheticGenerator,
    envs: Sequence[dict],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, x_tilde)`` sharing ``z_inv`` with independent ``z_env`` draws.

    The two nuisance draws come from two independently sampled environments of
    the training family, so the pair probes stability in the sense of
    Definition 3.3 rather than mere augmentation invariance.
    """
    z_inv = gen.sample_inv(n, rng)
    ea = rng.integers(0, len(envs), size=n)
    eb = rng.integers(0, len(envs), size=n)
    z_a = np.zeros((n, len(gen.env_factor_names)), dtype=np.int64)
    z_b = np.zeros_like(z_a)
    for j, env in enumerate(envs):
        ma, mb = ea == j, eb == j
        if ma.any():
            z_a[ma] = gen.sample_env(z_inv[ma], env, rng)
        if mb.any():
            z_b[mb] = gen.sample_env(z_inv[mb], env, rng)
    xa = gen.render(z_inv, z_a, rng)
    xb = gen.render(z_inv, z_b, rng)
    return xa, xb


def split_fit_val(n: int, val_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    return perm[n_val:], perm[:n_val]


class EnvBalancedLoader:
    """Yields environment-balanced minibatches (line 7 of Algorithm 1).

    Each step returns one equally sized chunk per training environment, which is
    what the cross-environment shift term S_t and the environment discriminator
    baselines both require.
    """

    def __init__(
        self,
        splits: Sequence[EnvSplit],
        batch_per_env: int,
        device: torch.device,
        seed: int = 0,
        steps: int | None = None,
    ) -> None:
        self.x = [torch.as_tensor(s.x) for s in splits]
        self.y = [torch.as_tensor(s.y) for s in splits]
        self.batch_per_env = batch_per_env
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.steps = steps or max(1, min(len(s) for s in splits) // batch_per_env)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for _ in range(self.steps):
            xs, ys, es = [], [], []
            for e, (x, y) in enumerate(zip(self.x, self.y)):
                idx = torch.as_tensor(self.rng.integers(0, len(x), size=self.batch_per_env))
                xs.append(x[idx])
                ys.append(y[idx])
                es.append(torch.full((self.batch_per_env,), e, dtype=torch.long))
            yield (
                torch.cat(xs).to(self.device),
                torch.cat(ys).to(self.device),
                torch.cat(es).to(self.device),
            )


def concat_splits(splits: Sequence[EnvSplit]) -> EnvSplit:
    return EnvSplit(
        env_name="+".join(s.env_name for s in splits),
        x=np.concatenate([s.x for s in splits]),
        y=np.concatenate([s.y for s in splits]),
        z_inv=np.concatenate([s.z_inv for s in splits]),
        z_env=np.concatenate([s.z_env for s in splits]),
        fit_idx=np.arange(sum(len(s) for s in splits)),
        val_idx=np.empty(0, dtype=np.int64),
    )
