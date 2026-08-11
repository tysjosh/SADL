"""Statistical analysis of Section 6.5.

  - paired SADL-minus-baseline effects over shared seeds
  - 95% hierarchical bootstrap confidence intervals that resample seeds,
    environments and test instances at their appropriate levels
  - Holm correction for the secondary pairwise family
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi}


def hierarchical_bootstrap_worst_acc(
    correctness_by_seed: list[dict[str, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """CI for worst-environment accuracy.

    ``correctness_by_seed[i][env]`` is a 0/1 vector of per-instance correctness.
    Resampling is nested: seeds, then environments within a seed, then instances
    within an environment.
    """
    rng = np.random.default_rng(seed)
    if not correctness_by_seed:
        return Interval(float("nan"), float("nan"), float("nan"))
    point = float(np.mean([min(np.mean(v) for v in rec.values()) for rec in correctness_by_seed]))
    draws = np.empty(n_boot)
    n_seeds = len(correctness_by_seed)
    env_names = list(correctness_by_seed[0].keys())
    for b in range(n_boot):
        seed_idx = rng.integers(0, n_seeds, n_seeds)
        vals = []
        for si in seed_idx:
            rec = correctness_by_seed[si]
            names = [env_names[j] for j in rng.integers(0, len(env_names), len(env_names))]
            env_accs = []
            for name in names:
                arr = rec[name]
                idx = rng.integers(0, len(arr), len(arr))
                env_accs.append(arr[idx].mean())
            vals.append(min(env_accs))
        draws[b] = np.mean(vals)
    return Interval(point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def paired_effect(
    treat_by_seed: list[float],
    base_by_seed: list[float],
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[Interval, float]:
    """Paired difference over shared seeds with a bootstrap CI and two-sided p."""
    a = np.asarray(treat_by_seed, dtype=float)
    b = np.asarray(base_by_seed, dtype=float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    if n == 0 or np.all(np.isnan(d)):
        return Interval(float("nan"), float("nan"), float("nan")), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    point = float(d.mean())
    lo, hi = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    # sign-flip permutation test (exact for small n)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (signs * d[None, :]).mean(1)
    p = float((np.abs(null) >= abs(point) - 1e-12).mean())
    return Interval(point, lo, hi), p


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction."""
    items = [(k, v) for k, v in pvals.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": p, "p_holm": adj, "reject": bool(adj < alpha)}
    for k, v in pvals.items():
        out.setdefault(k, {"p_raw": v, "p_holm": float("nan"), "reject": False})
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
