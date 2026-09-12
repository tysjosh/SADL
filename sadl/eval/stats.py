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
    # How many seeds the interval was actually computed from, and how many were
    # unusable.  A CI over 2 of 5 seeds is not the CI the protocol asks for, so the
    # count travels with the estimate rather than being inferred from the caller.
    n_seeds: int = 0
    n_seeds_dropped: int = 0

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "n_seeds": self.n_seeds,
            "n_seeds_dropped": self.n_seeds_dropped,
        }


def hierarchical_bootstrap_worst_acc(
    correctness_by_seed: list[dict[str, np.ndarray]],
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """CI for worst-environment accuracy.

    ``correctness_by_seed[i][env]`` is a 0/1 vector of per-instance correctness.
    Resampling is nested: seeds, then environments within a seed, then instances
    within an environment.

    Seeds with no usable vectors are dropped rather than crashing or being scored.
    ``load_correctness`` returns an empty dict when a record's ``.npz`` is absent --
    a cell interrupted between writing its json and its vectors, or records moved
    between machines without them -- and an empty record cannot contribute to a
    resample of instances.  Treating it as zero accuracy would be worse than
    dropping it, so the count of dropped seeds is returned alongside the interval
    and reported next to the table.
    """
    rng = np.random.default_rng(seed)
    n_given = len(correctness_by_seed)
    usable = [r for r in correctness_by_seed if r and all(np.size(v) for v in r.values())]
    dropped = n_given - len(usable)
    if not usable:
        return Interval(float("nan"), float("nan"), float("nan"), 0, dropped)
    # Resample only environments present in every usable seed: a seed missing an
    # environment would otherwise raise, and the minimum over a varying set of
    # environments is not comparable across seeds.
    env_names = sorted(set.intersection(*(set(r) for r in usable)))
    if not env_names:
        return Interval(float("nan"), float("nan"), float("nan"), 0, dropped)

    n_seeds = len(usable)
    point = float(np.mean([min(float(np.mean(rec[e])) for e in env_names) for rec in usable]))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for si in rng.integers(0, n_seeds, n_seeds):
            rec = usable[si]
            names = [env_names[j] for j in rng.integers(0, len(env_names), len(env_names))]
            env_accs = []
            for name in names:
                arr = rec[name]
                env_accs.append(arr[rng.integers(0, len(arr), len(arr))].mean())
            vals.append(min(env_accs))
        draws[b] = np.mean(vals)
    return Interval(
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        n_seeds,
        dropped,
    )


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
