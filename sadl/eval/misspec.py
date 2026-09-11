"""Split misspecification diagnostic (Section 6.2).

Salaudeen et al. observe that many domain-generalization splits show "accuracy on
the line": in-distribution and out-of-distribution accuracy move together across
models, so the held-out environment never stresses the spurious correlation and
ERM matches invariance methods by construction.  Such a split cannot discriminate
between the methods under comparison.

Operationalisation used here (an approximation of the published protocol, stated
plainly as such): fit an OLS line of worst-environment accuracy against
in-distribution validation accuracy over the *baseline* model pool only -- SADL
runs are excluded so the designation cannot depend on the method under test.  A
split is designated

  ``misspecified``   if the relation is significantly positive with slope at or
                     above ``SLOPE_THRESHOLD`` (accuracy is on the line),
  ``well_specified`` if the slope is significantly *below* ``SLOPE_THRESHOLD`` by
                     a one-sided test, so there is positive evidence that accuracy
                     is off the line,
  ``indeterminate``  if neither holds: too few models, essentially constant ID
                     accuracies, or a fit too noisy to place the slope on either
                     side of the threshold.

The third case is why the test is one-sided against the threshold rather than
two-sided against zero.  Failing to show that a split is on the line is not
evidence that it is off it, and ``well_specified`` is not a null result: it is the
subset Table 4 reports and it feeds the preregistered contrast of Equation 26, so
admitting a split on absent evidence changes the headline comparison.  A slope of
-0.26 with a standard error of 0.23 is not significantly different from zero
(p = 0.28) yet is comfortably below 0.5 (p = 0.002); a slope of -0.26 with a
standard error of 1.5 is neither, and belongs in ``indeterminate``.  Section 6.2
expects exactly this category, requiring that classifications and exclusions be
listed "including failed or indeterminate diagnostics".

The designation is written to disk with a hash of the inputs and is read back
unchanged when the SADL comparisons are made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

SLOPE_THRESHOLD = 0.5
MIN_POINTS = 6
MIN_ID_SPREAD = 0.02


def diagnose(id_acc: list[float], ood_acc: list[float], alpha: float = 0.05) -> dict:
    x = np.asarray(id_acc, dtype=float)
    y = np.asarray(ood_acc, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < MIN_POINTS or float(x.max() - x.min()) < MIN_ID_SPREAD:
        return {
            "designation": "indeterminate",
            "reason": "insufficient model spread",
            "n_models": int(len(x)),
            "id_spread": float(x.max() - x.min()) if len(x) else float("nan"),
        }
    res = stats.linregress(x, y)
    df = len(x) - 2

    # One-sided test of H0: slope >= SLOPE_THRESHOLD against H1: slope < it.
    # Rejecting H0 is the evidence that accuracy is off the line.
    se = float(res.stderr)
    if not np.isfinite(se) or se <= 0:
        # A perfectly collinear pool leaves no sampling variation to test; place
        # the slope by direct comparison rather than dividing by zero.
        p_below = 0.0 if res.slope < SLOPE_THRESHOLD else 1.0
        t_below = float("-inf") if res.slope < SLOPE_THRESHOLD else float("inf")
    else:
        t_below = (float(res.slope) - SLOPE_THRESHOLD) / se
        p_below = float(stats.t.cdf(t_below, df))

    on_the_line = bool(res.slope >= SLOPE_THRESHOLD and res.pvalue < alpha and res.rvalue > 0)
    off_the_line = bool(p_below < alpha)
    if on_the_line:
        designation, reason = "misspecified", "slope on the line and significantly positive"
    elif off_the_line:
        designation, reason = "well_specified", "slope significantly below the threshold"
    else:
        designation, reason = (
            "indeterminate",
            "fit too noisy to place the slope on either side of the threshold",
        )
    return {
        "designation": designation,
        "reason": reason,
        "slope": float(res.slope),
        "slope_stderr": se,
        "r": float(res.rvalue),
        "p": float(res.pvalue),
        "t_below_threshold": float(t_below),
        "p_below_threshold": p_below,
        "n_models": int(len(x)),
        "id_spread": float(x.max() - x.min()),
        "alpha": float(alpha),
        "criterion": (
            f"slope >= {SLOPE_THRESHOLD} and p < {alpha} and r > 0 => misspecified; "
            f"one-sided p(slope < {SLOPE_THRESHOLD}) < {alpha} => well_specified; "
            "otherwise indeterminate"
        ),
    }


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_designations(records: list[dict], exclude_prefix: str = "SADL") -> dict:
    """Designate every dataset present in ``records`` using baseline runs only."""
    by_ds: dict[str, list[dict]] = {}
    for r in records:
        if r.get("status") != "ok" or r["method"].startswith(exclude_prefix):
            continue
        by_ds.setdefault(r["dataset"], []).append(r)
    out: dict[str, dict] = {}
    for ds, rows in sorted(by_ds.items()):
        diag = diagnose([r["acc_id_val"] for r in rows], [r["acc_worst"] for r in rows])
        diag["pool"] = sorted({r["method"] for r in rows})
        diag["input_hash"] = _hash([[r["method"], r["seed"], r["acc_id_val"], r["acc_worst"]] for r in sorted(rows, key=lambda r: (r["method"], r["seed"]))])
        out[ds] = diag
    return out


def freeze(designations: dict, path: Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    path.write_text(json.dumps(designations, indent=2, sort_keys=True))
    return designations


def load(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}
