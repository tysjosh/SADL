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
                     above ``slope_threshold`` (accuracy is on the line),
  ``well_specified`` otherwise,
  ``indeterminate``  if there are too few models or the ID accuracies are
                     essentially constant, so the diagnostic cannot be evaluated.

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


def diagnose(id_acc: list[float], ood_acc: list[float]) -> dict:
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
    on_the_line = bool(res.slope >= SLOPE_THRESHOLD and res.pvalue < 0.05 and res.rvalue > 0)
    return {
        "designation": "misspecified" if on_the_line else "well_specified",
        "slope": float(res.slope),
        "r": float(res.rvalue),
        "p": float(res.pvalue),
        "n_models": int(len(x)),
        "id_spread": float(x.max() - x.min()),
        "criterion": f"slope >= {SLOPE_THRESHOLD} and p < 0.05 and r > 0 => misspecified",
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
