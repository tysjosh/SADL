"""Repair stability-gap values in records written before the degenerate-case fix.

Equation 25 scores a constant representation 0.0 -- a perfect stability gap
earned by encoding nothing.  A SADL run that accepted no distinction produces
exactly that, and averaging it into a method's mean turns a capacity failure into
its best-looking result.  ``stability_gap`` now returns NaN in that case, but
records written by the earlier code carry the misleading 0.0.

A degenerate run is identifiable without retraining: ``n_accepted == 0``.  This
rewrites those records in place, so the sweep does not have to be repeated.

    python scripts/patch_degenerate_gap.py --dry-run
    python scripts/patch_degenerate_gap.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patched, checked = 0, 0
    for path in sorted(Path(args.runs).glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        checked += 1
        # Only SADL-family records carry n_accepted; a baseline is never degenerate
        # in this sense.
        if rec.get("n_accepted") != 0:
            continue
        if rec.get("stability_gap") is None:
            continue
        print(
            f"{'would patch' if args.dry_run else 'patching'} {path.name}: "
            f"stability_gap {rec.get('stability_gap')} -> null (n_accepted=0)"
        )
        patched += 1
        if args.dry_run:
            continue
        rec["stability_gap"] = None
        rec["stability_gap_per_dim"] = []
        rec["stability_gap_max"] = None
        rec["stability_gap_degenerate"] = True
        path.write_text(json.dumps(rec, indent=2, sort_keys=True))

    print(f"\n{patched} degenerate record(s) of {checked} ok record(s)")
    if patched and not args.dry_run:
        print("re-render tables with: python -m sadl.experiments.rqs tables --budget gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
