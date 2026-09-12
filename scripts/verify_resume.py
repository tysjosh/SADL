"""Check that downloaded run records can be resumed by this checkout.

A sweep resumes by filename: ``results/runs/<dataset>__<method>__seed<n>__<budget>[__<tag>].json``
is the cache key, so records only count if they land there intact.  Two things can
go wrong silently after moving records between machines, and both are worse than a
crash because the sweep continues and the tables render anyway:

1. the code differs from the code that produced the records.  Every record stores
   ``source_fingerprint()``; the runner warns once and then mixes versions.
2. a record survived without its ``.npz`` of per-instance correctness vectors --
   the usual cause is a machine dying mid-cell.  ``load_correctness`` returns an
   empty dict for those, and the hierarchical bootstrap of Section 6.5 then
   resamples nothing for that cell instead of failing.

    python scripts/verify_resume.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="defaults to $SADL_RESULTS/runs or results/runs")
    args = ap.parse_args()

    from sadl.utils import source_fingerprint

    runs = Path(args.runs) if args.runs else Path(os.environ.get("SADL_RESULTS", ROOT / "results")) / "runs"
    if not runs.is_dir():
        print(f"FAIL  no run directory at {runs}")
        return 2

    mine = source_fingerprint()
    paths = sorted(runs.glob("*.json"))
    status: collections.Counter = collections.Counter()
    prints: collections.Counter = collections.Counter()
    npe: collections.Counter = collections.Counter()
    budgets: collections.Counter = collections.Counter()
    missing_npz: list[str] = []
    reusable = 0

    for p in paths:
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  CORRUPT (truncated mid-write?): {p.name}")
            status["corrupt"] += 1
            continue
        st = rec.get("status", "?")
        status[st] += 1
        if st != "ok":
            continue
        reusable += 1
        prints[rec.get("fingerprint", "none")] += 1
        budgets[f"{rec.get('budget')}{'/' + rec['tag'] if rec.get('tag') else ''}"] += 1
        npe[str((rec.get("dataset_kwargs") or {}).get("n_per_env"))] += 1
        if not p.with_suffix(".npz").exists():
            missing_npz.append(p.name)

    print(f"records at {runs}")
    print(f"  {len(paths)} json, {reusable} reusable (only status=ok is reused)")
    print("  status:  " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
    print("  budget:  " + ", ".join(f"{k}={v}" for k, v in sorted(budgets.items())))
    print("  n_per_env: " + ", ".join(f"{k}={v}" for k, v in sorted(npe.items())))

    ok = True
    print(f"\nthis checkout's source fingerprint: {mine}")
    for fp, n in prints.most_common():
        mark = "match" if fp == mine else "MISMATCH -> records were produced by different code"
        print(f"  {fp}  {n:4d} record(s)   {mark}")
        if fp != mine:
            ok = False
    if not ok:
        print("\n  Fix by checking out the commit that produced them, or re-run those cells with")
        print("  --overwrite. Do not mix: a table built from two versions of the method is not")
        print("  a comparison.")

    if len(npe) > 1:
        ok = False
        print("\n  MIXED n_per_env: sample count is not part of the cache key, so these cells")
        print("  were trained on different amounts of data and will be aggregated together.")

    if missing_npz:
        ok = False
        print(f"\n  {len(missing_npz)} ok record(s) missing their .npz correctness vectors:")
        for n in missing_npz[:10]:
            print(f"    {n}")
        if len(missing_npz) > 10:
            print(f"    ... and {len(missing_npz) - 10} more")
        print("  The hierarchical bootstrap resamples those vectors; without them the cell")
        print("  contributes nothing to the CI. Re-run these cells with --overwrite.")

    print("\n" + ("READY to resume" if ok else "NOT clean -- read the warnings above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
