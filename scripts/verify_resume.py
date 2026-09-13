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
    ap.add_argument(
        "--rerun-script",
        default="/tmp/sadl_rerun.sh",
        help="where to write the --overwrite commands for incomparable cells",
    )
    ap.add_argument(
        "--git-depth",
        type=int,
        default=50,
        help="how many commits to replay when classifying records that predate per-group "
        "fingerprints (0 to skip). Cheap -- a few tenths of a second -- and must reach far "
        "enough back to cover the oldest records still in results/runs",
    )
    args = ap.parse_args()

    from sadl.provenance import classify_record, legacy_fingerprints, source_fingerprints
    from sadl.utils import source_fingerprint

    runs = Path(args.runs) if args.runs else Path(os.environ.get("SADL_RESULTS", ROOT / "results")) / "runs"
    if not runs.is_dir():
        print(f"FAIL  no run directory at {runs}")
        return 2

    mine = source_fingerprint()
    mine_groups = source_fingerprints()
    legacy = legacy_fingerprints(args.git_depth) if args.git_depth else {}
    if args.git_depth and not legacy:
        print("  (no git history available; records without group fingerprints stay 'unknown')")
    paths = sorted(runs.glob("*.json"))
    status: collections.Counter = collections.Counter()
    prints: collections.Counter = collections.Counter()
    prov_status: collections.Counter = collections.Counter()
    incomparable: collections.defaultdict = collections.defaultdict(list)
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
        flat = rec.get("fingerprint", "none")
        prints[flat] += 1
        prov = classify_record(rec, mine_groups, legacy)
        prov_status[prov["status"]] += 1
        if prov["material"]:
            # Key on the full cache key minus the seed: budget and tag must not be
            # collapsed, or `smoke` scratch cells appear in the same row as the
            # reporting cells and the list stops being actionable.
            npe_of = (rec.get("dataset_kwargs") or {}).get("n_per_env")
            incomparable[
                (rec["dataset"], rec["method"], rec.get("budget"), rec.get("tag") or "", npe_of)
            ].append(rec["seed"])
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
    print(f"\nsource provenance (this checkout: flat {mine})")
    for g, h in sorted(mine_groups.items()):
        print(f"    {g:8s} {h}")
    print("  records by comparability:")
    for st, n in prov_status.most_common():
        note = {
            "ok": "identical source",
            "compatible": "differs only where it cannot change a record's numbers",
            "incomparable": "differs in train/metric code -- MUST be re-run",
            "unknown": "predates per-group fingerprints and no matching commit found",
        }[st]
        print(f"    {st:13s} {n:4d}   {note}")
    if incomparable:
        ok = False
        n = sum(len(v) for v in incomparable.values())
        print(f"\n  {n} record(s) this code cannot reproduce:")
        cmds = []
        for k in sorted(incomparable, key=lambda k: (k[2] != "gpu", k[3] != "", k)):
            ds, m, budget, tag, npe_of = k
            seeds = sorted(incomparable[k])
            label = f"{budget}{'/' + tag if tag else ''}"
            print(f"    {ds:22s} {m:18s} {label:16s} n_per_env={str(npe_of):5s} seeds {seeds}")
            for s in seeds:
                # Reuse the cell's own n_per_env: it is not part of the cache key, so
                # re-running with a different value would silently change the cell
                # rather than reproduce it.
                cmds.append(
                    f".venv/bin/python -m sadl.experiments.run --dataset {ds} --method {m} "
                    f"--seed {s} --budget {budget}"
                    + (f" --tag {tag}" if tag else "")
                    + (f" --n-per-env {npe_of}" if npe_of else "")
                    + " --overwrite"
                )
        script = Path(args.rerun_script)
        script.write_text("#!/usr/bin/env bash\nset -uo pipefail\n" + "\n".join(cmds) + "\n")
        print(f"\n  Re-run commands written to {script} ({len(cmds)} cells).")
        print("  Reporting cells are the `gpu` ones with no tag; the rest are scratch and can be")
        print("  deleted instead. Records that are merely 'compatible' do NOT need re-running.")

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
