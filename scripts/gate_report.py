"""Read the acceptance traces of the preflight gate and say what they decide.

The gate answers one question before the sweep is launched: at the budget the
sweep will actually use, does the first Separator game produce a distinction that
satisfies Equation 19?  It has to be the first game, because ``SADL.fit`` breaks
out of the outer loop when a step exhausts its restarts -- if ``t = 0`` never
accepts, the sequence is empty, the representation is a column of zeros, and every
metric downstream of it is a chance-level artefact rather than a result.

This script does not decide anything on its own; it lays out which of the three
criteria bound, by how much, and whether the margin was closing across restarts,
which is what distinguishes "needs more restarts" from "needs a different gate".

    python scripts/gate_report.py --tag gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _budget_of(rec: dict):
    """The Budget the cell actually ran with: preset plus any --set overrides."""
    import sys

    sys.path.insert(0, str(ROOT))
    from sadl.methods import BUDGET_PRESETS

    bud = BUDGET_PRESETS[rec["budget"]]
    overrides = rec.get("budget_overrides") or {}
    return bud.with_(**overrides) if overrides else bud


def _attempts(rec: dict) -> list[dict]:
    return [t for t in (rec.get("sadl_trace") or []) if "restart" in t]


def _stops(rec: dict) -> list[str]:
    return [t["stop_reason"] for t in (rec.get("sadl_trace") or []) if "stop_reason" in t]


def _criteria(attempt: dict, bud) -> list[tuple[str, bool, float, str]]:
    """(name, passed, margin, rendered) for each clause of Equation 19.

    ``margin`` is signed so that positive means passing, in the units of the
    quantity itself.
    """
    nov = attempt.get("val_novelty", float("nan"))
    shift = attempt.get("val_shift", float("nan"))
    adv = attempt.get("val_adv_flip", float("nan"))
    if bud.sadl_compression_gate:
        nov_ok, nov_margin, nov_txt = nov >= bud.sadl_gamma, nov - bud.sadl_gamma, f"N={nov:.3f} >= {bud.sadl_gamma}"
    else:
        nov_ok, nov_margin, nov_txt = nov > 0.0, nov, f"N={nov:.3f} > 0"
    return [
        ("novelty", bool(nov_ok), float(nov_margin), nov_txt),
        ("shift", bool(shift <= bud.sadl_eps_stab), float(bud.sadl_eps_stab - shift), f"S={shift:.3f} <= {bud.sadl_eps_stab}"),
        ("flip", bool(adv <= bud.sadl_eps_adv), float(bud.sadl_eps_adv - adv), f"A={adv:.3f} <= {bud.sadl_eps_adv}"),
    ]


def report_one(rec: dict) -> dict:
    key = f"{rec['dataset']} / {rec['method']} / seed{rec['seed']}"
    print(f"\n{'=' * 78}\n{key}")
    if rec.get("status") != "ok":
        print(f"  status={rec.get('status')}: {str(rec.get('error', ''))[:200]}")
        return {"key": key, "verdict": "did not run"}

    bud = _budget_of(rec)
    attempts = _attempts(rec)
    n_acc = rec.get("n_accepted")
    steps_per_game = max(20, bud.steps // max(1, bud.sadl_tmax))
    print(
        f"  budget={rec['budget']} overrides={rec.get('budget_overrides') or {}} "
        f"device={rec.get('device')} {rec.get('train_seconds', float('nan')):.0f}s"
    )
    print(
        f"  thresholds: gamma={bud.sadl_gamma} eps_stab={bud.sadl_eps_stab} eps_adv={bud.sadl_eps_adv} "
        f"| T_max={bud.sadl_tmax} R_max={bud.sadl_rmax} steps/game={steps_per_game} batch/env={bud.batch_per_env}"
    )
    print(f"  accepted={n_acc}  repr_dim={rec.get('repr_dim')}  stop={_stops(rec) or ['-']}")

    if not attempts:
        print("  no acceptance attempts recorded")
        return {"key": key, "verdict": "no trace"}

    print(f"\n  {'t':>2} {'r':>2} {'ok':>3} {'novelty':>8} {'shift':>7} {'flip':>7} "
          f"{'bank':>7} {'learn':>7} {'worst':>7} {'bal':>6} {'best':>6}  binding")
    for a in attempts:
        crit = _criteria(a, bud)
        failed = [c[0] for c in crit if not c[1]]
        g = lambda k: a.get(k, float("nan"))
        print(
            f"  {a.get('t', -1):>2} {a.get('restart', -1):>2} {'Y' if a.get('accepted') else 'n':>3} "
            f"{g('val_novelty'):>8.3f} {g('val_shift'):>7.3f} {g('val_adv_flip'):>7.3f} "
            f"{g('val_adv_flip_bank'):>7.3f} {g('val_adv_flip_learned'):>7.3f} "
            f"{g('val_adv_flip_worstcase'):>7.3f} {g('val_balance'):>6.3f} "
            f"{str(a.get('best_step')):>6}  {','.join(failed) if failed else '-'}"
        )

    if n_acc:
        print("\n  VERDICT: GO -- the first game accepted, so the sequence is not empty.")
        return {"key": key, "verdict": "go", "n_accepted": n_acc}

    # Nothing accepted: name the binding criterion and say whether it was closing.
    t0 = [a for a in attempts if a.get("t") == 0]
    counts: dict[str, int] = {}
    for a in t0:
        for name, ok, _, _ in _criteria(a, bud):
            if not ok:
                counts[name] = counts.get(name, 0) + 1
    binding = max(counts, key=lambda k: counts[k]) if counts else None
    print(f"\n  VERDICT: NO-GO at t=0. binding criterion: {binding or 'unknown'} "
          f"(failed in {counts.get(binding, 0)}/{len(t0)} restarts)")

    last = _criteria(t0[-1], bud)
    for name, ok, margin, txt in last:
        print(f"    {name:8s} {'pass' if ok else 'FAIL'}  {txt}  (margin {margin:+.3f})")

    if binding == "flip":
        series = [a.get("val_adv_flip", float("nan")) for a in t0]
        trend = "declining" if len(series) > 1 and series[-1] < series[0] else "flat or rising"
        print(f"    flip rate across restarts: {[round(v, 3) for v in series]} -> {trend}")
        gap = bud.sadl_eps_adv - series[-1]
        if trend == "declining" and gap > -0.05:
            print("    close and still improving: more restarts is the cheap thing to try, and the")
            print("    trunk carries across restarts so they compound rather than reset.")
        else:
            print("    not converging towards eps_adv: this is a gate calibration question, not a")
            print("    budget question. Compare adv_flip_bank against adv_flip_learned -- if the bank")
            print("    alone is doing the rejecting, the bank may be leaving T on this dataset.")
    elif binding == "novelty":
        bal = [a.get("val_balance", float("nan")) for a in t0]
        print(f"    balance across restarts: {[round(v, 3) for v in bal]}")
        if any(b < 0.02 or b > 0.98 for b in bal if b == b):
            print("    balance at an extreme with novelty at zero is head collapse despite the")
            print("    non-collapse floors: report it as an optimisation failure with the trace.")
    elif binding == "shift":
        print("    the head is still environment-dependent: lambda_stab is not binding hard enough,")
        print("    or the two training environments differ too little to constrain it.")
    return {"key": key, "verdict": "no-go", "binding": binding}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="defaults to $SADL_RESULTS/runs or results/runs")
    ap.add_argument("--tag", default="gate", help="only records carrying this tag")
    args = ap.parse_args()

    import os

    runs = Path(args.runs) if args.runs else Path(os.environ.get("SADL_RESULTS", ROOT / "results")) / "runs"
    if not runs.is_dir():
        print(f"no run directory at {runs}")
        return 2
    paths = sorted(p for p in runs.glob("*.json") if args.tag in p.stem.split("__"))
    if not paths:
        print(f"no records tagged {args.tag!r} under {runs}")
        return 2

    verdicts = [report_one(json.loads(p.read_text())) for p in paths]
    print(f"\n{'=' * 78}\nsummary")
    for v in verdicts:
        print(f"  {v['key']:52s} {v['verdict']}" + (f"  (binding: {v['binding']})" if v.get("binding") else ""))
    go = [v for v in verdicts if v["verdict"] == "go"]
    print(f"\n  {len(go)}/{len(verdicts)} cell(s) accepted at least one distinction")
    if not go:
        print("  Nothing accepted. A full sweep from here fills the SADL rows with empty")
        print("  representations at chance accuracy; settle the binding criterion first.")
        return 1
    if len(go) < len(verdicts):
        print("  Mixed. The datasets that accepted are worth sweeping; the others will report")
        print("  n_accepted=0 with a stability gap of NaN rather than a misleading 0.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
