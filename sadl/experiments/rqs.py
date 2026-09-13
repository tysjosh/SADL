"""Drivers for RQ1-RQ6 (Section 6.1) and the tables of Section 6.6.

Usage:

    python -m sadl.experiments.rqs all --budget quick --seeds 0 1 2
    python -m sadl.experiments.rqs rq2 --budget standard --seeds 0 1 2 3 4

Each driver writes its raw records and a rendered markdown table under
``results/``.  Runs are cached by key, so re-running only fills gaps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data import REAL, SYNTHETIC, env_specs_for
from ..eval.misspec import build_designations, freeze, load
from ..methods import BASELINES, TABLE2_METHODS
from ..utils import RESULTS_DIR, write_json
from .run import run_grid
from .tables import (
    absolute_cost_table,
    accepted_per_hour_table,
    cell,
    confirmatory_analysis,
    cost_table,
    group,
    markdown_table,
    metric_table,
)

# Set from --shard / --tables; every driver goes through _grid so that sharded
# execution and pure aggregation share one code path.
SHARD: tuple[int, int] | None = None
CACHED_ONLY = False

SYNTH_FACTORS = ["dSprites", "Causal3DIdent-lite"]
MAIN_DATASETS = ["ColoredMNIST", "dSprites", "Causal3DIdent-lite"] + REAL
CONFUSER_FAMILIES = [
    ("SADL-conf-none", "None (shift only)"),
    ("SADL-conf-fixed3", "Fixed bank: 3"),
    ("SADL-conf-fixed10", "Fixed bank: 10"),
    ("SADL-conf-learned", "Learned mixture"),
]
ABLATIONS = [
    ("SADL", "SADL (full)"),
    ("SADL-lite", "SADL-lite (fixed bank)"),
    ("SADL-no-gate", "No compression gate"),
    ("SADL-joint", "Joint rather than sequential"),
    ("SADL-T1", "T_max = 1"),
    ("SADL-T8", "T_max = 8"),
    ("SADL-warmstart", "+ Confuser-consistent trunk warm start"),
]
ENV_COUNTS = [1, 2, 3, 5, 8]

# Table 10 is measured on the two synthetic benchmarks that every other driver
# also runs, and with the same keys, so it reuses their cached cells instead of
# training anything of its own.  That is also what makes the ratios meaningful:
# "identical hardware and matched input budgets" holds only if the numerator and
# the denominator come from the same sweep.
COST_DATASETS = ["ColoredMNIST", "dSprites"]
# The row set of the paper's Table 10, in its order.
COST_METHODS = ["ERM", "SimCLR", "MAE", "IRMv1", "DANN", "SADL-lite", "SADL"]
COST_METHODS_EXTRA = [m for m in TABLE2_METHODS if m not in COST_METHODS]
COST_SADL_VARIANTS = ["SADL-lite", "SADL"]

# Methods that train their own classifier, and can therefore be scored the way
# their source papers score them as well as under the shared frozen readout.
OWN_CLF_METHODS = ["ERM", "IRMv1", "VREx", "DANN"]


ONLY_DATASETS: list[str] | None = None
ONLY_METHODS: list[str] | None = None
DRY_RUN = False


def _grid(datasets, methods, seeds, **kwargs) -> list[dict]:
    """Every driver funnels through here, so filters, sharding, aggregation and
    dry-run behave identically for all six research questions."""
    if ONLY_DATASETS is not None:
        datasets = [d for d in datasets if d in ONLY_DATASETS]
    if ONLY_METHODS is not None:
        methods = [m for m in methods if m in ONLY_METHODS]
    kwargs.setdefault("shard", SHARD)
    kwargs.setdefault("cached_only", CACHED_ONLY)
    if DRY_RUN:
        return _list_cells(datasets, methods, seeds, kwargs)
    return run_grid(datasets, methods, seeds, **kwargs)


def _list_cells(datasets, methods, seeds, kwargs) -> list[dict]:
    """Print which cells would run, and whether each is already cached."""
    from ..utils import RESULTS_DIR
    from .run import run_key, shard_of

    budget = kwargs.get("budget", "standard")
    tag = kwargs.get("tag", "")
    i, todo, cached = 0, 0, 0
    for d in datasets:
        for m in methods:
            for s in seeds:
                i += 1
                if not shard_of(i - 1, kwargs.get("shard")):
                    continue
                key = run_key(d, m, s, budget, tag)
                exists = (RESULTS_DIR / "runs" / f"{key}.json").exists()
                cached += exists
                todo += not exists
                print(f"  [{'cached' if exists else ' todo '}] {key}")
    print(f"  -> {todo} to run, {cached} cached")
    return []


def _tables_only(args: argparse.Namespace) -> int:
    """Re-render every table from cached records without training anything."""
    global CACHED_ONLY
    CACHED_ONLY = True
    rq1(args.budget, args.seeds, args.n_per_env)
    recs = rq2(args.budget, args.seeds, args.n_per_env)
    rq3(args.budget, args.seeds, args.n_per_env)
    rq4(args.budget, args.seeds, args.n_total)
    rq6(args.budget, args.seeds, args.n_per_env)
    table10(args.budget, args.seeds, args.n_per_env)
    if recs:
        rq5(recs)
    return 0


def _out(name: str) -> Path:
    return RESULTS_DIR / name


def _write(name: str, payload: dict, text: str) -> None:
    if DRY_RUN:
        return
    write_json(_out(f"{name}.json"), payload)
    path = _out(f"{name}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"\n{text}\n-> {path}")


# --------------------------------------------------------------------- RQ1
def rq1(budget: str, seeds: list[int], n_per_env: int) -> list[dict]:
    recs = _grid(SYNTH_FACTORS, TABLE2_METHODS, seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env})
    g = group(recs)
    header = ["Method"]
    for d in SYNTH_FACTORS:
        header += [f"{d} FR", f"{d} worst acc."]
    rows = []
    for m in TABLE2_METHODS:
        row = [m]
        for d in SYNTH_FACTORS:
            row += [cell(g.get((d, m), []), "fr", pct=False, nd=3), cell(g.get((d, m), []), "acc_worst")]
        rows.append(row)
    text = (
        "### Table 3: ground-truth factor recovery and worst-environment accuracy (RQ1)\n\n"
        f"budget={budget}, seeds={seeds}, n_per_env={n_per_env}. FR is Equation 24 (NMI, "
        "one-to-one matching); accuracy is in percent, mean ± sd over seeds.\n\n"
        + markdown_table(header, rows)
    )
    _write("table3_rq1_factor_recovery", {"records": recs}, text)
    return recs


# --------------------------------------------------------------------- RQ2
def rq2(budget: str, seeds: list[int], n_per_env: int) -> list[dict]:
    datasets = MAIN_DATASETS
    # Stage 1: baselines only, so the misspecification designation cannot depend
    # on the method under test.
    base_recs = _grid(datasets, BASELINES, seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env})
    built = build_designations(base_recs)
    if CACHED_ONLY or not built:
        # Aggregation must never create the frozen file: an empty or partial
        # designation written here would then be read back as authoritative and
        # would silently decide which splits Table 4 reports.
        designations = load(_out("split_designations.json")) or built
    else:
        designations = freeze(built, _out("split_designations.json"))
    print("\nsplit designations:")
    for d, v in designations.items():
        if "slope" in v:
            detail = (
                f"slope={v['slope']:+.3f}±{v.get('slope_stderr', float('nan')):.3f} "
                f"r={v.get('r', float('nan')):+.2f} p={v.get('p', float('nan')):.3f} "
                f"p(slope<{0.5})={v.get('p_below_threshold', float('nan')):.4f}"
            )
        else:
            detail = v.get("reason", "")
        print(f"  {d:20s} {v['designation']:15s} {detail}")

    # Stage 2: SADL and its fixed-bank ablation.
    sadl_recs = _grid(datasets, ["SADL-lite", "SADL"], seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env})
    recs = base_recs + sadl_recs

    well = [d for d in datasets if designations.get(d, {}).get("designation") == "well_specified"]
    analysis = confirmatory_analysis(recs, datasets, BASELINES)
    delta_spec = _delta_spec(recs, datasets, well)

    # A hierarchical bootstrap computed from fewer seeds than were run is not the
    # interval Section 6.5 specifies, so say so next to the table instead of
    # leaving it to be discovered in the JSON.
    short = []
    for d, v in analysis["per_dataset"].items():
        for arm in ("treatment_bootstrap", "baseline_bootstrap"):
            b = v.get(arm) or {}
            if b.get("n_seeds_dropped"):
                short.append(f"{d} {arm.split('_')[0]}: {b['n_seeds']} seed(s) used, {b['n_seeds_dropped']} dropped")
    boot_note = (
        "\n**Hierarchical bootstrap coverage.** Some cells had no per-instance correctness "
        "vectors (`.npz` absent), so their seeds are excluded from the bootstrap; the paired "
        "effect and its p-value are unaffected because they read `acc_worst` from the record. "
        "Re-run those cells with `--overwrite` to restore full coverage.\n\n  - "
        + "\n  - ".join(short)
        + "\n"
        if short
        else ""
    )

    t4 = metric_table(recs, well, TABLE2_METHODS, "acc_worst", add_mean=True) if well else "_no split was designated well specified_"
    t5 = metric_table(recs, datasets, TABLE2_METHODS, "acc_worst", add_mean=True)
    t5b = metric_table(recs, datasets, TABLE2_METHODS, "acc_avg", add_mean=True)
    text = (
        "### Split designations (Section 6.2)\n\n"
        "Frozen from baseline runs only, before any SADL run was evaluated. Section 6.2 requires "
        "the classifications and exclusions to be listed, including failed and indeterminate "
        "diagnostics, so every candidate split appears here whether or not it reaches Table 4.\n\n"
        + _designation_table(designations)
        + "\n\n### Table 4: worst-environment accuracy on splits designated well specified (RQ2)\n\n"
        f"budget={budget}, seeds={seeds}. Only the splits designated `well_specified` above are "
        "reported here; the rest are excluded rather than assigned a result.\n\n" + t4 + "\n\n"
        "### Table 5: worst-environment accuracy on all splits (RQ2)\n\n" + t5 + "\n\n"
        "### Table 5b: average-environment accuracy on all splits\n\n" + t5b + "\n\n"
        "### Table 5c: worst-environment accuracy under each method's OWN classifier\n\n"
        "Tables 4, 5 and 5b use the shared frozen-feature readout of Section 6.3, which is the "
        "preregistered protocol and the only one SADL can be scored under. That protocol discards "
        "the classifier IRM and VREx actually train, and refits an ERM readout on their frozen "
        "features -- so the shortcut those methods suppress in the *classifier* is simply "
        "relearned. Below, the methods that train their own classifier are scored with it, which "
        "is how their source papers report them. Cite this column when comparing against "
        "published numbers, and Table 5 when comparing representations.\n\n"
        + metric_table(recs, datasets, OWN_CLF_METHODS, "acc_worst_own", add_mean=True)
        + "\n\n_Blank cells are runs recorded before this second protocol existed; re-run them "
        "with `--overwrite` to fill it._\n\n"
        "### Confirmatory comparison\n\n" + _analysis_table(analysis) + "\n"
        + boot_note + "\n"
        f"Preregistered contrast (Equation 26): Delta_spec = {delta_spec['delta_spec']}\n"
    )
    _write(
        "table4_5_rq2_ood",
        {"records": recs, "designations": designations, "analysis": analysis, "delta_spec": delta_spec},
        text,
    )
    return recs


def _designation_table(designations: dict) -> str:
    """Every candidate split with the numbers behind its designation."""
    header = [
        "Split",
        "Designation",
        "In Table 4",
        "Slope",
        "SE",
        "r",
        "p (slope != 0)",
        "p (slope < 0.5)",
        "Models",
        "Reason",
    ]
    rows = []
    f = lambda v, nd=3: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
    for d, v in sorted(designations.items()):
        des = v.get("designation", "?")
        rows.append(
            [
                d,
                des,
                "yes" if des == "well_specified" else "no",
                f(v.get("slope")),
                f(v.get("slope_stderr")),
                f(v.get("r")),
                f(v.get("p")),
                f(v.get("p_below_threshold"), 4),
                str(v.get("n_models", "n/a")),
                str(v.get("reason", "")),
            ]
        )
    if not rows:
        return "_no split was designated: no baseline records were available_"
    return markdown_table(header, rows)


def _analysis_table(analysis: dict) -> str:
    header = ["Dataset", "Reference baseline", "SADL", "Baseline", "Paired diff [95% CI]", "p (Holm)"]
    rows = []
    for d, v in analysis["per_dataset"].items():
        if v.get("status") == "unavailable":
            rows.append([d, str(v.get("reference")), "n/a", "n/a", "n/a", "n/a"])
            continue
        eff = v["paired_effect"]
        holm = analysis["holm"].get(d, {})
        rows.append(
            [
                d,
                v["reference"],
                f"{v['treatment_mean'] * 100:.1f}",
                f"{v['baseline_mean'] * 100:.1f}",
                f"{eff['point'] * 100:+.1f} [{eff['ci_lo'] * 100:+.1f}, {eff['ci_hi'] * 100:+.1f}]",
                f"{holm.get('p_holm', float('nan')):.3f}",
            ]
        )
    return markdown_table(header, rows)


def _delta_spec(recs: list[dict], datasets: list[str], well: list[str]) -> dict:
    """Equation 26: SADL's margin on well-specified splits minus its margin overall."""
    a_all = confirmatory_analysis(recs, datasets, BASELINES)["per_dataset"]
    deltas = {d: v["paired_effect"]["point"] for d, v in a_all.items() if v.get("status") != "unavailable"}
    if not deltas:
        return {"delta_spec": None}
    d_well = [v for k, v in deltas.items() if k in well]
    d_all = list(deltas.values())
    if not d_well:
        return {"delta_spec": None, "per_dataset": deltas}
    return {
        "delta_spec": float(np.mean(d_well) - np.mean(d_all)),
        "delta_well_specified": float(np.mean(d_well)),
        "delta_all": float(np.mean(d_all)),
        "per_dataset": deltas,
    }


# --------------------------------------------------------------------- RQ3
def rq3(budget: str, seeds: list[int], n_per_env: int) -> list[dict]:
    datasets = ["ColoredMNIST", "dSprites"]
    methods = [m for m, _ in CONFUSER_FAMILIES]
    recs = _grid(datasets, methods, seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env})
    g = group(recs)
    header = ["Confuser family"]
    for d in datasets:
        header += [f"{d} stability gap", f"{d} worst acc.", f"{d} accepted T"]
    rows = []
    for m, label in CONFUSER_FAMILIES:
        row = [label]
        for d in datasets:
            recs_d = g.get((d, m), [])
            row += [
                cell(recs_d, "stability_gap", pct=False, nd=3),
                cell(recs_d, "acc_worst"),
                cell(recs_d, "n_accepted", pct=False, nd=2),
            ]
        rows.append(row)
    text = (
        "### Table 7: stability gap versus nested Confuser families (RQ3)\n\n"
        "Families are nested; each contains the previous one. Stability gap is "
        "Equation 25 (lower is better).\n\n" + markdown_table(header, rows)
    )
    _write("table7_rq3_confuser", {"records": recs}, text)
    return recs


# --------------------------------------------------------------------- RQ4
def rq4(budget: str, seeds: list[int], n_total: int) -> list[dict]:
    dataset = "ColoredMNIST"
    recs: list[dict] = []
    for k in ENV_COUNTS:
        kwargs = {
            "n_per_env": max(500, n_total // k),
            "n_train_envs": k,
            "env_specs": env_specs_for(dataset, k),
        }
        recs += _grid([dataset], ["ERM", "SADL"], seeds, budget=budget, dataset_kwargs=kwargs, tag=f"E{k}")
    header = ["|E|", "SADL stability gap", "SADL worst acc.", "SADL accepted T", "ERM worst acc."]
    rows = []
    for k in ENV_COUNTS:
        sel = lambda m: [r for r in recs if r.get("tag") == f"E{k}" and r["method"] == m]
        rows.append(
            [
                str(k),
                cell(sel("SADL"), "stability_gap", pct=False, nd=3),
                cell(sel("SADL"), "acc_worst"),
                cell(sel("SADL"), "n_accepted", pct=False, nd=2),
                cell(sel("ERM"), "acc_worst"),
            ]
        )
    text = (
        "### Table 8: sensitivity to the number of training environments (RQ4)\n\n"
        f"Total sample count is held at {n_total} as |E| changes, so environment count is not "
        "confounded with data volume. |E| = 1 is the condition Theorem 4.2 speaks to.\n\n"
        + markdown_table(header, rows)
    )
    _write("table8_rq4_env_count", {"records": recs}, text)
    return recs


# --------------------------------------------------------------------- RQ5
def rq5_standalone(budget: str, seeds: list[int], n_per_env: int) -> None:
    """RQ5 scores frozen representations, so it reuses the RQ2 cells from cache."""
    global CACHED_ONLY
    prev, CACHED_ONLY = CACHED_ONLY, True
    recs = _grid(
        MAIN_DATASETS, TABLE2_METHODS, seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env}
    )
    CACHED_ONLY = prev
    rq5(recs)


def rq5(records: list[dict], datasets: list[str] | None = None) -> None:
    datasets = datasets or ["ColoredMNIST", "dSprites", "Causal3DIdent-lite"]
    text = (
        "### Composability of frozen representations (RQ5)\n\n"
        "Equation 22 on probe tasks withheld from pretraining and from model selection; "
        "1.0 is oracle-level, 0.0 is the constant predictor.\n\n"
        + metric_table(records, datasets, TABLE2_METHODS, "composability", pct=False, nd=3)
        + "\n\n#### Empirical stability gap (Equation 25, lower is better)\n\n"
        + metric_table(records, datasets, TABLE2_METHODS, "stability_gap", pct=False, nd=3)
    )
    _write("table9_rq5_composability", {"n_records": len(records)}, text)


# --------------------------------------------------------------------- RQ6
def rq6(budget: str, seeds: list[int], n_per_env: int) -> list[dict]:
    datasets = ["ColoredMNIST", "dSprites"]
    methods = [m for m, _ in ABLATIONS]
    recs = _grid(datasets, methods + ["ERM"], seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env})
    g = group(recs)
    erm_time = {}
    for d in datasets:
        ok = [r for r in g.get((d, "ERM"), []) if r.get("status") == "ok"]
        erm_time[d] = float(np.mean([r["train_seconds"] for r in ok])) if ok else float("nan")
    header = ["Variant", "Worst acc.", "Factor recovery", "Composability", "Accepted T", "Time / ERM"]
    rows = []
    for m, label in ABLATIONS:
        recs_m = [r for d in datasets for r in g.get((d, m), [])]
        ok = [r for r in recs_m if r.get("status") == "ok"]
        ratios = [r["train_seconds"] / erm_time[r["dataset"]] for r in ok if not np.isnan(erm_time[r["dataset"]])]
        rows.append(
            [
                label,
                cell(recs_m, "acc_worst"),
                cell(recs_m, "fr", pct=False, nd=3),
                cell(recs_m, "composability", pct=False, nd=3),
                cell(recs_m, "n_accepted", pct=False, nd=2),
                f"{np.mean(ratios):.1f}x" if ratios else "n/a",
            ]
        )
    text = (
        "### Table 6: component ablations (RQ6)\n\n"
        f"Averaged over {datasets}, budget={budget}, seeds={seeds}.\n\n" + markdown_table(header, rows)
    )
    _write("table6_rq6_ablations", {"records": recs}, text)
    return recs


# ----------------------------------------------------------------- Table 10
def table10(budget: str, seeds: list[int], n_per_env: int) -> list[dict]:
    """Table 10: computational cost relative to ERM.

    Section 6.4 asks for wall-clock time, peak accelerator memory and
    forward/backward evaluations, all relative to ERM; Section 5.7 adds accepted
    distinctions per accelerator-hour, which only applies to the SADL rows.

    The cells are the same keys RQ1 and RQ2 use, so this trains nothing once
    either of those has run.
    """
    recs = _grid(
        COST_DATASETS, TABLE2_METHODS, seeds, budget=budget, dataset_kwargs={"n_per_env": n_per_env}
    )
    if DRY_RUN:
        return recs
    # _grid applies the --datasets filter internally; the caption and the ratios
    # have to describe what was actually aggregated, not the unfiltered default.
    datasets = [d for d in COST_DATASETS if ONLY_DATASETS is None or d in ONLY_DATASETS]

    n_mem = sum(1 for r in recs if r.get("status") == "ok" and str(r.get("device", "")).startswith("cuda"))
    mem_note = (
        ""
        if n_mem
        else (
            "\nThe memory column is `n/a`: a true peak is only available from "
            "`torch.cuda.max_memory_allocated`, which is reset before each fit. MPS exposes no "
            "peak counter in torch 2.4 and CPU none at all, so no residual allocation is "
            "reported here as though it were a peak.\n"
        )
    )
    missing_evals = [
        r for r in recs if r.get("status") == "ok" and r.get("evals_per_step") is None
    ]
    stale_note = (
        f"\n{len(missing_evals)} cached record(s) predate the evaluation counter and carry no "
        "`evals_per_step`; re-run those cells with `--overwrite` to fill that column.\n"
        if missing_evals
        else ""
    )

    text = (
        "### Table 10: computational cost relative to ERM\n\n"
        f"budget={budget}, seeds={seeds}, datasets={datasets}. Ratios are formed inside each "
        "(dataset, seed) cell and then averaged, so numerator and denominator always share "
        "hardware and input budget; mean ± sd is over those cells. `Evals/step` counts shared-trunk "
        "forward plus backward evaluations per environment-balanced minibatch, relative to ERM, and "
        "includes SADL's periodic acceptance audits and compression probe because Algorithm 1 does.\n"
        "\nRead `Time/ERM` for the SADL rows together with the `Steps` column below: a sequence that "
        "stops early on `restarts_exhausted` runs fewer games and so costs less, which makes the "
        "time ratio a measurement of what the run did rather than of the method's budget.\n"
        + mem_note
        + stale_note
        + "\n"
        + cost_table(recs, datasets, COST_METHODS)
        + "\n\n#### Remaining Table 2 methods\n\n"
        + cost_table(recs, datasets, COST_METHODS_EXTRA)
        + "\n\n#### Absolute figures behind the ratios\n\n"
        + absolute_cost_table(recs, datasets, TABLE2_METHODS)
        + "\n\n#### Accepted distinctions per accelerator-hour (Section 5.7)\n\n"
        + accepted_per_hour_table(recs, datasets, COST_SADL_VARIANTS)
    )
    _write("table10_cost", {"records": recs}, text)
    return recs


# --------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="run the SADL research questions")
    ap.add_argument(
        "which",
        choices=["rq1", "rq2", "rq3", "rq4", "rq5", "rq6", "cost", "tables", "all"],
        help="'cost' renders Table 10 from the RQ1/RQ2 cells",
    )
    ap.add_argument("--budget", default="quick")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-per-env", type=int, default=4000)
    ap.add_argument("--n-total", type=int, default=8000)
    ap.add_argument(
        "--shard",
        default=None,
        help="i/n: run only cells i (mod n); use one shard per GPU, then rerun with "
        "`tables` to aggregate everything",
    )
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict this RQ to these datasets")
    ap.add_argument("--methods", nargs="+", default=None, help="restrict this RQ to these methods")
    ap.add_argument("--dry-run", action="store_true", help="list the cells and exit without training")
    args = ap.parse_args()
    global SHARD, ONLY_DATASETS, ONLY_METHODS, DRY_RUN
    if args.shard:
        i, n = args.shard.split("/")
        SHARD = (int(i), int(n))
    ONLY_DATASETS, ONLY_METHODS, DRY_RUN = args.datasets, args.methods, args.dry_run
    if args.which == "tables":
        return _tables_only(args)
    if args.which == "rq5":
        rq5_standalone(args.budget, args.seeds, args.n_per_env)
        return 0

    main_recs: list[dict] = []
    if args.which in ("rq1", "all"):
        rq1(args.budget, args.seeds, args.n_per_env)
    if args.which in ("rq2", "all"):
        main_recs = rq2(args.budget, args.seeds, args.n_per_env)
    if args.which in ("rq3", "all"):
        rq3(args.budget, args.seeds, args.n_per_env)
    if args.which in ("rq4", "all"):
        rq4(args.budget, args.seeds, args.n_total)
    if args.which in ("rq6", "all"):
        rq6(args.budget, args.seeds, args.n_per_env)
    if args.which in ("cost", "all"):
        table10(args.budget, args.seeds, args.n_per_env)
    if args.which == "all" and main_recs:
        rq5(main_recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
