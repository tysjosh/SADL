"""Aggregation and markdown rendering for the tables of Section 6.6."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..eval.stats import hierarchical_bootstrap_worst_acc, holm, mean_std, paired_effect
from ..utils import read_json
from .run import load_correctness


def group(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        out[(r["dataset"], r["method"])].append(r)
    return {k: sorted(v, key=lambda r: r["seed"]) for k, v in out.items()}


def cell(records: list[dict], metric: str, pct: bool = True, nd: int = 1) -> str:
    """``mean ± sd`` over seeds, or a status marker when no run succeeded."""
    if not records:
        return "not run"
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        statuses = {r.get("status") for r in records}
        if statuses == {"not_run"}:
            return "not run"
        if "data_unavailable" in statuses:
            return "n/a (no data)"
        return "failed"
    vals = [r.get(metric) for r in ok]
    vals = [float("nan") if v is None else float(v) for v in vals]
    m, s = mean_std(vals)
    if np.isnan(m):
        return "n/a"
    scale = 100.0 if pct else 1.0
    return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i]) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def metric_table(
    records: list[dict],
    datasets: list[str],
    methods: list[str],
    metric: str,
    pct: bool = True,
    nd: int = 1,
    add_mean: bool = False,
) -> str:
    g = group(records)
    header = ["Method"] + datasets + (["Mean"] if add_mean else [])
    rows = []
    for m in methods:
        row = [m]
        means = []
        for d in datasets:
            recs = g.get((d, m), [])
            row.append(cell(recs, metric, pct, nd))
            ok = [r for r in recs if r.get("status") == "ok" and r.get(metric) is not None]
            if ok:
                means.append(np.nanmean([float(r[metric]) for r in ok]))
        if add_mean:
            row.append(f"{np.mean(means) * (100 if pct else 1):.{nd}f}" if means else "n/a")
        rows.append(row)
    return markdown_table(header, rows)


def strongest_baseline(records: list[dict], dataset: str, baselines: list[str]) -> str | None:
    """Baseline with the best training-domain validation accuracy (Section 6.5)."""
    g = group(records)
    best, best_id = None, -np.inf
    for b in baselines:
        ok = [r for r in g.get((dataset, b), []) if r.get("status") == "ok"]
        if not ok:
            continue
        v = float(np.mean([r["acc_id_val"] for r in ok]))
        if v > best_id:
            best, best_id = b, v
    return best


def confirmatory_analysis(
    records: list[dict],
    datasets: list[str],
    baselines: list[str],
    treatment: str = "SADL",
    metric: str = "acc_worst",
) -> dict:
    """Paired effect against the validation-selected baseline, plus Holm family."""
    g = group(records)
    out: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    for d in datasets:
        ref = strongest_baseline(records, d, baselines)
        t_ok = [r for r in g.get((d, treatment), []) if r.get("status") == "ok"]
        if ref is None or not t_ok:
            out[d] = {"status": "unavailable", "reference": ref}
            continue
        b_ok = [r for r in g.get((d, ref), []) if r.get("status") == "ok"]
        seeds = sorted({r["seed"] for r in t_ok} & {r["seed"] for r in b_ok})
        tv = [next(r[metric] for r in t_ok if r["seed"] == s) for s in seeds]
        bv = [next(r[metric] for r in b_ok if r["seed"] == s) for s in seeds]
        eff, p = paired_effect(tv, bv)
        boot_t = hierarchical_bootstrap_worst_acc([load_correctness(r) for r in t_ok])
        boot_b = hierarchical_bootstrap_worst_acc([load_correctness(r) for r in b_ok])
        out[d] = {
            "reference": ref,
            "n_seeds": len(seeds),
            "treatment_mean": float(np.mean(tv)),
            "baseline_mean": float(np.mean(bv)),
            "paired_effect": eff.as_dict(),
            "p_value": p,
            "treatment_bootstrap": boot_t.as_dict(),
            "baseline_bootstrap": boot_b.as_dict(),
        }
        pvals[d] = p
    return {"per_dataset": out, "holm": holm(pvals)}


def load_records(paths: list[str]) -> list[dict]:
    return [read_json(p) for p in paths]
