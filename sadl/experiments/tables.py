"""Aggregation and markdown rendering for the tables of Section 6.6."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..eval.cost import memory_is_peak
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


# ------------------------------------------------------- cost (Table 10)
def _by_cell(records: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {
        (r["dataset"], r["method"], r["seed"]): r for r in records if r.get("status") == "ok"
    }


def paired_ratios(
    records: list[dict],
    datasets: list[str],
    method: str,
    metric: str,
    reference: str = "ERM",
    peak_memory_only: bool = False,
) -> list[float]:
    """Ratios of ``metric`` to the reference method, paired within (dataset, seed).

    Pairing matters: the two synthetic benchmarks differ in resolution and sample
    count, so a ratio of pooled means would be dominated by whichever dataset is
    slower.  A ratio formed inside a (dataset, seed) cell compares runs that used
    the same hardware, the same input budget and the same data, which is the
    "identical hardware and matched input budgets" Table 10 asks for.
    """
    cells = _by_cell(records)
    out: list[float] = []
    for d in datasets:
        for (dd, mm, seed), rec in sorted(cells.items()):
            if dd != d or mm != method:
                continue
            ref = cells.get((d, reference, seed))
            if ref is None:
                continue
            if peak_memory_only and not (memory_is_peak(rec.get("device")) and memory_is_peak(ref.get("device"))):
                continue
            num, den = rec.get(metric), ref.get(metric)
            if num is None or den is None:
                continue
            num, den = float(num), float(den)
            if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
                continue
            out.append(num / den)
    return out


def _ratio_cell(ratios: list[float], nd: int = 2, suffix: str = "") -> str:
    if not ratios:
        return "n/a"
    m, s = mean_std(ratios)
    if np.isnan(m):
        return "n/a"
    if len(ratios) < 2 or np.isnan(s):
        return f"{m:.{nd}f}{suffix}"
    return f"{m:.{nd}f} ± {s:.{nd}f}{suffix}"


def cost_table(records: list[dict], datasets: list[str], methods: list[str], reference: str = "ERM") -> str:
    """Table 10: time, peak memory and trunk evaluations, all relative to ERM."""
    header = ["Method", "Time/ERM", "Memory/ERM", "Evals/step"]
    rows = []
    for m in methods:
        rows.append(
            [
                m,
                _ratio_cell(paired_ratios(records, datasets, m, "train_seconds", reference)),
                _ratio_cell(
                    paired_ratios(records, datasets, m, "peak_mem_mb", reference, peak_memory_only=True)
                ),
                _ratio_cell(paired_ratios(records, datasets, m, "evals_per_step", reference)),
            ]
        )
    return markdown_table(header, rows)


def absolute_cost_table(records: list[dict], datasets: list[str], methods: list[str]) -> str:
    """The same runs in absolute units, so the ratios above can be checked."""
    g = group(records)
    header = ["Method", "Train (s)", "Peak mem (MB)", "Trunk fwd", "Trunk bwd", "Steps", "Evals/step"]
    rows = []
    for m in methods:
        recs = [r for d in datasets for r in g.get((d, m), [])]
        ok = [r for r in recs if r.get("status") == "ok"]
        mem = [r for r in ok if memory_is_peak(r.get("device"))]
        rows.append(
            [
                m,
                cell(recs, "train_seconds", pct=False, nd=1),
                cell(mem, "peak_mem_mb", pct=False, nd=0) if mem else "n/a",
                cell(recs, "n_encoder_forward", pct=False, nd=0),
                cell(recs, "n_encoder_backward", pct=False, nd=0),
                cell(recs, "n_train_steps", pct=False, nd=0),
                cell(recs, "evals_per_step", pct=False, nd=2),
            ]
        )
    return markdown_table(header, rows)


def accepted_per_hour_table(records: list[dict], datasets: list[str], methods: list[str]) -> str:
    """Accepted distinctions per accelerator-hour (Section 5.7)."""
    g = group(records)
    header = ["Variant", "Accepted T", "Train (s)", "Accepted / accelerator-hour"]
    rows = []
    for m in methods:
        recs = [r for d in datasets for r in g.get((d, m), [])]
        ok = [r for r in recs if r.get("status") == "ok" and r.get("n_accepted") is not None]
        rates = [
            r["n_accepted"] / (r["train_seconds"] / 3600.0)
            for r in ok
            if r.get("train_seconds")
        ]
        rows.append(
            [
                m,
                cell(recs, "n_accepted", pct=False, nd=2),
                cell(recs, "train_seconds", pct=False, nd=1),
                _ratio_cell(rates, nd=1) if rates else "n/a",
            ]
        )
    return markdown_table(header, rows)
