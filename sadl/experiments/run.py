"""Single-run driver: (method, dataset, seed) -> one result record.

Every record is cached as JSON under ``results/runs`` and the per-instance
correctness vectors are cached alongside as ``.npz`` so that the hierarchical
bootstrap of Section 6.5 can resample instances without retraining.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch

from ..data import DataUnavailable, get_dataset
from ..eval import composability, count_encoder_evals, evaluate_ood, factor_recovery, stability_gap
from ..methods import BUDGET_PRESETS, Budget, build_method, method_meta
from ..provenance import compare as compare_provenance
from ..provenance import source_fingerprints
from ..utils import (
    RESULTS_DIR,
    Timer,
    get_device,
    peak_memory_mb,
    read_json,
    set_seed,
    source_fingerprint,
    write_json,
)

_FINGERPRINT = source_fingerprint()
_FINGERPRINTS = source_fingerprints()
_WARNED_STALE: set[str] = set()


def run_key(dataset: str, method: str, seed: int, budget: str, tag: str = "") -> str:
    base = f"{dataset}__{method}__seed{seed}__{budget}"
    return f"{base}__{tag}" if tag else base


def run_one(
    dataset: str,
    method: str,
    seed: int,
    budget: str = "standard",
    dataset_kwargs: dict | None = None,
    budget_overrides: dict | None = None,
    out_dir: Path | None = None,
    tag: str = "",
    overwrite: bool = False,
    device: torch.device | None = None,
    cached_only: bool = False,
) -> dict:
    out_dir = Path(out_dir or RESULTS_DIR / "runs")
    key = run_key(dataset, method, seed, budget, tag)
    jpath = out_dir / f"{key}.json"
    if cached_only and not jpath.exists():
        return {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "budget": budget,
            "tag": tag,
            "status": "not_run",
        }
    if jpath.exists() and not overwrite:
        cached = read_json(jpath)
        # Only successful cells are cached.  A failure is usually an environment
        # problem -- missing data, an unavailable device -- and caching it means a
        # retry after the fix replays the same failure instead of doing the work.
        if cached.get("status") != "ok":
            cached = None
    else:
        cached = None
    if cached is not None:
        prov = compare_provenance(cached.get("fingerprints"), _FINGERPRINTS)
        # Only warn when the difference could change the record's numbers.  A
        # blanket warning on every source edit is why a real invalidation was
        # dismissed as noise once already; see sadl/provenance.py.
        if prov["material"] and prov["status"] not in _WARNED_STALE:
            _WARNED_STALE.add(prov["status"])
            print(
                f"[warn] reusing cached records that this code cannot reproduce: {prov['reason']}. "
                f"Re-run the affected cells with --overwrite, or accept that the table mixes "
                f"two versions of the method.",
                flush=True,
            )
        cached["stale_cache"] = prov["status"] != "ok"
        cached["provenance"] = prov
        return cached

    device = device or get_device()
    dataset_kwargs = dict(dataset_kwargs or {})
    dataset_kwargs.setdefault("seed", seed)
    bud = BUDGET_PRESETS[budget]
    if budget_overrides:
        bud = bud.with_(**budget_overrides)

    record: dict = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "budget": budget,
        "tag": tag,
        "device": str(device),
        "fingerprint": _FINGERPRINT,
        "fingerprints": _FINGERPRINTS,
        "dataset_kwargs": {k: v for k, v in dataset_kwargs.items() if k != "env_specs"},
        "budget_overrides": budget_overrides or {},
        **method_meta(method),
    }

    try:
        set_seed(seed)
        ds = get_dataset(dataset, **dataset_kwargs)
        record["dataset_notes"] = ds.notes
        record["n_train_envs"] = ds.n_train_envs
        record["train_envs"] = [e.env_name for e in ds.train_envs]
        record["test_envs"] = [e.env_name for e in ds.test_envs]

        m = build_method(method, bud, device, seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # Table 10 needs forward/backward evaluations as well as time and memory,
        # and only the training phase counts: the counter closes before the
        # readout and the metrics are computed, so it measures the objective
        # rather than the shared evaluation protocol.
        with count_encoder_evals() as costs, Timer() as t:
            log = m.fit(ds)
        record["train_seconds"] = t.elapsed
        record["peak_mem_mb"] = peak_memory_mb(device)
        record.update(costs.as_dict())
        record["n_params"] = m.n_params()
        record["repr_dim"] = m.repr_dim
        record["n_accepted"] = log.get("n_accepted")
        record["sadl_trace"] = log.get("trace")

        ood = evaluate_ood(m, ds)
        correctness = ood.pop("correctness")
        record.update(ood)

        rep_train = np.concatenate([m.features(e.fit().x) for e in ds.train_envs])
        z_test = np.concatenate([e.z_inv for e in ds.test_envs]) if ds.fr_factor_names else None
        if z_test is not None:
            rep_test = np.concatenate([m.features(e.x) for e in ds.test_envs])
            fr = factor_recovery(rep_train, rep_test, z_test, ds.fr_factor_names, ds.inv_factor_names, ds.fr_bins)
            record["fr"] = fr["fr"]
            record["fr_matched"] = fr["matched"]
        else:
            record["fr"] = float("nan")

        record.update(stability_gap(m, ds, seed=seed))
        record.update(composability(m, ds))
        record["status"] = "ok"

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{key}.npz", **{k: v for k, v in correctness.items()})
    except DataUnavailable as exc:
        record["status"] = "data_unavailable"
        record["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - recorded, not raised
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    write_json(jpath, record)
    return record


def load_correctness(record: dict, out_dir: Path | None = None) -> dict[str, np.ndarray]:
    out_dir = Path(out_dir or RESULTS_DIR / "runs")
    key = run_key(record["dataset"], record["method"], record["seed"], record["budget"], record.get("tag", ""))
    path = out_dir / f"{key}.npz"
    if not path.exists():
        return {}
    with np.load(path) as blob:
        return {k: blob[k] for k in blob.files}


def shard_of(index: int, shard: tuple[int, int] | None) -> bool:
    """Whether cell ``index`` belongs to this shard, for multi-GPU splitting."""
    if shard is None:
        return True
    which, total = shard
    return index % total == which


def run_grid(
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    budget: str = "standard",
    dataset_kwargs: dict | None = None,
    budget_overrides: dict | None = None,
    tag: str = "",
    overwrite: bool = False,
    verbose: bool = True,
    shard: tuple[int, int] | None = None,
    cached_only: bool = False,
) -> list[dict]:
    """Run a (dataset x method x seed) grid.

    Cells are independent and cached, so several processes can share the grid by
    passing disjoint ``shard=(i, n)`` values -- one per GPU.  Every process still
    reads the whole grid back when it aggregates, so a shard that has not finished
    shows up as ``not run`` rather than corrupting a table.
    """
    device = get_device()
    out: list[dict] = []
    total = len(datasets) * len(methods) * len(seeds)
    i = 0
    for dname in datasets:
        for mname in methods:
            for seed in seeds:
                i += 1
                if not cached_only and not shard_of(i - 1, shard):
                    continue
                rec = run_one(
                    dname,
                    mname,
                    seed,
                    budget=budget,
                    dataset_kwargs=dataset_kwargs,
                    budget_overrides=budget_overrides,
                    tag=tag,
                    overwrite=overwrite,
                    device=device,
                    cached_only=cached_only,
                )
                out.append(rec)
                if verbose and not cached_only:
                    if rec["status"] == "ok":
                        print(
                            f"[{i}/{total}] {dname:20s} {mname:16s} s{seed} "
                            f"worst={rec['acc_worst']:.3f} avg={rec['acc_avg']:.3f} "
                            f"id={rec['acc_id_val']:.3f} fr={rec.get('fr', float('nan')):.3f} "
                            f"gap={rec.get('stability_gap', float('nan')):.3f} "
                            f"({rec['train_seconds']:.0f}s)",
                            flush=True,
                        )
                    else:
                        print(f"[{i}/{total}] {dname:20s} {mname:16s} s{seed} -> {rec['status']}: {rec.get('error','')[:120]}", flush=True)
    return out


def _coerce(name: str, value: str):
    """Cast a ``--set key=value`` override to the type declared on Budget."""
    from dataclasses import fields

    types = {f.name: f.type for f in fields(Budget)}
    if name not in types:
        raise SystemExit(f"unknown budget field {name!r}; see sadl/methods/base.py:Budget")
    declared = str(types[name])
    try:
        if "bool" in declared:
            return value.lower() in ("1", "true", "yes", "on")
        if "int" in declared:
            return int(value)
        if "float" in declared:
            return float(value)
    except ValueError:
        raise SystemExit(f"{name} expects {declared}, got {value!r}") from None
    return value


def summarize(rec: dict) -> str:
    if rec.get("status") != "ok":
        return f"status={rec.get('status')} {rec.get('error', '')}"
    g = lambda k: rec.get(k)
    lines = [
        f"status      {g('status')}   device={g('device')}   {g('train_seconds'):.0f}s   repr_dim={g('repr_dim')}",
        f"worst acc   {g('acc_worst'):.3f}   (chance {g('acc_chance_worst'):.3f})",
        f"avg acc     {g('acc_avg'):.3f}   (chance {g('acc_chance_avg'):.3f})",
        f"id val acc  {g('acc_id_val'):.3f}",
        f"per env     " + ", ".join(f"{k}={v:.3f}" for k, v in (g("acc_per_env") or {}).items()),
        f"factor rec. {g('fr')}",
        f"stab. gap   {g('stability_gap')}",
        f"composab.   {g('composability')}",
    ]
    if g("n_accepted") is not None:
        lines.append(f"accepted T  {g('n_accepted')}")
        for tr in g("sadl_trace") or []:
            if "stop_reason" in tr:
                lines.append(f"  t={tr.get('t')} stop: {tr['stop_reason']}")
                continue
            lines.append(
                "  t={t} r={r} accepted={a} | N={n:.3f} S={s:.3f} A={adv:.3f} "
                "(bank {bank:.3f} / learned {learn:.3f} / worst-case {wc:.3f}) bal={bal:.3f} best_step={bs}".format(
                    t=tr.get("t"),
                    r=tr.get("restart"),
                    a=int(bool(tr.get("accepted"))),
                    n=tr.get("val_novelty", float("nan")),
                    s=tr.get("val_shift", float("nan")),
                    adv=tr.get("val_adv_flip", float("nan")),
                    bank=tr.get("val_adv_flip_bank", float("nan")),
                    learn=tr.get("val_adv_flip_learned", float("nan")),
                    wc=tr.get("val_adv_flip_worstcase", float("nan")),
                    bal=tr.get("val_balance", float("nan")),
                    bs=tr.get("best_step"),
                )
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="run one (dataset, method, seed) cell",
        epilog=(
            "examples:\n"
            "  --list\n"
            "  --dataset dSprites --method SADL --seed 0 --budget smoke\n"
            "  --dataset ColoredMNIST --method SADL --budget gpu --set sadl_tmax=6 --set sadl_eps_adv=0.3 --tag epsweep\n"
            "  --dataset ColoredMNIST --method SADL --n-train-envs 1 --tag E1     # the |E|=1 cell of RQ4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="standard", choices=list(BUDGET_PRESETS))
    ap.add_argument("--n-per-env", type=int, default=None)
    ap.add_argument("--n-train-envs", type=int, default=None, help="restrict/extend |E| (RQ4)")
    ap.add_argument("--tag", default="", help="suffix for the cache key; use for one-off variants")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE", help="override any Budget field")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="list datasets, methods and budgets, then exit")
    args = ap.parse_args()

    if args.list:
        from ..data import ALL_DATASETS, REAL
        from ..methods import ALL_METHODS, TABLE2_METHODS

        print("datasets (Table 2 comparison):", ", ".join(d for d in ALL_DATASETS if d not in REAL))
        print("datasets (local data required):", ", ".join(REAL))
        print("methods  (Table 2):", ", ".join(TABLE2_METHODS))
        print("methods  (ablations):", ", ".join(m for m in ALL_METHODS if m not in TABLE2_METHODS))
        print("budgets:", ", ".join(BUDGET_PRESETS))
        return 0

    if not args.dataset or not args.method:
        ap.error("--dataset and --method are required (or pass --list)")

    from ..data import ALL_DATASETS
    from ..methods import ALL_METHODS

    if args.dataset not in ALL_DATASETS:
        ap.error(f"unknown dataset {args.dataset!r}; see --list")
    if args.method not in ALL_METHODS:
        ap.error(f"unknown method {args.method!r}; see --list")

    dk: dict = {}
    if args.n_per_env:
        dk["n_per_env"] = args.n_per_env
    if args.n_train_envs:
        from ..data import env_specs_for

        dk["n_train_envs"] = args.n_train_envs
        dk["env_specs"] = env_specs_for(args.dataset, args.n_train_envs)

    overrides = {}
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects FIELD=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = _coerce(k, v)

    rec = run_one(
        args.dataset,
        args.method,
        args.seed,
        args.budget,
        dataset_kwargs=dk or None,
        budget_overrides=overrides or None,
        tag=args.tag,
        overwrite=args.overwrite,
    )
    print(summarize(rec))
    print(f"\nrecord: results/runs/{run_key(args.dataset, args.method, args.seed, args.budget, args.tag)}.json")
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
