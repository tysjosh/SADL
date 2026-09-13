"""Which source changes can invalidate a cached record, and which cannot.

``source_fingerprint()`` hashes every ``.py`` in the package, so any edit re-bases
it and every cached record is reported as stale.  That is useless in both
directions: a change to table rendering fires the same warning as a change to the
optimisation budget, so the warning gets ignored, and then a change that really
does invalidate 77 records slips through as noise.  That happened -- ``81fcef8``
altered the ``gpu`` preset (``steps`` 6000 to 8000, ``sadl_rmax`` 3 to 6) and the
cached cells kept being reused, with ``accepted T`` differing by a factor of four
between the two versions of the same cell.

Files are therefore grouped by what a change to them can affect:

``train``   the fitted model itself -- methods, models, datasets, seeding.
``metric``  the numbers computed from a fitted model and written to the record.
``driver``  orchestration: which cells run, what is recorded, how it is cached.
``stats``   post-hoc analysis only: bootstraps, corrections, table rendering.

A cached record is *incomparable* with a fresh one when ``train`` or ``metric``
differs, because its numbers would not reproduce.  A ``driver`` difference is
worth mentioning and rarely matters.  A ``stats`` difference cannot affect a
record at all, since every statistic there is recomputed from the records on each
aggregation.

This module is excluded from every group: changing the provenance machinery must
not invalidate the records it describes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PKG = Path(__file__).resolve().parent

# Longest prefix wins, so a specific file overrides its directory.
_GROUPS: dict[str, tuple[str, ...]] = {
    "train": ("methods/", "models/", "data/", "utils.py"),
    "metric": ("eval/metrics.py", "eval/cost.py"),
    "stats": ("eval/stats.py", "eval/misspec.py", "experiments/tables.py", "experiments/rqs.py"),
    "driver": ("experiments/",),
}
# A change to these cannot make a cached record incomparable with a fresh one.
MATERIAL_GROUPS = ("train", "metric")
_EXCLUDED = ("provenance.py",)


def _group_of(rel: str) -> str | None:
    if rel in _EXCLUDED:
        return None
    best, best_len = None, -1
    for group, prefixes in _GROUPS.items():
        for p in prefixes:
            if rel == p or rel.startswith(p):
                if len(p) > best_len:
                    best, best_len = group, len(p)
    # An unclassified file is treated as train: assuming the worst is the safe
    # default, so a new module cannot silently escape invalidation.
    return best or "train"


def source_fingerprints(pkg: Path | None = None) -> dict[str, str]:
    """Per-group short hashes of the package source, plus a combined ``all``."""
    pkg = pkg or PKG
    buckets: dict[str, hashlib._Hash] = {g: hashlib.sha256() for g in _GROUPS}
    combined = hashlib.sha256()
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        group = _group_of(rel)
        if group is None:
            continue
        blob = path.read_bytes()
        buckets[group].update(rel.encode())
        buckets[group].update(blob)
        combined.update(rel.encode())
        combined.update(blob)
    out = {g: h.hexdigest()[:12] for g, h in buckets.items()}
    out["all"] = combined.hexdigest()[:12]
    return out


def compare(recorded: dict | None, current: dict | None = None) -> dict:
    """Classify a record's provenance against the running code.

    ``recorded`` is the record's ``fingerprints`` dict.  Records written before
    this module existed have none; those are ``unknown`` rather than ``ok``,
    because nothing in them says which groups differ.
    """
    current = current or source_fingerprints()
    if not recorded:
        return {
            "status": "unknown",
            "material": True,
            "differs": [],
            "reason": "record predates per-group fingerprints; cannot tell what changed",
        }
    differs = [g for g in _GROUPS if recorded.get(g) != current.get(g)]
    material = [g for g in differs if g in MATERIAL_GROUPS]
    if not differs:
        return {"status": "ok", "material": False, "differs": [], "reason": "identical source"}
    if material:
        return {
            "status": "incomparable",
            "material": True,
            "differs": differs,
            "reason": "changed: " + ", ".join(differs) + " -- these can change a record's numbers",
        }
    return {
        "status": "compatible",
        "material": False,
        "differs": differs,
        "reason": "changed only: " + ", ".join(differs) + " -- cannot change a record's numbers",
    }
