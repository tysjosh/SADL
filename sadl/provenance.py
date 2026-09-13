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


def legacy_fingerprints(depth: int = 15, repo: Path | None = None) -> dict[str, dict[str, str]]:
    """Group hashes for commits, indexed by the flat ``source_fingerprint()`` hash.

    Records written before this module existed carry only the flat hash, which says
    nothing about *which* files changed.  Replaying recent commits recovers the
    group hashes, so a legacy record can be classified exactly as a fresh one.
    Returns an empty map when git is unavailable, in which case legacy records stay
    ``unknown``.
    """
    import hashlib
    import subprocess
    import tempfile

    repo = repo or PKG.parent
    try:
        shas = subprocess.run(
            ["git", "log", "--format=%H", "-n", str(depth)],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {}

    out: dict[str, dict[str, str]] = {}
    for sha in shas:
        try:
            tar = subprocess.run(
                ["git", "archive", sha, PKG.name], cwd=repo, capture_output=True, check=True
            ).stdout
        except subprocess.CalledProcessError:
            continue
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["tar", "-x", "-C", td], input=tar, check=True)
            pkg = Path(td) / PKG.name
            if not pkg.is_dir():
                continue
            flat = hashlib.sha256()
            for p in sorted(pkg.rglob("*.py")):
                flat.update(p.name.encode())
                flat.update(p.read_bytes())
            out.setdefault(flat.hexdigest()[:12], source_fingerprints(pkg))
    return out


def classify_record(
    record: dict,
    current: dict | None = None,
    legacy: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Classify a whole record, preferring its group hashes and falling back to the
    flat hash via ``legacy``.

    Use this rather than calling ``compare`` directly on ``record["fingerprints"]``:
    every record written before this module has no such key, so a bare ``compare``
    reports all of them ``unknown`` and therefore material, which is both wrong and
    alarming.
    """
    groups = record.get("fingerprints")
    if not groups and legacy:
        groups = legacy.get(record.get("fingerprint", ""))
    return compare(groups, current)


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
        other = [g for g in differs if g not in MATERIAL_GROUPS]
        return {
            "status": "incomparable",
            "material": True,
            "differs": differs,
            "material_differs": material,
            "reason": (
                ", ".join(material)
                + " changed, which can alter a record's numbers"
                + (f" (also {', '.join(other)}, which cannot)" if other else "")
            ),
        }
    return {
        "status": "compatible",
        "material": False,
        "differs": differs,
        "reason": "changed only: " + ", ".join(differs) + " -- cannot change a record's numbers",
    }
