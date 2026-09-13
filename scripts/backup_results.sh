#!/usr/bin/env bash
# Push results to the Hub and prove the round trip kept everything.
#
#   HF_REPO=<user>/SADL-results bash scripts/backup_results.sh
#   HF_REPO=... MSG="rq1+rq2 complete" bash scripts/backup_results.sh
#
# Run this before a leased machine expires.  `results/` is gitignored, so the Hub
# copy is the only copy, and an upload that silently drops files is worse than no
# upload at all: it looks like a backup.
#
# That is not hypothetical.  A previous round trip lost 69 `.npz` files of
# per-instance correctness vectors -- the ones the hierarchical bootstrap of
# Section 6.5 resamples -- and it was only noticed when aggregation crashed much
# later.  So this script counts before, uploads, downloads to a scratch directory,
# counts again, and fails loudly if the two disagree.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
: "${HF_REPO:?set HF_REPO=<user>/SADL-results}"
MSG=${MSG:-"results snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)"}
SCRATCH=${SCRATCH:-/tmp/sadl_backup_verify}

count() { find "$1" -name "*.$2" 2>/dev/null | wc -l | tr -d ' '; }

J_LOCAL=$(count results/runs json)
N_LOCAL=$(count results/runs npz)
T_LOCAL=$(find results -maxdepth 1 -name "table*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "local:   $J_LOCAL json, $N_LOCAL npz, $T_LOCAL rendered table(s)"
if [ "$J_LOCAL" -eq 0 ]; then
  echo "nothing to back up"; exit 1
fi
echo "commit:  $(git rev-parse HEAD)"
echo "         record this in the dataset card -- results/ is gitignored, so it is the"
echo "         only link between these records and the code that produced them."

echo
echo "=== uploading to $HF_REPO"
$PY -m huggingface_hub.commands.huggingface_cli version >/dev/null 2>&1 || true
hf upload "$HF_REPO" ./results . --repo-type=dataset \
  --exclude="archive_local_mps/*" --commit-message="$MSG" || {
    echo "UPLOAD FAILED -- do not release the machine"; exit 1; }

echo
echo "=== verifying the round trip into $SCRATCH"
rm -rf "$SCRATCH"
hf download "$HF_REPO" --repo-type=dataset --local-dir "$SCRATCH" >/dev/null || {
    echo "DOWNLOAD FAILED -- treat the backup as unverified"; exit 1; }

J_REMOTE=$(count "$SCRATCH/runs" json)
N_REMOTE=$(count "$SCRATCH/runs" npz)
T_REMOTE=$(find "$SCRATCH" -maxdepth 1 -name "table*.md" 2>/dev/null | wc -l | tr -d ' ')
echo "remote:  $J_REMOTE json, $N_REMOTE npz, $T_REMOTE rendered table(s)"

fail=0
[ "$J_REMOTE" -lt "$J_LOCAL" ] && { echo "MISSING $((J_LOCAL - J_REMOTE)) json file(s) on the Hub"; fail=1; }
[ "$N_REMOTE" -lt "$N_LOCAL" ] && { echo "MISSING $((N_LOCAL - N_REMOTE)) npz file(s) on the Hub"; fail=1; }
[ "$T_REMOTE" -lt "$T_LOCAL" ] && { echo "MISSING $((T_LOCAL - T_REMOTE)) table(s) on the Hub"; fail=1; }

echo
if [ "$fail" -eq 0 ]; then
  echo "backup verified: every local file round-tripped. Safe to release the machine."
  echo "restore on a new box with:  bash scripts/restore_results.sh"
  rm -rf "$SCRATCH"
else
  echo "BACKUP INCOMPLETE -- $SCRATCH kept for inspection. Do not release the machine."
fi
exit "$fail"
