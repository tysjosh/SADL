#!/usr/bin/env bash
# Bring a fresh GPU machine up to the state the sweep left off at.
#
#   HF_REPO=<user>/SADL-results bash scripts/restore_results.sh
#
# Assumes the repo is already cloned and you are inside it. Fetches the datasets
# (gitignored, ~80 MB) and the run records (gitignored, on the Hub), then checks
# that this checkout can actually reproduce what it downloaded before you resume.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
: "${HF_REPO:?set HF_REPO=<user>/SADL-results}"

echo "=== checkout"
git rev-parse HEAD
echo "  If the records were produced at a different commit, verify_resume.py below"
echo "  will say so per group -- it does not require an exact match, only that the"
echo "  train and metric code agree."

echo
echo "=== environment"
if [ ! -x "$PY" ]; then
  echo "creating .venv"
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/pip install -q -U huggingface_hub
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "WARNING: no CUDA. requirements.txt pins the cu121 build of torch 2.4.1; for a"
  echo "different CUDA install torch first from the matching index, then the rest."
}

echo
echo "=== datasets"
$PY scripts/download_data.py

echo
echo "=== run records from $HF_REPO"
# --local-dir is essential: without it files land in the hashed HF cache and the
# runner sees no cached cells at all, silently restarting the sweep from zero.
hf download "$HF_REPO" --repo-type=dataset --local-dir ./results || {
  echo "DOWNLOAD FAILED -- resuming now would retrain everything"; exit 1; }
echo "  $(find results/runs -name '*.json' 2>/dev/null | wc -l | tr -d ' ') json, $(find results/runs -name '*.npz' 2>/dev/null | wc -l | tr -d ' ') npz restored"

echo
echo "=== comparability"
$PY scripts/verify_resume.py
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "READY. Resume with:"
  echo "  RQS=\"rq1 rq2\" NPE=8000 nohup bash scripts/run_all.sh > results/logs/sweep.log 2>&1 &"
else
  echo "Read the warnings above first. Incomparable cells have a generated re-run script;"
  echo "cells that are merely 'compatible' need nothing."
fi
exit "$status"
