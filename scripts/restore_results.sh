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
# A stale checkout cost 1.3 GPU-hours once: 60 cells were re-run to record a metric
# whose implementation had not been pulled, so they reproduced values that already
# existed and produced nothing. Provenance cannot catch this -- the records are
# internally consistent, they just lack fields the newer code would have written.
if git fetch -q origin 2>/dev/null; then
  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if [ "${behind:-0}" -gt 0 ]; then
    echo
    echo "  !! this checkout is $behind commit(s) BEHIND origin/main:"
    git log --oneline HEAD..origin/main | sed 's/^/     /'
    echo "     Run 'git pull' before training anything. A cell run against old code"
    echo "     silently omits any metric added since, and looks successful doing it."
    echo
  else
    echo "  up to date with origin/main"
  fi
else
  echo "  (could not reach origin; cannot confirm the checkout is current)"
fi
echo "  verify_resume.py below reports per-group comparability of the downloaded"
echo "  records -- it does not require an exact commit match, only that the train"
echo "  and metric code agree."

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
