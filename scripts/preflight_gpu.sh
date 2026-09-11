#!/usr/bin/env bash
# Stage 1 on the GPU box: does SADL accept anything at the reporting budget?
#
#   bash scripts/preflight_gpu.sh
#   SEED=1 NPE=8000 bash scripts/preflight_gpu.sh
#
# Runs the *first* Separator game only, at the same per-game step count, batch
# size and audit density the `gpu` preset uses for the sweep.  That is the whole
# question: SADL.fit breaks out of the outer loop when a step exhausts its
# restarts, so if t = 0 never accepts, the sequence is empty and every SADL cell
# in the sweep reports a column of zeros at chance accuracy.
#
# Cost: T_max = 1 instead of 8, so roughly one eighth of a single SADL cell per
# dataset, against a full sweep of several hundred cells.  Cells are tagged
# `gate`, so they never collide with a reporting cell and are not read by the
# tables.
#
# `smoke` is deliberately not used here.  It runs 120 steps at T=2, R=1 and
# accepts nothing even on ColoredMNIST, so it cannot distinguish "the method does
# not clear the gate" from "the budget was too small to try".
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEED=${SEED:-0}
NPE=${NPE:-8000}                     # matches run_all.sh
DATASETS=${DATASETS:-"dSprites ColoredMNIST Causal3DIdent-lite"}
METHOD=${METHOD:-SADL}

# gpu preset: steps=8000 over sadl_tmax=8 -> 1000 steps per game.  Pinning
# steps=1000 with sadl_tmax=1 reproduces one game of the sweep exactly, rather
# than a shortened version of it.
STEPS_PER_GAME=${STEPS_PER_GAME:-1000}

echo "=== environment"
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.device_count(), 'device(s)', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "WARNING: CUDA is not available. The gate will run on the fallback device and"
  echo "the timings will not represent the sweep. Set SADL_DEVICE=cuda once a GPU is visible."
fi
$PY scripts/download_data.py

echo
echo "=== acceptance gate: first game only, gpu per-game budget, seed $SEED"
for d in $DATASETS; do
  echo
  echo "--- $d"
  $PY -W ignore -u -m sadl.experiments.run \
    --dataset "$d" --method "$METHOD" --seed "$SEED" --budget gpu \
    --set steps="$STEPS_PER_GAME" --set sadl_tmax=1 \
    --n-per-env "$NPE" --tag gate || echo "  cell failed; see the record"
done

echo
echo "=== verdict"
$PY scripts/gate_report.py --tag gate
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "Gate passed somewhere. Launch the sweep with:  NGPU=<gpus> bash scripts/run_all.sh"
else
  echo "Gate did not pass. Read the binding criterion above before spending sweep hours."
  echo "Select any change on training-domain validation only -- the held-out environments"
  echo "stay untouched until final evaluation."
fi
exit "$status"
