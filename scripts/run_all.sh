#!/usr/bin/env bash
# Full experiment sweep, intended for a GPU machine.
#
#   bash scripts/run_all.sh                  # single GPU, budget=gpu, 5 seeds
#   BUDGET=standard SEEDS="0 1 2" bash scripts/run_all.sh
#   NGPU=4 bash scripts/run_all.sh           # shard the grid across 4 GPUs
#
# Cells are cached individually, so this is safe to interrupt and re-run: it
# fills gaps rather than repeating work.  With NGPU > 1 each GPU takes a disjoint
# shard and the tables are aggregated at the end from all of them.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
BUDGET=${BUDGET:-gpu}
SEEDS=${SEEDS:-0 1 2 3 4}
NPE=${NPE:-8000}
NTOTAL=${NTOTAL:-16000}
NGPU=${NGPU:-1}
RQS=${RQS:-"rq1 rq2 rq3 rq4 rq6"}

mkdir -p results/logs
echo "budget=$BUDGET seeds=$SEEDS n_per_env=$NPE gpus=$NGPU"

$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'devices')"
$PY scripts/download_data.py

for rq in $RQS; do
  echo "=== $rq ==="
  if [ "$NGPU" -le 1 ]; then
    $PY -W ignore -u -m sadl.experiments.rqs "$rq" \
      --budget "$BUDGET" --seeds $SEEDS --n-per-env "$NPE" --n-total "$NTOTAL" \
      2>&1 | tee "results/logs/${rq}_${BUDGET}.log"
  else
    pids=()
    for ((g = 0; g < NGPU; g++)); do
      CUDA_VISIBLE_DEVICES=$g SADL_DEVICE=cuda \
      $PY -W ignore -u -m sadl.experiments.rqs "$rq" \
        --budget "$BUDGET" --seeds $SEEDS --n-per-env "$NPE" --n-total "$NTOTAL" \
        --shard "$g/$NGPU" > "results/logs/${rq}_${BUDGET}_gpu${g}.log" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  fi
done

# Re-render every table from all cached records, including cells other shards ran.
$PY -W ignore -u -m sadl.experiments.rqs tables \
  --budget "$BUDGET" --seeds $SEEDS --n-per-env "$NPE" --n-total "$NTOTAL" \
  2>&1 | tee "results/logs/tables_${BUDGET}.log"

echo "done; tables in results/*.md"
