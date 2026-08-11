#!/usr/bin/env bash
# Stage 0: prove every method variant runs before committing GPU hours to a sweep.
#
#   bash scripts/smoke_all.sh
#
# One cell per method on ColoredMNIST at the `smoke` budget, plus the two 64x64
# datasets for SADL, which is where the acceptance gate has failed before.
# Records are tagged `smoke`, so they never collide with a reporting sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

ALL="ERM SimCLR MAE IRMv1 VREx beta-VAE FactorVAE DANN SADL-lite SADL \
SADL-no-gate SADL-warmstart SADL-joint SADL-T1 SADL-T8 \
SADL-conf-none SADL-conf-fixed3 SADL-conf-fixed10 SADL-conf-learned"

fail=0
for m in $ALL; do
  printf '%-22s ' "$m"
  if out=$($PY -W ignore -m sadl.experiments.run --dataset ColoredMNIST --method "$m" \
            --budget smoke --n-per-env 1200 --tag smoke 2>&1); then
    echo "$out" | awk '/^worst acc/{w=$3} /^accepted T/{t=$3} END{printf "ok   worst=%s acceptedT=%s\n", w, (t==""?"-":t)}'
  else
    echo "FAILED"; echo "$out" | tail -3 | sed 's/^/    /'; fail=$((fail + 1))
  fi
done

for d in dSprites Causal3DIdent-lite; do
  printf '%-22s ' "SADL on $d"
  if out=$($PY -W ignore -m sadl.experiments.run --dataset "$d" --method SADL \
            --budget smoke --n-per-env 1200 --tag smoke 2>&1); then
    echo "$out" | awk '/^accepted T/{printf "ok   acceptedT=%s\n", $3}'
  else
    echo "FAILED"; echo "$out" | tail -3 | sed 's/^/    /'; fail=$((fail + 1))
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "all variants ran; safe to start the sweep"
else
  echo "$fail variant(s) failed -- fix before sweeping"
fi
exit "$fail"
