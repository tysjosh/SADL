#!/usr/bin/env bash
# Stage 1 of the v2 method paper: does each correction move the metric it targets?
#
#   bash scripts/v2_viability.sh
#   SEEDS="0 1 2 3 4" BUDGET=gpu bash scripts/v2_viability.sh
#
# The v1 sweep produced three failures with three distinct causes, and each binds
# on a different dataset.  So each fix is tested where its cause binds, rather than
# paying for a full grid before knowing whether any of them work.
#
#   fix1  marginal-preserving Confuser   ColoredMNIST   accepted balance, novelty
#         The learned adversary collapses to a constant saturating attack, making
#         the hard flip rate identically min(p, 1-p) -- 21 of 22 restarts -- so the
#         gate caps novelty at Hb(eps_adv)=0.811 and forbids balanced distinctions.
#         Binds on ColoredMNIST, where the gate read learned 0.459 / bank 0.146.
#
#   fix2  label-free sufficiency term    ColoredMNIST   composability, factor rec.
#         Equation 18 has no sufficiency term; Theorem 4.1 requires one.  v1 gives a
#         generalisation gap of -0.005 while capturing 19% of the available signal,
#         with composability 0.228 against SimCLR's 0.898.
#
#   fix3  coverage-complete audit        dSprites       stability gap
#         env_resample -- the operation Proposition 5.1's condition (i) is stated in
#         terms of -- existed only at the `learned` level, so the parameter-free
#         auditor never applied it.  Binds where the bank is the stronger adversary:
#         dSprites (gap 0.271) and Causal3DIdent (0.247), bank in ~9 of 10 rows.
#
#   SimCLR-bank                          both           the bar v2 must clear
#         Same transformation family, none of the algorithm.  If it matches
#         SADL-lite there is no method claim, however well the fixes work.
#
# Reference values to beat, from the verified v1 sweep at budget=gpu, 5 seeds:
#   ColoredMNIST  SADL-lite  gap 0.007  worst 54.2  comp 0.228  fr 0.085
#   ColoredMNIST  SADL       gap 0.039  worst 46.2  comp 0.163  fr 0.109
#   dSprites      SADL-lite  gap 0.271  worst 33.2  comp 0.004  fr 0.006
#   dSprites      SADL       gap 0.291  worst 30.8  comp 0.008  fr 0.007
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
BUDGET=${BUDGET:-gpu}
SEEDS=${SEEDS:-0 1 2 3 4}
NPE=${NPE:-8000}
mkdir -p results/logs

run () {  # dataset method
  for s in $SEEDS; do
    printf '%-20s %-16s s%s  ' "$1" "$2" "$s"
    out=$($PY -W ignore -m sadl.experiments.run --dataset "$1" --method "$2" --seed "$s" \
            --budget "$BUDGET" --n-per-env "$NPE" 2>&1) || { echo "FAILED"; continue; }
    echo "$out" | awk '
      /^worst acc/   {w=$3; ch=$5}
      /^factor rec/  {fr=$3}
      /^stab. gap/   {g=$3}
      /^composab/    {c=$3}
      /^accepted T/  {t=$3}
      END {printf "worst=%s (chance %s) fr=%.3f gap=%s comp=%.3f T=%s\n", w, ch, fr, g, c, (t==""?"-":t)}'
  done
}

echo "=== fix1: novelty ceiling, on ColoredMNIST"
run ColoredMNIST SADL-v2-fix1
echo
echo "=== fix2: sufficiency, on ColoredMNIST"
run ColoredMNIST SADL-v2-fix2
echo
echo "=== fix3: audit coverage, on dSprites"
run dSprites SADL-v2-fix3
echo
echo "=== control: same family, none of the algorithm"
run ColoredMNIST SimCLR-bank
run dSprites SimCLR-bank

echo
echo "Read fix1 against v1 SADL on ColoredMNIST (gap 0.039, worst 46.2): the balance"
echo "of accepted distinctions and novelty should rise. Read fix2 against comp 0.163."
echo "Read fix3 against v1 SADL-lite on dSprites (gap 0.271). Read SimCLR-bank against"
echo "SADL-lite (ColoredMNIST gap 0.007, dSprites 0.271) -- if it matches, stop."
