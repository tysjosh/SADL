# SADL experiments

Runnable implementation of the experimental programme in *Sequential Adversarial
Distinction Learning: Environment-Invariant Pretraining for Out-of-Distribution
Generalization* (the PDF in this directory). The paper's Section 6 specifies the
tests and leaves every result cell as `TBD`; this repository implements the
method, the baselines, the metrics, the statistical protocol, and the six research
questions, and fills those cells in.

```
sadl/
  data/        multi-environment datasets with a known generative decomposition
  models/      shared encoder family, Separator heads, Confuser bank
  methods/     SADL, SADL-lite, ablations, and every baseline of Table 2
  eval/        metrics of Section 6.4, statistics of Section 6.5, split diagnostic
  experiments/ RQ1-RQ6 drivers and the table builders of Section 6.6
scripts/       dataset download and preparation
results/       run records, rendered tables, logs (created on first run)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_data.py        # dSprites + MNIST, ~80 MB
```

`requirements.txt` pins `torch==2.4.1`, which installs the CUDA 12.1 build from
PyPI. For a different CUDA version, install torch first from the matching index
and then the rest:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

The device is chosen automatically (CUDA, then MPS, then CPU) and can be forced
with `SADL_DEVICE=cuda`. Data and results locations are `$SADL_DATA` and
`$SADL_RESULTS`.

## Running on a GPU machine

```bash
bash scripts/run_all.sh                    # single GPU, budget=gpu, 5 seeds
NGPU=4 bash scripts/run_all.sh             # shard the grid across 4 GPUs
BUDGET=standard SEEDS="0 1 2" bash scripts/run_all.sh
```

Sanity-check the pipeline before committing hours to it:

```bash
.venv/bin/python -m sadl.experiments.run --dataset ColoredMNIST --method SADL --seed 0 --budget smoke
.venv/bin/python -m sadl.experiments.rqs tables --budget gpu   # aggregates only, trains nothing
```

## Running experiments individually

Three granularities, all sharing one cache, so anything run individually counts
towards the full sweep and nothing is computed twice.

**One research question.** `rq1` … `rq6` each run standalone and write their own
table. `--dry-run` lists the cells and exits, which is the cheap way to see what a
command would cost before starting it.

```bash
.venv/bin/python -m sadl.experiments.rqs rq3 --budget gpu --seeds 0 1 2 3 4
.venv/bin/python -m sadl.experiments.rqs rq2 --budget gpu --seeds 0 1 2 3 4 --dry-run

# narrow an RQ to particular datasets or methods
.venv/bin/python -m sadl.experiments.rqs rq6 --datasets dSprites --methods SADL SADL-warmstart

# re-render every table from whatever is cached; trains nothing
.venv/bin/python -m sadl.experiments.rqs tables --budget gpu --seeds 0 1 2 3 4
```

`rq5` scores frozen representations from the RQ2 cells, so run `rq2` first; it
then reads from cache and trains nothing itself.

**One cell.** `(dataset, method, seed)` on its own, with any `Budget` field
overridable from the command line. Use `--tag` for one-off variants so they get
their own cache entry instead of colliding with a sweep cell.

```bash
.venv/bin/python -m sadl.experiments.run --list

.venv/bin/python -m sadl.experiments.run --dataset dSprites --method SADL --seed 0 --budget smoke

.venv/bin/python -m sadl.experiments.run --dataset ColoredMNIST --method SADL --budget gpu \
    --set sadl_tmax=6 --set sadl_eps_adv=0.3 --tag epsweep

# the |E| = 1 cell of RQ4 on its own
.venv/bin/python -m sadl.experiments.run --dataset ColoredMNIST --method SADL --n-train-envs 1 --tag E1
```

A single cell prints worst / average / in-distribution accuracy against the chance
reference, factor recovery, stability gap, composability, and — for SADL — the
full acceptance trace: per step and per restart the novelty, shift, and flip rate
broken out by adversary (bank, learned, per-example worst case), the balance, the
selected snapshot step, and the stop reason.

**Everything.** `scripts/run_all.sh`, or `rqs all`.

Budgets are `smoke`, `quick`, `standard`, `full`, `gpu`; report from `gpu`. Runs
are cached by `(dataset, method, seed, budget, tag)` under `results/runs`, so a
sweep is interruptible and resumable, and shards are disjoint by construction.
Each record carries a fingerprint of the package source, and the runner warns
when it reads a record produced by different code rather than mixing versions
silently. Every run also stores per-instance correctness vectors, which is what
the hierarchical bootstrap resamples.

`results/archive_local_mps/` holds an earlier partial sweep from a laptop (MPS,
`quick` budget, 3 seeds, older code). It is kept for reference only and is not
read by the runner.

## What is implemented

**SADL** (`methods/sadl.py`) follows Algorithm 1: a Separator head per step
conditioned on the frozen outputs of earlier heads (Equation 13-14), a Confuser
that ascends the flip objective on a faster timescale (Equation 17), the
alternating minimax objective of Equation 18, the acceptance test of Equation 19
on held-out environment-balanced validation data, restarts, and the compression
gate and patience rule of Equation 20.

**Confuser** (`models/confuser.py`) is a bank of photometric, textural and
environment-conditional operations, none of which alters spatial content, so the
invariant factors of the synthetic generators are preserved by construction. Two
variants: a parameter-free per-sample worst case over the bank (`SADL-lite`) and
an amortised spectrally normalised network that emits per-sample mixture weights
and bounded strengths (`SADL`). `env_resample` is the AdaIN-style nuisance
resampling that Proposition 5.1 idealises. The families are nested
(`none ⊂ fixed3 ⊂ fixed7 ⊂ fixed10 ⊂ learned`) for RQ3.

**Baselines** (Table 2): ERM, SimCLR, MAE, IRMv1, VREx, β-VAE, FactorVAE, DANN.
All methods share the encoder trunk, input resolution, optimisation budget, and
downstream readout class; accuracy is always measured by fitting the same
logistic readout on frozen features, with the L2 strength chosen on
training-domain validation splits.

**Metrics** (Section 6.4): worst- and average-environment accuracy (Equation 23),
factor recovery by NMI with maximum-weight one-to-one matching (Equation 24), the
empirical stability gap on pairs that share `z_inv` and resample `z_env`
(Equation 25), and composability against oracle and constant-predictor references
(Equation 22). Continuous representations are quantile-discretised and binarised
at their training median so a 128-dimensional feature vector and a 4-bit code are
scored by the same rule.

**Statistics** (Section 6.5): training-domain validation for all model selection,
shared seeds across methods, mean ± sd, paired SADL-minus-baseline effects with
bootstrap CIs, a 95% hierarchical bootstrap over seeds / environments / instances,
and Holm correction for the secondary family.

## Datasets

| Dataset | Source | Environments | Shortcut |
|---|---|---|---|
| ColoredMNIST | MNIST, standard construction | train `e ∈ {0.1, 0.2}`, test `{0.9, 0.5}` | colour channel |
| dSprites | official dSprites archive + environment interventions | train tint agreement `{0.90, 0.80}`, test `{0.10, 0.33}` | object tint |
| Causal3DIdent-lite | procedural renderer (see below) | train background agreement `{0.90, 0.80}`, test `{0.10, 0.33}` | background hue |
| PACS, TerraIncognita, Camelyon17, iWildCam | local directories only | leave-one-domain-out | — |

Environment specifications, intervention ranges, and the invariant/nuisance
designation are fixed in code before any model is trained. The invariant marginal
is identical across environments; only the nuisance law shifts.

The four naturalistic benchmarks are tens of gigabytes and are never downloaded
automatically. `sadl/data/real.py` reads them from `$SADL_DATA` if present and
otherwise raises `DataUnavailable`, which the runner records as
`status="data_unavailable"` so the table cell stays empty rather than being
quietly filled. `scripts/prepare_real_data.py` converts a DomainBed or WILDS
installation into the expected layout.

## Deviations from the paper, and why

These are choices the paper does not specify, or places where a literal reading
does not run. Each is isolated and switchable.

1. **Causal3DIdent-lite is a substitute, not the published dataset.** The original
   renders Blender scenes and ships as a multi-gigabyte archive. The replacement
   reproduces the content/style structure the paper's Assumption 3.1 relies on
   with an analytic renderer. Numbers are not comparable to the original.

2. **The naturalistic benchmarks are not run here.** No result is reported for
   PACS, TerraIncognita, Camelyon17 or iWildCam; the cells are marked
   `n/a (no data)`. The loaders exist and will run if the data is provided.

3. **The misspecification diagnostic is an operationalisation.** Salaudeen et
   al.'s protocol is approximated by fitting worst-environment accuracy against
   in-distribution validation accuracy over the baseline model pool and
   designating a split `misspecified` when the relation is significantly positive
   with slope ≥ 0.5 ("accuracy on the line"). SADL runs are excluded from the fit,
   and the designation is written to `results/split_designations.json` with a hash
   of its inputs before any SADL comparison is made.

4. **The compression gate probe is label-free.** Equation 20 stops the sequence
   when the marginal probe gain falls below a threshold. Using downstream labels
   there would leak task information into pretraining, which Section 5.4 forbids,
   so the probe measures the residual variance of a ridge regression from the
   retained code to a fixed random projection of the input instead.

5. **Novelty is estimated as a mutual information.** `H(d_t | D_{t-1})` is
   estimated as `Σ_v w_v Hb(E[q | v]) − E[Hb(q)]`, which coincides with the
   conditional entropy of the hard distinction when `q` is 0/1 and is what the
   acceptance test evaluates. Without the subtracted term, a head that outputs
   `q = 1/2` everywhere scores a full bit of novelty while its hard decision is
   constant.

6. **Three additions keep the minimax game from answering stability pressure
   with a constant head.** A constant distinction satisfies both penalties in
   Equation 18 exactly, and in a plain implementation that is the solution
   gradient descent finds. The additions are: the Separator's score is bounded and
   standardised by running statistics, so "score above its own mean" cannot be
   constant and neither shrinking nor saturating the logit reduces the penalties;
   a variance floor on the score and a VICReg-style variance/covariance anchor on
   the trunk (Section 3.2 points at exactly these non-collapse constraints); and
   the penalties are ramped in rather than applied from step 0. All are label-free
   and environment-agnostic. `sadl_anchor = 0` turns the anchors off.

7. **The acceptance test audits with the fixed bank as well as the learned
   Confuser.** A learned adversary that has not yet found a perturbation
   certifies distinctions it has merely failed to break; the discrete bank needs
   no training and cannot be under-optimised. `T` in the objective is likewise the
   union of the learned mixture and the bank, subsampled per step for cost. The
   record stores the learned and bank flip rates separately, so the two roles --
   searching for a perturbation and certifying its absence -- can be read apart.

8. **The flip rate is gated in its literal form and reported in a stricter one.**
   Equation 17 defines `A_t` as `E_X |q(tau(x)) - q(x)|` for one transformation,
   so the gate uses `max_tau E_X[1{d(tau(x)) != d(x)}]`: the strongest single
   transformation, scored on the population. The per-example worst case
   `E_X[max_tau ...]` is strictly larger and, with a bank of about forty settings,
   nearly always finds some transformation that flips some example; it is recorded
   as `adv_flip_worstcase` rather than used to reject. Which of the two is used
   changes the accept/reject outcome, so both are reported.

9. **Acceptance is evaluated periodically inside each game, and the passing
   snapshot is kept.** Adversarial games are not monotone: a candidate that
   satisfies Equation 19 mid-run is routinely destroyed later. Selection uses
   training-environment validation data only.

10. **A Confuser-consistent trunk warm start is available but not part of SADL.**
   Equation 18 gives the trunk no reason to retain content. The optional warm
   start pretrains it with InfoNCE over two views drawn from the Confuser's own
   bank. It is reported as the separate `SADL-warmstart` variant, because it is an
   addition to the published objective and because it turns part of the comparison
   with SimCLR into a comparison of augmentation policies.

11. **ColoredMNIST uses two held-out environments** (`e = 0.9` inverts the
    shortcut, `e = 0.5` removes it) so that worst-environment accuracy is a
    minimum over more than one environment.

## Reading the results

`results/` contains, after a run:

- `table3_rq1_factor_recovery.md` — RQ1
- `table4_5_rq2_ood.md` — RQ2, plus the confirmatory comparison and Equation 26
- `table6_rq6_ablations.md` — RQ6
- `table7_rq3_confuser.md` — RQ3
- `table8_rq4_env_count.md` — RQ4
- `table9_rq5_composability.md` — RQ5
- `split_designations.json` — frozen split diagnostic
- `runs/*.json` — one record per cell, including the full SADL acceptance trace
  (per-step novelty, shift, flip rate, balance, restart, stop reason)

`FINDINGS.md` summarises what the runs in this repository actually produced.
