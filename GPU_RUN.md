# GPU run: what to check, in what order

Nothing has been run under the current code. An earlier partial sweep from a
laptop (MPS, `quick` budget, 3 seeds, older method code) is archived under
`results/archive_local_mps/` for reference; the runner does not read it.

## Changes made after that sweep, and why they matter

1. **The acceptance gate now uses the literal reading of Equation 17.**
   Previously `A_hat` was the per-example worst case over the whole bank, which
   with ~40 settings is a much harder bar than `max_tau E_X[...]`. This is the
   single change most likely to alter the headline result: under the old gate,
   dSprites and Causal3DIdent-lite accepted **no** distinctions at any budget, so
   both representations were empty and accuracy was at chance. Both readings are
   now recorded (`adv_flip`, `adv_flip_worstcase`).

2. **The learned and bank flip rates are recorded separately**
   (`adv_flip_learned`, `adv_flip_bank`). On the laptop sweep the learned Confuser
   reported ~0.0 on colour-based heads that the parameter-free bank broke with a
   rate of 1.0. That decomposition is the evidence for splitting the Confuser's
   search role from its certification role, and it is what RQ6 should report.

3. **A chance reference is stored per run** (`acc_chance_worst`,
   `acc_chance_avg`). Worst-environment accuracy is not interpretable without it:
   in the laptop sweep SADL beat the best baseline by 22 points on ColoredMNIST
   while sitting exactly at chance, and most baselines landed *below* chance.

4. **Interim acceptance checks audit a bank subsample; the accepting decision
   re-runs the full bank.** Roughly halves SADL-lite cost, which was 600 s per
   run against 184 s for SADL.

5. **Sharding, fingerprinting, resumability.** `--shard i/n` splits a grid across
   GPUs; `rqs tables` aggregates every cached cell without training; each record
   carries a source fingerprint and the runner warns rather than silently mixing
   code versions.

## Order of operations

```bash
# 1. environment
bash -c 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
.venv/bin/python scripts/download_data.py
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 2. does anything get accepted now?  ~2 minutes.  This is the go/no-go check.
.venv/bin/python -m sadl.experiments.run --dataset dSprites --method SADL --seed 0 --budget smoke
.venv/bin/python -m sadl.experiments.run --dataset ColoredMNIST --method SADL --seed 0 --budget smoke
```

Read `n_accepted` and the `sadl_trace` in `results/runs/*.json`. Per step the
trace holds `val_novelty`, `val_shift`, `val_adv_flip`, `val_adv_flip_bank`,
`val_adv_flip_learned`, `val_adv_flip_worstcase`, `val_balance`, `best_step`, and
the stop reason.

- If `n_accepted > 0` on dSprites, the old gate was the binding constraint and the
  full sweep is worth its compute.
- If `n_accepted == 0` and `val_adv_flip` sits just above `sadl_eps_adv = 0.25`,
  the game is close; raise `sadl_check_every` density and `sadl_rmax` before
  concluding, and check whether `val_balance` is extreme (lopsided splits were
  what passed on ColoredMNIST).
- If `n_accepted == 0` with `val_novelty == 0` and `val_balance` at 0 or 1, the
  head collapsed despite the non-collapse floors; report it as an optimisation
  failure with the trace attached rather than tuning it away silently.

```bash
# 3. full sweep
NGPU=<gpus> bash scripts/run_all.sh          # budget=gpu, seeds 0-4
```

Expect the SADL cells to dominate the cost: a sequential game costs about `T`
times a single adversarial game, and the laptop measured 20-40x ERM per run.

## What the sweep is for

The primary claim to test is **not** worst-environment accuracy. It is that
SADL's accepted distinctions are stable: on ColoredMNIST the laptop sweep gave a
stability gap of 0.013-0.022 against 0.169 for ERM and ~0.25 for IRMv1, VREx and
DANN. That is the mechanism working, and it does not depend on sufficiency.

Worst-environment accuracy inherits a dependency the objective does not control:
Equation 18 optimises novelty, cross-environment shift, and flip rate, and none
of them prefers a stable separation that carries label information. Expect
stable-but-uninformative codes. `SADL-warmstart` (label-free InfoNCE over the
Confuser's own bank) is in the ablation set to test whether trunk
content-poverty is the cause; it is deliberately **not** part of SADL.

## Do not do this on the GPU box

Do not tune SADL hyperparameters against test-environment accuracy. All model
selection, including the acceptance snapshot inside each game, uses
training-environment validation splits only, and the held-out environments stay
untouched until the final evaluation. If SADL needs different hyperparameters,
select them on training-domain validation and say so.
