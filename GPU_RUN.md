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

# 2. every variant executes at all.  ~10 minutes, CPU-cheap.
bash scripts/smoke_all.sh

# 3. does anything get accepted at the budget the sweep will use?  GO/NO-GO.
bash scripts/preflight_gpu.sh
```

Step 3 runs the **first Separator game only** — `sadl_tmax=1` with the `gpu`
preset's per-game step count (1000), batch size (256) and audit density — on
dSprites, ColoredMNIST and Causal3DIdent-lite. That is the whole question:
`SADL.fit` breaks out of the outer loop when a step exhausts its restarts, so if
`t = 0` never accepts, the sequence is empty, the representation is a column of
zeros, and every SADL cell in the sweep is a chance-level artefact. Cost is about
one eighth of a single SADL cell per dataset against a sweep of several hundred.
Cells are tagged `gate`, so they never collide with a reporting cell.

Do not use `smoke` as the gate. It runs 120 steps at `T=2, R=1` and accepts
nothing even on ColoredMNIST, so it cannot separate "the method does not clear the
gate" from "the budget was too small to try".

`scripts/gate_report.py` reads the traces and prints, per restart, `val_novelty`,
`val_shift`, `val_adv_flip` with its `bank` / `learned` / `worst-case`
decomposition, `val_balance` and `best_step`, then names which clause of
Equation 19 bound and by how much. Thresholds are read from the budget the cell
actually ran with, including `--set` overrides. It exits non-zero when nothing
accepted anywhere.

- **`go`** — the sequence is not empty; the sweep is worth its compute.
- **`no-go`, binding `flip`, declining across restarts** — close and still
  improving. More restarts is the cheap thing to try, and the trunk carries across
  restarts, so they compound rather than reset.
- **`no-go`, binding `flip`, flat** — a gate calibration question, not a budget
  one. Compare `adv_flip_bank` against `adv_flip_learned`: if the bank alone is
  rejecting, the bank may be leaving `T` on that dataset.
- **`no-go`, binding `novelty` with `balance` at 0 or 1** — head collapse despite
  the non-collapse floors. Report it as an optimisation failure with the trace
  attached rather than tuning it away.
- **`no-go`, binding `shift`** — the head is still environment-dependent.

```bash
# 4. full sweep
NGPU=<gpus> bash scripts/run_all.sh          # budget=gpu, seeds 0-4
```

Expect the SADL cells to dominate the cost: a sequential game costs about `T`
times a single adversarial game, and the laptop measured 20-40x ERM per run.
`results/table10_cost.md` reports this directly once the sweep has run — time,
peak memory and shared-trunk forward/backward evaluations, each relative to ERM,
plus accepted distinctions per accelerator-hour. Table 10 trains nothing of its
own; it reuses the RQ1 and RQ2 cells, which is also what makes the ratios valid.
Measured at `smoke` on one dataset, `Evals/step` was 47x ERM for SADL-lite and
18x for SADL, the learned Confuser being the cheaper of the two because it applies
one amortised transformation where the bank takes a maximum over a subsample.

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
