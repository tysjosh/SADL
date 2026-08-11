"""Sequential Adversarial Distinction Learning (Section 5, Algorithm 1).

Per step ``t`` the Separator solves

    min_theta  -N_t(theta) + lambda_stab * S_t(theta) + lambda_adv * max_phi A_t(theta, phi)

with

    N_t  conditional novelty  H(d_t(X) | D_{t-1}(X))                    (Eq. 15)
    S_t  cross-environment marginal shift  max_{e != e'} |E_e q - E_e' q| (Eq. 16)
    A_t  Confuser soft flip  E_X |q(tau_phi(X)) - q(X)|                  (Eq. 17)

and the candidate is accepted only if the held-out hard-assignment statistics
satisfy ``N >= gamma``, ``S <= eps_stab``, ``A <= eps_adv``    (Eq. 19).

The compression gate (Eq. 20) uses a *label-free* probe: the residual variance of
a ridge regression from the retained code to a fixed random projection of the
input.  Using downstream labels here would leak task information into
pretraining, which Section 5.4 explicitly forbids, so marginal information gain
is measured against an unsupervised target instead.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ..data import MultiEnvDataset
from ..models import MLP, ConvEncoder, FrozenDistinction, SeparatorHead, gumbel_sigmoid, make_confuser
from ..utils import entropy_bits
from .base import Method


def conditional_novelty(q: torch.Tensor, prev_bits: torch.Tensor | None, conf_weight: float = 1.0) -> torch.Tensor:
    """Estimate of H(d_t(X) | D_{t-1}(X)) in bits (Equation 15).

    Written as a mutual-information estimator,

        sum_v w_v * Hb(E[q | D_{t-1} = v])  -  E[Hb(q)],

    which equals the conditional entropy of the hard distinction exactly when
    ``q`` is 0/1 and is used unchanged for the hard-assignment acceptance test.
    The subtracted per-sample term matters during optimisation: without it a head
    that outputs ``q = 1/2`` everywhere scores a full bit of novelty while its
    hard decision is constant, and that degenerate solution is the one gradient
    descent finds first.

    Two changes are made for optimisation, neither of which affects the value on
    hard assignments:

    1. ``conf_weight < 1`` down-weights the per-sample term, as in
       mutual-information clustering objectives.  With equal weights a constant
       head and an undecided head both score zero, leaving no gradient out of the
       collapsed basin.
    2. the per-sample uncertainty ``Hb(q)`` is replaced by the linear surrogate
       ``1 - 2|q - 1/2|``.  Both vanish on hard assignments, but ``Hb`` also has
       zero derivative at ``q = 1/2``, which makes the undecided head a stationary
       point: the gain from becoming input-dependent is second order there while
       the shift and flip penalties are first order, so the collapsed head is a
       local minimum.  The linear surrogate has a non-zero subgradient at
       ``q = 1/2`` and breaks the symmetry immediately.
    """
    per_sample = conf_weight * entropy_bits(q).mean()
    if prev_bits is None or prev_bits.shape[1] == 0:
        return entropy_bits(q.mean()) - per_sample
    codes = (prev_bits > 0.5).to(torch.long)
    weights = torch.tensor([2**i for i in range(codes.shape[1])], device=q.device)
    cell = (codes * weights).sum(1)
    total = q.new_zeros(())
    n = len(q)
    for v in cell.unique():
        m = cell == v
        w = m.sum().to(q.dtype) / n
        total = total + w * entropy_bits(q[m].mean())
    return total - per_sample


def env_shift(q: torch.Tensor, env: torch.Tensor, n_envs: int) -> torch.Tensor:
    """max_{e != e'} |E_{P^e}[q] - E_{P^e'}[q]| (Equation 16)."""
    means = []
    for e in range(n_envs):
        m = env == e
        if m.any():
            means.append(q[m].mean())
    if len(means) < 2:
        return q.new_zeros(())
    mm = torch.stack(means)
    return mm.max() - mm.min()


class SADL(Method):
    name = "SADL"
    category = "learned Confuser"
    pretraining_labels = "env. only"
    repr_kind = "binary"

    def __init__(self, budget, device, seed=0, confuser: str | None = None) -> None:
        super().__init__(budget, device, seed)
        self.confuser_kind = confuser or budget.sadl_confuser
        self.accepted: list[FrozenDistinction] = []
        self.trace: list[dict] = []
        self._built = False

    # ---- structure --------------------------------------------------------
    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        self.trunk = ConvEncoder(ds.in_shape[0], b.feat_dim, b.width).to(self.device)
        self.n_ch = ds.in_shape[0]
        self.n_envs = ds.n_train_envs
        # Parameter-free bank: part of T during optimisation and the auditor in
        # the acceptance test.
        self._audit_confuser = make_confuser(b.sadl_audit_bank, self.n_ch, self.n_envs).to(self.device)
        self._obj_bank = self._audit_confuser
        self._built = True

    @property
    def repr_dim(self) -> int:
        return max(1, len(self.accepted))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.accepted:
            return torch.zeros(len(x), 1, device=x.device)
        return self.prev_bits(x, len(self.accepted))

    @torch.no_grad()
    def prev_bits(self, x: torch.Tensor, upto: int) -> torch.Tensor:
        bits = torch.zeros(len(x), 0, device=x.device)
        for j in range(upto):
            b = self.accepted[j](x, bits if j else None).unsqueeze(1)
            bits = torch.cat([bits, b], 1)
        return bits

    # ---- objective pieces -------------------------------------------------
    def _q(self, head: SeparatorHead, x: torch.Tensor, prev: torch.Tensor | None, tau: float, noisy: bool) -> torch.Tensor:
        logit = head(self.trunk(x), prev)
        return gumbel_sigmoid(logit, tau=tau, hard=False, noisy=noisy)

    def _logit(self, head: SeparatorHead, x: torch.Tensor, prev: torch.Tensor | None, bn_eval: bool) -> torch.Tensor:
        was_training = head.norm.training
        head.norm.train(not bn_eval)
        out = head(self.trunk(x), prev)
        head.norm.train(was_training)
        return out

    def _anchor(self, feat: torch.Tensor) -> torch.Tensor:
        """Variance / covariance non-collapse anchor on the shared trunk.

        Equation 18 alone lets the Separator satisfy the shift and flip penalties
        by destroying the trunk's features: a constant representation is perfectly
        stable.  Section 3.2 anticipates this and points at the non-collapse
        constraints used in non-contrastive learning; this is that constraint,
        in its VICReg form.  It touches only the trunk, uses no labels and no
        environment identifiers, and is switched off by ``sadl_anchor = 0``.
        """
        b = self.budget
        if b.sadl_anchor <= 0:
            return feat.new_zeros(())
        z = feat - feat.mean(0, keepdim=True)
        var = torch.relu(1.0 - torch.sqrt(z.var(0) + 1e-6)).mean()
        cov = (z.T @ z / max(1, len(z) - 1)).pow(2)
        cov = (cov.sum() - torch.diagonal(cov).sum()) / feat.shape[1]
        return b.sadl_anchor * (var + b.sadl_anchor_cov * cov)

    def _warmstart(self, ds: MultiEnvDataset, steps: int) -> None:
        """Optional label-free trunk warm start on Confuser-consistent views.

        Equation 18 gives the trunk no reason to retain content: the Separator can
        satisfy it with any invariant feature, informative or not.  This stage
        pretrains the trunk with an InfoNCE loss whose two views are drawn from the
        same transformation bank the Confuser draws from, so the trunk becomes
        invariant to exactly the nuisances in T while keeping whatever separates
        one input from another.  It uses no labels and no environment identifiers.

        Off by default: it is reported as the ``SADL-warmstart`` variant rather
        than folded into SADL, because it is an addition to the published
        objective, and because it makes part of the comparison with SimCLR a
        comparison of augmentation policies.
        """
        b = self.budget
        proj = MLP(b.feat_dim, 64, hidden=128, depth=1).to(self.device)
        opt = torch.optim.Adam(list(self.trunk.parameters()) + list(proj.parameters()), lr=b.lr)
        bank = self._obj_bank
        self.trunk.train()
        for x, _, _ in self.loader(ds, steps=steps):
            ref = torch.roll(x, shifts=b.batch_per_env, dims=0)
            v1, v2 = bank.candidates(x, ref, subsample=2) if len(bank.bank) else (x, x)
            z1 = torch.nn.functional.normalize(proj(self.trunk(v1)), dim=1)
            z2 = torch.nn.functional.normalize(proj(self.trunk(v2)), dim=1)
            z = torch.cat([z1, z2])
            sim = z @ z.t() / 0.2
            sim.fill_diagonal_(-1e4)
            n = len(z1)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(x.device)
            loss = torch.nn.functional.cross_entropy(sim, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.trunk.parameters()) + list(proj.parameters()), 5.0)
            opt.step()
        self.log["warmstart_loss"] = float(loss.detach())

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        if b.sadl_warmstart_steps:
            self._warmstart(ds, b.sadl_warmstart_steps)
        steps_per_t = max(20, b.steps // max(1, b.sadl_tmax))
        probe_loss_prev = self._probe_loss(ds)
        small_gain_streak = 0

        for t in range(b.sadl_tmax):
            accepted_this_step = False
            for r in range(b.sadl_rmax):
                head, confuser, stats = self._run_game(ds, t, steps_per_t, restart=r)
                ok, val = self._acceptance(ds, head, confuser, t)
                self.trace.append(
                    {"t": t, "restart": r, "accepted": bool(ok), **stats, **{f"val_{k}": v for k, v in val.items()}}
                )
                if ok:
                    self.accepted.append(FrozenDistinction(copy.deepcopy(self.trunk), copy.deepcopy(head)))
                    accepted_this_step = True
                    break
            if not accepted_this_step:
                self.trace.append({"t": t, "stop_reason": "restarts_exhausted"})
                break
            if b.sadl_compression_gate:
                probe_loss = self._probe_loss(ds)
                gain = probe_loss_prev - probe_loss
                self.trace[-1]["probe_gain"] = float(gain)
                probe_loss_prev = probe_loss
                small_gain_streak = small_gain_streak + 1 if gain < b.sadl_eta else 0
                if small_gain_streak >= b.sadl_patience:
                    self.trace.append({"t": t, "stop_reason": "patience"})
                    break

        self.log["n_accepted"] = len(self.accepted)
        self.log["trace"] = self.trace
        return self.log

    def _run_game(self, ds: MultiEnvDataset, t: int, steps: int, restart: int):
        """Alternating minimax for one candidate distinction (lines 5-10)."""
        b = self.budget
        torch.manual_seed(self.seed * 1000 + t * 17 + restart)
        head = SeparatorHead(b.feat_dim, n_prev=t, logit_cap=b.sadl_logit_cap).to(self.device)
        confuser = make_confuser(self.confuser_kind, self.n_ch, self.n_envs).to(self.device)
        theta = list(self.trunk.parameters()) + list(head.parameters())
        opt_theta = torch.optim.Adam(theta, lr=b.lr, weight_decay=b.weight_decay)
        opt_phi = (
            torch.optim.Adam(confuser.parameters(), lr=b.lr * b.sadl_confuser_lr_mult, betas=(0.5, 0.9))
            if getattr(confuser, "learned", False)
            else None
        )
        self.trunk.train()
        head.train()
        hist = {"novelty": [], "shift": [], "adv": []}
        best: dict | None = None
        n_checks = 0

        for step, (x, _, env) in enumerate(self.loader(ds, steps=steps)):
            prev = self.prev_bits(x, t) if t else None
            ref = torch.roll(x, shifts=b.batch_per_env, dims=0)

            # The shift and flip terms use the head's running normalisation
            # statistics, so q(x) and q(tau(x)) are standardised identically and
            # the flip measures a change in the decision rather than a change in
            # the batch statistics.
            def q_fn(inp: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(self._logit(head, inp, prev, bn_eval=True) / b.sadl_tau)

            # --- Confuser ascent on A_t (line 8), on the faster timescale
            if opt_phi is not None:
                for _ in range(b.sadl_confuser_steps):
                    adv = confuser.flip_objective(q_fn, x, ref, env)
                    opt_phi.zero_grad(set_to_none=True)
                    (-adv).backward(inputs=list(confuser.parameters()))
                    nn.utils.clip_grad_norm_(confuser.parameters(), 5.0)
                    opt_phi.step()

            # --- Separator descent on Equation 18 (line 9)
            feat = self.trunk(x)
            logit = head(feat, prev)  # train mode: updates the running statistics
            q = gumbel_sigmoid(logit / b.sadl_tau, tau=1.0, hard=False, noisy=b.sadl_gumbel_noise)
            novelty = conditional_novelty(q, prev, b.sadl_conf_weight)
            shift = env_shift(q_fn(x), env, self.n_envs)
            adv_term = confuser.flip_objective(q_fn, x, ref, env)
            if getattr(confuser, "learned", False) and self._obj_bank.bank:
                # T is the union of the learned mixture and the discrete bank:
                # the learned adversary starts from scratch every step and needs
                # time before it finds anything, and until then it certifies
                # distinctions it has simply not learned to break.
                adv_term = torch.maximum(
                    adv_term,
                    self._obj_bank.flip_objective(q_fn, x, ref, subsample=b.sadl_bank_subsample),
                )
            # The invariance penalties are ramped in: a constant head satisfies
            # them exactly, so full-strength penalties from step 0 reward collapse
            # before any candidate separation exists to be tested.
            ramp = min(1.0, (step + 1) / max(1.0, b.sadl_penalty_warmup_frac * steps))
            # Non-collapse floor on the head's own score.  Normalising the logit is
            # not enough on its own: if the pre-normalisation score becomes
            # constant, the running variance collapses and the evaluation-mode
            # distinction degenerates even though the training-mode one looks fine.
            score_floor = torch.relu(1.0 - head.score(feat, prev).std())
            loss = (
                -novelty
                + ramp * (b.sadl_lambda_stab * shift + b.sadl_lambda_adv * adv_term)
                + self._anchor(feat)
                + b.sadl_anchor * score_floor
            )
            opt_theta.zero_grad(set_to_none=True)
            loss.backward(inputs=theta)
            nn.utils.clip_grad_norm_(theta, 5.0)
            opt_theta.step()

            hist["novelty"].append(float(novelty.detach()))
            hist["shift"].append(float(shift.detach()))
            hist["adv"].append(float(adv_term.detach()))

            # --- periodic acceptance test on training-environment validation data
            last = step == steps - 1
            if (step + 1) % b.sadl_check_every == 0 or last:
                n_checks += 1
                # Interim checks audit a random subset of the bank; the accepting
                # decision below re-runs the full bank on more validation data.
                ok, st = self._acceptance(
                    ds, head, confuser, t, n_val=b.sadl_check_n, audit_subsample=b.sadl_audit_subsample
                )
                if ok and (best is None or st["novelty"] > best["stats"]["novelty"]):
                    best = {
                        "stats": st,
                        "step": step,
                        "trunk": copy.deepcopy(self.trunk.state_dict()),
                        "head": copy.deepcopy(head.state_dict()),
                    }
                self.trunk.train()
                head.train()

        # Adversarial games are not monotone: a candidate that satisfies
        # Equation 19 mid-run is routinely destroyed later.  The snapshot that
        # passed is what the step returns, selected on training-environment
        # validation data only.
        if best is not None:
            self.trunk.load_state_dict(best["trunk"])
            head.load_state_dict(best["head"])

        tail = lambda a: float(np.mean(a[-20:])) if a else float("nan")
        return (
            head,
            confuser,
            {
                "train_novelty": tail(hist["novelty"]),
                "train_shift": tail(hist["shift"]),
                "train_adv": tail(hist["adv"]),
                "n_checks": n_checks,
                "best_step": best["step"] if best else None,
            },
        )

    # ---- acceptance -------------------------------------------------------
    @torch.no_grad()
    def _hard_stats(
        self,
        ds: MultiEnvDataset,
        head: SeparatorHead,
        confuser,
        t: int,
        n_val: int = 512,
        audit_subsample: int | None = None,
    ) -> dict:
        self.trunk.eval()
        head.eval()
        qs, ds_bits, envs = [], [], []
        a_learn, a_bank, a_worst = [], [], []
        for e, split in enumerate(ds.train_envs):
            v = split.val()
            n = min(len(v), n_val)
            x = torch.as_tensor(v.x[:n]).to(self.device)
            prev = self.prev_bits(x, t) if t else None
            logit = head(self.trunk(x), prev)
            bits = (logit > 0).to(torch.float32)
            qs.append(torch.sigmoid(logit))
            ds_bits.append(bits)
            env_ids = torch.full((n,), e, dtype=torch.long, device=self.device)
            envs.append(env_ids)
            ref = torch.roll(x, shifts=max(1, n // 2), dims=0)
            bit_fn = lambda z: (head(self.trunk(z), prev) > 0).to(torch.float32)
            # The acceptance test uses the strongest available attack: the
            # co-trained Confuser *and* the parameter-free bank.  Using the learned
            # Confuser alone lets an under-optimised adversary certify a
            # distinction it simply has not learned to break yet, so the two are
            # recorded separately as well as combined.
            # When the Confuser *is* the bank (SADL-lite) the auditor already
            # covers it, so it is not evaluated twice.
            l_op = (
                confuser.flip_rates(bit_fn, x, ref, env_ids)[0]
                if getattr(confuser, "learned", False)
                else x.new_zeros(())
            )
            k_op, k_sample = self._audit_confuser.flip_rates(bit_fn, x, ref, env_ids, subsample=audit_subsample)
            a_learn.append(l_op.reshape(1))
            a_bank.append(k_op.reshape(1))
            a_worst.append(k_sample.reshape(1))
        q = torch.cat(qs)
        bits = torch.cat(ds_bits)
        env = torch.cat(envs)
        prev_all = None
        if t:
            xs = torch.cat([torch.as_tensor(s.val().x[: min(len(s.val()), n_val)]).to(self.device) for s in ds.train_envs])
            prev_all = self.prev_bits(xs, t)
        novelty = float(conditional_novelty(bits, prev_all, conf_weight=1.0))
        shift = float(env_shift(bits, env, self.n_envs))
        adv_learned = float(torch.cat(a_learn).mean())
        adv_bank = float(torch.cat(a_bank).mean())
        return {
            "novelty": novelty,
            "shift": shift,
            # Gate: strongest single transformation, scored on the population.
            "adv_flip": max(adv_learned, adv_bank),
            "adv_flip_learned": adv_learned,
            "adv_flip_bank": adv_bank,
            # Diagnostic: per-example worst case over the bank.
            "adv_flip_worstcase": float(torch.cat(a_worst).mean()),
            "balance": float(bits.mean()),
        }

    def _acceptance(
        self,
        ds: MultiEnvDataset,
        head: SeparatorHead,
        confuser,
        t: int,
        n_val: int = 512,
        audit_subsample: int | None = None,
    ) -> tuple[bool, dict]:
        b = self.budget
        s = self._hard_stats(ds, head, confuser, t, n_val=n_val, audit_subsample=audit_subsample)
        ok_novel = s["novelty"] >= b.sadl_gamma if b.sadl_compression_gate else s["novelty"] > 0.0
        ok = bool(ok_novel and s["shift"] <= b.sadl_eps_stab and s["adv_flip"] <= b.sadl_eps_adv)
        return ok, s

    # ---- label-free compression probe (Equation 20) -----------------------
    @torch.no_grad()
    def _probe_loss(self, ds: MultiEnvDataset, n: int = 1500, proj_dim: int = 32) -> float:
        rng = np.random.default_rng(self.seed + 4242)
        xs = np.concatenate([s.val().x[: n // len(ds.train_envs)] for s in ds.train_envs])
        flat = xs.reshape(len(xs), -1)
        if not hasattr(self, "_proj"):
            self._proj = rng.normal(size=(flat.shape[1], proj_dim)).astype(np.float32) / np.sqrt(flat.shape[1])
        target = flat @ self._proj
        target = (target - target.mean(0)) / (target.std(0) + 1e-6)
        if not self.accepted:
            return float((target**2).mean())
        bits = self.encode(torch.as_tensor(xs).to(self.device)).cpu().numpy()
        design = np.concatenate([bits, np.ones((len(bits), 1), dtype=np.float32)], 1)
        gram = design.T @ design + 1e-3 * np.eye(design.shape[1], dtype=np.float32)
        w = np.linalg.solve(gram, design.T @ target)
        resid = target - design @ w
        return float((resid**2).mean())


class SADLLite(SADL):
    """SADL-lite: max over a fixed transformation bank instead of max_phi."""

    name = "SADL-lite"
    category = "fixed-bank ablation"

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed, confuser="fixed10")


class SADLJoint(SADL):
    """Ablation: all heads trained jointly in one game, no conditioning or gating."""

    name = "SADL-joint"
    category = "ablation"

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        T = b.sadl_tmax
        torch.manual_seed(self.seed * 1000 + 77)
        heads = nn.ModuleList([SeparatorHead(b.feat_dim, n_prev=0) for _ in range(T)]).to(self.device)
        confuser = make_confuser(self.confuser_kind, self.n_ch, self.n_envs).to(self.device)
        theta = list(self.trunk.parameters()) + list(heads.parameters())
        opt_theta = torch.optim.Adam(theta, lr=b.lr, weight_decay=b.weight_decay)
        opt_phi = (
            torch.optim.Adam(confuser.parameters(), lr=b.lr * b.sadl_confuser_lr_mult, betas=(0.5, 0.9))
            if getattr(confuser, "learned", False)
            else None
        )
        self.trunk.train()

        for x, _, env in self.loader(ds, steps=b.steps):
            ref = torch.roll(x, shifts=b.batch_per_env, dims=0)

            def q_all(inp: torch.Tensor) -> torch.Tensor:
                f = self.trunk(inp)
                return torch.sigmoid(torch.stack([h(f) for h in heads], 1) / b.sadl_tau)

            if opt_phi is not None:
                adv = (q_all(confuser(x, ref, env)) - q_all(x)).abs().mean()
                opt_phi.zero_grad(set_to_none=True)
                (-adv).backward(inputs=list(confuser.parameters()))
                opt_phi.step()

            q = q_all(x)
            novelty = sum(entropy_bits(q[:, k].mean()) for k in range(T)) / T
            # decorrelation stands in for the sequential conditioning
            qc = q - q.mean(0, keepdim=True)
            corr = (qc.T @ qc / len(q)).abs()
            redundancy = (corr.sum() - torch.diagonal(corr).sum()) / max(1, T * (T - 1))
            shift = sum(env_shift(q[:, k], env, self.n_envs) for k in range(T)) / T
            if opt_phi is not None:
                adv_term = (q_all(confuser(x, ref, env)) - q).abs().mean()
            else:
                adv_term = confuser.flip_objective(lambda z: q_all(z).mean(1), x, ref)
            loss = -novelty + redundancy + b.sadl_lambda_stab * shift + b.sadl_lambda_adv * adv_term
            opt_theta.zero_grad(set_to_none=True)
            loss.backward(inputs=theta)
            nn.utils.clip_grad_norm_(theta, 5.0)
            opt_theta.step()

        snap = copy.deepcopy(self.trunk)
        self.accepted = [FrozenDistinction(snap, copy.deepcopy(h)) for h in heads]
        for fd in self.accepted:  # joint heads are unconditional
            fd.head.n_prev = 0
        self.log["n_accepted"] = len(self.accepted)
        self.log["trace"] = [{"mode": "joint", "T": T}]
        return self.log

    @torch.no_grad()
    def prev_bits(self, x: torch.Tensor, upto: int) -> torch.Tensor:
        bits = [self.accepted[j](x, None).unsqueeze(1) for j in range(upto)]
        return torch.cat(bits, 1) if bits else torch.zeros(len(x), 0, device=x.device)
