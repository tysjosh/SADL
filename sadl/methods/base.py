"""Method interface and the shared training budget.

Every method implements ``fit`` (representation learning) and ``features``
(frozen representation).  Downstream accuracy is always measured by fitting the
same readout class on frozen features, so no method benefits from a larger
downstream capacity than another.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..data import EnvBalancedLoader, MultiEnvDataset


@dataclass
class Budget:
    """Optimisation budget shared by all methods (Section 6.3)."""

    steps: int = 1200
    batch_per_env: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    feat_dim: int = 128
    width: int = 32
    latent_dim: int = 16
    eval_batch: int = 512
    # method-specific knobs
    irm_lambda: float = 100.0
    irm_anneal_frac: float = 0.25
    vrex_lambda: float = 100.0
    dann_lambda: float = 1.0
    simclr_temp: float = 0.2
    mae_mask_ratio: float = 0.6
    beta_vae_beta: float = 4.0
    factorvae_gamma: float = 10.0
    # SADL
    sadl_tmax: int = 4
    sadl_rmax: int = 2
    # Penalty weights are strong enough to push a candidate off a shortcut it has
    # already found; the non-collapse floors on the trunk and on the head's score
    # are what keep the game from answering that pressure with a constant head.
    sadl_lambda_stab: float = 2.0
    sadl_lambda_adv: float = 2.0
    sadl_bank_subsample: int = 8
    sadl_anchor: float = 1.0
    sadl_anchor_cov: float = 0.04
    sadl_check_every: int = 25
    sadl_check_n: int = 256
    sadl_audit_bank: str = "fixed10"
    sadl_audit_subsample: int = 12
    sadl_warmstart_steps: int = 0
    sadl_gamma: float = 0.3
    sadl_eps_stab: float = 0.15
    sadl_eps_adv: float = 0.25
    sadl_tau: float = 1.0
    sadl_conf_weight: float = 1.0
    sadl_penalty_warmup_frac: float = 1.0
    sadl_gumbel_noise: bool = False
    sadl_logit_cap: float = 4.0
    sadl_confuser: str = "learned"
    sadl_confuser_lr_mult: float = 2.0
    sadl_confuser_steps: int = 2
    sadl_patience: int = 2
    sadl_eta: float = 0.01
    sadl_sequential: bool = True
    sadl_compression_gate: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def with_(self, **kw: Any) -> "Budget":
        return replace(self, **kw)


BUDGET_PRESETS = {
    # ``steps`` is the number of optimisation steps for a baseline; SADL splits it
    # into ``steps / sadl_tmax`` steps per sequential game, so the total number of
    # gradient steps is comparable across methods.
    "smoke": Budget(steps=120, batch_per_env=32, sadl_tmax=2, sadl_rmax=1, sadl_check_every=20),
    "quick": Budget(steps=900, batch_per_env=64, sadl_tmax=3, sadl_rmax=2, sadl_check_every=25),
    "standard": Budget(steps=1800, batch_per_env=64, sadl_tmax=4, sadl_rmax=2, sadl_check_every=25),
    "full": Budget(steps=4800, batch_per_env=96, sadl_tmax=8, sadl_rmax=3, sadl_check_every=25),
    # For an accelerator: larger batches, more restarts, and a denser acceptance
    # check, which is what actually decides whether a candidate is ever caught in
    # a passing state. This is the preset the reported tables should use.
    #
    # sadl_rmax=6 and steps=8000 are set from a direct measurement, not a guess:
    # on dSprites at tmax=1 (750 steps/restart previously), the adversarial flip
    # rate declined monotonically across restarts (0.289, 0.285, 0.264, 0.257,
    # 0.250) and only cleared eps_adv=0.25 on the 5th attempt (restart index 4).
    # The trunk is shared and keeps training across restarts -- only the head and
    # Confuser reinitialize -- so a restart is a continuation of the same fight,
    # not an independent retry, and rmax=3 was one short of what was needed here.
    "gpu": Budget(
        steps=8000,
        batch_per_env=256,
        sadl_tmax=8,
        sadl_rmax=6,
        sadl_check_every=20,
        sadl_check_n=512,
        sadl_audit_subsample=16,
        sadl_bank_subsample=12,
        sadl_confuser_steps=3,
        eval_batch=2048,
    ),
}


class Method:
    name: str = "base"
    category: str = ""
    pretraining_labels: str = "none"  # 'none' | 'source' | 'source + env.' | 'env. only'
    repr_kind: str = "continuous"

    def __init__(self, budget: Budget, device: torch.device, seed: int = 0) -> None:
        self.budget = budget
        self.device = device
        self.seed = seed
        self.log: dict[str, Any] = {}

    # ---- to implement -----------------------------------------------------
    def fit(self, ds: MultiEnvDataset) -> dict:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def repr_dim(self) -> int:
        raise NotImplementedError

    # ---- shared helpers ---------------------------------------------------
    def loader(self, ds: MultiEnvDataset, steps: int | None = None) -> EnvBalancedLoader:
        return EnvBalancedLoader(
            [e.fit() for e in ds.train_envs],
            self.budget.batch_per_env,
            self.device,
            seed=self.seed,
            steps=steps or self.budget.steps,
        )

    def modules_(self) -> list[nn.Module]:
        return [m for m in vars(self).values() if isinstance(m, nn.Module)]

    def eval_mode(self) -> None:
        for m in self.modules_():
            m.eval()

    def train_mode(self) -> None:
        for m in self.modules_():
            m.train()

    @torch.no_grad()
    def features(self, x: np.ndarray) -> np.ndarray:
        """Frozen representation for a numpy image batch."""
        self.eval_mode()
        out = []
        bs = self.budget.eval_batch
        for i in range(0, len(x), bs):
            xb = torch.as_tensor(x[i : i + bs]).to(self.device)
            out.append(self.encode(xb).detach().float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.repr_dim), dtype=np.float32)

    def n_params(self) -> int:
        return sum(sum(p.numel() for p in m.parameters()) for m in self.modules_())
