"""Label-consuming baselines: ERM, IRMv1, VREx, DANN.

These consume source-task labels (and, except for ERM, environment identifiers)
during representation learning, which Table 2 records as an advantage over the
task-agnostic methods.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import MultiEnvDataset
from ..models import ConvEncoder, MLP, grad_reverse
from .base import Method


class ERM(Method):
    name = "ERM"
    category = "pooled risk minimization"
    pretraining_labels = "source"

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed)
        self._built = False

    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        self.encoder = ConvEncoder(ds.in_shape[0], b.feat_dim, b.width).to(self.device)
        self.head = nn.Linear(b.feat_dim, ds.n_classes).to(self.device)
        self._built = True

    @property
    def repr_dim(self) -> int:
        return self.budget.feat_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def penalty(self, logits: torch.Tensor, y: torch.Tensor, env: torch.Tensor, n_envs: int) -> torch.Tensor:
        return logits.new_zeros(())

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()), lr=b.lr, weight_decay=b.weight_decay
        )
        self.train_mode()
        losses = []
        for step, (x, y, e) in enumerate(self.loader(ds)):
            logits = self.head(self.encoder(x))
            loss = F.cross_entropy(logits, y)
            pen = self.penalty(logits, y, e, ds.n_train_envs)
            lam = self._lambda(step)
            total = loss + lam * pen
            if lam > 1.0:
                total = total / lam
            opt.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.head.parameters()), 5.0
            )
            opt.step()
            losses.append(float(loss.detach()))
        self.log["train_loss"] = float(sum(losses[-50:]) / max(1, len(losses[-50:])))
        return self.log

    def _lambda(self, step: int) -> float:
        return 0.0


class IRMv1(ERM):
    name = "IRMv1"
    category = "invariant"
    pretraining_labels = "source + env."

    def _lambda(self, step: int) -> float:
        warm = int(self.budget.irm_anneal_frac * self.budget.steps)
        return self.budget.irm_lambda if step >= warm else 1.0

    def penalty(self, logits: torch.Tensor, y: torch.Tensor, env: torch.Tensor, n_envs: int) -> torch.Tensor:
        """IRMv1 gradient penalty w.r.t. a dummy scale, averaged over environments."""
        total = logits.new_zeros(())
        for e in range(n_envs):
            m = env == e
            if m.sum() < 2:
                continue
            scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
            loss = F.cross_entropy(logits[m] * scale, y[m])
            g = torch.autograd.grad(loss, [scale], create_graph=True)[0]
            total = total + (g**2).sum()
        return total / max(1, n_envs)


class VREx(ERM):
    name = "VREx"
    category = "invariant"
    pretraining_labels = "source + env."

    def _lambda(self, step: int) -> float:
        warm = int(self.budget.irm_anneal_frac * self.budget.steps)
        return self.budget.vrex_lambda if step >= warm else 1.0

    def penalty(self, logits: torch.Tensor, y: torch.Tensor, env: torch.Tensor, n_envs: int) -> torch.Tensor:
        risks = []
        for e in range(n_envs):
            m = env == e
            if m.sum() < 1:
                continue
            risks.append(F.cross_entropy(logits[m], y[m]))
        if len(risks) < 2:
            return logits.new_zeros(())
        r = torch.stack(risks)
        return ((r - r.mean()) ** 2).mean()


class DANN(Method):
    name = "DANN"
    category = "domain adversarial"
    pretraining_labels = "source + env."

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed)
        self._built = False

    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        self.encoder = ConvEncoder(ds.in_shape[0], b.feat_dim, b.width).to(self.device)
        self.head = nn.Linear(b.feat_dim, ds.n_classes).to(self.device)
        self.disc = MLP(b.feat_dim, ds.n_train_envs, hidden=128, depth=2).to(self.device)
        self._built = True

    @property
    def repr_dim(self) -> int:
        return self.budget.feat_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        params = list(self.encoder.parameters()) + list(self.head.parameters()) + list(self.disc.parameters())
        opt = torch.optim.Adam(params, lr=b.lr, weight_decay=b.weight_decay)
        self.train_mode()
        for step, (x, y, e) in enumerate(self.loader(ds)):
            p = step / max(1, b.steps)
            lam = b.dann_lambda * (2.0 / (1.0 + float(torch.exp(torch.tensor(-10.0 * p)))) - 1.0)
            f = self.encoder(x)
            loss_cls = F.cross_entropy(self.head(f), y)
            loss_dom = F.cross_entropy(self.disc(grad_reverse(f, lam)), e)
            opt.zero_grad(set_to_none=True)
            (loss_cls + loss_dom).backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        self.log["final_domain_loss"] = float(loss_dom.detach())
        return self.log
