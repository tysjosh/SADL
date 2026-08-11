"""Separator heads and the straight-through relaxation of Equation 14."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import MLP


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0, hard: bool = True, noisy: bool = True) -> torch.Tensor:
    """Binary concrete relaxation with a straight-through hard forward pass.

    ``noisy=False`` recovers the deterministic sigmoid used at evaluation time, so
    the accepted distinction is exactly the hard decision of Equation 14.
    """
    if noisy:
        u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
        logits = logits + torch.log(u) - torch.log1p(-u)
    q = torch.sigmoid(logits / tau)
    if not hard:
        return q
    hard_q = (q > 0.5).to(q.dtype)
    return hard_q + q - q.detach()


class SeparatorHead(nn.Module):
    """s_{theta,t}: X x {0,1}^{t-1} -> R (Equation 13).

    The raw score is standardised by a running mean and variance before the
    sigmoid, and then squashed to ``[-cap, cap]`` by ``cap * tanh(.)``.  Both
    steps are about the geometry of the minimax game, not about the definition of
    a distinction:

    * Standardisation removes the two ways a Separator can satisfy the stability
      penalties without being stable.  Shrinking the logit toward zero drives
      ``|q(tau(x)) - q(x)|`` to zero for every perturbation, and pushing the logit
      far to one side saturates the sigmoid to the same effect.  Both produce a
      constant hard distinction, and in an un-normalised parameterisation they are
      the paths gradient descent takes.  After standardisation the hard decision is
      "score above its own running mean", so a constant distinction is not
      reachable and the penalties can only be reduced by genuine invariance.
    * The ``tanh`` cap keeps ``dq/dx`` non-zero, since a saturated head leaves the
      Confuser no gradient to ascend.

    In evaluation mode the running statistics are used, so the accepted
    distinction is a deterministic, batch-independent function of a single input.
    """

    def __init__(self, feat_dim: int, n_prev: int, hidden: int = 128, logit_cap: float = 4.0) -> None:
        super().__init__()
        self.n_prev = n_prev
        self.logit_cap = logit_cap
        self.net = MLP(feat_dim + n_prev, 1, hidden=hidden, depth=2)
        self.norm = nn.BatchNorm1d(1, affine=False, momentum=0.05)

    def score(self, feat: torch.Tensor, prev_bits: torch.Tensor | None = None) -> torch.Tensor:
        """Bounded pre-normalisation score."""
        if self.n_prev:
            assert prev_bits is not None and prev_bits.shape[1] == self.n_prev
            feat = torch.cat([feat, prev_bits], dim=1)
        raw = self.net(feat).squeeze(-1)
        cap = self.logit_cap
        if cap and cap > 0:
            # Cap *before* normalising: an uncapped score drifts faster than the
            # running statistics track it, which silently turns the evaluation-mode
            # distinction constant even while the training-mode one looks healthy.
            raw = cap * torch.tanh(raw / cap)
        return raw

    def forward(self, feat: torch.Tensor, prev_bits: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.score(feat, prev_bits).unsqueeze(-1)
        if self.training and len(raw) < 2:  # BatchNorm needs a batch
            self.norm.eval()
            s = self.norm(raw).squeeze(-1)
            self.norm.train()
        else:
            s = self.norm(raw).squeeze(-1)
        cap = self.logit_cap
        if cap and cap > 0:
            s = cap * torch.tanh(s / cap)
        return s


class FrozenDistinction(nn.Module):
    """An accepted distinction: a snapshot of the trunk plus its head.

    Snapshotting the trunk is what makes ``D_{t-1}`` genuinely frozen while the
    live trunk keeps training for step ``t`` (Section 5.3).
    """

    def __init__(self, trunk: nn.Module, head: SeparatorHead) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, prev_bits: torch.Tensor | None = None) -> torch.Tensor:
        logit = self.head(self.trunk(x), prev_bits)
        return (logit > 0).to(x.dtype)


class LinearReadout(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
