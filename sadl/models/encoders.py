"""Shared encoder family used by every method (Section 6.3).

All methods -- SADL and baselines alike -- use the same convolutional trunk,
input resolution and readout class so that differences in reported accuracy come
from the objective rather than from capacity.  Normalisation is GroupNorm rather
than BatchNorm: batches are environment-mixed and, for SADL and DANN, partly
adversarially perturbed, which makes batch statistics an information leak.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int = 128, width: int = 32) -> None:
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1),
            nn.GroupNorm(8, w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, 2 * w, 3, 2, 1),
            nn.GroupNorm(8, 2 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1),
            nn.GroupNorm(8, 4 * w),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 4 * w, 3, 1, 1),
            nn.GroupNorm(8, 4 * w),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * w, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.fc(self.pool(h).flatten(1))


class ConvDecoder(nn.Module):
    """Mirror decoder for MAE and the VAE baselines."""

    def __init__(self, out_ch: int, out_res: int, latent_dim: int, width: int = 32) -> None:
        super().__init__()
        w = width
        self.start = 4 if out_res % 8 == 0 else 7  # 64 -> 4x4, 28 -> 7x7
        self.n_up = 4 if out_res // self.start == 16 else 2
        self.fc = nn.Linear(latent_dim, 4 * w * self.start * self.start)
        self.w = w
        layers: list[nn.Module] = []
        ch = 4 * w
        for i in range(self.n_up):
            nxt = max(w, ch // 2)
            layers += [
                nn.ConvTranspose2d(ch, nxt, 4, 2, 1),
                nn.GroupNorm(8, nxt),
                nn.ReLU(inplace=True),
            ]
            ch = nxt
        layers += [nn.Conv2d(ch, out_ch, 3, 1, 1)]
        self.net = nn.Sequential(*layers)
        self.out_res = out_res

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 4 * self.w, self.start, self.start)
        x = self.net(h)
        if x.shape[-1] != self.out_res:
            x = F.interpolate(x, size=(self.out_res, self.out_res), mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)
