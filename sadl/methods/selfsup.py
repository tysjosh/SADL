"""Task-agnostic baselines: SimCLR, MAE, beta-VAE, FactorVAE.

None of these sees a task label or an environment identifier during
pretraining, matching the "none" row of Table 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import MultiEnvDataset
from ..models import ConvDecoder, ConvEncoder, MLP
from .base import Method


def simclr_augment(x: torch.Tensor) -> torch.Tensor:
    """A generic augmentation policy: crop/resize, flip, mild photometric jitter.

    Deliberately generic.  A policy tuned to erase the shortcut (for example
    randomising the colour channel on ColoredMNIST) would encode the answer into
    the augmentation, which is exactly the dependence on augmentation design that
    Section 2.1 identifies.
    """
    n, c, h, w = x.shape
    # random resized crop (scale 0.7-1.0), implemented with affine grid sampling
    scale = 0.7 + 0.3 * torch.rand(n, device=x.device)
    tx = (1 - scale) * (2 * torch.rand(n, device=x.device) - 1)
    ty = (1 - scale) * (2 * torch.rand(n, device=x.device) - 1)
    flip = torch.where(torch.rand(n, device=x.device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = scale * flip
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (n, c, h, w), align_corners=False)
    # zeros padding: MPS does not implement border padding for grid_sample
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    # mild photometric jitter
    gain = 1.0 + 0.2 * (2 * torch.rand(n, c, 1, 1, device=x.device) - 1)
    bias = 0.1 * (2 * torch.rand(n, 1, 1, 1, device=x.device) - 1)
    return (out * gain + bias).clamp(0.0, 1.0)


class SimCLR(Method):
    name = "SimCLR"
    category = "contrastive"
    pretraining_labels = "none"

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed)
        self._built = False

    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        self.encoder = ConvEncoder(ds.in_shape[0], b.feat_dim, b.width).to(self.device)
        self.proj = MLP(b.feat_dim, 64, hidden=128, depth=1).to(self.device)
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
        params = list(self.encoder.parameters()) + list(self.proj.parameters())
        opt = torch.optim.Adam(params, lr=b.lr, weight_decay=b.weight_decay)
        self.train_mode()
        for x, _, _ in self.loader(ds):
            z1 = F.normalize(self.proj(self.encoder(simclr_augment(x))), dim=1)
            z2 = F.normalize(self.proj(self.encoder(simclr_augment(x))), dim=1)
            z = torch.cat([z1, z2])
            sim = z @ z.t() / b.simclr_temp
            n = len(z1)
            sim.fill_diagonal_(-1e4)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(x.device)
            loss = F.cross_entropy(sim, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        self.log["final_loss"] = float(loss.detach())
        return self.log


class MAE(Method):
    name = "MAE"
    category = "reconstructive"
    pretraining_labels = "none"

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed)
        self._built = False

    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        c, h, _ = ds.in_shape
        self.encoder = ConvEncoder(c, b.feat_dim, b.width).to(self.device)
        self.decoder = ConvDecoder(c, h, b.feat_dim, b.width).to(self.device)
        self.patch = 7 if h % 7 == 0 else 8
        self._built = True

    @property
    def repr_dim(self) -> int:
        return self.budget.feat_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch
        m = (torch.rand(n, 1, h // p, w // p, device=x.device) > self.budget.mae_mask_ratio).to(x.dtype)
        return F.interpolate(m, size=(h, w), mode="nearest")

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=b.lr, weight_decay=b.weight_decay)
        self.train_mode()
        for x, _, _ in self.loader(ds):
            m = self._mask(x)
            rec = self.decoder(self.encoder(x * m))
            loss = (((rec - x) ** 2) * (1 - m)).sum() / (1 - m).sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        self.log["final_loss"] = float(loss.detach())
        return self.log


class BetaVAE(Method):
    name = "beta-VAE"
    category = "disentanglement"
    pretraining_labels = "none"

    def __init__(self, budget, device, seed=0) -> None:
        super().__init__(budget, device, seed)
        self._built = False

    def _build(self, ds: MultiEnvDataset) -> None:
        b = self.budget
        c, h, _ = ds.in_shape
        self.encoder = ConvEncoder(c, 2 * b.latent_dim, b.width).to(self.device)
        self.decoder = ConvDecoder(c, h, b.latent_dim, b.width).to(self.device)
        self._built = True

    @property
    def repr_dim(self) -> int:
        return self.budget.latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)[:, : self.budget.latent_dim]

    def _posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        d = self.budget.latent_dim
        return h[:, :d], h[:, d:].clamp(-8, 8)

    def _elbo(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._posterior(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = self.decoder(z)
        rec_loss = F.binary_cross_entropy(rec.clamp(1e-5, 1 - 1e-5), x.clamp(0, 1), reduction="none").sum((1, 2, 3)).mean()
        kl = (-0.5 * (1 + logvar - mu**2 - logvar.exp())).sum(1).mean()
        return rec_loss, kl, z

    def extra_loss(self, z: torch.Tensor, opt_aux: torch.optim.Optimizer | None) -> torch.Tensor:
        return z.new_zeros(())

    def fit(self, ds: MultiEnvDataset) -> dict:
        if not self._built:
            self._build(ds)
        b = self.budget
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=b.lr, weight_decay=b.weight_decay)
        opt_aux = self._aux_optimizer()
        self.train_mode()
        for x, _, _ in self.loader(ds):
            rec, kl, z = self._elbo(x)
            loss = rec + b.beta_vae_beta * kl + self.extra_loss(z, opt_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            self._aux_step(x, opt_aux)
        self.log["final_rec"] = float(rec.detach())
        self.log["final_kl"] = float(kl.detach())
        return self.log

    def _aux_optimizer(self) -> torch.optim.Optimizer | None:
        return None

    def _aux_step(self, x: torch.Tensor, opt_aux: torch.optim.Optimizer | None) -> None:
        return None


class FactorVAE(BetaVAE):
    name = "FactorVAE"
    category = "disentanglement"
    pretraining_labels = "none"

    def _build(self, ds: MultiEnvDataset) -> None:
        super()._build(ds)
        self.tc_disc = MLP(self.budget.latent_dim, 2, hidden=128, depth=2).to(self.device)

    def _aux_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.tc_disc.parameters(), lr=self.budget.lr, betas=(0.5, 0.9))

    def extra_loss(self, z: torch.Tensor, opt_aux) -> torch.Tensor:
        logits = self.tc_disc(z)
        tc = (logits[:, 0] - logits[:, 1]).mean()
        return self.budget.factorvae_gamma * tc

    def _aux_step(self, x: torch.Tensor, opt_aux) -> None:
        with torch.no_grad():
            mu, logvar = self._posterior(x)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            z_perm = torch.stack([z[torch.randperm(len(z), device=z.device), j] for j in range(z.shape[1])], 1)
        d_z = self.tc_disc(z)
        d_p = self.tc_disc(z_perm)
        ones = torch.ones(len(z), dtype=torch.long, device=z.device)
        loss_d = 0.5 * (F.cross_entropy(d_z, ones * 0) + F.cross_entropy(d_p, ones))
        opt_aux.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_aux.step()
