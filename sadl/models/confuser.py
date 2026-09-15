"""The Confuser: constrained structure-preserving perturbations (Section 5.1).

The transformation bank ``T`` contains photometric, textural and
environment-conditional operations.  None of them moves, rescales or reshapes
image content, so the invariant factors of the synthetic generators (shape,
position, size, rotation, digit identity) are preserved by construction, and the
constraint set is enforced by bounding every strength parameter.

Two Confusers are provided:

``FixedBankConfuser``   takes the per-sample worst case over a discrete bank
                        (the ``SADL-lite`` ablation, and the nested families of
                        RQ3).
``LearnedConfuser``     an amortised, spectrally normalised network that emits
                        per-sample mixture weights and strengths over the bank,
                        trained by ascent on the flip objective A_t.

``env_resample`` implements the environment-conditional resampling of nuisance
statistics that Proposition 5.1 idealises: the per-channel first and second
moments of an image are replaced by those of a reference image drawn from a
different environment (an AdaIN-style style swap).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class _SpectralNorm(nn.Module):
    """Spectral normalisation via power iteration.

    Hand-rolled instead of ``torch.nn.utils.parametrizations.spectral_norm``
    because the stock implementation calls ``aten::vdot``, which the MPS backend
    does not implement.
    """

    def __init__(self, weight: torch.Tensor, n_power: int = 1, eps: float = 1e-9) -> None:
        super().__init__()
        rows = weight.shape[0]
        cols = weight.numel() // rows
        self.register_buffer("u", F.normalize(torch.randn(rows), dim=0, eps=eps))
        self.register_buffer("v", F.normalize(torch.randn(cols), dim=0, eps=eps))
        self.n_power = n_power
        self.eps = eps

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        w = weight.reshape(weight.shape[0], -1)
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_power):
                    self.v.copy_(F.normalize(w.t() @ self.u, dim=0, eps=self.eps))
                    self.u.copy_(F.normalize(w @ self.v, dim=0, eps=self.eps))
        sigma = (self.u * (w @ self.v)).sum().clamp_min(self.eps)
        return weight / sigma


def spectral_norm(module: nn.Module) -> nn.Module:
    parametrize.register_parametrization(module, "weight", _SpectralNorm(module.weight))
    return module


@dataclass
class Op:
    name: str
    n_strength: int
    fn: Callable[..., torch.Tensor]
    max_strength: float
    # Discrete settings used by the fixed bank.  ``None`` means "the scalar grid
    # applied to every strength coordinate".
    presets: Callable[[int], list[list[float]]] | None = None


def _brightness(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x + s[:, 0].view(-1, 1, 1, 1)


def _contrast(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return m + (x - m) * (1.0 + s[:, 0].view(-1, 1, 1, 1))


def _channel_gain(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    g = 1.0 + s[:, : x.shape[1]].view(x.shape[0], x.shape[1], 1, 1)
    return x * g


def _channel_bias(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x + s[:, : x.shape[1]].view(x.shape[0], x.shape[1], 1, 1)


def _gamma(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    g = torch.exp(s[:, 0].view(-1, 1, 1, 1))
    return x.clamp_min(1e-4) ** g


def _noise(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x + s[:, 0].view(-1, 1, 1, 1).abs() * torch.randn_like(x)


def _texture(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    coarse = torch.rand(n, 1, 8, 8, device=x.device, dtype=x.dtype)
    tex = F.interpolate(coarse, size=(h, w), mode="nearest") - 0.5
    return x + s[:, 0].view(-1, 1, 1, 1) * 2.0 * tex


def _blur(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([1.0, 2.0, 1.0], device=x.device, dtype=x.dtype)
    k = (k[:, None] * k[None, :]) / 16.0
    k = k.expand(x.shape[1], 1, 3, 3)
    blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])
    a = s[:, 0].view(-1, 1, 1, 1).abs().clamp(0, 1)
    return (1 - a) * x + a * blurred


def _channel_mix(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Linear colour-space rotation: a cheap style-transfer surrogate."""
    n, c = x.shape[0], x.shape[1]
    m = s[:, : c * c].view(n, c, c)
    mat = torch.eye(c, device=x.device, dtype=x.dtype).unsqueeze(0) + m
    return torch.einsum("nij,njhw->nihw", mat, x)


def _soft_occlude(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Low-frequency multiplicative shading field (vignetting / soft occlusion)."""
    n, c, h, w = x.shape
    field = torch.rand(n, 1, 4, 4, device=x.device, dtype=x.dtype)
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False) - 0.5
    return x * (1.0 + s[:, 0].view(-1, 1, 1, 1) * field)


def _env_resample(x: torch.Tensor, s: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Replace per-channel nuisance statistics with those of a reference image."""
    eps = 1e-5
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    sd_x = x.std(dim=(2, 3), keepdim=True) + eps
    mu_r = ref.mean(dim=(2, 3), keepdim=True)
    sd_r = ref.std(dim=(2, 3), keepdim=True) + eps
    swapped = (x - mu_x) / sd_x * sd_r + mu_r
    a = s[:, 0].view(-1, 1, 1, 1).abs().clamp(0, 1)
    return (1 - a) * x + a * swapped


def _per_channel_presets(n: int) -> list[list[float]]:
    """Boost or suppress one channel at a time, plus the two global settings."""
    out = [[1.0] * n, [-1.0] * n]
    for i in range(n):
        v = [0.0] * n
        v[i] = 1.0
        out.append(list(v))
        v[i] = -1.0
        out.append(list(v))
    return out


def _perm_presets(n: int) -> list[list[float]]:
    """Colour-channel permutations, expressed as ``P - I`` for the linear map."""
    import itertools

    out = []
    eye = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for perm in itertools.permutations(range(n)):
        if list(perm) == list(range(n)):
            continue
        p = [[1.0 if perm[i] == j else 0.0 for j in range(n)] for i in range(n)]
        out.append([p[i][j] - eye[i][j] for i in range(n) for j in range(n)])
    # plus two mild global mixes
    out.append([0.3] * (n * n))
    out.append([-0.3] * (n * n))
    return out


def build_bank(n_ch: int, level: str = "learned") -> list[Op]:
    """Nested transformation families for RQ3 (each level contains the previous).

    ``channel_mix`` is bounded by 1, so the constraint set contains full
    colour-channel permutations.  That matters: a distinction that reads the
    colour channel an object occupies is only exposed as unstable if the bank can
    re-assign colour.  Every operation leaves the spatial content untouched, so
    shape, size, position, rotation and digit identity are preserved exactly.

    Strengths of the additive and multiplicative operations are deliberately
    modest.  The objects in dSprites and Causal3DIdent-lite occupy a few percent
    of the frame at moderate contrast, and a brightness or occlusion strength
    large enough to invert local contrast destroys shape, position and size --
    that is outside T, and a bank that leaves T makes every distinction
    unacceptable for the right reason but the wrong cause.  Colour permutation and
    environment-conditional resampling are exact on spatial content and so run at
    full strength; they are also the operations that break colour shortcuts.
    """
    core = [
        Op("brightness", 1, _brightness, 0.15),
        Op("contrast", 1, _contrast, 0.30),
        Op("channel_gain", n_ch, _channel_gain, 0.35, _per_channel_presets),
    ]
    mid = core + [
        Op("channel_bias", n_ch, _channel_bias, 0.12, _per_channel_presets),
        Op("gamma", 1, _gamma, 0.40),
        Op("noise", 1, _noise, 0.08),
        Op("texture", 1, _texture, 0.10),
    ]
    full = mid + [
        Op("blur", 1, _blur, 0.60),
        Op("channel_mix", n_ch * n_ch, _channel_mix, 1.00, _perm_presets),
        Op("soft_occlude", 1, _soft_occlude, 0.30),
    ]
    rich = full + [Op("env_resample", 1, _env_resample, 1.00)]
    # ``fixed11`` is ``rich`` as a *parameter-free* bank.  Proposition 5.1's
    # condition (i) is about conditional nuisance resampling, and ``env_resample``
    # is the operation that models it -- yet it appeared only at the ``learned``
    # level, so the parameter-free auditor could never apply the operation the
    # theory is stated in terms of.  This level exists so that it can.
    return {
        "none": [],
        "fixed3": core,
        "fixed7": mid,
        "fixed10": full,
        "fixed11": rich,
        "learned": rich,
    }[level]


class FixedBankConfuser(nn.Module):
    """Per-sample worst case over a discrete bank; no learned parameters.

    ``max_phi A_t`` is replaced by a maximum over ``bank x strength grid``, which
    is the cheaper, more stable lower-capacity comparison referred to in
    Section 5.7.
    """

    learned = False

    def __init__(self, n_ch: int, level: str = "fixed10", grid: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)) -> None:
        super().__init__()
        self.bank = build_bank(n_ch, level)
        self.grid = grid
        self.level = level

    def specs(self, n_ch: int) -> list[tuple[Op, list[float]]]:
        out: list[tuple[Op, list[float]]] = []
        for op in self.bank:
            settings = (
                op.presets(n_ch)
                if op.presets is not None
                else [[g] * max(1, op.n_strength) for g in self.grid]
            )
            for vec in settings:
                out.append((op, vec))
        return out

    def _transform(self, spec: tuple[Op, list[float]], x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        op, vec = spec
        s = torch.tensor(vec, device=x.device, dtype=x.dtype) * op.max_strength
        return op.fn(x, s.unsqueeze(0).expand(x.shape[0], -1), ref).clamp(0.0, 1.0)

    def candidates(self, x: torch.Tensor, ref: torch.Tensor, subsample: int | None = None) -> list[torch.Tensor]:
        specs = self.specs(x.shape[1])
        if subsample is not None and subsample < len(specs):
            idx = torch.randperm(len(specs))[:subsample].tolist()
            specs = [specs[i] for i in idx]
        return [self._transform(s, x, ref) for s in specs]

    def flip_objective(
        self,
        q_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        ref: torch.Tensor,
        env: torch.Tensor | None = None,
        subsample: int | None = None,
    ) -> torch.Tensor:
        """E_X[max_tau |q(tau(x)) - q(x)|]: the per-sample worst-case flip.

        ``subsample`` evaluates a random subset of the bank per step, which keeps
        the cost of using the full bank inside the training loop bounded.
        """
        if not self.bank:
            return x.new_zeros(())
        q0 = q_fn(x)
        worst = None
        for xc in self.candidates(x, ref, subsample):
            d = (q_fn(xc) - q0).abs()
            worst = d if worst is None else torch.maximum(worst, d)
        return worst.mean()

    @torch.no_grad()
    def flip_rates(
        self,
        bit_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        ref: torch.Tensor,
        env: torch.Tensor | None = None,
        subsample: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two readings of the hard-assignment flip rate.

        ``per_op`` is ``max_tau E_X[1{d(tau(x)) != d(x)}]``, the literal reading of
        Equation 17 under the maximum over the transformation family: the strongest
        single transformation, scored on the population.

        ``per_sample`` is ``E_X[max_tau 1{...}]``, the per-example worst case.  It
        is strictly larger and a much harder bar -- with a bank of ~40 settings,
        almost any head has *some* transformation that flips *some* example.  The
        acceptance test uses ``per_op``; ``per_sample`` is reported as a stricter
        diagnostic.
        """
        if not self.bank:
            z = x.new_zeros(())
            return z, z
        b0 = bit_fn(x)
        worst_sample = torch.zeros_like(b0)
        worst_op = x.new_zeros(())
        for xc in self.candidates(x, ref, subsample):
            flip = (bit_fn(xc) != b0).to(x.dtype)
            worst_sample = torch.maximum(worst_sample, flip)
            worst_op = torch.maximum(worst_op, flip.mean())
        return worst_op, worst_sample.mean()

    @torch.no_grad()
    def flip_rate(self, bit_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        return self.flip_rates(bit_fn, x, ref, env)[0]

    def perturb(self, x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if not self.bank:
            return x
        cands = self.candidates(x, ref, subsample=1)
        return cands[0]


class LearnedConfuser(nn.Module):
    """Amortised mixture over the bank, trained by ascent on A_t.

    Emits per-sample mixture weights ``w(x)`` (softmax over the bank, with a
    learnable temperature-free logit scale) and bounded strengths ``s(x)``.  The
    output is the convex combination ``sum_k w_k(x) op_k(x; s_k(x))``, which stays
    inside the constraint set ``T`` because every operation does.
    """

    learned = True

    def __init__(self, n_ch: int, level: str = "learned", width: int = 24, env_dim: int = 8, n_envs: int = 2) -> None:
        super().__init__()
        self.bank = build_bank(n_ch, level)
        self.level = level
        self.n_ops = len(self.bank)
        self.strength_dims = [max(1, op.n_strength) for op in self.bank]
        n_out = self.n_ops + sum(self.strength_dims)
        self.trunk = nn.Sequential(
            spectral_norm(nn.Conv2d(n_ch, width, 3, 2, 1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(width, 2 * width, 3, 2, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.env_emb = nn.Embedding(max(1, n_envs), env_dim)
        self.head = spectral_norm(nn.Linear(2 * width + env_dim, n_out))
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if not self.bank:
            return x
        h = self.trunk(x)
        e = self.env_emb(env if env is not None else torch.zeros(len(x), dtype=torch.long, device=x.device))
        out = self.head(torch.cat([h, e], 1))
        logits, rest = out[:, : self.n_ops], out[:, self.n_ops :]
        w = torch.softmax(logits, dim=1)
        y = torch.zeros_like(x)
        pos = 0
        for k, op in enumerate(self.bank):
            d = self.strength_dims[k]
            s = torch.tanh(rest[:, pos : pos + d]) * op.max_strength
            pos += d
            y = y + w[:, k].view(-1, 1, 1, 1) * op.fn(x, s, ref).clamp(0.0, 1.0)
        return y.clamp(0.0, 1.0)

    def flip_objective(self, q_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if not self.bank:
            return x.new_zeros(())
        return (q_fn(self.forward(x, ref, env)) - q_fn(x)).abs().mean()

    @torch.no_grad()
    def flip_rates(
        self,
        bit_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        ref: torch.Tensor,
        env: torch.Tensor | None = None,
        subsample: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The learned mixture is a single transformation, so both readings agree."""
        if not self.bank:
            z = x.new_zeros(())
            return z, z
        r = self.flip_rate(bit_fn, x, ref, env)
        return r, r

    @torch.no_grad()
    def flip_rate(self, bit_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        if not self.bank:
            return x.new_zeros(())
        b0 = bit_fn(x)
        return (bit_fn(self.forward(x, ref, env)) != b0).to(x.dtype).mean()

    def perturb(self, x: torch.Tensor, ref: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x, ref, env)


def make_confuser(kind: str, n_ch: int, n_envs: int) -> nn.Module:
    if kind == "none":
        return FixedBankConfuser(n_ch, level="none")
    if kind.startswith("fixed"):
        return FixedBankConfuser(n_ch, level=kind)
    if kind == "learned":
        return LearnedConfuser(n_ch, level="learned", n_envs=n_envs)
    raise KeyError(f"unknown confuser kind {kind!r}")
