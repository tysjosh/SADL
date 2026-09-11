"""Computational cost instrumentation for Table 10.

Section 6.4 asks for "wall-clock time, peak accelerator memory, and
forward/backward evaluations relative to ERM", and Section 5.7 additionally for
accepted distinctions per accelerator-hour.  Time and memory are recorded
directly by the runner; the third quantity has to be counted while the method
trains, which is what this module does.

``forward/backward evaluations`` is operationalised as evaluations of the shared
convolutional trunk, counted on the class rather than on an instance:

  * every method in Table 2 builds exactly one ``ConvEncoder`` (SADL calls it
    ``trunk``, the baselines call it ``encoder``), so the trunk is the one unit of
    work all methods have in common and the only one whose count is comparable
    across objectives;
  * SADL's accepted distinctions hold ``deepcopy``s of the trunk that are created
    *during* ``fit``, and its flip objective re-evaluates the trunk once per
    transformation in the bank.  Patching the class catches both; hooks registered
    on the instance built at the start of ``fit`` would miss the copies and so
    would understate SADL's cost by roughly the size of the bank.

A step is one environment-balanced minibatch drawn from ``EnvBalancedLoader``,
which is how every method advances its optimiser.  The count covers everything
inside ``fit``, including SADL's periodic acceptance audits and its label-free
compression probe: those are part of Algorithm 1 and they cost what they cost.
Evaluation after ``fit`` is outside the counter, so readout fitting and the
metrics are excluded from every method equally.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CostCounts:
    encoder_forward: int = 0
    encoder_backward: int = 0
    train_steps: int = 0

    @property
    def evals_per_step(self) -> float:
        """Trunk forward + backward evaluations per environment-balanced minibatch."""
        if self.train_steps <= 0:
            return float("nan")
        return (self.encoder_forward + self.encoder_backward) / self.train_steps

    def as_dict(self) -> dict:
        return {
            "n_encoder_forward": self.encoder_forward,
            "n_encoder_backward": self.encoder_backward,
            "n_train_steps": self.train_steps,
            "evals_per_step": self.evals_per_step,
        }


class count_encoder_evals:
    """Context manager counting trunk evaluations and training steps during ``fit``.

    Patches ``ConvEncoder.forward`` and ``EnvBalancedLoader.__iter__`` for the
    duration of the block and restores both on exit, including on exception.  The
    imports are deferred to ``__enter__`` so that ``sadl.eval`` does not have to
    import the model and data packages at module load.
    """

    def __init__(self) -> None:
        self.counts = CostCounts()
        self._patched: list[tuple[type, str, object]] = []

    def __enter__(self) -> CostCounts:
        from ..data.base import EnvBalancedLoader
        from ..models.encoders import ConvEncoder

        counts = self.counts
        orig_forward = ConvEncoder.forward
        orig_iter = EnvBalancedLoader.__iter__
        self._patched = [
            (ConvEncoder, "forward", orig_forward),
            (EnvBalancedLoader, "__iter__", orig_iter),
        ]

        def _count_backward(grad: torch.Tensor) -> None:
            # A tensor hook that leaves the gradient untouched; returning None
            # means "unchanged" rather than "zero".
            counts.encoder_backward += 1
            return None

        def forward(module, x: torch.Tensor) -> torch.Tensor:
            counts.encoder_forward += 1
            out = orig_forward(module, x)
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(_count_backward)
            return out

        def __iter__(loader):
            for batch in orig_iter(loader):
                counts.train_steps += 1
                yield batch

        ConvEncoder.forward = forward  # type: ignore[method-assign]
        EnvBalancedLoader.__iter__ = __iter__  # type: ignore[method-assign]
        return counts

    def __exit__(self, *exc: object) -> None:
        for cls, name, original in self._patched:
            setattr(cls, name, original)
        self._patched = []


def memory_is_peak(device: str | None) -> bool:
    """Whether a recorded ``peak_mem_mb`` is genuinely a peak.

    ``torch.cuda.max_memory_allocated`` is a true high-water mark and is reset
    before each fit.  MPS exposes no peak counter in torch 2.4, so the recorded
    value is the *currently* allocated bytes read after ``fit`` returned, which is
    not a cost measurement; CPU reports nothing at all.  Table 10 renders the
    memory column only for records where this is true, rather than presenting a
    residual allocation as a peak.
    """
    return bool(device) and str(device).startswith("cuda")
