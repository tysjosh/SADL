"""Evaluate a method with its own classifier, not just a refit frozen readout.

Section 6.3 measures every method by fitting one logistic readout on frozen
features, so that differences come from the objective rather than from readout
capacity.  That is the right protocol for representation learners -- SimCLR, MAE,
the VAEs, SADL -- whose entire output *is* a representation.  It quietly destroys
the mechanism of the invariance methods.

IRM does not remove a shortcut from the features; it finds a *classifier* whose
gradient is simultaneously optimal across environments.  Discard that classifier,
refit an ERM readout on the same frozen features pooled over training
environments, and the shortcut is simply relearned.  VREx is affected the same
way, its penalty acting on the joint model's per-environment risks.  DANN largely
is not: its invariance is enforced on the features themselves.

The measured consequence on ColoredMNIST is a 57-point discrepancy with the
literature.  This harness reports IRMv1 at 10.0% worst-environment accuracy --
below the 49.3% chance line and worse than ERM -- where Arjovsky et al. report IRM
at 66.9%, ERM at 17.1% and a grayscale oracle at 73.0%.  Worse, the confirmatory
comparison selects its reference baseline by in-distribution validation accuracy,
which picked IRMv1 on ColoredMNIST: the headline margin was measured against the
one method the protocol most disadvantages.

So both numbers are reported.  The frozen-readout column stays primary, because it
is the preregistered protocol and the only one SADL can be scored under.  The
own-classifier column is what the source papers report, and it is the one to cite
when comparing against them.

This module lives under ``experiments/`` rather than ``eval/`` on purpose.  It only
*adds* fields to a record and touches no existing metric, so a cached record's
``acc_worst``, ``fr``, ``stability_gap`` and ``composability`` remain exactly
reproducible.  Placing it here keeps the change in the ``driver`` provenance group,
which ``sadl/provenance.py`` correctly treats as non-material, instead of marking
every previously computed record incomparable for a change that cannot affect it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ..data import MultiEnvDataset


def own_classifier(method, ds: MultiEnvDataset) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """The method's own class-logit function, or None if it has no classifier.

    Detected structurally rather than by method name: an ``encoder`` plus a ``head``
    projecting to ``n_classes``.  ERM, IRMv1, VREx and DANN match; the
    representation learners and SADL do not, and must not -- scoring SADL with a
    head it never trained would be meaningless.
    """
    enc = getattr(method, "encoder", None)
    head = getattr(method, "head", None)
    if not isinstance(enc, nn.Module) or not isinstance(head, nn.Module):
        return None
    out_features = getattr(head, "out_features", None)
    if out_features != ds.n_classes:
        return None

    def logits(x: torch.Tensor) -> torch.Tensor:
        return head(enc(x))

    return logits


@torch.no_grad()
def _predict(method, fn: Callable[[torch.Tensor], torch.Tensor], x: np.ndarray) -> np.ndarray:
    method.eval_mode()
    out = []
    bs = method.budget.eval_batch
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i : i + bs]).to(method.device)
        out.append(fn(xb).argmax(1).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def own_classifier_metrics(method, ds: MultiEnvDataset) -> dict:
    """Worst / average / in-distribution accuracy under the method's own classifier.

    Returns ``{}`` for methods without one, so the fields are simply absent from
    those records rather than present and meaningless.  Keys are suffixed ``_own``
    and the per-instance correctness vectors come back under ``correctness_own``
    for the hierarchical bootstrap.
    """
    fn = own_classifier(method, ds)
    if fn is None:
        return {}

    per_env: dict[str, float] = {}
    correctness: dict[str, np.ndarray] = {}
    for e in ds.test_envs:
        c = (_predict(method, fn, e.x) == e.y).astype(np.uint8)
        per_env[e.env_name] = float(c.mean())
        correctness[e.env_name] = c

    y_val = np.concatenate([e.val().y for e in ds.train_envs])
    pred_val = np.concatenate([_predict(method, fn, e.val().x) for e in ds.train_envs])
    accs = np.array(list(per_env.values()))
    return {
        "acc_per_env_own": per_env,
        "acc_worst_own": float(accs.min()),
        "acc_avg_own": float(accs.mean()),
        "acc_id_val_own": float((pred_val == y_val).mean()),
        "correctness_own": correctness,
    }
