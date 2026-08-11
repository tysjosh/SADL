"""Method registry: the rows of Table 2 plus the ablations of Table 6."""
from __future__ import annotations

from .base import BUDGET_PRESETS, Budget, Method
from .sadl import SADL, SADLJoint, SADLLite
from .selfsup import MAE, BetaVAE, FactorVAE, SimCLR
from .supervised import DANN, ERM, IRMv1, VREx

_BUILDERS: dict[str, callable] = {
    "ERM": lambda b, d, s: ERM(b, d, s),
    "SimCLR": lambda b, d, s: SimCLR(b, d, s),
    "MAE": lambda b, d, s: MAE(b, d, s),
    "IRMv1": lambda b, d, s: IRMv1(b, d, s),
    "VREx": lambda b, d, s: VREx(b, d, s),
    "beta-VAE": lambda b, d, s: BetaVAE(b, d, s),
    "FactorVAE": lambda b, d, s: FactorVAE(b, d, s),
    "DANN": lambda b, d, s: DANN(b, d, s),
    "SADL-lite": lambda b, d, s: SADLLite(b, d, s),
    "SADL": lambda b, d, s: SADL(b, d, s),
}

# Ablations (Table 6) and Confuser families (Table 7).
_ABLATIONS: dict[str, dict] = {
    "SADL-no-gate": {"base": "SADL", "budget": {"sadl_compression_gate": False}},
    "SADL-warmstart": {"base": "SADL", "budget": {"sadl_warmstart_steps": 600}},
    "SADL-joint": {"base": "SADL-joint", "budget": {}},
    "SADL-T1": {"base": "SADL", "budget": {"sadl_tmax": 1}},
    "SADL-T8": {"base": "SADL", "budget": {"sadl_tmax": 8}},
    # RQ3: nested Confuser families.  The acceptance audit uses the same family
    # under test, so each row reflects that family's capacity end to end.
    "SADL-conf-none": {
        "base": "SADL",
        "budget": {"sadl_confuser": "none", "sadl_lambda_adv": 0.0, "sadl_audit_bank": "none"},
    },
    "SADL-conf-fixed3": {"base": "SADL", "budget": {"sadl_confuser": "fixed3", "sadl_audit_bank": "fixed3"}},
    "SADL-conf-fixed10": {"base": "SADL", "budget": {"sadl_confuser": "fixed10", "sadl_audit_bank": "fixed10"}},
    "SADL-conf-learned": {"base": "SADL", "budget": {"sadl_confuser": "learned", "sadl_audit_bank": "fixed10"}},
}

TABLE2_METHODS = [
    "ERM",
    "SimCLR",
    "MAE",
    "IRMv1",
    "VREx",
    "beta-VAE",
    "FactorVAE",
    "DANN",
    "SADL-lite",
    "SADL",
]
BASELINES = [m for m in TABLE2_METHODS if not m.startswith("SADL")]
ALL_METHODS = TABLE2_METHODS + list(_ABLATIONS)


def build_method(name: str, budget: Budget, device, seed: int = 0) -> Method:
    if name in _ABLATIONS:
        spec = _ABLATIONS[name]
        budget = budget.with_(**spec["budget"])
        base = spec["base"]
        if base == "SADL-joint":
            return SADLJoint(budget, device, seed)
        return _BUILDERS[base](budget, device, seed)
    if name not in _BUILDERS:
        raise KeyError(f"unknown method {name!r}; known: {ALL_METHODS}")
    return _BUILDERS[name](budget, device, seed)


def method_meta(name: str) -> dict:
    spec = _ABLATIONS.get(name)
    base = spec["base"] if spec else name
    cls_map = {
        "ERM": ERM,
        "SimCLR": SimCLR,
        "MAE": MAE,
        "IRMv1": IRMv1,
        "VREx": VREx,
        "beta-VAE": BetaVAE,
        "FactorVAE": FactorVAE,
        "DANN": DANN,
        "SADL-lite": SADLLite,
        "SADL": SADL,
        "SADL-joint": SADLJoint,
    }
    cls = cls_map[base]
    return {
        "name": name,
        "category": cls.category,
        "pretraining_labels": cls.pretraining_labels,
        "repr_kind": cls.repr_kind,
    }


__all__ = [
    "ALL_METHODS",
    "BASELINES",
    "BUDGET_PRESETS",
    "Budget",
    "Method",
    "TABLE2_METHODS",
    "build_method",
    "method_meta",
]
