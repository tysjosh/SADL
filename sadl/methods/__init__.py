"""Method registry: the rows of Table 2 plus the ablations of Table 6."""
from __future__ import annotations

from .base import BUDGET_PRESETS, Budget, Method
from .sadl import SADL, SADLJoint, SADLLite
from .selfsup import MAE, BetaVAE, FactorVAE, SimCLR, SimCLRBank
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
    # The control for every SADL claim: same transformation family, none of the
    # algorithm.  Not part of Table 2; reported alongside it.
    "SimCLR-bank": lambda b, d, s: SimCLRBank(b, d, s),
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
    # ---- v2: one entry per correction, plus the combination ------------------
    # Each targets a failure observed in the v1 sweep, and each binds on a
    # different dataset, so the single-fix rows are informative on their own.
    #
    # fix1 -- marginal-preserving Confuser.  Targets the novelty ceiling, which
    # binds where the *learned* adversary is the stronger one: ColoredMNIST, where
    # the gate read learned 0.459 against bank 0.146.
    "SADL-v2-fix1": {"base": "SADL", "budget": {"sadl_conf_marginal": 5.0}},
    # fix2 -- label-free sufficiency term.  Targets composability 0.228 and factor
    # recovery 0.006, which fail on every dataset because Equation 18 has no
    # sufficiency term while Theorem 4.1 requires one.
    "SADL-v2-fix2": {"base": "SADL", "budget": {"sadl_lambda_suf": 1.0}},
    # fix3 -- coverage-complete audit.  Targets the stability gaps of 0.271 and
    # 0.247, which occur where the *bank* is the binding adversary (dSprites and
    # Causal3DIdent, bank in ~9 of 10 trace rows) and the bank lacks env_resample.
    "SADL-v2-fix3": {
        "base": "SADL",
        "budget": {"sadl_audit_bank": "fixed11", "sadl_audit_cross_env": True},
    },
    "SADL-v2": {
        "base": "SADL",
        "budget": {
            "sadl_conf_marginal": 5.0,
            "sadl_lambda_suf": 1.0,
            "sadl_audit_bank": "fixed11",
            "sadl_audit_cross_env": True,
        },
    },
    # Leave-one-out from the full v2, for the ablation table.
    "SADL-v2-no-fix1": {
        "base": "SADL",
        "budget": {"sadl_lambda_suf": 1.0, "sadl_audit_bank": "fixed11", "sadl_audit_cross_env": True},
    },
    "SADL-v2-no-fix2": {
        "base": "SADL",
        "budget": {"sadl_conf_marginal": 5.0, "sadl_audit_bank": "fixed11", "sadl_audit_cross_env": True},
    },
    "SADL-v2-no-fix3": {"base": "SADL", "budget": {"sadl_conf_marginal": 5.0, "sadl_lambda_suf": 1.0}},
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
# Controls: not methods under test and not ablations of one, but the comparisons
# a claim has to survive.  Kept separate so they never enter Table 2 by accident.
CONTROL_METHODS = ["SimCLR-bank"]
V2_VARIANTS = [m for m in _ABLATIONS if m.startswith("SADL-v2")]
ALL_METHODS = TABLE2_METHODS + list(_ABLATIONS) + CONTROL_METHODS


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
        "SimCLR-bank": SimCLRBank,
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
