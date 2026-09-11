"""Metrics of Section 6.4.

  Acc_worst / Acc_avg          Equation 23
  factor recovery FR           Equation 24 (NMI + maximum-weight matching)
  empirical stability gap      Equation 25
  composability Comp(D)        Equation 22

All representations, continuous or binary, are scored with the same rules:
continuous dimensions are discretised with quantile boundaries estimated on
training data only, and the downstream readout class is identical for every
method.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from ..data import MultiEnvDataset, stability_pairs

READOUT_C_GRID = (0.01, 0.1, 1.0, 10.0)


# ----------------------------------------------------------------- readout
def _fit_logreg(x: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(x, y)
    return clf


def fit_readout(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler, float]:
    """Frozen-feature readout with L2 strength chosen on training-domain validation."""
    scaler = StandardScaler().fit(x_fit)
    xf, xv = scaler.transform(x_fit), scaler.transform(x_val)
    if len(np.unique(y_fit)) < 2:
        raise ValueError("readout needs at least two classes")
    best, best_acc, best_c = None, -1.0, READOUT_C_GRID[0]
    for C in READOUT_C_GRID:
        clf = _fit_logreg(xf, y_fit, C)
        acc = float((clf.predict(xv) == y_val).mean())
        if acc > best_acc:
            best, best_acc, best_c = clf, acc, C
    return best, scaler, best_c


def evaluate_ood(method, ds: MultiEnvDataset) -> dict:
    """Fit the readout on training environments, evaluate on held-out ones."""
    feats_fit = np.concatenate([method.features(e.fit().x) for e in ds.train_envs])
    y_fit = np.concatenate([e.fit().y for e in ds.train_envs])
    feats_val = np.concatenate([method.features(e.val().x) for e in ds.train_envs])
    y_val = np.concatenate([e.val().y for e in ds.train_envs])

    if feats_fit.shape[1] == 0 or np.allclose(feats_fit.std(0), 0):
        # degenerate representation: fall back to the majority-class predictor
        maj = np.bincount(y_fit).argmax()
        per_env, correctness = {}, {}
        for e in ds.test_envs:
            c = (np.full(len(e.y), maj) == e.y).astype(np.uint8)
            per_env[e.env_name] = float(c.mean())
            correctness[e.env_name] = c
        id_acc = float((np.full(len(y_val), maj) == y_val).mean())
    else:
        clf, scaler, C = fit_readout(feats_fit, y_fit, feats_val, y_val)
        id_acc = float((clf.predict(scaler.transform(feats_val)) == y_val).mean())
        per_env, correctness = {}, {}
        for e in ds.test_envs:
            f = scaler.transform(method.features(e.x))
            c = (clf.predict(f) == e.y).astype(np.uint8)
            per_env[e.env_name] = float(c.mean())
            correctness[e.env_name] = c

    # Chance reference: the majority class of the training environments, scored on
    # the held-out environments.  Worst-environment accuracy is not interpretable
    # without it -- a method can look far ahead of every baseline while sitting at
    # chance, and baselines that encode the shortcut land *below* chance because
    # the readout inverts out of distribution.
    maj = int(np.bincount(y_fit).argmax())
    chance = {e.env_name: float((e.y == maj).mean()) for e in ds.test_envs}

    accs = np.array(list(per_env.values()))
    return {
        "acc_per_env": per_env,
        "acc_worst": float(accs.min()),
        "acc_avg": float(accs.mean()),
        "acc_id_val": id_acc,
        "acc_chance_worst": float(min(chance.values())),
        "acc_chance_avg": float(np.mean(list(chance.values()))),
        "correctness": correctness,
    }


# --------------------------------------------------------- factor recovery
def _discretize(r: np.ndarray, ref: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-discretise a representation column using training-data boundaries."""
    uniq = np.unique(ref)
    if len(uniq) <= n_bins:
        return np.searchsorted(np.sort(uniq), r)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, r)


def factor_recovery(
    rep_train: np.ndarray,
    rep_eval: np.ndarray,
    z_eval: np.ndarray,
    factor_names: list[str],
    all_names: list[str],
    fr_bins: dict[str, int],
    n_bins_cont: int = 8,
) -> dict:
    """Equation 24: one-to-one NMI matching between dimensions and true factors."""
    if not factor_names or rep_eval.shape[1] == 0:
        return {"fr": float("nan"), "matrix": [], "matched": {}}
    m = len(factor_names)
    p = rep_eval.shape[1]
    cols = []
    for j in range(p):
        n_bins = min(n_bins_cont, max(2, len(np.unique(rep_train[:, j]))))
        cols.append(_discretize(rep_eval[:, j], rep_train[:, j], n_bins))
    score = np.zeros((p, m))
    for j, col in enumerate(cols):
        for k, fname in enumerate(factor_names):
            z = z_eval[:, all_names.index(fname)]
            score[j, k] = normalized_mutual_info_score(z, col)
    size = max(p, m)
    padded = np.zeros((size, size))
    padded[:p, :m] = score
    rows, cite = linear_sum_assignment(padded, maximize=True)
    matched = {}
    total = 0.0
    for r, c in zip(rows, cite):
        if c < m:
            val = padded[r, c]
            matched[factor_names[c]] = {"dim": int(r) if r < p else None, "nmi": float(val)}
            total += val
    return {"fr": float(total / m), "matrix": score.tolist(), "matched": matched}


# ---------------------------------------------------------- stability gap
def stability_gap(method, ds: MultiEnvDataset, n: int = 2000, seed: int = 0) -> dict:
    """Equation 25 on pairs sharing z_inv with independently resampled z_env.

    Continuous representations are binarised at their training median so that a
    128-dimensional feature vector and a 4-bit code are scored by the same rule.
    """
    if ds.generator is None or not ds.train_env_specs:
        return {"stability_gap": float("nan")}
    rng = np.random.default_rng(seed + 31337)
    xa, xb = stability_pairs(ds.generator, ds.train_env_specs, n, rng)
    ra, rb = method.features(xa), method.features(xb)
    if ra.shape[1] == 0:
        return {"stability_gap": float("nan")}
    ref = np.concatenate([method.features(e.fit().x[:1000]) for e in ds.train_envs])
    active = np.array([len(np.unique(ref[:, j])) > 1 for j in range(ref.shape[1])])
    if not active.any():
        # A constant representation is trivially invariant, so Equation 25 would
        # score it 0.0 -- a perfect stability gap earned by encoding nothing.  A
        # SADL run that accepted no distinction lands here, and averaging that
        # 0.0 into a method's mean turns a capacity failure into its best result.
        # Report it as undefined so it is excluded rather than counted.
        return {
            "stability_gap": float("nan"),
            "stability_gap_per_dim": [],
            "stability_gap_max": float("nan"),
            "stability_gap_degenerate": True,
        }
    if getattr(method, "repr_kind", "continuous") == "binary":
        thr = np.full(ra.shape[1], 0.5)
    else:
        thr = np.median(ref, axis=0)
    ba, bb = (ra > thr), (rb > thr)
    per_dim = (ba != bb).mean(axis=0)
    used = per_dim[active]
    return {
        "stability_gap": float(used.mean()),
        "stability_gap_per_dim": per_dim.tolist(),
        "stability_gap_max": float(used.max()),
        "stability_gap_degenerate": False,
    }


# ---------------------------------------------------------- composability
def _logloss(prob: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    rows = np.array([idx.get(v, -1) for v in y])
    ok = rows >= 0
    p = np.clip(prob[np.arange(len(y))[ok], rows[ok]], 1e-12, 1.0)
    return float(-np.log(p).mean())


def composability(method, ds: MultiEnvDataset) -> dict:
    """Equation 22 with frozen probes on tasks withheld from pretraining."""
    if not ds.aux_task_names or ds.generator is None:
        return {"composability": float("nan"), "per_task": {}}
    x_fit = np.concatenate([e.fit().x for e in ds.train_envs])
    z_fit = np.concatenate([e.fit().z_inv for e in ds.train_envs])
    f_fit = method.features(x_fit)
    test_feats = [method.features(e.x) for e in ds.test_envs]
    test_z = [e.z_inv for e in ds.test_envs]

    oracle_names = [n for n in ds.fr_factor_names]
    per_task: dict[str, dict] = {}
    scores = []
    for task in ds.aux_task_names:
        ti = ds.inv_factor_names.index(task)
        y_fit = z_fit[:, ti]
        classes = np.unique(y_fit)
        if len(classes) < 2:
            continue
        # probe on the frozen representation
        degenerate = f_fit.shape[1] == 0 or np.allclose(f_fit.std(0), 0)
        if not degenerate:
            sc = StandardScaler().fit(f_fit)
            probe = _fit_logreg(sc.transform(f_fit), y_fit, 1.0)
            l_probe = float(
                np.mean([_logloss(probe.predict_proba(sc.transform(f)), z[:, ti], probe.classes_) for f, z in zip(test_feats, test_z)])
            )
        else:
            l_probe = None  # a degenerate representation scores as the constant predictor
        # oracle probe: a readout of the same class on the true generative factors
        oh_fit = _onehot(z_fit, ds.inv_factor_names, oracle_names)
        oracle = _fit_logreg(oh_fit, y_fit, 1.0)
        l_star = float(
            np.mean(
                [
                    _logloss(oracle.predict_proba(_onehot(z, ds.inv_factor_names, oracle_names)), z[:, ti], oracle.classes_)
                    for z in test_z
                ]
            )
        )
        # constant predictor
        prior = np.bincount(y_fit, minlength=int(classes.max()) + 1) / len(y_fit)
        l_base = float(np.mean([-np.log(np.clip(prior[z[:, ti]], 1e-12, 1)).mean() for z in test_z]))
        if l_probe is None:
            l_probe = l_base
        denom = l_base - l_star
        if denom < 1e-2:  # the probe task is not identifiable from the factors
            per_task[task] = {"probe": l_probe, "oracle": l_star, "base": l_base, "normalized": None}
            continue
        norm = (l_probe - l_star) / denom
        per_task[task] = {"probe": l_probe, "oracle": l_star, "base": l_base, "normalized": norm}
        scores.append(norm)
    if not scores:
        return {"composability": float("nan"), "per_task": {}}
    return {"composability": float(1.0 - np.mean(scores)), "per_task": per_task}


def _onehot(z: np.ndarray, all_names: list[str], names: list[str]) -> np.ndarray:
    cols = []
    for n in names:
        v = z[:, all_names.index(n)]
        k = int(v.max()) + 1
        oh = np.zeros((len(v), k), dtype=np.float32)
        oh[np.arange(len(v)), v] = 1.0
        cols.append(oh)
    return np.concatenate(cols, 1) if cols else np.zeros((len(z), 1), dtype=np.float32)
