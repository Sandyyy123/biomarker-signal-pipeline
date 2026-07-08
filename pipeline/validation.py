"""Validation utilities: subject-wise (LOSO) vs row-wise, Bland-Altman, bootstrap CIs.

This module is the whole argument of the repo: it shows, on the SAME data, that
row-wise cross-validation reports an optimistic score and leave-one-subject-out
reports the honest one.
"""

from __future__ import annotations

import numpy as np


# ---- agreement statistics -------------------------------------------------

def bland_altman(measured: np.ndarray, reference: np.ndarray) -> dict:
    """Bland-Altman bias and 95% limits of agreement."""
    measured, reference = np.asarray(measured), np.asarray(reference)
    diff = measured - reference
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return {
        "bias": bias,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "sd_diff": sd,
    }


def icc_2way(measured: np.ndarray, reference: np.ndarray) -> float:
    """ICC(2,1) consistency between measured and reference — a standard
    agreement metric for a measure vs its reference standard."""
    data = np.vstack([np.asarray(measured), np.asarray(reference)]).T
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = ((data - data.mean(axis=1, keepdims=True)
              - data.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_err = resid / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom != 0 else float("nan")


# ---- classification metrics ----------------------------------------------

def sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(y_true)
    return {"sensitivity": sens, "specificity": spec, "accuracy": acc}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap 95% CI for the mean of a metric vector."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = [rng.choice(values, len(values), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---- cross-validation schemes --------------------------------------------

def _fit_predict(Xtr, ytr, Xte, l2: float = 1.0):
    """Tiny logistic-regression via gradient descent (no sklearn dependency),
    standardised features + L2. Deliberately low-capacity for small n."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    w = np.zeros(Xtr.shape[1])
    for _ in range(500):
        p = 1 / (1 + np.exp(-Xtr @ w))
        grad = Xtr.T @ (p - ytr) / len(ytr)
        grad[1:] += l2 * w[1:] / len(ytr)
        w -= 0.3 * grad
    return (1 / (1 + np.exp(-Xte @ w)) >= 0.5).astype(int)


def loso_scores(X, y, sid, l2: float = 1.0):
    """Leave-one-subject-out: honest, generalisation-facing accuracy."""
    accs = []
    for s in np.unique(sid):
        tr, te = sid != s, sid == s
        pred = _fit_predict(X[tr], y[tr], X[te], l2)
        accs.append(sens_spec(y[te], pred)["accuracy"])
    return np.array(accs)


def rowwise_scores(X, y, k: int = 5, l2: float = 1.0, seed: int = 0):
    """Row-wise k-fold: ignores subject structure -> optimistic accuracy."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    accs = []
    for f in folds:
        te = np.zeros(len(y), bool)
        te[f] = True
        pred = _fit_predict(X[~te], y[~te], X[te], l2)
        accs.append(sens_spec(y[te], pred)["accuracy"])
    return np.array(accs)
