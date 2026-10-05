"""Distribution-shift detection between training data and a production window.

Two complementary tests on CLIP image embeddings:

* **MMD** (maximum mean discrepancy, RBF kernel, median-distance bandwidth) with a permutation
  test: a calibrated p-value for "these two samples come from the same distribution".
* **Domain classifier AUC**: a cross-validated logistic regression trying to tell reference from
  production. 0.5 = indistinguishable; the AUC is an effect size that is easy to explain.

Plus a population stability index (PSI) on image brightness, the classic tabular drift metric,
so the dashboard can show which interpretable feature moved.
"""

from __future__ import annotations

import numpy as np


def _rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(d, 0))


def mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    kxx, kyy, kxy = _rbf(x, x, gamma), _rbf(y, y, gamma), _rbf(x, y, gamma)
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2 * kxy.mean())


def mmd_test(x: np.ndarray, y: np.ndarray, permutations: int = 500, seed: int = 0) -> dict:
    z = np.concatenate([x, y]).astype(np.float64)
    d = (z * z).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2 * z @ z.T
    med = np.median(d[np.triu_indices(len(z), 1)])
    gamma = 1.0 / max(med, 1e-12)
    k = np.exp(-gamma * np.maximum(d, 0))
    n = len(x)

    def stat(idx: np.ndarray) -> float:
        a, b = idx[:n], idx[n:]
        kaa, kbb, kab = k[np.ix_(a, a)], k[np.ix_(b, b)], k[np.ix_(a, b)]
        return float((kaa.sum() - np.trace(kaa)) / (len(a) * (len(a) - 1))
                     + (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1)) - 2 * kab.mean())

    obs = stat(np.arange(len(z)))
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(len(z))) for _ in range(permutations)])
    p = (1 + (null >= obs).sum()) / (permutations + 1)
    return {"mmd2": obs, "p_value": float(p), "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95))}


def domain_auc(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    z = np.concatenate([x, y])
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    folds = int(min(5, len(x), len(y)))
    if folds < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(clf, z, lab, cv=folds, scoring="roc_auc").mean())


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())


def drift_report(ref_emb: np.ndarray, cur_emb: np.ndarray, ref_bright: np.ndarray, cur_bright: np.ndarray,
                 permutations: int = 500, alpha: float = 0.01, seed: int = 0) -> dict:
    t = mmd_test(ref_emb, cur_emb, permutations, seed)
    auc = domain_auc(ref_emb, cur_emb, seed)
    p = psi(ref_bright, cur_bright)
    return {**t, "domain_auc": auc, "psi_brightness": p, "alpha": alpha, "drift": bool(t["p_value"] < alpha),
            "n_reference": int(len(ref_emb)), "n_current": int(len(cur_emb)),
            "brightness_ref_mean": float(np.mean(ref_bright)), "brightness_cur_mean": float(np.mean(cur_bright))}
