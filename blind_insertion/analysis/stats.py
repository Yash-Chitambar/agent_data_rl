"""Statistics helpers used in every phase (plan §3.2)."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


def bootstrap_median(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), (float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2)))


def newcombe_diff(k1, n1, k2, n2, z=1.96):
    """Newcombe hybrid score interval for p1 - p2."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, (lo, hi)


def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    zv = (k1 / n1 - k2 / n2) / se
    return float(zv), float(2 * st.norm.sf(abs(zv)))


def paired(a, b, n_boot=10000, seed=0):
    """Paired comparison on matched starts: fraction a>b, median diff + bootstrap CI, Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    med, ci = bootstrap_median(d, n_boot, seed)
    p = st.wilcoxon(a, b).pvalue if len(d) > 0 and np.any(d != 0) else 1.0
    return dict(n=int(len(d)), frac_a_gt_b=float((d > 0).mean()), median_diff=med, ci=ci, wilcoxon_p=float(p))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
