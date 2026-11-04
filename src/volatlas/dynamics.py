"""PCA decomposition of day-to-day smile movements.

The dynamics panel holds the nearest-expiry smile for 22 consecutive
trading days around the window's biggest move. Each day's smile is
interpolated onto a fixed log-moneyness grid; the matrix of daily
*changes* (in vol points) is decomposed by SVD (PCA without a library
call — the covariance eigenproblem is the SVD of the demeaned change
matrix). Each principal component is then *classified by its loading
shape*, not by wishful naming:

  * level shift  — loadings (almost) one sign and flat across k
  * skew rotation — loadings monotone in k (they cross zero)
  * curvature    — loadings U / inverted-U shaped (track k^2)

Classification uses the correlation of the loadings with 1, k and k^2
plus a sign-change count, and is reported with those diagnostics so the
reader can disagree with the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GRID = np.round(np.linspace(-0.10, 0.10, 15), 4)


def smile_matrix(panel: pd.DataFrame, grid=DEFAULT_GRID,
                 panel_name: str = "dynamics"):
    """(dates, IV matrix) — each dynamics day's smile on a fixed k-grid."""
    days = sorted(panel[panel["panel"] == panel_name]["trade_date"]
                  .astype(str).unique())
    rows = []
    for day in days:
        sub = panel[(panel["panel"] == panel_name)
                    & (panel["trade_date"].astype(str) == day)]
        sub = sub.sort_values("log_moneyness")
        rows.append(np.interp(grid, sub["log_moneyness"], sub["iv"]))
    return days, np.array(rows) * 100.0  # vol points


def pca_on_changes(matrix_vol_pts: np.ndarray, grid=DEFAULT_GRID,
                   n_components: int = 3) -> dict:
    """PCA of daily changes; returns variance split + classified PCs."""
    changes = np.diff(matrix_vol_pts, axis=0)
    centred = changes - changes.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s**2
    var_pct = var / var.sum() * 100.0
    pcs = []
    for i in range(min(n_components, len(var_pct))):
        load = vt[i]
        if load[np.argmax(np.abs(load))] < 0:  # stable sign convention
            load = -load
        corr_k = float(np.corrcoef(load, grid)[0, 1])
        corr_k2 = float(np.corrcoef(load, grid**2)[0, 1])
        sign_changes = int(np.sum(np.diff(np.sign(load)) != 0))
        mean_level = float(np.mean(load))
        if sign_changes == 0 and abs(corr_k) < 0.7:
            ratio = float(load.max() / load.min()) if load.min() > 0 else float("nan")
            label = ("parallel level shift (wing-amplified)"
                     if corr_k2 >= 0.85 else "parallel level shift")
        elif abs(corr_k) >= 0.7:
            label = "skew rotation (twist)"
        elif abs(corr_k2) >= 0.6:
            label = "curvature (wings vs belly)"
        else:
            label = "mixed shape change"
        pcs.append({"pc": i + 1, "variance_pct": float(var_pct[i]),
                    "label": label, "corr_with_k": corr_k,
                    "corr_with_k2": corr_k2, "sign_changes": sign_changes,
                    "mean_loading": mean_level,
                    "loadings": [float(x) for x in load]})
    return {"n_changes": int(len(changes)),
            "total_variance_pct_accounted": float(var_pct[:len(pcs)].sum()),
            "variance_pct_all": [float(x) for x in var_pct],
            "pcs": pcs, "grid": [float(x) for x in grid]}


def component_scores(panel: pd.DataFrame, grid=DEFAULT_GRID):
    """PC scores per day-change (for the dashboard / case discussion)."""
    days, mat = smile_matrix(panel, grid)
    changes = np.diff(mat, axis=0)
    centred = changes - changes.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    scores = u * s
    return days, scores[:, :3], changes
