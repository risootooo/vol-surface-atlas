"""Raw SVI parameterisation (Gatheral) fitted per day / expiry.

Total variance w(k) = IV(k)^2 * T is modelled as

    w(k) = a + b * [ rho * (k - m) + sqrt((k - m)^2 + sigma^2) ]

with a >= 0, b >= 0, |rho| < 1, sigma > 0. Fits are by least squares on
total variance, reported back in vol points (RMSE of IV).

No-arbitrage sanity: a static smile is butterfly-arbitrage-free iff
Gatheral's g-function is non-negative everywhere,

    g(k) = (1 - k w'/(2w))^2 - (w')^2 / 4 * (1/w + 1/4) + w''/2  >= 0.

We evaluate g on a grid and report violations *honestly* — a least-
squares SVI fit to settlement prints is a description of the smile, not
a guarantee of an arbitrage-free surface, and the calendar condition
(w non-decreasing in T) is checked in the atlas tables, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_iv(k, params, T):
    """Implied vol from SVI parameters at log-moneyness k."""
    w = svi_total_variance(k, *params)
    return np.sqrt(np.maximum(w, 0.0) / T)


def _derivs(k, a, b, rho, m, sigma):
    u = np.asarray(k, float) - m
    s = np.sqrt(u**2 + sigma**2)
    w = a + b * (rho * u + s)
    wp = b * (rho + u / s)
    wpp = b * sigma**2 / s**3
    return w, wp, wpp


def g_function(k, params):
    """Gatheral's g(k); butterfly arbitrage exists where g < 0."""
    w, wp, wpp = _derivs(k, *params)
    return (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2


def fit_svi(k, iv, T, max_nfev: int = 5000) -> dict:
    """Least-squares raw-SVI fit. Returns params + RMSE in vol points."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w_obs = iv**2 * T
    p0 = np.array([max(w_obs.min() * 0.9, 1e-4), 0.10, -0.5,
                   float(k[np.argmin(w_obs)]), 0.10])
    lo = [0.0, 0.0, -0.999, -0.5, 0.005]
    hi = [2.0, 2.0, 0.999, 0.5, 1.0]
    sol = least_squares(lambda p: svi_total_variance(k, *p) - w_obs,
                        np.clip(p0, lo, hi), bounds=(lo, hi), max_nfev=max_nfev)
    params = tuple(float(x) for x in sol.x)
    rmse_vol_pts = float(np.sqrt(np.mean((svi_iv(k, params, T) - iv) ** 2)) * 100.0)
    return {"params": dict(zip(PARAM_NAMES, params)),
            "params_tuple": params,
            "rmse_vol_pts": rmse_vol_pts, "n_points": int(len(k))}


def arbitrage_check(params, k_grid=None) -> dict:
    """Butterfly no-arb check for one fitted slice."""
    if k_grid is None:
        k_grid = np.linspace(-0.30, 0.30, 601)
    g = g_function(k_grid, tuple(params[name] for name in PARAM_NAMES)
                   if isinstance(params, dict) else params)
    return {"min_g": float(g.min()), "frac_negative": float((g < 0).mean()),
            "k_at_min": float(k_grid[int(np.argmin(g))]),
            "butterfly_ok": bool(g.min() >= 0)}
