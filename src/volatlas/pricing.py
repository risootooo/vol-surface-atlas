"""Black-Scholes pricing for European index options (cash-settled, q = 0).

NIFTY options are European-style and cash-settled on the index level, so
plain Black-Scholes with the index as the underlying and zero dividend
yield (the standard desk approximation over weekly/monthly horizons) is
the right workhorse for implied-vol extraction.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

DEFAULT_RATE = 0.065  # 6.5% INR risk-free proxy used throughout the atlas


def _d1_d2(S, K, T, r, sigma, q=0.0):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S, K, T, r, sigma, kind, q=0.0):
    """Black-Scholes price. ``kind`` is 'CE' (call) or 'PE' (put)."""
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    if T <= 0:
        intrinsic = np.maximum(S - K, 0.0) if kind == "CE" else np.maximum(K - S, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if kind == "CE":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, kind, q=0.0):
    """Black-Scholes delta (calls in (0, 1), puts in (-1, 0))."""
    d1, _ = _d1_d2(S, K, T, r, np.asarray(sigma, float), q)
    if kind == "CE":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega per 1.00 of sigma (not per vol point)."""
    d1, _ = _d1_d2(S, K, T, r, np.asarray(sigma, float), q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
