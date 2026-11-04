"""Implied-volatility solver with honest no-arbitrage handling.

Method: Newton iteration on vega from a Brenner-Subrahmanyam seed, with a
bisection fallback on sigma in [1e-4, 4.0] (BS price is monotone in sigma,
so bisection always brackets when a solution exists).

NSE *settlement* prints (ClsPric in the bhavcopy) regularly sit below the
discounted intrinsic value, especially ITM. No BSM volatility produces
such a price, so the solver returns NaN rather than inventing a number --
and VolSurface Atlas solves IVs from **out-of-the-money options only**,
where the prints are time-value dominated and arb-consistent.
"""

from __future__ import annotations

import numpy as np

from .pricing import bs_price, bs_vega

SIGMA_LO, SIGMA_HI = 1e-4, 4.0


def _lower_bound(S, K, T, r, kind, q=0.0):
    """Discounted intrinsic value — the BSM no-arbitrage price floor."""
    fwd_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
    return max(fwd_intrinsic, 0.0) if kind == "CE" else max(-fwd_intrinsic, 0.0)


def implied_vol(price, S, K, T, r, kind, q=0.0, tol=1e-10):
    """Implied vol of one option, or NaN if the price is arb-inconsistent."""
    if price is None or not np.isfinite(price) or T <= 0 or price < 0:
        return float("nan")
    lower = _lower_bound(S, K, T, r, kind, q)
    if price < lower - 1e-9:
        return float("nan")
    if price <= lower + 1e-12:
        return 0.0
    sigma = np.sqrt(2 * np.pi / T) * price / S if S > 0 else 0.2
    sigma = float(np.clip(sigma, 0.05, SIGMA_HI))
    for _ in range(100):
        diff = float(bs_price(S, K, T, r, sigma, kind, q)) - price
        if abs(diff) < tol:
            return float(sigma)
        vega = float(bs_vega(S, K, T, r, sigma, q))
        if vega < 1e-12:
            break
        sigma -= diff / vega
        if not (SIGMA_LO < sigma < SIGMA_HI):
            break
    lo, hi = SIGMA_LO, SIGMA_HI
    if not (float(bs_price(S, K, T, r, lo, kind, q)) <= price
            <= float(bs_price(S, K, T, r, hi, kind, q))):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(bs_price(S, K, T, r, mid, kind, q)) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return float(0.5 * (lo + hi))


def implied_vols(prices, S, K, T, r, kinds, q=0.0):
    """Vector wrapper (plain loop — chains are a few hundred rows)."""
    return np.array([implied_vol(p, S, k, T, r, kd, q)
                     for p, k, kd in zip(prices, K, kinds)], dtype=float)
