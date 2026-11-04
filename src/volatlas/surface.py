"""Smile and term-structure metrics for one trading day.

The smile is taken in log-moneyness space, k = ln(K/F) with F the spot
(the forward differs from spot by <0.1% at weekly horizons; stated so
the approximation is visible). 25-delta risk reversal / butterfly are
computed the desk way but *approximately*: the strike whose Black-
Scholes delta equals ±0.25 is located on a dense k-grid using the
smile's own interpolated IV (two passes are enough for convergence at
these maturities), then the smile IV is read at that strike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import DEFAULT_RATE, bs_delta


def smile_for(panel: pd.DataFrame, trade_date, expiry=None,
              panel_name: str = "atlas") -> pd.DataFrame:
    """Smile rows (sorted by log-moneyness) for one day / expiry."""
    sub = panel[(panel["trade_date"].astype(str) == str(trade_date))
                & (panel["panel"] == panel_name)]
    if expiry is None:
        expiry = sorted(sub["expiry"].unique())[0]
    smile = sub[sub["expiry"].astype(str) == str(expiry)]
    return smile.sort_values("log_moneyness").reset_index(drop=True)


def expiries_for(panel: pd.DataFrame, trade_date, panel_name: str = "atlas"):
    sub = panel[(panel["trade_date"].astype(str) == str(trade_date))
                & (panel["panel"] == panel_name)]
    return sorted(sub["expiry"].unique())


def iv_at(smile: pd.DataFrame, k):
    """Interpolated IV at log-moneyness k (flat outside the smile)."""
    return np.interp(k, smile["log_moneyness"].values, smile["iv"].values)


def atm_iv(smile: pd.DataFrame) -> float:
    return float(iv_at(smile, 0.0))


def _strike_for_delta(smile: pd.DataFrame, target_abs_delta: float,
                      kind: str, r: float = DEFAULT_RATE) -> float:
    """Log-moneyness whose BS delta (at the smile's own IV) hits target."""
    spot = float(smile["spot"].iloc[0])
    T = float(smile["T"].iloc[0])
    k_grid = np.linspace(-0.35, 0.35, 1401)
    ivs = iv_at(smile, k_grid)
    strikes = spot * np.exp(k_grid)
    deltas = np.abs(bs_delta(spot, strikes, T, r, ivs, kind))
    j = int(np.argmin(np.abs(deltas - target_abs_delta)))
    return float(k_grid[j])


def risk_reversal_butterfly(smile: pd.DataFrame, delta: float = 0.25):
    """(RR, BF) in vol points for the given delta (default 25-delta).

    RR = IV(25d call) - IV(25d put)  (negative in equity index: puts bid)
    BF = (IV(25d call) + IV(25d put))/2 - IV(ATM)
    """
    k_c = _strike_for_delta(smile, delta, "CE")
    k_p = _strike_for_delta(smile, delta, "PE")
    iv_c, iv_p, iv_a = iv_at(smile, k_c), iv_at(smile, k_p), atm_iv(smile)
    return float((iv_c - iv_p) * 100.0), float(((iv_c + iv_p) / 2 - iv_a) * 100.0)


def skew_slope(smile: pd.DataFrame, half_width: float = 0.05) -> float:
    """OLS slope of IV on k over |k| <= half_width, in vol points per
    10% of log-moneyness (i.e. slope * 0.10 * 100). Equity smiles slope
    down: this is normally negative."""
    m = smile["log_moneyness"].abs() <= half_width
    if m.sum() < 3:
        return float("nan")
    slope = np.polyfit(smile.loc[m, "log_moneyness"], smile.loc[m, "iv"], 1)[0]
    return float(slope * 0.10 * 100.0)


def front_expiry(panel: pd.DataFrame, trade_date, panel_name: str = "atlas",
                 min_dte: int = 3):
    """The 'front usable' expiry: nearest with DTE >= min_dte.

    NIFTY's weekly cycle means the literal nearest expiry very often
    has DTE 1, where settlement-print IVs are dominated by pin/expiry
    noise (one atlas day printed a 42.8% ATM IV at DTE 1 against ~20%
    a week out). Surface *levels* are therefore quoted on the first
    expiry with at least ``min_dte`` days left; the DTE-1 slice is kept
    in the panel and analysed separately (see CASES.md, Case C).
    """
    sub = panel[(panel["trade_date"].astype(str) == str(trade_date))
                & (panel["panel"] == panel_name)]
    by_exp = sub.groupby(sub["expiry"].astype(str))["days_to_expiry"].first()
    for expiry, dte in by_exp.items():
        if dte >= min_dte:
            return expiry
    return by_exp.index[0]


def day_metrics(panel: pd.DataFrame, trade_date, panel_name: str = "atlas") -> dict:
    """Headline surface metrics for one day (front usable expiry +
    term slope to the next expiry after it)."""
    exps = expiries_for(panel, trade_date, panel_name)
    front = front_expiry(panel, trade_date, panel_name)
    near = smile_for(panel, trade_date, front, panel_name)
    rr, bf = risk_reversal_butterfly(near)
    out = {
        "date": str(trade_date),
        "spot": float(near["spot"].iloc[0]),
        "near_expiry": str(front),
        "near_dte": int(near["days_to_expiry"].iloc[0]),
        "atm_iv_near_pct": atm_iv(near) * 100.0,
        "rr25_vol_pts": rr,
        "bf25_vol_pts": bf,
        "skew_slope_vol_pts_per_10pct": skew_slope(near),
        "n_smile_points": int(len(near)),
    }
    later = [e for e in exps if str(e) > str(front)]
    if later:
        nxt = smile_for(panel, trade_date, later[0], panel_name)
        out["next_expiry"] = str(later[0])
        out["next_dte"] = int(nxt["days_to_expiry"].iloc[0])
        out["atm_iv_next_pct"] = atm_iv(nxt) * 100.0
        out["term_slope_vol_pts"] = out["atm_iv_next_pct"] - out["atm_iv_near_pct"]
    return out
