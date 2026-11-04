"""NSE F&O bhavcopy loading and NIFTY chain extraction.

The bhavcopy is the exchange's official end-of-day file (one CSV per
trading day, distributed here as ``fo_YYYYMMDD.zip`` archives). Columns
used: TradDt, FinInstrmTp ('IDO' = index options), TckrSymb, XpryDt,
StrkPric, OptnTp, ClsPric (settlement/close print — the premium source),
UndrlygPric (index level), OpnIntrst, TtlTradgVol.

Conventions:
  * Premium = ClsPric (settlement print). This is a *settlement* price,
    not a traded mid — see README limitations.
  * Rate r = 6.5%, dividend yield q = 0.
  * IVs are computed from OUT-OF-THE-MONEY options only: puts with
    K < F and calls with K >= F, F the spot (forward ≈ spot at these
    horizons). ITM settlement prints on this feed frequently violate the
    discounted-intrinsic floor and have no implied vol.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .iv import implied_vols
from .pricing import DEFAULT_RATE

BHAV_COLUMNS = ["TradDt", "FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric",
                "OptnTp", "ClsPric", "UndrlygPric", "OpnIntrst", "TtlTradgVol"]


def load_bhavcopy(source) -> pd.DataFrame:
    """Load one bhavcopy from a .zip archive or .csv path (all symbols)."""
    if str(source).endswith(".zip"):
        with zipfile.ZipFile(source) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            raw = z.read(name)
        return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in BHAV_COLUMNS)
    return pd.read_csv(source, usecols=lambda c: c in BHAV_COLUMNS)


def nifty_spot(df: pd.DataFrame) -> float:
    """The day's NIFTY index level as printed on the option rows."""
    sub = df[(df["TckrSymb"] == "NIFTY") & (df["FinInstrmTp"] == "IDO")]
    if sub.empty:
        raise ValueError("no NIFTY index options in frame")
    return float(sub["UndrlygPric"].iloc[0])


def list_expiries(df: pd.DataFrame, trade_date, min_days: int = 1):
    """Sorted expiries with at least ``min_days`` left (T=0 has no IV)."""
    sub = df[(df["TckrSymb"] == "NIFTY") & (df["FinInstrmTp"] == "IDO")]
    expiries = sorted(pd.to_datetime(sub["XpryDt"]).dt.date.unique())
    return [e for e in expiries if (e - trade_date).days >= min_days]


def extract_chain(df: pd.DataFrame, expiry, symbol: str = "NIFTY") -> pd.DataFrame:
    """One expiry's chain: strike, kind, premium, spot, oi, volume, T."""
    sub = df[(df["TckrSymb"] == symbol) & (df["FinInstrmTp"] == "IDO")].copy()
    sub["TradDt"] = pd.to_datetime(sub["TradDt"]).dt.date
    sub["XpryDt"] = pd.to_datetime(sub["XpryDt"]).dt.date
    trade_date = sub["TradDt"].iloc[0]
    chain = sub[sub["XpryDt"] == expiry]
    days = (expiry - trade_date).days
    out = pd.DataFrame({
        "trade_date": trade_date,
        "expiry": expiry,
        "strike": chain["StrkPric"].astype(float),
        "kind": chain["OptnTp"],
        "premium": chain["ClsPric"].astype(float),
        "spot": chain["UndrlygPric"].astype(float),
        "oi": chain["OpnIntrst"].astype(float),
        "volume": chain["TtlTradgVol"].astype(float),
        "days_to_expiry": days,
        "T": days / 365.0,
    }).sort_values(["strike", "kind"]).reset_index(drop=True)
    return out


def chain_iv(chain: pd.DataFrame, r: float = DEFAULT_RATE,
             max_abs_log_moneyness: float = 0.20) -> pd.DataFrame:
    """OTM IV smile rows for one chain.

    Keeps OTM rows with OI > 0 within ``max_abs_log_moneyness`` of the
    forward, solves IV, drops arb-inconsistent prints (NaN IV).
    """
    out = chain.copy()
    spot = float(out["spot"].iloc[0])
    T = float(out["T"].iloc[0])
    out["log_moneyness"] = np.log(out["strike"] / spot)
    otm = np.where(out["kind"] == "CE", out["strike"] >= spot,
                   out["strike"] < spot)
    out = out[otm & (out["oi"] > 0)
              & (out["log_moneyness"].abs() <= max_abs_log_moneyness)].copy()
    out["iv"] = implied_vols(out["premium"].values, spot, out["strike"].values,
                             T, r, out["kind"].values)
    out = out[out["iv"].notna() & (out["iv"] > 0)].reset_index(drop=True)
    return out[["trade_date", "expiry", "days_to_expiry", "T", "spot",
                "strike", "kind", "premium", "oi", "volume",
                "log_moneyness", "iv"]]
