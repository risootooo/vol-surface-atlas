"""VolSurface Atlas tests.

Groups: Black-Scholes pricing (textbook values, parity), IV solver
(round-trip, no-arb honesty), SVI (parameter recovery on a synthetic
SVI smile, g-function), surface metrics (toy smile with known slope),
PCA (variance accounting, level-shift detection), and invariants of
the shipped REAL bhavcopy panel.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volatlas.dynamics import pca_on_changes
from volatlas.events import biggest_moves
from volatlas.iv import implied_vol
from volatlas.pricing import bs_delta, bs_price
from volatlas.surface import atm_iv, iv_at, skew_slope
from volatlas.svi import arbitrage_check, fit_svi, g_function, svi_iv

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- pricing
def test_bs_known_value():
    # Hull's textbook example: S=K=100, T=1, r=5%, sigma=20% -> call 10.4506
    assert bs_price(100, 100, 1.0, 0.05, 0.20, "CE") == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    S, K, T, r, s = 23810.0, 24000.0, 0.08, 0.065, 0.18
    c = bs_price(S, K, T, r, s, "CE")
    p = bs_price(S, K, T, r, s, "PE")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)


def test_delta_bounds_and_atm():
    d_c = bs_delta(100, 100, 0.25, 0.065, 0.2, "CE")
    d_p = bs_delta(100, 100, 0.25, 0.065, 0.2, "PE")
    assert 0.5 < d_c < 0.65 and -0.5 < d_p < -0.35
    assert bs_delta(100, 200, 0.25, 0.065, 0.2, "CE") < 0.05  # deep OTM


# -------------------------------------------------------------------- IV
@pytest.mark.parametrize("sigma", [0.08, 0.15, 0.23, 0.45])
def test_iv_round_trip(sigma):
    S, K, T, r = 23000.0, 23500.0, 0.05, 0.065
    price = bs_price(S, K, T, r, sigma, "CE")
    assert implied_vol(price, S, K, T, r, "CE") == pytest.approx(sigma, abs=1e-6)


def test_iv_put_round_trip():
    price = bs_price(23000.0, 22500.0, 0.1, 0.065, 0.19, "PE")
    assert implied_vol(price, 23000.0, 22500.0, 0.1, 0.065, "PE") == pytest.approx(0.19, abs=1e-6)


def test_iv_below_intrinsic_is_nan_not_fantasy():
    # Settlement prints below the discounted intrinsic floor have no IV.
    assert np.isnan(implied_vol(10.0, 23000.0, 20000.0, 0.05, 0.065, "CE"))


# ------------------------------------------------------------------- SVI
TRUE_PARAMS = (0.045, 0.16, -0.55, 0.01, 0.11)
T_SYN = 30 / 365.0


def _synthetic_smile():
    k = np.linspace(-0.15, 0.15, 25)
    return k, svi_iv(k, TRUE_PARAMS, T_SYN)


def test_svi_recovers_known_parameters():
    k, iv = _synthetic_smile()
    fit = fit_svi(k, iv, T_SYN)
    got = fit["params"]
    assert fit["rmse_vol_pts"] < 0.05
    assert got["a"] == pytest.approx(TRUE_PARAMS[0], abs=0.01)
    assert got["b"] == pytest.approx(TRUE_PARAMS[1], abs=0.03)
    assert got["rho"] == pytest.approx(TRUE_PARAMS[2], abs=0.10)
    assert got["sigma"] == pytest.approx(TRUE_PARAMS[4], abs=0.04)


def test_g_function_flat_smile_is_arbitrage_free():
    flat = (0.04, 0.0, 0.0, 0.0, 0.1)  # constant total variance: w'=w''=0
    g = g_function(np.linspace(-0.2, 0.2, 41), flat)
    assert np.allclose(g, 1.0)
    assert arbitrage_check(flat)["butterfly_ok"]


def test_svi_iv_matches_total_variance_identity():
    k = np.array([-0.05, 0.0, 0.05])
    w = TRUE_PARAMS[0] + TRUE_PARAMS[1] * (
        TRUE_PARAMS[2] * (k - TRUE_PARAMS[3])
        + np.sqrt((k - TRUE_PARAMS[3]) ** 2 + TRUE_PARAMS[4] ** 2))
    assert np.allclose(svi_iv(k, TRUE_PARAMS, T_SYN), np.sqrt(w / T_SYN))


# ------------------------------------------------------- surface metrics
def _toy_smile():
    k = np.linspace(-0.10, 0.10, 21)
    return pd.DataFrame({"log_moneyness": k, "iv": 0.20 - 0.50 * k,
                         "spot": 23000.0, "T": 0.05})


def test_atm_iv_and_interp_on_toy_smile():
    smile = _toy_smile()
    assert atm_iv(smile) == pytest.approx(0.20)
    assert iv_at(smile, -0.10) == pytest.approx(0.25)


def test_skew_slope_on_toy_smile_with_known_slope():
    # iv = 0.20 - 0.50 k  ->  slope = -0.50 * 0.10 * 100 = -5.0 vol pts / 10%
    assert skew_slope(_toy_smile()) == pytest.approx(-5.0, abs=0.05)


# ------------------------------------------------------------------- PCA
def test_pca_variance_sums_to_100():
    rng = np.random.default_rng(7)
    mat = 20 + np.cumsum(rng.normal(0, 0.4, size=(22, 15)), axis=0) \
        + rng.normal(0, 0.05, size=(22, 15))
    out = pca_on_changes(mat)
    assert sum(out["variance_pct_all"]) == pytest.approx(100.0, abs=1e-6)


def test_pca_pure_level_shifts_are_pc1():
    grid = np.linspace(-0.10, 0.10, 15)
    levels = np.linspace(18, 26, 22) + np.sin(np.arange(22)) * 1.5
    mat = np.tile(levels[:, None], (1, 15)) + 0.02 * grid[None, :] ** 2 * 100
    out = pca_on_changes(mat)
    assert out["pcs"][0]["variance_pct"] > 99.0
    assert out["pcs"][0]["sign_changes"] == 0


# ----------------------------------------------------------------- events
def test_biggest_moves_on_toy_series():
    spots = pd.DataFrame({"date": ["d1", "d2", "d3", "d4"],
                          "spot": [100.0, 103.0, 99.0, 101.0],
                          "ret_pct": [np.nan, 3.0, -3.88, 2.02]})
    up, down = biggest_moves(spots, 1)
    assert up.iloc[0]["date"] == "d2" and down.iloc[0]["date"] == "d3"


# -------------------------------------------- real shipped panel (data/)
@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(ROOT / "data" / "iv_panel_sample.csv")


def test_panel_covers_at_least_12_atlas_days(panel):
    days = panel[panel["panel"] == "atlas"]["trade_date"].unique()
    assert len(days) >= 12


def test_panel_otm_convention(panel):
    ce = panel[panel["kind"] == "CE"]
    pe = panel[panel["kind"] == "PE"]
    assert (ce["log_moneyness"] >= -1e-9).all()   # calls at/above spot
    assert (pe["log_moneyness"] < 1e-9).all()     # puts below spot


def test_panel_iv_sane_ranges(panel):
    assert panel["iv"].between(0.02, 2.0).all()
    assert (panel["premium"] > 0).all()
    assert (panel["T"] > 0).all()


def test_panel_t_consistent_within_slice(panel):
    grouped = panel.groupby(["panel", "trade_date", "expiry"])["T"].nunique()
    assert (grouped == 1).all()
    dte = panel.groupby(["panel", "trade_date", "expiry"])["days_to_expiry"].first()
    assert (dte >= 1).all()


def test_spot_series_is_the_real_220_day_window():
    spots = pd.read_csv(ROOT / "data" / "nifty_spot_series.csv")
    assert len(spots) == 220
    assert spots["date"].iloc[0] == "2025-11-04"
    assert spots["date"].iloc[-1] == "2026-11-03"
    assert spots["spot"].between(20000, 27000).all()
