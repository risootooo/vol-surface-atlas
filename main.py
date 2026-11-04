"""VolSurface Atlas CLI.

Runs the full analysis on the shipped real-data panel
(data/iv_panel_sample.csv) and prints the event cases, the PCA of
surface dynamics, and the SVI fit-quality summary. With --dashboard,
also (re)generates docs/dashboard.html and docs/results.json.

    python main.py               # print the cases
    python main.py --dashboard   # + regenerate dashboard & results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from volatlas.dynamics import pca_on_changes, smile_matrix  # noqa: E402
from volatlas.events import biggest_moves, event_window  # noqa: E402
from volatlas.svi import arbitrage_check, fit_svi  # noqa: E402
from volatlas.surface import day_metrics, smile_for  # noqa: E402


def fmt_window(rows):
    lines = []
    for r in rows:
        lines.append(
            f"  {r['date']}  off {r['offset']:+d}  spot {r['spot_ret_pct']:+6.2f}%  "
            f"ATM {r['atm_iv_near_pct']:5.2f}%  RR25 {r['rr25_vol_pts']:+6.2f}  "
            f"BF25 {r['bf25_vol_pts']:+5.2f}  skew {r['skew_slope_vol_pts_per_10pct']:+6.2f}  "
            f"term {r.get('term_slope_vol_pts', float('nan')):+5.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard", action="store_true")
    args = ap.parse_args()

    panel = pd.read_csv(ROOT / "data" / "iv_panel_sample.csv")
    spots = pd.read_csv(ROOT / "data" / "nifty_spot_series.csv")
    print("VolSurface Atlas — NIFTY IV surfaces from real NSE bhavcopy")
    print(f"spot series {spots['date'].iloc[0]} .. {spots['date'].iloc[-1]} "
          f"({len(spots)} trading days); panel rows {len(panel)}")

    up, down = biggest_moves(spots, 1)
    for label, ev in (("BIGGEST DOWN DAY", down.iloc[0]),
                      ("BIGGEST UP DAY", up.iloc[0])):
        print(f"\n{label}: {ev['date']}  {ev['ret_pct']:+.2f}%  (spot {ev['spot']:,.2f})")
        print(fmt_window(event_window(panel, spots, ev['date'])))

    days, mat = smile_matrix(panel)
    pca = pca_on_changes(mat)
    print(f"\nPCA OF DAILY SMILE CHANGES ({pca['n_changes']} changes, "
          f"{days[0]} .. {days[-1]})")
    for pc in pca["pcs"]:
        print(f"  PC{pc['pc']}: {pc['variance_pct']:5.1f}%  {pc['label']}")

    atlas_days = sorted(panel[panel["panel"] == "atlas"]["trade_date"]
                        .astype(str).unique())
    fits = []
    for day in atlas_days:
        for expiry in sorted(panel[(panel["panel"] == "atlas")
                                   & (panel["trade_date"].astype(str) == day)]
                             ["expiry"].unique()):
            smile = smile_for(panel, day, expiry)
            if len(smile) < 8:
                continue
            f = fit_svi(smile["log_moneyness"].values, smile["iv"].values,
                        float(smile["T"].iloc[0]))
            f.update({"date": day, "expiry": str(expiry),
                      "dte": int(smile["days_to_expiry"].iloc[0]),
                      **arbitrage_check(f["params"])})
            fits.append(f)
    import numpy as np
    rmse = pd.Series([f["rmse_vol_pts"] for f in fits])
    dte = pd.Series([f["dte"] for f in fits])
    print(f"\nSVI FITS: {len(fits)} slices  RMSE median {rmse.median():.2f} vol pts "
          f"(DTE>=3: {rmse[dte >= 3].median():.2f} | DTE<3: {rmse[dte < 3].median():.2f})  "
          f"butterfly violations {sum(0 if f['butterfly_ok'] else 1 for f in fits)}/{len(fits)}")

    metrics = [day_metrics(panel, d) for d in atlas_days]
    results = {"window": {"first_day": str(spots["date"].iloc[0]),
                          "last_day": str(spots["date"].iloc[-1]),
                          "n_trading_days": int(len(spots)),
                          "n_atlas_days": len(atlas_days), "atlas_days": atlas_days},
               "day_metrics": metrics,
               "svi_fits": [{k: v for k, v in f.items() if k != "params_tuple"}
                            for f in fits],
               "svi_summary": {
                   "n_slices": len(fits), "rmse_min": float(rmse.min()),
                   "rmse_median": float(rmse.median()), "rmse_max": float(rmse.max()),
                   "rmse_median_dte_ge3": float(rmse[dte >= 3].median()),
                   "rmse_median_dte_lt3": float(rmse[dte < 3].median()),
                   "n_butterfly_violations": int(sum(
                       0 if f["butterfly_ok"] else 1 for f in fits))},
               "pca": {k: v for k, v in pca.items() if k != "grid"},
               "pca_grid": pca["grid"],
               "events": {
                   "biggest_up": {"date": str(up.iloc[0]["date"]),
                                  "ret_pct": float(up.iloc[0]["ret_pct"]),
                                  "window": event_window(panel, spots, up.iloc[0]["date"])},
                   "biggest_down": {"date": str(down.iloc[0]["date"]),
                                    "ret_pct": float(down.iloc[0]["ret_pct"]),
                                    "window": event_window(panel, spots, down.iloc[0]["date"])}}}
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nwrote docs/results.json")
    if args.dashboard:
        from volatlas.report import build_dashboard
        html = build_dashboard(panel, results)
        (ROOT / "docs" / "dashboard.html").write_text(html)
        print(f"wrote docs/dashboard.html ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
