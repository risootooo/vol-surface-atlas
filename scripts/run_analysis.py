"""Run the full atlas analysis on data/iv_panel_sample.csv.

Writes docs/results.json (every number used by the README, CASES.md and
dashboard) and prints the headline tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volatlas.dynamics import pca_on_changes, smile_matrix  # noqa: E402
from volatlas.events import biggest_moves, event_window  # noqa: E402
from volatlas.surface import day_metrics, smile_for  # noqa: E402
from volatlas.svi import arbitrage_check, fit_svi  # noqa: E402


def main():
    panel = pd.read_csv(ROOT / "data" / "iv_panel_sample.csv")
    spots = pd.read_csv(ROOT / "data" / "nifty_spot_series.csv")
    atlas_days = sorted(panel[panel["panel"] == "atlas"]["trade_date"]
                        .astype(str).unique())

    metrics = [day_metrics(panel, d) for d in atlas_days]

    svi_rows = []
    for day in atlas_days:
        for expiry in sorted(panel[(panel["panel"] == "atlas")
                                   & (panel["trade_date"].astype(str) == day)]
                             ["expiry"].unique()):
            smile = smile_for(panel, day, expiry)
            if len(smile) < 8:
                continue
            fit = fit_svi(smile["log_moneyness"].values, smile["iv"].values,
                          float(smile["T"].iloc[0]))
            arb = arbitrage_check(fit["params"])
            svi_rows.append({"date": day, "expiry": str(expiry),
                             "dte": int(smile["days_to_expiry"].iloc[0]),
                             "n_points": fit["n_points"],
                             "rmse_vol_pts": fit["rmse_vol_pts"],
                             "params": fit["params"], **arb})

    days, mat = smile_matrix(panel)
    pca = pca_on_changes(mat)

    up, down = biggest_moves(spots, 5)
    ev_up_date, ev_down_date = str(up.iloc[0]["date"]), str(down.iloc[0]["date"])
    events = {
        "biggest_up": {"date": ev_up_date,
                       "ret_pct": float(up.iloc[0]["ret_pct"]),
                       "window": event_window(panel, spots, ev_up_date)},
        "biggest_down": {"date": ev_down_date,
                         "ret_pct": float(down.iloc[0]["ret_pct"]),
                         "window": event_window(panel, spots, ev_down_date)},
        "top_up": up.assign(date=up["date"].astype(str)).to_dict("records"),
        "top_down": down.assign(date=down["date"].astype(str)).to_dict("records"),
    }

    results = {
        "window": {"first_day": str(spots["date"].iloc[0]),
                   "last_day": str(spots["date"].iloc[-1]),
                   "n_trading_days": int(len(spots)),
                   "n_atlas_days": len(atlas_days),
                   "atlas_days": atlas_days},
        "day_metrics": metrics,
        "svi_fits": svi_rows,
        "svi_summary": {
            "n_slices": len(svi_rows),
            "rmse_min": float(min(r["rmse_vol_pts"] for r in svi_rows)),
            "rmse_median": float(np.median([r["rmse_vol_pts"] for r in svi_rows])),
            "rmse_max": float(max(r["rmse_vol_pts"] for r in svi_rows)),
            "rmse_median_dte_ge3": float(np.median(
                [r["rmse_vol_pts"] for r in svi_rows if r["dte"] >= 3])),
            "rmse_median_dte_lt3": float(np.median(
                [r["rmse_vol_pts"] for r in svi_rows if r["dte"] < 3])),
            "n_butterfly_violations": int(sum(
                0 if r["butterfly_ok"] else 1 for r in svi_rows)),
        },
        "pca": {k: v for k, v in pca.items() if k != "grid"},
        "pca_grid": pca["grid"],
        "events": events,
    }
    out = ROOT / "docs" / "results.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {out}")

    svi_df = pd.DataFrame(svi_rows)
    print("\nSVI RMSE (vol pts): min %.3f  median %.3f  max %.3f"
          % (svi_df.rmse_vol_pts.min(), svi_df.rmse_vol_pts.median(),
             svi_df.rmse_vol_pts.max()))
    worst = svi_df.nlargest(3, "rmse_vol_pts")[["date", "expiry", "rmse_vol_pts"]]
    print("worst fits:\n", worst.to_string(index=False))
    print("butterfly violations:", int((~svi_df.butterfly_ok).sum()),
          "of", len(svi_df), "slices")
    print("\nPCA variance %:", [round(x, 2) for x in pca["variance_pct_all"][:5]])
    for pc in pca["pcs"]:
        print(f"  PC{pc['pc']}: {pc['variance_pct']:.1f}%  {pc['label']}"
              f"  (corr k {pc['corr_with_k']:+.2f}, k^2 {pc['corr_with_k2']:+.2f},"
              f" sign changes {pc['sign_changes']})")
    for name in ("biggest_down", "biggest_up"):
        ev = events[name]
        print(f"\n{name}: {ev['date']} {ev['ret_pct']:+.2f}%")
        for r in ev["window"]:
            print(f"  {r['date']} off {r['offset']:+d} ret {r['spot_ret_pct']:+.2f}%"
                  f" ATM {r['atm_iv_near_pct']:.2f}% RR25 {r['rr25_vol_pts']:+.2f}"
                  f" skew {r['skew_slope_vol_pts_per_10pct']:+.2f}"
                  f" term {r.get('term_slope_vol_pts', float('nan')):+.2f}")


if __name__ == "__main__":
    main()
