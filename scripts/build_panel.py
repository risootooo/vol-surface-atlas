"""Regenerate the atlas data from raw NSE bhavcopy archives.

Usage:
    python scripts/build_panel.py --bhavcopy-dir /path/to/bhavcopy \
        --out-dir data

Reads every ``fo_YYYYMMDD.zip`` in the bhavcopy directory, builds the
full NIFTY spot series (to find the real biggest-move days *in the
data*), then extracts IV panels for:

  * **atlas days** — one representative day per month plus the biggest
    up/down event clusters (±2 trading days), nearest THREE expiries
    (three, so a DTE>=3 "front usable" expiry and a next expiry both
    exist even in expiry week, when the nearest print has DTE 1);
  * **dynamics days** — 22 consecutive trading days centred on the
    biggest absolute move, nearest expiry (for the PCA of daily surface
    changes).

Outputs: ``iv_panel_sample.csv`` and ``nifty_spot_series.csv`` in the
output directory, and prints the selection for the record.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volatlas.chain import (chain_iv, extract_chain, list_expiries,  # noqa: E402
                            load_bhavcopy, nifty_spot)


def scan_spots(bhav_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(bhav_dir / "fo_*.zip"))):
        df = load_bhavcopy(path)
        date = pd.to_datetime(df["TradDt"]).dt.date.iloc[0]
        rows.append({"date": date, "spot": nifty_spot(df)})
    spots = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    spots["ret_pct"] = spots["spot"].pct_change() * 100.0
    return spots


def select_days(spots: pd.DataFrame):
    dates = list(spots["date"])
    idx = {d: i for i, d in enumerate(dates)}
    rets = spots.set_index("date")["ret_pct"]
    biggest_up = rets.idxmax()
    biggest_down = rets.idxmin()
    biggest_abs = rets.abs().idxmax()

    atlas = set()
    by_month = spots.groupby(spots["date"].apply(lambda d: (d.year, d.month)))
    for _, grp in by_month:  # representative day: closest to the 15th
        atlas.add(min(grp["date"], key=lambda d: abs(d.day - 15)))
    for centre in (biggest_up, biggest_down):  # event clusters ±2 days
        for off in (-2, -1, 0, 1, 2):
            j = idx[centre] + off
            if 0 <= j < len(dates):
                atlas.add(dates[j])
    i = idx[biggest_abs]  # dynamics window: 22 consecutive days
    lo, hi = max(0, i - 10), min(len(dates), i + 12)
    dynamics = dates[lo:hi]
    return sorted(atlas), dynamics, biggest_up, biggest_down, biggest_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bhavcopy-dir", required=True)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    bhav_dir, out_dir = Path(args.bhavcopy_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("scanning bhavcopy spots ...", flush=True)
    spots = scan_spots(bhav_dir)
    spots_out = spots.copy()
    spots_out["date"] = spots_out["date"].astype(str)
    spots_out.to_csv(out_dir / "nifty_spot_series.csv", index=False)
    print(f"spot series: {len(spots)} days "
          f"{spots['date'].iloc[0]} .. {spots['date'].iloc[-1]}")

    atlas_days, dyn_days, up, down, big_abs = select_days(spots)
    rets = spots.set_index("date")["ret_pct"]
    print(f"biggest up   : {up}  {rets[up]:+.2f}%")
    print(f"biggest down : {down}  {rets[down]:+.2f}%")
    print(f"biggest |move|: {big_abs}  {rets[big_abs]:+.2f}%")
    print(f"atlas days ({len(atlas_days)}): {[str(d) for d in atlas_days]}")
    print(f"dynamics days ({len(dyn_days)}): "
          f"{dyn_days[0]} .. {dyn_days[-1]}")

    frames = []
    for day in sorted(set(atlas_days) | set(dyn_days)):
        path = bhav_dir / f"fo_{day:%Y%m%d}.zip"
        df = load_bhavcopy(path)
        expiries = list_expiries(df, day, min_days=1)
        panels = []
        if day in atlas_days:
            panels.append(("atlas", expiries[:3]))
        if day in dyn_days:
            panels.append(("dynamics", expiries[:1]))
        for panel, exps in panels:
            for expiry in exps:
                rows = chain_iv(extract_chain(df, expiry))
                rows.insert(0, "panel", panel)
                frames.append(rows)
        print(f"  {day}: done", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(str)
    panel["expiry"] = panel["expiry"].astype(str)
    panel = panel.round({"premium": 4, "log_moneyness": 6, "iv": 6})
    out_path = out_dir / "iv_panel_sample.csv"
    panel.to_csv(out_path, index=False)
    print(f"panel rows: {len(panel)}  "
          f"size: {out_path.stat().st_size / 1e6:.2f} MB")
    print(panel.groupby("panel").size().to_string())


if __name__ == "__main__":
    main()
