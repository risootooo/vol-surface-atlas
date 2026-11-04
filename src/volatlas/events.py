"""Real event cases: the biggest NIFTY moves in the bhavcopy window.

Events are detected from the data itself — the full 220-day spot series
rebuilt from the same bhavcopy files — never from a news calendar. For
each event, surface metrics (ATM IV, 25-delta risk reversal, skew slope,
term-structure slope) are tabulated over a ±2-trading-day window from
the atlas panel, so the vol surface's behaviour *into* and *out of* the
move is a measurement, not a story.
"""

from __future__ import annotations

import pandas as pd

from .surface import day_metrics


def biggest_moves(spots: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Top-n up and down single-day moves in the spot series."""
    s = spots.dropna(subset=["ret_pct"])
    cols = ["date", "spot", "ret_pct"]
    up = s.nlargest(n, "ret_pct")[cols]
    down = s.nsmallest(n, "ret_pct")[cols]
    return up, down


def event_window(panel: pd.DataFrame, spots: pd.DataFrame, event_date,
                 half_window: int = 2) -> list[dict]:
    """day_metrics for event_date ± half_window *trading* days."""
    dates = list(spots["date"].astype(str))
    centre = dates.index(str(event_date))
    rets = spots.set_index(spots["date"].astype(str))["ret_pct"]
    rows = []
    for j in range(max(0, centre - half_window),
                   min(len(dates), centre + half_window + 1)):
        day = dates[j]
        m = day_metrics(panel, day)
        m["spot_ret_pct"] = float(rets[day]) if pd.notna(rets[day]) else None
        m["offset"] = j - centre
        rows.append(m)
    return rows
