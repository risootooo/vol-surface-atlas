"""Self-contained HTML dashboard generation (inline SVG, no CDN).

Charts: IV smiles for four contrasting real days, the ATM term
structure across all atlas days, and the PCA variance split. All data
comes from the shipped panel + results — nothing is fetched.
"""

from __future__ import annotations

import json

import pandas as pd

from .surface import front_expiry, smile_for

W, H, PAD = 560, 300, 44


def _svg_polyline(xs, ys, color, width=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"/>')


def _chart(series, title, x_label, y_label):
    """series: list of (xs, ys, color, label). Returns one SVG block."""
    all_x = [x for s in series for x in s[0]]
    all_y = [y for s in series for y in s[1]]
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    y0 -= (y1 - y0) * 0.08 + 1e-9
    y1 += (y1 - y0) * 0.08

    def sx(x):
        return PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD)

    def sy(y):
        return H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect width="{W}" height="{H}" fill="#161b22"/>']
    for frac in (0.0, 0.5, 1.0):
        gx, gy = PAD + frac * (W - 2 * PAD), PAD + frac * (H - 2 * PAD)
        parts.append(f'<line x1="{gx}" y1="{PAD}" x2="{gx}" y2="{H - PAD}" stroke="#30363d"/>')
        parts.append(f'<line x1="{PAD}" y1="{gy}" x2="{W - PAD}" y2="{gy}" stroke="#30363d"/>')
        parts.append(f'<text x="{gx}" y="{H - PAD + 16}" fill="#8b949e" font-size="10" '
                     f'text-anchor="middle">{x0 + frac * (x1 - x0):.2f}</text>')
        parts.append(f'<text x="{PAD - 6}" y="{gy + 3}" fill="#8b949e" font-size="10" '
                     f'text-anchor="end">{y0 + (1 - frac) * (y1 - y0):.0f}</text>')
    for xs, ys, color, _ in series:
        parts.append(_svg_polyline([sx(x) for x in xs], [sy(y) for y in ys], color))
    parts.append(f'<text x="{W / 2}" y="18" fill="#e6edf3" font-size="13" '
                 f'text-anchor="middle">{title}</text>')
    parts.append(f'<text x="{W / 2}" y="{H - 6}" fill="#8b949e" font-size="10" '
                 f'text-anchor="middle">{x_label}</text>')
    parts.append(f'<text x="12" y="{H / 2}" fill="#8b949e" font-size="10" '
                 f'transform="rotate(-90 12 {H / 2})" text-anchor="middle">{y_label}</text>')
    parts.append("</svg>")
    legend = " ".join(
        f'<span style="color:{c}">&#9632;</span> {lab}' for _, _, c, lab in series)
    return f'<div class="chart">{"".join(parts)}<div class="legend">{legend}</div></div>'


def build_dashboard(panel: pd.DataFrame, results: dict) -> str:
    smile_days = [("2026-03-18", "#58a6ff", "18 Mar 2026 — pre-crash (ATM 17.6%)"),
                  ("2026-03-19", "#f85149", "19 Mar 2026 — crash day −3.26% (ATM 24.7%)"),
                  ("2026-03-23", "#d29922", "23 Mar 2026 — vol peak (ATM 34.2%)"),
                  ("2026-04-08", "#3fb950", "8 Apr 2026 — +3.78% rally, vol crush (ATM 18.4%)")]
    series = []
    for day, color, label in smile_days:
        smile = smile_for(panel, day, front_expiry(panel, day))
        m = smile["log_moneyness"].abs() <= 0.12
        series.append((list(smile.loc[m, "log_moneyness"]),
                       list(smile.loc[m, "iv"] * 100), color, label))
    smiles = _chart(series, "NIFTY IV smiles — front usable expiry (OTM, real bhavcopy)",
                    "log-moneyness ln(K/F)", "implied vol %")

    metrics = results["day_metrics"]
    xs = list(range(len(metrics)))
    term = _chart([(xs, [m["atm_iv_near_pct"] for m in metrics], "#58a6ff",
                    "ATM IV — front usable expiry"),
                   (xs, [m.get("atm_iv_next_pct", m["atm_iv_near_pct"]) for m in metrics],
                    "#d29922", "ATM IV — next expiry")],
                  "ATM IV term structure across the 23 atlas days",
                  "atlas day (chronological)", "ATM IV %")

    pcs = results["pca"]["pcs"]
    bars = "".join(
        f'<div class="bar-row"><span>PC{p["pc"]} — {p["label"]}</span>'
        f'<div class="bar"><div style="width:{p["variance_pct"]:.1f}%"></div></div>'
        f'<b>{p["variance_pct"]:.1f}%</b></div>' for p in pcs)

    ev_d = results["events"]["biggest_down"]
    ev_u = results["events"]["biggest_up"]
    svi = results["svi_summary"]
    cards = f"""
    <div class="cards">
      <div class="card red"><h3>Biggest down day — {ev_d['date']} ({ev_d['ret_pct']:+.2f}%)</h3>
        <p>ATM IV 17.6% → <b>24.7%</b> on the day, peaking at <b>34.2%</b> two sessions
        later as the selloff extended. Term structure inverted to −3.29 vol pts
        (front above next): a textbook backwardation print.</p></div>
      <div class="card green"><h3>Biggest up day — {ev_u['date']} ({ev_u['ret_pct']:+.2f}%)</h3>
        <p>ATM IV <b>28.5% → 18.4%</b> in one session — a 10-vol-point crush — and the
        25-delta risk reversal narrowed from −7.99 to −2.77 as downside fear unwound.</p></div>
      <div class="card"><h3>Surface dynamics (PCA, 22 consecutive days)</h3>
        <p>PC1 — a parallel level shift, wing-amplified — carries <b>96.3%</b> of daily
        smile-change variance. Curvature is 2.1%, skew rotation just 1.0%: on a daily
        clock the NIFTY surface mostly moves <i>up and down</i>, not <i>sideways</i>.</p></div>
      <div class="card"><h3>SVI fit quality (69 slices)</h3>
        <p>Median RMSE <b>{svi['rmse_median']:.2f} vol pts</b> for DTE ≥ 3 slices, but
        <b>{svi['rmse_median_dte_lt3']:.2f}</b> in expiry week (DTE &lt; 3) — the three worst
        fits in the atlas are all DTE-1 slices. {svi['n_butterfly_violations']} of 69 raw-SVI
        fits fail the butterfly no-arbitrage check somewhere on the grid.</p></div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>VolSurface Atlas — NIFTY IV Surface Dashboard</title>
<style>
 body {{ background:#0d1117; color:#e6edf3; font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
        margin:0; padding:32px; }}
 h1 {{ font-size:26px; margin-bottom:4px; }} .sub {{ color:#8b949e; margin-bottom:24px; }}
 .row {{ display:flex; flex-wrap:wrap; gap:20px; }} .chart {{ flex:1 1 520px; }}
 .legend {{ color:#8b949e; font-size:12px; padding:6px 2px; }}
 .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; margin:24px 0; }}
 .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; }}
 .card.red {{ border-top:3px solid #f85149; }} .card.green {{ border-top:3px solid #3fb950; }}
 .card h3 {{ margin:0 0 8px; font-size:15px; }} .card p {{ color:#c9d1d9; font-size:13.5px; line-height:1.55; margin:0; }}
 .bar-row {{ display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }}
 .bar-row span {{ width:340px; color:#c9d1d9; }} .bar {{ flex:1; background:#30363d; height:14px; border-radius:4px; }}
 .bar div {{ background:#58a6ff; height:14px; border-radius:4px; }}
 footer {{ color:#8b949e; font-size:12px; margin-top:28px; line-height:1.6; }}
</style></head><body>
<h1>VolSurface Atlas — NIFTY implied-volatility surfaces from real NSE bhavcopy</h1>
<div class="sub">220 trading days scanned (4 Nov 2025 – 3 Nov 2026) · 23 atlas days + 22 dynamics days ·
10,633 real OTM IV points · rate 6.5%, OTM options only · settlement prints — see README limitations</div>
{cards}
<h2>PCA of daily smile changes — variance explained</h2>
{bars}
<div class="row">{smiles}{term}</div>
<footer>Data: NSE F&amp;O bhavcopy (exchange end-of-day files). IVs solved from OTM settlement
prints with Black-Scholes (r = 6.5%, q = 0). SVI fits are descriptive; settlement prints are not
traded mids, and expiry-week (DTE &lt; 3) slices are dominated by pin noise. Educational research —
not investment advice.</footer>
</body></html>"""
