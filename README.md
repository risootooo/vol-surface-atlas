# VolSurface Atlas — NIFTY Implied-Volatility Surfaces from Real NSE Data

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)
![Data](https://img.shields.io/badge/data-NSE%20bhavcopy%20(real)-orange)
![Slices](https://img.shields.io/badge/SVI%20slices%20fitted-69-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**One line:** 220 real trading days of NSE bhavcopy, turned into
implied-volatility surfaces — SVI fits, a PCA of how the surface
actually moves, and the two biggest days of the year dissected
strike-by-strike.

> The NIFTY surface on a daily clock is **96.3% a level shift** — skew
> rotation is under 1%. The worst day of the year was preceded by the
> *calmest* surface of its week. And no smooth smile model can fit
> expiry week, because at DTE 1 the data stops being a smile at all.
> All three claims are measured below, from the exchange's own files.

## Headline numbers (all real, all reproduced by `python main.py`)

| Quantity | Value |
|---|---|
| Trading days scanned | 220 (4 Nov 2025 – 3 Nov 2026) |
| Real OTM IV points extracted | 10,633 (23 atlas days + 22 dynamics days) |
| Biggest up day | **8 Apr 2026, +3.78%** — ATM IV crushed 28.5% → 18.4% in one session |
| Biggest down day | **19 Mar 2026, −3.26%** — ATM IV 17.6% → 24.7%, peaking at 34.2% two sessions later |
| PC1 share of daily smile-change variance | **96.32%** (level shift, wing-amplified) |
| SVI fit quality | median RMSE **1.06 vol pts** (DTE ≥ 3) vs **3.63** in expiry week |
| Butterfly no-arb violations | 12 of 69 raw-SVI slices (reported, not hidden) |

## Quickstart

```bash
pip install -r requirements.txt   # numpy, pandas, scipy (pytest for tests)
python main.py                    # event cases + PCA + SVI summary
python main.py --dashboard        # also regenerates docs/dashboard.html
python -m pytest tests            # 27 tests
```

Open [`docs/dashboard.html`](docs/dashboard.html) — a single
self-contained file (inline SVG, no CDN, works offline): smiles for
the pre-crash / crash / vol-peak / rally days, the ATM term structure
across all atlas days, and the PCA variance split. Full case studies
with before/after tables: [`docs/CASES.md`](docs/CASES.md).

## The cases

- **Case A — the biggest days of the year.** Into the −3.26% crash,
  ATM IV *fell* for two sessions (19.60% → 17.55%), spiked +7.15 pts
  on impact while the skew *flattened* (the put wing was already
  expensive), and peaked two sessions later at 34.16% with the term
  structure inverted to −3.29 vol pts. Into the +3.78% rally, the
  mirror image: a 10.08-point one-day vol crush and RR25 unwinding
  from −7.99 to −1.49 in 48 hours.
- **Case B — PCA of surface dynamics.** On 22 consecutive days around
  those events, PC1 (parallel level shift, wings moving ≈2.6× the
  belly) explains 96.32% of daily smile-change variance; curvature
  2.15%; skew rotation 0.95%. PC labels are assigned from the loading
  shapes (correlations with k and k², sign changes) — diagnostics
  included, so you can argue with them.
- **Case C — where SVI breaks.** All three worst fits are DTE-1
  slices (RMSE up to 4.52 vol pts vs 0.31 for the best monthly
  slices). Expiry-week settlement prints are pin noise, not a smile —
  which is why every level here is quoted on the first expiry with
  DTE ≥ 3 ("front usable expiry" — see below).

## The maths

**Smile metrics** (per day, front usable expiry, in log-moneyness
k = ln(K/F)):

- ATM IV: smile interpolated at k = 0.
- 25-delta risk reversal `RR25 = IV(Δ=+0.25 call) − IV(Δ=−0.25 put)` and
  butterfly `BF25 = (IV_call25 + IV_put25)/2 − IV_ATM`, with the
  delta strikes located on a dense k-grid using the smile's own IV
  (the standard desk approximation, stated as such).
- Skew slope: OLS of IV on k over |k| ≤ 0.05, quoted in vol points
  per 10% moneyness.
- Term slope: ATM IV(next expiry) − ATM IV(front), vol points.

**SVI (raw, Gatheral).** Total variance
`w(k) = a + b[ρ(k−m) + √((k−m)² + σ²)]`, fitted per day/expiry by least
squares on w. Butterfly no-arbitrage is checked with Gatheral's

```
g(k) = (1 − k·w′/(2w))² − (w′)²/4 · (1/w + 1/4) + w″/2  ≥ 0
```

evaluated on a ±30% grid; violations are reported per slice in
`docs/results.json`.

**PCA of dynamics.** Daily changes of the smile on a fixed
15-point k-grid, demeaned, decomposed by SVD (the PCA of the change
matrix). Variance shares are sᵢ²/Σs².

## Data provenance & regeneration

Source: NSE F&O **bhavcopy** — the exchange's official end-of-day
file, one `fo_YYYYMMDD.zip` per trading day (220 files, Nov 2025 –
Nov 2026), columns TradDt / FinInstrmTp (`IDO`) / TckrSymb / XpryDt /
StrkPric / OptnTp / ClsPric / UndrlygPric / OpnIntrst / TtlTradgVol.
The biggest-move days were found by rebuilding the spot series from
these same files (`data/nifty_spot_series.csv`), never from a news
calendar.

```bash
python scripts/build_panel.py --bhavcopy-dir /path/to/bhavcopy --out-dir data
```

produces the shipped `data/iv_panel_sample.csv` (1.18 MB, 10,633
rows): atlas days = one representative day per month + the two event
clusters (±2 trading days), nearest three expiries; dynamics days =
22 consecutive days centred on the biggest absolute move, nearest
expiry.

## Conventions that matter (the lessons are in the data)

- **OTM options only.** Puts below spot, calls at/above. ITM
  settlement prints on this feed regularly sit below discounted
  intrinsic value — there is *no* implied vol for them, and tools
  that print one are inventing it. The solver here returns NaN.
- **Premium = ClsPric** (settlement print), rate **6.5%**, q = 0.
- **Front usable expiry = DTE ≥ 3** for all quoted levels. The
  literal front expiry in expiry week printed a 42.8% ATM IV on
  23 Mar 2026 where the DTE-7 expiry said 34.2% — expiry noise,
  excluded from levels but analysed in Case C.

## Limitations — read before citing a number

- Settlement prints are not traded mids: no bid/ask, no staleness
  filter beyond OI > 0. IV *levels* carry that noise; the event and
  PCA results are differences/patterns and are more robust than any
  single print.
- SVI fits to these prints are descriptive, not arbitrage-free
  surfaces (12/69 slices fail the butterfly check somewhere — see
  Case C). Short-DTE SVI parameters can pin at bounds and lose
  economic meaning even when the fit is tight.
- The PCA window is a high-vol regime (it contains both of the
  year's biggest moves by construction). The 96.3% level share is
  that window's measurement, not a universal constant.
- 25-delta strikes use the smile's own interpolated IV on a grid —
  an approximation to a full delta inversion, adequate at these
  maturities.

## Structure

```
src/volatlas/   pricing.py (BS price/delta/vega) · iv.py (solver, no-arb honest)
                chain.py (bhavcopy -> OTM IV panel) · surface.py (smile metrics)
                svi.py (raw SVI + Gatheral g) · dynamics.py (PCA via SVD)
                events.py (data-detected event windows) · report.py (dashboard)
scripts/        build_panel.py (raw bhavcopy -> data/) · run_analysis.py
data/           iv_panel_sample.csv (1.18 MB) · nifty_spot_series.csv (220 days)
docs/           CASES.md · dashboard.html · results.json (every quoted number)
tests/          27 tests: pricing, IV round-trips, SVI recovery, PCA, chain extraction, real-panel invariants
main.py         CLI — prints the cases, writes docs/results.json
```

Educational / research work — not investment advice.

## Author

Samrat Banerjee — [github.com/risootooo](https://github.com/risootooo)
