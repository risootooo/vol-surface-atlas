# Case Studies — Real Events in the NIFTY Vol Surface

All numbers below are computed by this repo from real NSE F&O bhavcopy
files (see README for provenance). Events were *detected from the data*
— the biggest single-day moves of the 220-trading-day window
(4 Nov 2025 – 3 Nov 2026) — not picked from a news calendar.

Conventions for every table: ATM IV / RR25 / BF25 / skew / term slope
are quoted on the **front usable expiry** (first expiry with DTE ≥ 3;
expiry-week DTE-1 prints are expiry noise, see Case C). RR25 and BF25
in vol points; skew = vol points per 10% log-moneyness; term slope =
ATM IV(next expiry) − ATM IV(front), vol points (negative = the front
is bid above the back — backwardation).

---

## Case A — The two biggest days of the year

### A1. Biggest down day: Thursday 19 Mar 2026, NIFTY −3.26% (spot 23,002.15)

| Date | Offset | Spot Δ | ATM IV | RR25 | BF25 | Skew | Term slope |
|---|---|---|---|---|---|---|---|
| 17 Mar | −2 | +0.74% | 19.60% | −5.96 | +0.58 | −12.32 | +0.20 |
| 18 Mar | −1 | +0.83% | 17.55% | −5.76 | +0.85 | −11.33 | −0.04 |
| **19 Mar** | **0** | **−3.26%** | **24.70%** | **−4.18** | **+0.60** | **−8.61** | **−1.10** |
| 23 Mar | +1 | −2.13% | 34.16% | −6.98 | −0.72 | −8.43 | −3.29 |
| 25 Mar | +2 | +3.53% | 26.02% | −6.45 | −0.04 | −11.73 | −0.00 |

What the surface actually did:

- **Vol fell into the crash.** Two up days beforehand bled ATM IV from
  19.60% to 17.55% — the surface was at its calmest the session before
  the worst day of the year.
- **Impact day: +7.15 vol points** (17.55% → 24.70%), but the *skew
  flattened* (−11.33 → −8.61) and RR25 *narrowed* (−5.76 → −4.18):
  the whole surface lifted, so the put wing — already expensive —
  rose least in relative terms. A crash does not steepen an already
  steep smile; it shifts it.
- **The worst vol print came after**: with losses extending on 23 Mar
  (−2.13%), ATM IV peaked at 34.16% and the term structure inverted
  hard (−3.29) — front-month panic, the classic backwardation print.
- The rebound (+3.53% on 25 Mar) knocked 8 points off ATM IV in one
  session, but RR25 stayed wide (−6.45): spot recovered faster than
  fear did.

### A2. Biggest up day: Wednesday 8 Apr 2026, NIFTY +3.78% (spot 23,997.35)

| Date | Offset | Spot Δ | ATM IV | RR25 | BF25 | Skew | Term slope |
|---|---|---|---|---|---|---|---|
| 6 Apr | −2 | +1.12% | 30.59% | −5.78 | +0.18 | −10.08 | −2.77 |
| 7 Apr | −1 | +0.68% | 28.52% | −7.99 | +0.22 | −13.94 | −2.14 |
| **8 Apr** | **0** | **+3.78%** | **18.44%** | **−2.77** | **+1.12** | **−7.66** | **−0.18** |
| 9 Apr | +1 | −0.93% | 19.73% | −1.49 | +0.16 | −4.35 | +0.38 |
| 10 Apr | +2 | +1.16% | 16.91% | −2.90 | −0.07 | −6.92 | +1.02 |

- The mirror image, with one difference: **vol was still elevated
  from the March stress** (28–31%) when the rally hit, so the crush
  was larger than the crash-day spike — ATM IV fell **10.08 points in
  one session**.
- RR25 unwound violently: −7.99 → −2.77 → −1.49. Downside protection
  that was bid all through March was dumped in 48 hours.
- Term structure healed from backwardation (−2.14) to contango
  (+1.02) in three sessions — the surface declaring the emergency over
  before spot had fully retraced.

---

## Case B — PCA of surface dynamics: the surface moves up/down, not sideways

22 consecutive trading days (18 Mar – 24 Apr 2026, spanning the crash
and the rally), nearest-expiry smile interpolated onto a 15-point
log-moneyness grid (k ∈ [−0.10, +0.10]), PCA on the 21 daily IV changes
(vol points), via SVD of the demeaned change matrix.

| PC | Variance explained | Shape (from its loadings) |
|---|---|---|
| PC1 | **96.32%** | Parallel level shift, wing-amplified (all loadings positive, 0.139–0.369; wings move ≈2.6× the belly) |
| PC2 | 2.15% | Curvature — belly vs wings (loadings hump-shaped, corr with k² = −0.88) |
| PC3 | 0.95% | Skew rotation — twist (loadings corr with k = −0.81, one sign change) |
| PC4–PC15 | 0.58% combined | residual shape noise |

**Reading:** on a daily clock, 96% of what the NIFTY surface *does* is
re-price the level of volatility — and it re-prices the wings more
than the belly. Skew rotation, the thing option desks talk about most,
is a rounding error at this horizon (0.95%). Anyone running a
"skew trade" on daily NIFTY data is trading the remaining 4%.

(Caveat, stated plainly: this window is a high-vol regime — Mar–Apr
2026 contained both of the year's biggest moves. A calm-regime window
would likely show a smaller PC1 share. The method, not the exact
96.3%, is the portable result.)

---

## Case C — SVI fit quality: expiry week is un-fittable

Raw SVI fitted to all 69 (day × expiry) slices in the atlas panel
(least squares on total variance; RMSE back in vol points):

| Slice set | Slices | Median RMSE |
|---|---|---|
| All | 69 | 1.09 vol pts |
| DTE ≥ 3 | 63 | 1.06 vol pts |
| DTE < 3 (expiry week) | 6 | **3.63 vol pts** |

The three worst fits in the atlas are all DTE-1 slices:

| Date | Expiry | DTE | RMSE |
|---|---|---|---|
| 15 Dec 2025 | 16 Dec 2025 | 1 | 4.52 |
| 15 Jun 2026 | 16 Jun 2026 | 1 | 4.28 |
| 16 Mar 2026 | 17 Mar 2026 | 1 | 3.89 |

The three best are all long-dated monthly slices (DTE 18–20,
RMSE 0.31–0.34). This is a *data* result, not a model failure: at
DTE 1 the settlement print of a near-ATM option is dominated by pin
risk and discrete-tick effects, and no smooth 5-parameter smile can —
or should — fit it. It is exactly why every level in this repo is
quoted on the front usable (DTE ≥ 3) expiry.

**No-arbitrage honesty:** 12 of 69 raw-SVI fits violate the butterfly
condition (Gatheral's g < 0) somewhere on a ±30% log-moneyness grid
(worst min g = −0.176; among violators the negative region covers a
median ~4% of the grid). Least-squares SVI on settlement prints is a
*description* of the smile, not an arbitrage-free surface — the repo
reports the violations per slice in `docs/results.json` instead of
hiding them.

One more honest wrinkle: on the crash day itself (19 Mar, DTE-5
slice) SVI fits *well* (RMSE 0.95, g-clean) but its `a` parameter pins
at the zero bound — at 5 days to expiry total variance is so small
that the parameterisation degenerates. Fit quality and parameter
meaning are different things; both are reported.
