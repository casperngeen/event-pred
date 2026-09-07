# Research Log

Running log of empirical findings. Newest entry first. Companion to
`research_summary.md` (the standing plan) and `data_prep_plan.md` (data prep).

Scripts backing every claim here live in `analysis/exploratory_2026_08/`.

---

## 2026-08-20 — Pearson fails, sign survives

**One-line summary:** every magnitude-weighted test of cross-market underreaction
came back null or unstable; a sign-based test on the same data survives a
correctly-specified permutation null at p≈0.001–0.006. The signal, if real, is in
the *direction* of the surprise, not its size.

All analysis below is **in-sample only** (pre-2026). The 2026 OOS block was not
touched — see `data/MANIFEST_new_pulls.md`.

---

### 1. Why Pearson kept failing

Four separate magnitude-based attempts, all of which dissolved on inspection:

**(a) Pairwise sweep — 32 tests, zero significant.** 9 trigger→target pairs × 4
trade-time horizons, correlating `|surprise|` against `|response|`. Max |t| = 1.64;
Bonferroni at 32 tests needs |t| ≈ 3.5. Full table in `pairwise_results.json`.

**(b) The highest-powered pairs were the emptiest — and were also broken.**
WTI→CPI (n=221) and WTI→FED (n=211) had ~10× the power of anything else and showed
|r| ≤ 0.084. But these results should be **discarded, not treated as informative
nulls**: WTI is 7,818/9,222 *bucket* contracts, and the threshold parser silently
dropped all of them (§2 below). The WTI ladder used for bracketing was the 15% that
happened to parse.

**(c) Sign instability at n≈32.** CPI→FEDDECISION flipped from r ≈ +0.20/+0.25 to
r ≈ −0.03/−0.22 between two runs differing only by one boundary filter and a single
dropped observation. At this sample size the magnitude estimate is not stable enough
to read directionally in either direction.

**(d) An apparently significant co-movement result was three data points.**
`corr(FEDDECISION response, FED response)` for a shared CPI trigger: r = 0.493,
t = 3.05, n = 31. Dropping the top 1/2/3 events by combined |response| collapses it
to 0.301 / 0.227 / **0.066**. Several of the top events move in *opposite*
directions (`KXCPI-24NOV`: FEDDECISION +2 vs FED −12), and same-sign fraction sits
*below* chance (35–45%) throughout. Retracted. Separately, FED and FEDDECISION are
close to two measurements of one quantity, so their co-movement was never a clean
test of multi-target diffusion anyway.

### 2. The diagnosis: magnitude is the noisy component

CPI (month-over-month) and CPIYOY (year-over-year) are **separate contract ladders
resolving from the same BLS print on the same day**. If the PDF-based surprise
measure is real, they must agree. Matched on release date, n = 37:

| Cross-check | r | t |
|---|---|---|
| **Signed surprise** | **0.686** | **5.58** |
| implied_std | 0.354 | 2.24 |
| **\|surprise\|** | 0.242 | 1.48 |

Direction agrees strongly; magnitude does not. Concrete cases: 2023-01-12 gives CPI
+0.009 vs CPIYOY +0.112 (same sign, 12× apart); 2023-03-14 gives −0.013 vs −0.096
(same sign, 7× apart).

This validates the surprise measure *and* invalidates the test design. Pearson is
magnitude-weighted, so it dilutes a sign-only signal toward zero. It also explains
the CPI/CPIYOY contradictions in (c) — those were never conflicting findings, just
two noisy magnitude estimates of one thing.

### 3. The sign result

**Design.** 12 trigger→target cells with directions fixed *in advance* by theory:
hawkish surprise (higher inflation / stronger jobs) raises P(hike), lowers P(cut);
**U3 signs flip**, since higher unemployment is dovish. "Aligned surprise" =
signed surprise × predicted direction, so the prediction is uniformly positive.
Statistics are rank/sign-based (Spearman, sign-agreement), hence magnitude-robust.

**Significance via permutation**, not parametric tests: shuffle surprises while
holding all target responses fixed, breaking only the trigger→target pairing. This
preserves the real dependence structure (shared target contracts, overlapping
windows) which the earlier per-cell binomial treatment did not.

**Dormant horizon (next 3 trades after trigger resolution):**

| Cell | n | sign% | Spearman |
|---|---|---|---|
| CPI→FEDDECISION/hike | 30 | 90% | 0.397 |
| CPIYOY→FEDDECISION/hike | 30 | 80% | 0.472 |
| CPIYOY→FEDDECISION/cut | 29 | 74% | 0.418 |
| CPI→FED | 46 | 68% | 0.492 |
| PAYROLLS→FED | 20 | 76% | 0.386 |
| PAYROLLS→FEDDECISION/cut | 20 | 72% | 0.179 |
| CPI→FEDDECISION/cut | 29 | 63% | 0.298 |
| U3→FEDDECISION/hike | 26 | 62% | 0.048 |
| U3→FED | 42 | 61% | 0.042 |
| CPIYOY→FED | 35 | 53% | 0.185 |
| PAYROLLS→FEDDECISION/hike | 18 | 50% | 0.257 |
| U3→FEDDECISION/cut | 25 | 47% | 0.011 |
| **POOLED** | **224** | **65.6%** | **0.265** |

**Results under both permutation schemes:**

| Horizon | Variant | n | sign% | Spearman | p(sign) | p(rho) |
|---|---|---|---|---|---|---|
| dormant | all cells | 224 | 65.6% | 0.265 | **0.0030** | **0.0056** |
| dormant | dedup | 149 | 66.4% | 0.285 | **0.0012** | **0.0042** |
| liquid | all cells | 270 | 59.3% | 0.173 | 0.042 | 0.064 |
| liquid | dedup | 173 | 59.0% | 0.205 | 0.087 | 0.064 |

p-values above use **block permutation by release period**, which preserves the
CPI↔CPIYOY (r=0.686) and PAYROLLS↔U3 correlations. This matters: the naive
within-trigger shuffle gave a null sd of 2.9% and p < 0.0001, while the correct
block null gives sd 5.5% and p = 0.0030. **The naive version overstated
significance by roughly an order of magnitude** — worth remembering as a general
trap in this setting.

### 4. Two design worries, checked

**Hike/cut are NOT near-duplicates.** Expected `corr(R_hike, R_cut) ≈ −1` for the
same FOMC event; measured **−0.002** (dormant) and **+0.047** (liquid). Pooling
both sides was legitimate; n was not padded. Deduplicating to one contract per
(trigger, event) *raises* the result (66.4%, ρ=0.285) rather than lowering it —
the opposite of what redundancy-driven inflation would do.

**Dormant > liquid, contradicting the liquidity-gated hypothesis.** The idea that
information sits unpriced until the target market wakes up predicts liquid should
dominate. It does not (65.6% vs 59.3%), and the liquid result does not survive the
corrected null (p = 0.04–0.09, weakening under dedup). The effect concentrates
*immediately* after the trigger, while the target is still thin. This argues for
fast partial repricing over delayed incorporation.

### 5. Caveats — read before citing any of this

- **This is a specification search.** Magnitude → signed → sign-only were tried in
  sequence, and the one that worked is reported. The FDR discipline demanded of the
  CA report's "22 relationships" applies here too. **Treat the sign result as an
  in-sample-generated hypothesis, not a finding.** The 2026 OOS block is the correct
  test and remains untouched.
- **The U3 falsification cell underperforms.** U3 was the strongest test — a real
  effect had to appear there with *flipped* signs, which a common artifact could not
  fake. In the dormant horizon it is ~zero (ρ = 0.048 / 0.011 / 0.042). The pooled
  dormant result is carried by CPI/CPIYOY/PAYROLLS. U3 looks better in the liquid
  window (0.388 / 0.216 / 0.284), but that is the horizon that failed overall.
- **Identification untouched.** Nothing here separates "information propagated
  through the market" from "both contracts respond to the same public release"
  (`research_summary.md` §5). A sign test cannot address this.
- **The surprise measure still rests on a contaminated ladder** — see §7.

### 6. Implications for the plan

1. **Stage-1 structure estimation should be sign/rank-based**, not correlation-based,
   with FDR across the full pair grid, IS only. This is the first evidence that it
   might find something.
2. **Permutation, not parametric tests**, for anything pooled across overlapping
   cells — and block-permute on the shared release period.
3. **Drop `|surprise|` as the primary feature.** `research_summary.md` §6.2 frames
   the falsifiable prediction as "drift magnitude scales with surprise magnitude";
   the data says magnitude is the unreliable half. Reframe around direction.
4. **The liquidity-gated reframe is not supported.** Worth stating as a tested and
   rejected alternative rather than silently dropping.

### 7. Measurement problems quantified (bear on everything above)

**Cross-sectional staleness.** `build_daily` forward-fills, and legs of the same
ladder go stale on *different* days — so a recovered "implied distribution" mixes
prices last traded days or months apart, and never existed at any instant.
Event-days with ≥2 legs:

| Series | Event-days | All legs fresh | Median max-stale | p90 | Median freshest↔stalest gap |
|---|---|---|---|---|---|
| CPI | 3,555 | 12.0% | 10d | 153d | 8d |
| CPIYOY | 2,338 | 10.3% | 13d | 127d | 5d |
| U3 | 2,235 | 12.2% | 9d | 102d | 8d |
| GDP | 1,632 | 18.4% | 6d | 42d | 6d |
| PAYROLLS | 755 | 28.5% | 2d | 12d | 2d |
| JOBLESSCLAIMS | 115 | 39.1% | 1d | 4d | 1d |

Trade-time indexing does **not** fix this — trade-time is per-market, so there is no
shared trade clock across a ladder's legs. It fixes the *horizon* half of the
problem only.

**Activity is concentrated at the end of life.** CPI, by days-to-close:

| Days to close | All legs fresh | Median trades/day |
|---|---|---|
| 0–3 | 58.3% | 44 |
| 4–7 | 24.5% | 24 |
| 15–30 | 17.9% | 14 |
| 31–60 | 7.2% | 1 |
| 61–120 | 0.4% | **0** |

Beyond ~30 days these markets are effectively dead. Restricting PDF construction to
≲14 days-to-close discards ~89% of rows but very little actual information.

**Spreads are wider than trade data implies.** On `KXFEDDECISION-26JAN-H0`
(117,769 order-book snapshots, the one macro ticker in `kalshi_orderbooks.jsonl`):

| Method | Spread |
|---|---|
| Order book (ground truth, continuous) | **8.0¢** |
| Naive trade bounce (consecutive-print diff) | 1.0¢ |
| Taker-pairing effective spread (≤60s) | **no valid pairs at all** |

Across 127 trades in 5 weeks, not once did opposite-direction trades land within 60s
of each other. Trade-based spread estimators only observe moments when trading
happens — precisely when spreads are tightest — so they are **selection-biased
lower bounds**, and here the bound is off by 8×. `research_summary.md` §4.2.1's 1–2¢
figures are trade-weighted and likely dominated by near-resolution front contracts.

**Resting-order fill probability** (same contract, 792 hourly placements at the touch;
optimistic = front of queue, conservative = back):

| Horizon | Optimistic | Conservative |
|---|---|---|
| 1 hour | 1.6% | 0.3% |
| 1 day | 20.8% | 11.9% |
| 7 days | 54.9% | 35.5% |
| Rest of window | 68.9% | 63.9% |

**Chasing the touch is worse than resting**: cancel-replacing on every touch shift
gives 13.3% / 5.4% (time-weighted) vs 54.9% / 35.5% for a static 7-day rest. The
touch moves every ~90s median (2,117 shifts in 5 weeks) while fill-eligible trades
arrive roughly every 2 days, so each replace discards queue priority that almost
never had time to mature. Argues against naive touch-chasing in §6.4.

*Caveat on all execution numbers: one thin, far-dated contract. Not representative
of a liquid near-meeting contract.*

### 8. Code fixed this session (`stg_infra/stg/events/implied.py`)

Three bugs, all of which silently corrupted the surprise measure:

1. **`parse_threshold` dropped entire series.** Required a literal `-T` prefix.
   JOBLESSCLAIMS 0/132, ISMPMI 0/49, ADP 143/150 — failing silently via
   `.filter(is_not_null())`. Now parses from `yes_sub_title` with ticker fallback:
   **100% coverage on all threshold ladders**.
2. **Contract types were conflated.** Added THRESHOLD / BUCKET / CATEGORICAL
   classification. Kalshi has three structurally different contract shapes and
   feeding a bucket ladder to `recover_pdf` produces nonsense. Measured:
   5,169 threshold / 7,875 bucket / 204 categorical. **WTI is 85% bucket** — which
   is why §1(b) above is discarded rather than reported.
3. **`resolved_value` hardcoded `spacing = 0.1`.** Correct for CPI/U3/GDP, wrong by
   ~6 orders of magnitude for PAYROLLS (~50,000 strike spacing). Affected 23
   one-sided events; e.g. PAYROLLS-24SEP 250,000.05 → **275,000**.
4. **Inclusive vs exclusive conventions coexisted** ("Above 4.3%" vs "At least
   225000" vs "216,000 or above"). Now detected and normalised to exclusive.

**Still outstanding:** the `build_daily` forward-fill refactor (§7); a PMF path for
bucket contracts, so WTI — the largest series by event count — is currently absent
from implied-mean output rather than handled; and the coherence-violation counter
(`np.clip` in `recover_pdf` discards exactly the monotonicity violations
`research_summary.md` §5 wants to measure).
