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

---

## 2026-09-07 — §9 The structure predicts direction; nothing else does

Full writeup: `direction_study.md`. Regenerate: `scripts/run_direction_study.py`.
New module `stg_infra/stg/direction/`; harness tested in `tests/test_direction.py`.

Stage-2 as the AGCRN post-mortem prescribed it: dormant horizon, direction
label, transparent ladder, and — the point of the exercise — **edges re-estimated
inside every walk-forward fold on training rows only**, so nothing consumes the
published in-sample adjacency. 4,816 labelled (trigger event → next target
event) rows over the same 140-pair grid; 3,487 scored out of fold; 8 expanding
folds, 21-day purge; block-permutation null within trigger series.

**(a) Pooled, nothing predicts.** Sign rule 50.6% (p = 0.35). Expected: the graph
is sparse, so most rows carry no edge. Any pooled evaluation of this task will
report a null regardless of what the structure contains — worth stating before
someone reads a pooled null as a refutation.

**(b) On structure-covered rows it does.** Where the *fold's own* training data
licensed a BH-surviving edge (n = 81): sign rule 63.0% vs 59.3% base, balanced
0.626, **p = 0.041**; at the looser p<0.05 gate (n = 221) 57.5%, **p = 0.022**.
This is the first predictive rather than descriptive evidence for the Stage-1
adjacency — the 75.9% in `artifacts/adjacency_report.md` conditions on in-sample
BH survival and cannot be read this way.

**(c) The edge does the work, not the surprise.** Ablation on the same 81 rows,
edge weight removed and surprise plus context kept: 49.4%, balanced 0.473, AUC
0.467, p = 0.75. Directional content lives in *which pair and with what sign*.

**(d) Capacity hurts, again, at the bottom of the ladder.** Accuracy: sign rule
(0 fitted weights) ≥ edge logit (2) ≥ feature logit (7) ≥ neighbour logit (8).
Same monotone pattern as the AGCRN rungs. Caveat: the logits rank better than
they classify (edge logit AUC 0.680 vs sign rule 0.576) — a fitted weight buys
ordering, not decisions, and at n = 81 the difference is inside noise.

**(e) Identification, quantified.** CPICORE→CPI — a *same-release* pair — supplies
13 of the 81 covered rows at 84.6%. Drop same-release pairs entirely and the
result is 60.7% on 61 rows, **p = 0.13**: point estimate survives, significance
does not. §5's identification problem now has a number attached to it.

**(f) WTI is mass without structure.** 56% of the panel's rows (2,705/4,816), one
BH edge contributing a single scored row. Excluding it as a *trigger*: 64.6% on
82 rows, **p = 0.007**, with the ablation still null (p = 0.14). Bears directly
on the open Universe-A decision — keep WTI as a target, report it separately as a
trigger.

**Reading.** The pre-specified cell was sign rule × BH gate (named in the
post-mortem before the run); the other 20 cells in the table are a family, not
independent tests. All of this is walk-forward *inside* the in-sample block. It
is a sharper hypothesis, not yet a finding — the holdout is untouched, and the
cell to spend it on is now specified (`TODO.md`).

---

## 2026-09-08 — §10 The edges are economically coherent and economically unavailable

Full writeup: `edge_economics.md`. Regenerate: `scripts/run_edge_economics.py`
(spreads cached in `artifacts/effective_spreads.parquet`; `--refresh-spreads`
to re-scan).

**(a) The graph is a policy hub.** Five of the eight BH survivors point at
`FED`/`FEDDECISION`; the rest sit inside the CPI family. The recovered
"structure" is one documented channel — macro data → policy path — plus a
same-release clique, not a rich network. Modest, but it is the channel whose
signs theory fixes in advance.

**(b) Sign restrictions pass, including the falsification cell.** Every hawkish
channel is positive (CPI→FED +0.51, CPICORE→FED +0.48, PAYROLLS→FED +0.53,
CPIYOY→PCECORE +0.72); both dovish cells flip as required (U3→FED −0.14,
U3→FEDDECISION/hike −0.40, neither significant). This is `research_summary.md`
§8.3 item 2's test and it is the check that would have caught the estimator
fitting generic co-movement.

**(c) The most mechanically certain edge in the grid is flat, and that explains
WTI.** WTI→CPIGAS: ρ̂ = +0.04, n = 80, p = 0.73. Oil passes into gasoline CPI by
construction, so this is not a power failure — **WTI has no information event**.
Its settle is public and continuously observable, so at resolution there is no
news; the "surprise" is an artefact of when the ladder was snapped. Consequence
for the universe: *a trigger needs a scheduled information release, not merely a
resolution timestamp.* This independently explains why dropping WTI as a trigger
takes the sign rule from p = 0.041 to p = 0.007 (§9f), and it downgrades
WTI→JOBLESSCLAIMS (ρ̂ = −0.76, n = 11) to a flagged false positive: an estimator
that cannot find oil in gasoline should not be believed finding oil in claims.

**(d) Not tradable — and the reason is not cost.** Ledger over the 81 out-of-fold
`bh`-covered signals, ~34/yr:

| measured from | cents/trade | hit rate |
|---|---|---|
| `p0` (last pre-resolution trade — what the estimator sees) | +0.86 | 0.630 |
| first post-resolution print (the earliest executable price) | **+0.01** | **0.395** |

The whole edge lives in the gap between the last pre-resolution trade and the
first post-resolution print. Median lag between them: **17.5 minutes**; median
hold to the exit print: 13 minutes. It is not a fast move being missed — no
tradable price exists in between, because these books do not print. Costs then
bury the remainder, and **fees dominate spread** (2.90c vs 1.38c per round trip;
Kalshi's p(1−p) schedule is worst near 50c, charged on both legs). Net −4.25c
taker (t = −4.94), −1.50c under the most favourable maker assumption.

`research_summary.md` §6.4's "be a maker not a taker" argument assumed a
multi-day drift and does **not** transfer to a ~13-minute window: an unfilled
resting order misses the move, and fill probability is lowest exactly when the
price is running away. `JOBLESSCLAIMS` could not be costed at all — not one pair
of adjacent opposite-direction trades within 60s exists in the IS block.

Read as §6.4 pre-committed: **economic significance, not profitability**. The
signal is real in the estimator's reference frame and simultaneously unavailable
to any trader, because the price at which it would be captured never exists. The
friction that creates the anomaly is the friction that makes it unexploitable —
a sharper limits-to-arbitrage result than "the margin was too thin".

**(e) The edge is a jump, not a drift — and staleness did not make it.**
Splitting the signed move at the first executable price: jump (`p0` → first
print, untradable) +0.85c at 54.3% hit; drift (first → third print, tradable)
+0.01c at **39.5%** — below chance. So there is no decay curve to arrive early
on; the market reprices once, completely, at its first print after resolution.
`p0` itself is old (median 8.8 h stale, 58% > 6 h), so the estimator's window and
a trader's window are not the same object.

Critically this is **not** §4.1's staleness artifact. If stale references
manufactured the edge, accuracy would rise with staleness; it falls monotonically
— 0.690 (<1h, n=29) / 0.615 (1–24h, n=39) / 0.538 (>24h, n=13), with
corr(staleness, |jump|) = +0.09. Stale prices dilute the finding. §4.1 was the
threat the plan said to check first; checked, and it survives.

And corr(|surprise|, |jump|) = **+0.02**: the first print moves in the right
direction more often than chance by an amount unrelated to the size of the news
— "sign survives, magnitude does not" (§1–2), now in the execution frame.

**(f) Pre-emptive holding: two paths closed, one open.** Holding *before*
resolution is the only way to capture a jump. Conditioning on the surprise is
impossible by construction. Conditioning on an ex-ante bias in the implied mean
fails empirically: no series shows one (max |t| = 1.66 over 17 series,
uncorrected) — **the ladder-recovered implied mean is well calibrated**, itself
worth reporting. What remains is pre-resolution *coherence*: trade the
disagreement between the trigger's implied distribution and the target's price,
and hold through resolution. That restores §6.4's maker argument (hours, not 17
minutes, to get filled) and is the coherence-violation contribution already
scoped in §5 / Phase 3 item 10 — which moves the Phase C panels onto the critical
path.

**(g) Reproducibility bug fixed.** Identical rebuilds of the pair panel differed
by a row: `representative_tickers` broke trade-count ties through an unordered
`group_by`, and `response_panel` broke close-time ties by list order. Both now
use lexicographic tiebreaks (`panel/targets.py`), verified by four separate
processes producing an identical panel hash, with a regression test. Every
artifact built before 2026-09-08 carries the old nondeterminism at the
one-row level.

---

## 2026-09-08 — §11 Widening coverage: the bottleneck is estimator design, not events

The direction result rests on 81 out-of-fold signals (~34/yr) concentrated in
three pairs — too thin to carry a claim. Where the coverage actually goes:

    1,435 macro events in the IS archive
      ->   799 usable trigger events (a real pre-resolution cross-section)
      -> 4,818 matched (trigger event -> next target event) rows
      -> 3,487 scored out of fold
      ->    81 covered by a per-pair BH edge          <- the bottleneck

The loss is at the last step, and it is a *choice*: 140 ordered pairs each
estimated as a free parameter, each needing n >= 10 and BH survival on its own.
Only ~5 clear it per fold.

**Channel pooling recovers 5.6x of that, tested.** Group pairs by economic
channel and impose one theory-fixed sign for the whole channel — hawkish
triggers +1, dovish (U3, JOBLESSCLAIMS) −1 — instead of estimating 29 separate
rho's. Over the data→policy channel (inflation→policy 318 rows/21 pairs, plus
labour→policy 138/8, same-release excluded):

| | value |
|---|---|
| rows | **456** (vs 81) |
| aligned sign agreement | **55.0%** |
| block-permutation null | 50.1% ± 2.3% |
| one-sided p | **0.021** |
| pooled within-pair-ranked rho | +0.085 (p = 0.07) |
| fitted parameters | **zero** — the sign is theory-imposed, not estimated |

The effect is smaller than the per-pair BH cell's 63% (which selects the
strongest pairs, in sample) but applies to 5.6x the rows, ~114 signals/yr rather
than 34. And because the channel rule fits *nothing*, there is no overfitting to
undo — the in-sample/out-of-sample gap that motivates the whole walk-forward
apparatus mostly collapses. Caveat: the channel *definition* was informed by
Stage-1's results, so an OOS confirmation is still owed.

This is `research_summary.md` §8.3 item 1 made concrete — economic structure as
a hard sparsity prior replacing parameters the data cannot support — and it
directly answers the "81 rows, three pairs" concentration objection.

**Other multipliers, measured:**

| lever | multiplier | note |
|---|---|---|
| channel pooling (above) | **5.6x rows** | free, tested, on-thesis |
| use the whole strike ladder | **~11x tickers** | 1,435 macro events carry 15,899 tickers; the pipeline collapses each event to its single most-traded one. Strikes within an event are correlated, so not 11x independent — but it multiplies *executable* opportunities, lets the response be measured at the near-the-money strike rather than the most-traded one, and turns on the moneyness axis `estimate_by_horizon` half-built. |
| sports | **~100x events** | MVENFLMULTIGAMEEXTENDED 132,094 events, MVENFLSINGLEGAME 31,291 — already local. Games *are* scheduled information reveals (unlike WTI), and the graph is mechanically known: game → season win total → division → championship. Verifiable edges rather than guessed ones. |
| more macro series | ~510 events | 141 candidate series outside the registry, mostly gas/oil variants — which by §10(c) lack information events. Low value. |

**Selection criterion carried over from §10(c):** a trigger needs a *scheduled
information release*, not merely a resolution timestamp. That is what makes
sports attractive and the gas/oil candidates unattractive.

### §11.1 Trading beyond the ATM contract — capacity, not power

**Depth.** The pipeline uses one representative ticker per target event. Measured
usable strikes per event: FED 9.7 traded / 7.5 with >=20 trades / 4.5 with >=100;
CPI 7.1 / 5.4 / 3.4; CPIYOY 9.6 / 5.1 / 2.2; PAYROLLS 5.1 / 4.2 / 1.9. So
**4-7 genuinely usable strikes per event**, a 4-7x multiplier. Off-ATM is not a
thin tail: the 15-30c band carries a *higher* median trade count than ATM for
CPICORE (80 vs 53), PAYROLLS (139 vs 110) and CPIYOY (61 vs 47).

**Spread by moneyness** (same §4.2.1 estimator, window-invariant in every band):

| band | median | mean | n pairs |
|---|---|---|---|
| ATM ±15c | 2.0c | 2.86 | 2,540 |
| 15-30c out | 2.0c | 2.21 | 4,567 |
| 30-40c out | 2.0c | 2.08 | 2,766 |
| >40c out | **1.0c** | 1.37 | 5,081 |

Spreads *narrow* away from the money rather than widening.

**No cost-optimal strike, though.** A surprise shifts the implied distribution, so
the move at strike k scales with the density there — maximal at ATM. The Kalshi
fee `0.07·p(1-p)` is *also* maximal at ATM and steps down (ceil-to-cent) once
p < ~18c. Combining move, fee and the measured spread:

| strike | price | move/0.1sd | fee | spread | cost | move/cost |
|---|---|---|---|---|---|---|
| ATM | 50c | 3.99c | 4c | 2c | 6.0c | **0.67** |
| 0.5sd | 31c | 3.52c | 4c | 2c | 6.0c | 0.59 |
| 1.0sd | 16c | 2.42c | 2c | 2c | 4.0c | **0.61** |
| 1.5sd | 7c | 1.30c | 2c | 1c | 3.0c | 0.43 |
| 2.0sd | 2c | 0.54c | 2c | 1c | 3.0c | 0.18 |

ATM and ~1sd out are within noise; beyond that it degrades. The fee saving
off-ATM is cancelled because the **spread has a 1c tick floor that does not scale
with sensitivity**, so it eats proportionally more of a smaller move. (A
fee-only calculation makes ~1sd look like a sweet spot at 1.21x — it is not, once
the measured spread is included.) Past ~2sd the move per 0.1sd shift falls below
the 1c tick and cannot be expressed at all.

**The load-bearing caveat.** Strikes on one event are all driven by the same shift
in the same distribution: 4-7 strikes is **one bet in larger size, not 4-7
independent observations**. The ladder therefore multiplies deployable *capacity*
and does nothing for the n=81 *significance* problem.

| lever | capacity | significance |
|---|---|---|
| strike ladder (4-7x) | yes | **no** |
| channel pooling (5.6x) | yes | **yes** |
| sports (~100x) | yes | **yes** |

So the order is **channel pooling → sports → ladder**, the ladder applied as a
capacity multiplier to whatever survives. Its one non-capacity use: measuring the
response at the *near-the-money* strike rather than the *most-traded* one is a
cleaner measurement, and it activates the moneyness axis beside the term
structure already in `structure/horizon.py`.

### §11.2 Correction: the §11/§11.1 population is not the signal window

The ledger numbers (81 signals, −4.25c, 17.5 min lag) are computed on the
covered rows. The ladder-depth and spread numbers in §11/§11.1 are computed on
the *whole life* of the six target series. Conditioning them on the moment a
signal actually fires changes both, in the same direction.

**(a) Spreads roughly double at release time.** Same §4.2.1 estimator, trades
split by proximity to a trigger resolution:

| window | median spread | mean | n pairs |
|---|---|---|---|
| within 0.5 h of a release | **2.0c** | 2.84c | 375 |
| within 6 h of a release | **2.0c** | 2.37c | 2,040 |
| quiet | **1.0c** | 1.98c | 14,514 |

The book widens exactly when the signal exists — the textbook inventory /
adverse-selection response to a release. **The ledger charged 1.38c, the
unconditional figure, so its net of −4.25c is optimistic**; at the
release-conditional median (2.0c) it is ≈ **−4.9c**, and at the conditional mean
≈ −5.7c. The conclusion is unchanged and strengthened; the reported number was
generous to the strategy.

**(b) The ladder is deep over a contract's life and shallow when it matters.**
Distinct strikes actually trading within a window after a resolution:

| window | median strikes/event | mean | share of events with >=3 |
|---|---|---|---|
| 30 min | **1.0** | 1.79 | 0.19 |
| 120 min | 2.0 | 2.16 | 0.29 |
| 360 min | 2.0 | 2.50 | 0.38 |

So §11.1's "4-7 usable strikes per event" is a *lifetime* count. In the dormant
window the median event has **one strike trading** — essentially the single
representative ticker the pipeline already uses. **The 4-7x capacity multiplier
does not exist for the post-resolution trade.**

It does exist for the pre-resolution coherence strategy (§10 Path C), where the
position is established over hours or days and the whole ladder is reachable.
That is a second, independent reason to prefer Path C, and it demotes the
ladder from an independent lever to a component of that one.

Both corrections point the same way: **at signal time these markets are
simultaneously thinner and wider than their own averages.** The friction that
sustains the anomaly peaks exactly where the anomaly is.
