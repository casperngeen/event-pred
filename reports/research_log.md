# Research Log

Running log of empirical findings, oldest entry first (§1 at the top, the
newest section at the bottom). Companion to `research_summary.md` (the standing
plan) and `data_prep_plan.md` (data prep).

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

> **SUPERSEDED by §14.10 (2026-09-14). Do not cite this section's headline.**
> Both halves of the title are now wrong. The sign rule's significance rested on
> a look-ahead in instrument choice (§14.5), an asymptotic p-value the data's
> 98% tie rate does not license (§14.4), and — for every re-run between
> 2026-09-08 and 2026-09-13 — a cached pair panel that never rebuilt (§14.7).
> On the corrected panel the sign rule clears no gate (p = 0.23 at `p05`), and
> the `no_structure` ablation that this section called "decisive" reverses for a
> mechanical reason: it was reading the target's bounded price level. The
> reasoning and the measurement design below stand; the numbers and the
> conclusion do not.

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

> **Partly superseded by §14.11 (2026-09-14).** The ledger is re-priced on the
> corrected panel: net −3.15c (was −4.25c), t = −4.31, entry lag 33.6 min (was
> 17.5). The *conclusion* — coherent, not exploitable — is unchanged and firmer.
> The sign-restriction results in this section are unaffected. But per §14.10
> the underlying rule no longer has demonstrated skill, so read the ledger as an
> upper bound on this class of signal rather than as costing a validated edge.

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

> **Arithmetic superseded by §14 (2026-09-14).** The funnel below (1,435 → 799 →
> 4,818 → 3,487 → 81) is pre-correction at every stage; it is now 1,435 → 519 →
> 3,127 → 2,347 → 45. The diagnosis — that the loss is concentrated in per-pair
> BH selection and is a *choice* — is unaffected, and channel pooling remains
> the proposed remedy.

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

---

## 2026-09-10 — §12 Data quality: the archive is a convenience sample

**One-line summary:** `data/trades/` was assembled by fetching one ticker at a
time from a supplied list, and that list was never complete. Whole series are
missing, and — the part that matters — several *registered* trigger series are
silently running on a third to two-thirds of their events. Every `n` in this log
is a floor set by collection, not a measurement of market activity.

Backing script: `analysis/data_quality_2026_09/coverage_audit.py`
(captured output beside it). All figures below are reproducible from it.

### §12.1 How this surfaced, and the wrong explanation

Registering INXU/INXD/NASDAQ100U as asset-price targets (commit `410f8fe`)
turned up three candidates — NASDAQ100D, TNOTED, USDJPYH — carrying real
aggregate `volume` in the markets metadata with **zero trades** in the local
archive. RATECUT, already registered, was the same: 9,758,096 aggregate volume
over 11 markets, no trades at all.

That was attributed to Kalshi's ~67-day trade retention (measured 2026-09-08:
earliest served 2026-07-02). **That explanation is wrong** and is corrected
here. Retention predicts absence tracks *date*. It does not:

| series | markets window | trades |
|---|---|---|
| NASDAQ100D | 2022-05 → 2024-01 | **0** |
| INXD | 2022-05 → 2024-12 | **565,941** (2022-05-12 → 2024-12-31) |

The archive holds trades back to **2021-06-30**. INXD's trades blanket exactly
the window in which NASDAQ100D has none. No retention cutoff deletes one series
and spares its contemporary. Absence tracks the **series**, not the date.

Retention is still real, but it answers a different question: it is why a series
omitted from the original pull **cannot be recovered now**, not why it is
missing. `kalshi_orderbooks.jsonl` is no help either — it is a live snapshot
stream beginning ~Oct 2025, so it carries no 2022–24 history for anything.

### §12.2 What actually happened: two independent, partial pulls

The two archives were built by different mechanisms and neither contains the
other:

|  | markets | trades |
|---|---|---|
| distinct series | 4,836 | 2,490 |
| distinct tickers | 4,252,445 | 420,341 |
| present here but not in the other | — | **227 series / 222,293 tickers** |

`data/markets/` is a bulk paginated sweep — broad, shallow (one metadata row per
ticker), and itself missing ~44% of its own page range (426 of 757 expected
chunks). `data/trades/` is assembled **per ticker**:
`fetch_kalshi_data.py::fetch_trades_for_ticker` calls
`/historical/trades?ticker=…` in a loop over a caller-supplied list. Coverage is
therefore exactly that list — about 10% of the market tickers.

The signature of a list-driven pull is all-or-nothing coverage per series, and
that is what the data shows. Across the 730 series with ≥20 market tickers:

| ticker coverage | series |
|---|---|
| [0, 0.1%) — nothing at all | **350** |
| [0.1%, 5%) | 25 |
| [5%, 25%) | 61 |
| [25%, 75%) | 129 |
| [75%, 95%) | 69 |
| [95%, 100%] — essentially complete | **96** |

Strongly bimodal. `TNOTE`, `USDJPY` and `RATECUT` have **no ticker prefix
whatsoever** in the trades archive, while `INX` has nine variants (`INX`,
`INXD`, `INXU`, `INXW`, `INXY`, `INXZ`, `KXINX`, …). They were never requested.

### §12.3 The consequence: registered series are silently short

`assert_trade_coverage()` (added in `410f8fe`) has an **absolute floor only**
— it catches total absence and lets partial coverage through. Five registered
*trigger* series are materially incomplete:

| series | events listed | events traded | coverage |
|---|---|---|---|
| WTIW | 161 | 51 | **32%** |
| CPIFOOD | 21 | 8 | **38%** |
| WTI | 702 | 452 | **64%** |
| CPIAPPAREL | 22 | 15 | **68%** |
| PAYROLLS | 46 | 33 | **72%** |

The rest of the macro universe (CPI, U3, CPICORE, CPIYOY, CPICOREYOY, FED, GDP,
PCECORE, ADP, ISMPMI) is at 100%, and the asset-price targets are at 87–99%.

**This is not thin markets — it is an incomplete fetch.** The distinction
matters for every claim in this log:

- **`n` is not a liquidity measurement.** Wherever a small `n` was read as
  evidence that a market is illiquid or an event under-traded, that inference is
  unsupported. PAYROLLS is short 13 of 46 events for collection reasons alone.
- **The covered events are a convenience sample, not a random one.** Whatever
  ordered the original ticker list — alphabetical, volume-ranked, or simply
  where a checkpointed run was interrupted (`fetch_all_econ_trades` checkpoints
  every 50 tickers and resumes by set difference) — is now an unmodelled
  selection mechanism upstream of every estimate. It is *not known* to be
  ignorable, and nothing in the pipeline currently tests it.
- **WTI is the sharpest case.** §1(b) and `edge_economics.md` §2(b) already
  discount WTI on two independent grounds (the bucket parse bug, no scheduled
  information event). Its 64% coverage — and WTIW's 32% — is a third.

`data/MANIFEST_new_pulls.md` had already noticed the symptom in passing
("PAYROLLS 2023 events, JOBLESSCLAIMS/ADP/ISMPMI pre-Oct-2025 history … the
original pull simply never captured") and `trades_backfill_is/`
(23,430 trades, 1,550 tickers, 2022-09-09 → 2025-12-31) was pulled to close it.
**It was never merged.** That is the cheapest available remedy and it is still
outstanding.

### §12.4 What to do

1. **Merge `trades_backfill_is_2022_2025.parquet` into the archive.** Zero
   overlap is claimed by construction; dedup against existing `trade_id` before
   concatenating. Then re-run coverage and re-estimate — the trigger panel's
   `n` may move, and everything downstream with it.
2. **Give `assert_trade_coverage()` a coverage-*fraction* bound**, not just the
   absolute floor, so 32% fails loudly instead of passing.
3. **Test the selection.** For a series at partial coverage, compare covered vs
   uncovered events on what the metadata *does* retain for both (`volume`,
   `open_interest`, `close_time`). If they differ systematically, the sample is
   not ignorable and the affected series need a stated caveat or exclusion.
4. **Correct the `registry.py::trade_coverage` docstring**, which still gives
   retention as the cause.

### §12.5 Which existing results this touches

> **Re-check against §14 (2026-09-14).** The audit below maps coverage gaps onto
> the *pre-correction* edge set. It partly resolves in the corrected table: of
> the three survivors this section flags as sitting on incompletely collected
> triggers, `WTI→JOBLESSCLAIMS` has left the grid entirely and
> `PAYROLLS→FEDDECISION/hike` and `PAYROLLS→FED` remain — so PAYROLLS at 72%
> trigger coverage is still load-bearing on **two of four** survivors, and that
> caveat stands undiminished. `CPI` and `CPICOREYOY`, carrying the other two,
> are at 100% coverage. The §3 pooled-sign and direction-study assessments below
> are superseded by §14.10.

The affected series are **not peripheral** — they are load-bearing in the
headline findings:

- **Stage-1 adjacency** (`artifacts/adjacency_report.md`, 8 BH-FDR survivors).
  The single strongest edge is **PAYROLLS→FED** (n=30, ρ=0.565, p=0.00029) at
  **72%** trigger coverage. **PAYROLLS→FEDDECISION/hike** (p=0.0038) and
  **WTI→JOBLESSCLAIMS** (n=13, ρ=−0.698, p=0.0012) are also survivors, the
  latter on a trigger at **64%**. Three of eight survivors sit on incompletely
  collected triggers.
- **§3 pooled sign result** (65.6% over 224 dormant rows, p=0.0030). Three of
  the twelve theory-specified cells are PAYROLLS cells (n=20, 20, 18). The
  other nine rest on CPI, CPIYOY, U3, FED and FEDDECISION — 100%, 100%, 100%,
  100% and 96% coverage respectively, so the *majority* of the pooled result is
  insulated, but not all of it.
- **Direction study** (`artifacts/direction_report.md`). WTI is the most
  frequent trigger in the pair table by a wide margin, which is why the
  "no WTI trigger" robustness cut already exists there. That cut now has a
  second, independent motivation.

**Nothing in §1–§11 is retracted on this basis.** Incomplete collection widens
the uncertainty on these estimates; it does not by itself reverse them, and the
direction of any bias is unknown until §12.4(3) is run. But the earlier reading
that these were simply low-`n` markets is wrong, and any writeup that cites
PAYROLLS→FED or WTI→JOBLESSCLAIMS should say so.

Until §12.4(1) and (3) are done, results involving WTI, WTIW, CPIFOOD,
CPIAPPAREL or PAYROLLS carry an explicit data-coverage caveat when cited. This
belongs in the final report as a limitations section, not only here.

---

## 2026-09-11 — §13 Horizon, cost and the tradability ceiling

> **Numbers superseded by §14.11 (2026-09-14)**; the argument is not. Every
> figure quoting 81 signals, −4.25c or a 17.5-minute lag is pre-correction.
> `analysis/horizon_2026_09/` was re-run on the corrected panel (all five
> scripts, captured outputs refreshed) and the ceiling argument **holds and
> firms up**:
>
> - Every horizon is net-negative at `bh`: 1h −4.50c, 1d −2.91c, 7d −3.58c,
>   14d −2.85c, settle −2.87c. Holding longer still does not help.
> - §13.5's clustered inference is now decisive rather than marginal. Clustering
>   on `target_event` at `p05`, the hold-to-settlement strategy is
>   **net −0.42c, t = −0.09, 95% CI [−9.2, +8.5]** over 91 rows in 47 events —
>   indistinguishable from zero. The "settlement buys breakeven" reading is
>   superseded by "settlement buys nothing measurable".
> - The `< 35c` entry bucket is the worst cell (−8.60c, t = −1.33), which is the
>   favourite–longshot direction §13.4 predicted.
>
> Read alongside §14.10: the rule being priced has no demonstrated skill, so a
> null here is the expected result rather than a surprising one.

**One-line summary:** the strongest recovered edge does not pay at any holding
horizon, and the two configurations that appeared to were each a handful of
events wearing a large `n`. Clustering on `target_event` — the honest unit, since
a settlement payoff is a function of the target's resolution and nothing else —
removes both. Nothing here retracts §1–§12; it closes the tradability question
§6.4 left open, in the direction §6.4 anticipated.

Scripts: `analysis/horizon_2026_09/` (see its README for run order). Fees are at
`CONTRACTS=100` throughout unless stated.

### §13.1 Why the strongest edge does not pay

`PAYROLLS→FED` is the largest Stage-1 survivor (ρ = 0.565, p = 0.00029, n = 30).
Decomposed on the direction panel (n = 25):

| quantity | value |
|---|---|
| Spearman ρ(surprise, response) | **+0.532** |
| sign agreement measured from `p0` | **0.840** |
| sign agreement at the executable entry | **0.440** |
| signed jump `p0` → first print (**untradable**) | +1.76c (65%) |
| signed drift first print → exit (**tradable**) | +0.96c |
| median lag, resolution → first print | 6.2 min |
| round-trip cost floor (spread 1.0c + fees) | **2.61c** |

Two separate things are going on and they compound.

**(a) ρ and the trade measure different quantities.** ρ ranks magnitudes across
events. The trade needs the *sign* to still be predictable at a price you can
transact at. Directionally the relationship is strong — 84% from `p0` — but that
accuracy lives entirely inside the jump; at the executable entry it is 0.440,
below chance. This is `edge_economics.md` §3's jump-not-drift finding reproduced
on a single pair.

**(b) The tradable remainder is smaller than the toll.** +0.96c against a 2.61c
floor. Conditioning on larger surprises does not rescue it (|z|≥1.0: drift
+1.38c, net −1.35c on n = 13).

### §13.2 Holding longer does not help

The hypothesis was that gross grows with horizon while cost is charged once per
round trip. Walk-forward OOF sign rule, direction held fixed, evaluated at each
horizon (gate `p05`, n = 204):

| horizon | gross | cost | **net** | t |
|---|---|---|---|---|
| 1h | 0.74 | 3.28 | −1.33 | −1.69 |
| 6h | 0.99 | 3.24 | −0.83 | −1.24 |
| **1d** | 0.98 | 3.27 | **−0.63** | −0.98 |
| 3d | 0.69 | 3.36 | −1.59 | −2.01 |
| 7d | 1.00 | 3.28 | −1.37 | −1.68 |
| 14d | 0.95 | 3.21 | −2.20 | −2.41 |
| **settle** | 2.61 | **1.84** | **+0.77** | 0.25 |

**Gross does not grow.** It peaks around one day at ~1–2c and then decays — under
`bh` the 14-day gross is *negative* (−0.46c). There is no continuation to
harvest; the repricing is complete at the first print and what follows partly
mean-reverts.

What helps at settlement is not the horizon but the **cost structure**: you cross
once and Kalshi charges no settlement fee, so cost falls 3.3c → 1.8c. That ~1.5c
is the entire improvement, and it buys breakeven, not profit.

### §13.3 Fees: the schedule is right, its application was not

Kalshi's published formula (verified 2026-09-11, schedule last updated
2026-07-07) is `round_up(0.07 × C × P × (1−P))`, matching `FEE_RATE = 0.07` on
every reference point ($0.0175 at 50c, $0.0063 at 10c/90c). The standing
"verify against the live schedule" TODO in `direction/tradability.py` is
discharged.

But `C` sits **inside** the round-up, and `tradability.py::fee` rounds *per
contract*. The two agree only at C = 1. Everything the pipeline has produced was
priced as though buying one contract at a time, costing ~0.25c/contract more at
50c than any realistic size:

| C | official | as coded | overcharge/contract |
|---|---|---|---|
| 1 | $0.02 | $0.02 | 0.00c |
| 100 | $1.76 | $2.00 | 0.24c |
| 1000 | $17.50 | $20.00 | 0.25c |

Correcting it is worth ~0.55c per leg. **No conclusion changes** — the round-trip
horizons remain clearly negative — but every published net figure is ~0.5c/leg
pessimistic, including `edge_economics.md`'s headline −4.25c ledger.

Size is not otherwise a lever: the fee is strictly proportional, the rounding
saving caps at 12.5% and is fully realised by ~25 contracts, and the books cannot
supply the size anyway — **66% of events see fewer than 100 contracts trade in
the hour after the signal fires** (median 0).

### §13.4 Conditioning on entry price

Two reasons to expect it to help, pulling together: the fee collapses away from
50c (0.57c at 80c+ vs 1.70c at the money), and Whelan et al. (2026) document a
favourite–longshot bias on Kalshi (§7.4). Trades entered at a mean of 50.8c sit
exactly on the fee maximum.

It does not help. At settlement, `p05`, bucketed on the price actually paid: no
bucket clears significance (five buckets tested, best permutation p = 0.067), and
the only clearly significant result is **negative** — the 80–101c bucket at
t = −2.07, precisely the bucket the fee argument predicted would be best. The fee
saving is ~1.1c while between-bucket differences are 15–25c: **once holding to
settlement, fees are second-order to whether the signal is right.**

### §13.5 The clustering correction — read this before citing §13.2–§13.4

At settlement the payoff is a deterministic function of the target event's
resolution, so rows sharing a `target_event` are **perfectly** correlated, not
merely dependent. Row-level inference overstates significance, and worst where a
fast trigger feeds a slow target.

The diagnostic case: the 0–35c bucket looked positive at +4.59c, driven by
`WTI→GDP` at +49.29c on "17 trades" with a 76% win rate. Those 17 rows are
**2 GDP contracts**, both of which resolved YES — 13 rows are the same contract
at the same price with the same outcome. It is one outcome counted thirteen times.

Clustering on `target_event` (point estimate = mean over event means, bootstrap
resamples events):

| subset | rows | **events** | net | t | 95% CI |
|---|---|---|---|---|---|
| ALL (`p05`) | 204 | **85** | −0.32 | −0.07 | [−9.4, 8.4] |
| excl WTI→GDP | 169 | 83 | −0.46 | −0.10 | [−9.7, 8.6] |
| price paid < 35c | 78 | **38** | +2.75 | 0.44 | [−9.0, 15.3] |
| price paid ≥ 65c | 76 | 36 | −10.48 | −1.57 | [−24.0, 1.7] |
| ALL (`bh`) | 81 | **47** | +0.38 | 0.06 | [−11.3, 12.0] |

Per pair, `p05`, clustered:

| pair | rows | events | net | t |
|---|---|---|---|---|
| PAYROLLS→FED | 15 | 10 | +11.93 | **1.74** |
| CPI→CPICORE * | 15 | 15 | +14.39 | 1.05 |
| CPICORE→CPI * | 13 | 13 | +13.72 | 0.82 |
| CPICORE→FED | 23 | 18 | −8.41 | −1.77 |
| CPI→FED | 24 | 18 | −7.63 | −1.61 |
| WTI→CPIUSEDCAR | 42 | **5** | −6.98 | −0.83 |
| WTI→GDP | 35 | **2** | +5.55 | *untestable* |

\* same-release, mechanical — not propagation

**Nothing is positive at conventional significance.** The strongest remaining
candidate is `PAYROLLS→FED` at t = 1.74 on 10 distinct outcomes. `WTI→GDP` cannot
be tested at all. The cheap-bucket result does not survive. Two of the three
positive cells are the same-release CPI clique.

This vindicates the existing `no WTI trigger` robustness cut in
`direction_report.md` on a second, independent ground: WTI's weekly cadence
against quarterly targets produces the worst clustering in the panel (42 rows /
5 outcomes; 35 rows / 2 outcomes).

**Action:** `_block_perm_p` in `direction/evaluate.py` already clusters for the
direction study. The settlement and ledger analyses do not, and should — every
tradability figure predating this entry carries row-level error bars.

### §13.6 What this settles

§6.4 asked for tradability to be treated as an open empirical question and
pre-authorised this landing: *"Reframe the trading evaluation as a test of
economic significance: how large is the signal relative to the frictions that
sustain it?"* The answer, on in-sample data and with honest clustering:

> The recovered structure is **economically significant and not economically
> exploitable**. Macro surprises propagate to the policy path in a
> theory-consistent, sign-restricted, out-of-fold-predictable way, but the
> repricing completes at the target's first post-resolution print — a median 6.2
> minutes after resolution and a median 8.8 hours after the stale reference the
> estimator uses. The component surviving to an executable price is ~1c against
> a ~2.6c floor. Holding longer does not help because there is no drift to hold;
> the only cost structure that changes the arithmetic is settlement, and it buys
> breakeven.

This quantifies the limits to arbitrage that explain the drift's persistence —
closing the §6.1 loop, as §6.4 argued a negative result would, and agreeing with
Angelini & De Angelis (2026) on Kalshi (§7.1).

One thread remains live: the frictions are not uniform. The fee vanishes at the
extremes and at settlement, and §13.4 tested only entry price, not **strike
selection** — expressing the same signal at a different point on the ladder.
That needs the ladder-aware panel rebuild also required for the cross-strike
long/short construction, and is the honest "future work" claim.

---

## 2026-09-13 — §14 Corrections: a measurement fix, five statistics bugs, and a look-ahead

**One-line summary:** a review of the data path and the study code found one
avoidable measurement error (the resolved value was being *inferred* when the
true printed value was on disk), five defects in `structure/stats.py`, and a
look-ahead in how the response instrument was chosen. Fixing them takes the
Stage-1 edge table from **8 BH survivors to 4** — and the 4 that remain are all
*non-same-release*, which is a better result than the 8. Nothing here was found
out of sample; the OOS wall was not approached.

Backing code: `stg/panel/_io.py`, `stg/panel/surprise.py`, `stg/panel/nodes.py`,
`stg/panel/targets.py`, `stg/structure/stats.py`, `stg/structure/estimator.py`,
`stg/direction/learners.py`, `stg/direction/tradability.py`. 110 tests pass,
including a regression test for each item below.

### §14.1 The resolved value was inferred when the true value was on disk

`implied.py::resolved_value` reconstructs each event's outcome as the midpoint
between the highest YES and the lowest NO strike, which is accurate only to
`spacing / 2`. Against CPI's median |surprise| of **0.0738 pp** that error is
**0.05 pp — 68% of the signal being measured** — and on one-sided ladders, where
the open tail is truncated at half a strike, it is far worse: measured errors up
to 0.35 (CPI), 0.25 (U3), 1.65 (GDP), 23,000 (JOBLESSCLAIMS), 20,000 (ADP).

`data/markets_api_pull/markets_api_pull_raw.jsonl` carries `expiration_value`,
the value actually printed, for **1,022 events (847 in sample)** once `%`, `$`
and thousands separators are parsed. Nothing read it. `panel/_io.py::
load_settlement_values` now does, `surprise.py` prefers it over the ladder
midpoint, and a new `resolved_source` column records which was used:
**442 of 519 panel rows get the true value**, 77 keep the fallback (WTIW and the
CPI subcomponents are absent from that pull).

The surprises moved by a median of **50–67% of their own previous magnitude**
(CPI 67%, CPICORE 62%, WTI 55%, U3 50%, PAYROLLS 22%). This is a change in the
dependent quantity, not a cosmetic one. Three events that the ladder could not
resolve at all also came back: GDP 3.8%, core PCE 0.2%, U3 4.4%.

**`settlement_ts` is not the release timestamp** and must not be used as one. On
the same file it runs a median **4.65 h after** `close_time` (WTI 21.75 h) — it
is Kalshi's administrative settlement. `close_time` is already the right
instant: CPI, U3 and PAYROLLS close at **12:25 UTC, five minutes before** the
8:30 ET BLS release, so `p0` is genuinely pre-news and §5's event-study window
is correctly anchored. This closes the `expiration_value` half of
`data_prep_plan.md` Phase B without a re-pull.

### §14.2 The implied distributions were not being checked for being distributions

`coverage` and `ladder_mass` were computed, documented as the coherence metric,
and then never filtered on. The bucket path enforced a one-sided `MIN_MASS = 0.5`
floor with **no ceiling**, so a WTI ladder whose mutually exclusive buckets summed
to **2.96** was accepted and divided by its own total. That does not recover a
distribution: the violation sits in the stale legs, not spread evenly. Measured
over the bucket rows, mass ran p50 = 1.26, p75 = 1.48, p95 = 1.85, with **23%
above 1.5**. Coverage was as low as 0.23 at the 5th percentile, and 166 events
were built from exactly three legs — a five-bin PDF with two open tails placed
half a spacing past the extreme strikes, which shrinks `implied_mean` toward the
ladder centre and *understates* `implied_std`. That understatement propagates,
because Stage 2's feature is `surprise / implied_std`.

`surprise.py::gate_panel` now applies a two-sided mass bound `[0.7, 1.5]`, a
`coverage >= 0.5` floor and a leg minimum, **identically to both contract
paths** (they previously disagreed). Cost: **799 → 519 rows and no trigger
series** — the 14 series clearing `usable_triggers(10)` are the same 14 before
and after, and `build_panels.py` now asserts that. `[0.85, 1.15]` would have left
WTI with 98 of 368 events, which is why it is not the default.

**The dropped rows are themselves a finding.** `update_2026_08.md` §1 proposes the
coherence violation (median mass 1.28 — WTI pricing to 128% of certainty) as a
candidate standalone contribution. Gating it away is right for a panel whose job
is a trustworthy implied mean and wrong for that study. Build it from
`build_surprise_panel(gated=False)`; the gated panel is bounded by construction,
so reading the coherence distribution off it understates the result.

### §14.3 The two panels disagreed about stale prices

`surprise.py` was careful — last day with `MIN_FRESH_LEGS` legs that actually
traded, never forward-filled. `nodes.py::_threshold_daily` did the opposite,
handing `build_daily`'s forward-filled `close` straight to
`build_daily_implied_means`. So **74.5% of node-feature rows** were implied
distributions assembled from legs last traded on *different days* (34.7% with a
leg over a week stale, p99 = 137 days) — the incoherent cross-section
`build_daily`'s own docstring warns about. One source, two opposite rules.

The node panel now applies the surprise panel's rule by default
(`freshness="fresh"`), emits `n_fresh_legs`, and keeps `freshness="filled"` for
the AGCRN-style use that needs a value every calendar day. At ticker-day level
the fresh rule keeps **23–59%** of days (CPI 4,972 → 1,358; CPICORE 2,562 → 590;
PAYROLLS 1,184 → 695). Note that `max_stale_days` changes meaning: the legs
*used* are now fresh by construction, so it is a ladder-completeness diagnostic
("how much of the ladder was dark"), not a contamination measure, and it stays
non-zero on most rows.

### §14.4 Five defects in `structure/stats.py`

1. **Ranks ignored ties.** `_rank` was `argsort(argsort(x))`, which hands
   arbitrary distinct ranks to equal values. `response` is a difference of
   integer cent prices — **103 distinct values over 5,315 rows, 98.1% tied** —
   and `surprise` is 86.2% tied. Against a tie-corrected Spearman the worst
   per-pair error was **|Δρ| = 0.238** (`JOBLESSCLAIMS→FED`: 0.329 vs 0.090).
   Now `scipy.rankdata`; ρ matches `scipy.spearmanr` exactly.
2. **The p-value used a normal where a t belongs.** `spearman_p` evaluated the t
   statistic against a standard normal — anti-conservative at n = 10–46
   (`CPIYOY→FEDDECISION/cut`: 0.014 reported, 0.062 actual, 4.3×). Now exact-t
   on `n - 2` df.
3. **BH was controlling the wrong p-value.** Both fixes above leave the
   asymptotic p an *approximation* whose assumptions this much tying does not
   meet. The permutation p is exact with the same statistic, ties included — and
   it was already being computed, but only for survivors and near-misses, so it
   could not be selected on. `estimate_adjacency` now computes it for **every**
   pair and selects on it by default (`select_on="permutation"`), reporting both.
4. **A permutation p of exactly 0.** `(|null| >= |obs|).mean()` has no `+1`
   correction, so `adjacency_report.md` printed `p = 0.0000` — not a possible
   estimate from 2,000 draws. Now `(1 + k) / (1 + n_perm)`, floor 1/2001.
5. **One shared mutable RNG.** A module-level `default_rng(0)` was consumed in
   sequence by every caller, so adding a pair — or a ladder rung — silently
   changed every p-value computed after it. Each pair now seeds its own
   generator from `crc32` of its identity (**not** `hash()`, which Python
   randomises per process and would have made this worse). `bh_critical` also
   gave tied p-values different thresholds; ties now take the largest rank in
   the group, consistent with BH's step-up.

### §14.5 The response instrument was chosen with hindsight — and it was holding up the strongest edge

`targets.py::representative_tickers` picked each target event's instrument as its
**most-traded leg over the event's whole life**, including every trade after the
trigger fired. Restricting the count to the first half of an event's life picks a
different leg for **56% of CPI events and 63% of WTI events**.

`target_frames` now returns every traded leg and `response_panel` chooses, per
trigger instant, the most-traded leg **among prints at or before `t_res`** — which
is also exactly the condition `p0` requires, so the `live` match rule and the leg
choice became one test. The trigger-independent work stays cached, so §11's INXU
performance fix survives.

The consequence is the most substantive result in this entry. **`CPICORE→CPI`,
the strongest edge in the published table at ρ = 0.744, p = 1.7e-7, falls to
ρ = 0.467, p_perm = 0.027 and does not survive.** `direction_study.md` already
flagged that pair as "arithmetic, not diffusion" — its 84.6% was the mechanical
same-release channel §5 worries about. The mechanism is now visible: the
full-life rule was selecting the CPI ladder leg that ended up nearest the
realised print, and for a same-release trigger that leg's move partly *is* the
trigger's own surprise restated. Removing the look-ahead removes most of it.

### §14.6 `neighbour_signal` split simultaneous triggers arbitrarily

The `nbr` feature promised "only rows with an earlier `t0` contribute" and
delivered a cumulative sum over *row order*, so a tied row saw whichever siblings
happened to sort before it. **993 of 5,315 rows matched neither a strict (`<`) nor
an inclusive (`<=`) past**, and **29.2% of rows share a `t0` with a sibling on the
same target event** — the CPI family, U3 and PAYROLLS all close at the same
instant as their co-release, which is precisely the dependence this project is
about. Now cumulated over distinct `t0` blocks: strictly-earlier by default,
with `include_simultaneous=True` for the other reading. Both are exact against a
brute-force reference and invariant to row order.

### §14.7 A cache with no staleness check served an eight-day-old panel

`run_direction_study.py` reused `artifacts/panels/pair_panel_dormant.parquet`
whenever the file merely *existed*, so the entire Stage-2 study ran on a
**8 September** panel through two rounds of upstream fixes. Stage 1 builds its
response panels in process and moved; Stage 2 did not, and that mismatch is what
exposed it. Both scripts now compare mtimes against `surprise_panel.parquet` —
the direction study rebuilds, and `run_edge_economics.py` refuses to price a
pair panel older than its inputs rather than doing it silently.

### §14.8 Three smaller defects in the feature panel

None of these change a headline, but two of them were silently wrong on most
rows of the node panel, which is the AGCRN input.

**Rolling windows counted rows, not days.** `recent_volume` and `net_flow` used
`rolling_sum(activity_days)` and momentum used `shift(momentum_days)`. The frame
has one row per day the series was *active*, so every gap — a weekend, a dark
ladder, and far more of them once the panel became fresh-only (§14.3) — stretched
"7 days of volume" across however long the gap ran. Switching to
`rolling_sum_by("date", "7d")`, and to an as-of join for momentum (a rolling
aggregate cannot give you a *level* 5 days back), changed:

| feature | rows changed | max \|Δ\| |
|---|---|---|
| `d_implied_mean` | 60.0% | 1.4e5 |
| `recent_volume` | 69.1% | 8.9e5 contracts |
| `net_flow` | 69.1% | 1.74 |

Nulls also fell (`recent_volume`/`net_flow` 97 → 0, `d_implied_mean` 81 → 45),
because a date window does not require a full complement of rows. That matters
because `build_tensor` writes a missing feature as `0.0`, which for
`max_stale_days` reads as "perfectly fresh" — a smaller footgun now, but still
one worth a sentinel.

**`clearance_days` never applied.** The filter read
`... if False else panel`, so a documented parameter defaulting to 16 did nothing
at any setting; `days_to_close` had a 5th percentile of 0, meaning nodes appeared
on their own resolution day. The parameter works now and the default is **0**,
which is what was actually in force — and is also right on the merits, so this is
a repair rather than a behaviour change. A node feature here is "the market's
current belief about the nearest unresolved event", and that belief is most
informative in the days just before the print; excluding the last 16 days of
every event's life would discard the part of the panel this thesis is about and
halve it (the median row sits 15 days out). The constraint clearance was reaching
for is a *label* constraint, and it already exists where it belongs, as
`PURGE_DAYS` in `splits.py`.

**Look-ahead in `aggregate_to_event_level`.** Its `*_norm` columns z-scored
against `mean().over("event_ticker")` — the event's entire life — so a feature on
day 3 was standardised using prices from day 40. Now expanding (prefix mean and
variance). Its only consumer is the legacy `pairs/pipeline.py`, so nothing
current was affected, but it was a loaded gun in a shared module.

### §14.9 Stage-1 edge table, before and after

| | published (2026-09-08) | corrected |
|---|---|---|
| ordered pairs searched | 158 | 141 |
| selected on | asymptotic p | **permutation p** |
| BH survivors at q = 0.1 | 8 | **4** |
| survivors that are same-release | 1 (the strongest) | **0** |

The four survivors, all macro → policy path:

| trigger | target | side | n | ρ | p_perm |
|---|---|---|---|---|---|
| CPI | FED | any | 36 | 0.617 | 0.0005 |
| PAYROLLS | FEDDECISION | hike | 26 | 0.596 | 0.0005 |
| PAYROLLS | FED | any | 30 | 0.583 | 0.0010 |
| CPICOREYOY | FEDDECISION | cut | 17 | — | 0.0025 |

Two published survivors left the grid entirely rather than failing BH:
`WTI→JOBLESSCLAIMS` (flagged in §12.5 as sitting on a 64%-coverage trigger) and
`CPIYOY→PCECORE` no longer clear `min_n = 10` after gating.

**This is a stronger claim than the eight it replaces**, and it should be written
up that way rather than as a retreat. The recovered structure is now entirely
scheduled-macro-surprise → policy-path — the Kuttner (2001) / GSS (2005) channel
`research_summary.md` §6.2 cites — with no mechanical same-print pair carrying
any of it, and with selection resting on an exact test rather than an
approximation whose assumptions the data violate. §5's identification problem is
no longer load-bearing on the headline.

### §14.10 Stage 2 reverses: neither the structure nor the surprise predicts direction, and the rung that "won" was reading the price level

This is the consequential half of §14 and it goes against the project's Stage-2
claim. With the pair panel rebuilt on the corrected surprise panel and the
causal instrument choice (§14.5), the ladder is **3,127 rows, 2,347 scored out
of fold** — against 5,312 / 3,836 before, because a usable row now needs a
pre-trigger print on the leg actually used.

| rung | `all` (n=2,347) | `p05` (n=92) | `bh` (n=45) |
|---|---|---|---|
| `sign_rule` | 50.9%, p = 0.39 | 58.7%, p = **0.23** | 68.9%, p = **0.059** |
| `no_structure` | 54.6%, p = 0.000 | 68.5%, p = 0.007 | 75.6%, p = 0.009 |

**(a) The sign rule is no longer significant at any gate.** §9(b) reported 63.0%
at p = 0.041 on `bh` and 57.5% at p = 0.022 on `p05`. The *point estimates barely
moved* — the samples collapsed (221 → 92 at `p05`, 81 → 45 at `bh`). The earlier
significance was resting on rows that the look-ahead in instrument choice had
manufactured, and on a stale cached panel (§14.7) that survived two rounds of
upstream fixes.

**(b) `no_structure` now beats every structure rung — and that is an artifact.**
§9(c) called the ablation "decisive" in the opposite direction: strip the edge
weight and accuracy fell *below* chance. It now wins everywhere, which would
invert the claim. It does not, because the win is entirely one feature. Ablating
`CONTEXT` on the `all` cell (n = 2,061, the only well-powered one):

| rung | acc | p |
|---|---|---|
| `p0c_only` | 53.5% | **0.001** |
| `p0c` + `dtc` | 54.0% | **0.000** |
| `no_structure` (full) | 54.4% | **0.000** |
| `no_structure` **minus `p0c`** | 49.2% | 0.77 |
| `z_only` | 49.4% | 0.72 |
| `abs_z_only` | 49.5% | 0.76 |
| `dtc_only` | 49.4% | 0.95 |

`p0c` is `(p0 - 50) / 50` — the target's pre-trigger **price level**. Kalshi
prices are bounded in [0, 100], so from a low `p0` the next move is
mechanically more likely to be up, and a logit handed `p0c` will find that. It
is bounded-support arithmetic, not cross-market propagation, and it carries
**all** of `no_structure`'s performance: remove it and the rung sits at chance.

**(c) So the honest reading is a null.** `z_only` and `abs_z_only` are at chance,
so the surprise predicts nothing on its own; `no_p0c` is at chance, so neither
does the context; and the sign rule clears no gate. **On the corrected panel the
dormant-horizon direction task shows no reliable predictive signal from either
the estimated structure or the surprise.**

**(d) `p0c` invalidates the ladder's internal comparisons, not just one rung.**
It sits in `CONTEXT`, so `feature_logit` and `neighbour_logit` contain it too —
their apparent accuracy is contaminated by the same effect, and the R4-minus-R3
contrast `learners.py` is built around ("what the estimated structure buys over
the surprise alone") is not identified while both sides carry a feature that
predicts the label mechanically. Any future run of this ladder should either drop
`p0c` from `CONTEXT` or re-specify the label to be orthogonal to the price level
(e.g. sign of the move net of the level-implied drift). That is a design change,
not a bug fix, so it is left as a decision rather than applied here.

**What survives.** Stage 1 is unaffected — it never used `p0c`, and its four
survivors rest on a signed rank correlation between surprise and response with
an exact permutation test (§14.9). The structure *exists*; what does not survive
is the claim that it was shown to predict direction out of fold. §9's title —
"The structure predicts direction; nothing else does" — is now wrong in both
halves and should be read as superseded by this section.

### §14.11 Ledger on the corrected panel

Re-priced on the `p05` gate (now the headline gate) with the corrected spread
estimator (§14.4 item 4 and the `frac_negative` fix):

| | §13 (published) | corrected |
|---|---|---|
| signals | 81 | 92 |
| costed | 81 | 53 |
| gross vs `p0` | 1.66c | 0.91c |
| gross vs first print | 0.84c | 0.47c |
| spread charged | ~2.0c | 1.19c |
| fees both legs | — | 2.60c |
| **net per trade (taker)** | **−4.25c** | **−3.15c** |
| net (maker bound) | −1.50c | −0.77c |
| t on net | −4.94 | **−4.31** (n = 92) |
| median lag to first print | 17.5 min | **33.6 min** |

The tradability conclusion is unchanged and if anything firmer: net is negative
at t = −4.31, and the executable gross has fallen to 0.47c against a 1.19c
spread. Note the entry lag doubled to 33.6 minutes, which is the causal
instrument choice showing up honestly — the leg a trader could actually identify
in advance is less liquid than the one hindsight picks. Three targets
(`CPICORE`, `FEDDECISION`, `GDP`) are now withheld for want of a usable spread,
which is itself the §13.4 illiquidity finding.

**Caveat that now matters more.** This ledger prices `sign_rule` signals, and
per (a) above that rule no longer has demonstrated skill. The ledger should
therefore be read as what it always literally was — an upper bound on what this
*class* of signal could pay after frictions — and not as costing a validated
edge.
