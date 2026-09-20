# Liquidity and underreaction: the right sign, the wrong sample size

*Written 2026-09-20. Scripts and captured output in
`analysis/leadlag_2026_09/liquidity.py`, `liquidity_gates.py` and
`liquidity_spread_diag.py`. In-sample only.*

*§4 **corrects §2 of this same document.** §2 records what the first pass
reported; §4 is what survives measuring it properly. Cite §4.*

## Summary

`leadlag_findings.md` and `move_filter.py` §C both point at **underreaction**:
the market moves part of the way on a trigger and the rest arrives later. If
that is the mechanism, the standard cross-sectional prediction follows —
Hong–Lim–Stein — that underreaction is larger where fewer people are watching.
This document tests it.

| question | answer |
|---|---|
| Do thin contracts reprice more slowly? | **Yes**, 30h to first print against 1h. Partly definitional. |
| Is the edge larger in thin contracts? | **Sign yes, magnitude no.** ~+3c thin−liquid, P(≤0) ≈ 0.3. |
| Was the first pass's +7.17c real? | **No.** It had no CI, and it was a selection draw. |
| Had the dataset already filtered on liquidity? | **Not in the pipeline.** Four implicit gates, three of them ours. |
| Does relaxing our gate kill the result? | **No — it mildly improves it.** |

The honest one-line version: **the sign has been positive in every cut tried,
the central estimate is about +3c, and it is not distinguishable from zero.**
That is the same conclusion the power curve reaches everywhere else in this
repo, and it should be stated in those terms rather than as a finding.

---

## 1. The mechanism, with no profit involved

`liquidity.py` §0. Liquidity is measured on the target leg's own tape
**strictly before `t_res`**, so nothing here can see the window the trade is
taken in.

Hours from trigger resolution to the target leg's first print:

| liquidity tercile | median h | mean h | median first move |
|---|---|---|---|
| thin | **29.6** | 72.6 | 2.0c |
| mid | 6.9 | 32.9 | 1.0c |
| liquid | **1.0** | 7.8 | 1.0c |

A 30× spread in repricing latency, consistent across all five liquidity
measures (`n_pre`, `vol_pre`, `days_pre`, `ev_vol`, `stale_h`).

**This is weaker evidence than it looks.** A leg with few prints has long gaps
between prints by arithmetic, so `n_pre → first_h` is partly definitional
rather than a fact about repricing. The `med_move` column is the less circular
half — thin legs move twice as far on their first post-trigger print — but a
longer interval also accumulates more news, so it is not clean either. Read §1
as establishing the shape, not the effect.

---

## 2. The first pass: a profit gradient — *superseded by §4*

`liquidity.py` §1, on the 843 spec-v2 positions:

| measure | thin | mid | liquid |
|---|---|---|---|
| prints before `t_res` | +8.04c [+0.56, +15.33] | +4.73c | −0.58c |
| volume before `t_res` | +8.36c [+1.90, +14.55] | +2.74c | +1.19c |
| distinct trading days | +5.19c | +4.86c | +2.06c |
| event-level volume | +7.12c [−0.71, +14.88] | +2.47c | +2.69c |
| staleness at `t_res` | +4.64c | +8.74c | −1.09c |

Monotone in four of five, with the liquid tercile indistinguishable from zero
in all five — i.e. the whole +4.10c headline appearing to come from the thin
two-thirds. On its face a Grossman–Stiglitz result: the mispricing survives
exactly where nobody is paid enough to arbitrage it.

**§4 shows this table was over-read.** It is kept here because the sign pattern
is real and reproduces; it is the *magnitude* that does not survive.

---

## 3. What liquidity is confounded with

Liquidity is substantially **series identity** (`liquidity.py` §2):

| tercile | median pre-volume | top targets | mean `p0` | mean entry | mean year |
|---|---|---|---|---|---|
| thin | 318 | U3, PAYROLLS, PCECORE | 49.6 | 71.3c | 2024.5 |
| mid | 3,013 | GDP, PAYROLLS, CPICOREYOY | 52.5 | 68.6c | 2024.4 |
| liquid | 21,865 | GDP, FED, CPI | 55.3 | 68.3c | 2024.6 |

A 70× volume range that maps almost onto a series list. So "thin pays more"
could be "the signal works on labour and PCE targets and not on Fed and CPI
targets". Year and moneyness are *not* the confound — both are flat across
terciles.

Two designs break it.

**Double sort on entry price** (`liquidity.py` §3) — thin beats liquid in every
band, so it is not moneyness:

| entry band | thin | liquid |
|---|---|---|
| 20–60c | +7.69c | −0.65c |
| 60–85c | +7.54c | +6.71c |
| 85–100c | +4.01c [+1.67, +5.78] | +0.79c |

**Within target event** (`liquidity.py` §4) — thin strikes against liquid
strikes of the *same ladder*, so series, date and release are fixed by
construction. Each event contributes one difference:

| set | events | thin − liquid | 95% CI | P(≤0) |
|---|---|---|---|---|
| spec v2 | 81 | **+5.60c** | [−1.70, +13.23] | **0.070** |
| signal only | 199 | +0.77c | [−2.79, +4.40] | 0.337 |

**This is the best-identified version of the claim in this document** and the
one to lead with. It still spans zero.

Note the second row: on the unfiltered signal-only set the gradient largely
vanishes across every measure. Either confirmation removes the noise that was
masking it, or confirmation is doing the work and liquidity rides along. This
sample cannot separate those.

---

## 4. The correction: the spread was never actually measured

`liquidity_spread_diag.py`. Two errors in how §2 and `liquidity_gates.py` §4
were read.

### 4a. No CI was ever put on the difference

§2 differenced two cells each carrying a ±7c interval and reported the
difference as a point estimate. Comparing two overlapping CIs is not a test of
their difference. With an event-clustered CI on the difference itself —
resampling the **union** of events, since one event contributes legs to both
arms:

| variant | thin | liquid | spread | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| k=3 (shipped) | +8.36c | +1.19c | 7.17c | [−1.56, +16.07] | 0.06 |
| k=2 | +6.48c | +4.25c | 2.23c | [−7.16, +11.45] | 0.32 |
| k=1 | +9.50c | +2.07c | 7.43c | [−0.21, +14.89] | 0.03 |
| no confirm | +2.88c | +1.94c | 0.94c | [−4.13, +6.33] | 0.35 |

Every interval is ~±9c wide. 7.17c and 2.23c sit inside each other. Those rows
are **one quantity measured four times with ±9c of noise**, not four answers.
The apparent instability was an artefact of printing estimates bare.

### 4b. Changing `k` changes the portfolio, not the price

`k` is how many post-trigger prints form the confirmation price (median of the
first `k`; fill on the `k+1`th). Counts barely move — 843 / 812 / 817 — which
invites the assumption that these are the same trades. They are not:

| Jaccard overlap | k=3 | k=2 | k=1 | no confirm |
|---|---|---|---|---|
| **k=3** | 1.00 | 0.74 | **0.61** | 0.27 |
| **k=1** | 0.61 | 0.67 | 1.00 | 0.27 |

Only 61% overlap between k=3 and k=1. The confirmation test *is* the
median-of-`k`, so changing `k` changes which legs clear 2c, in both directions
at once — roughly a third of the portfolio swaps while the count holds flat.

Also: `k` is not what sizes the sample. The print-count requirement costs
100–800 legs; the **2c confirmation filter costs ~6,300**.

### 4c. Freeze the population, vary only the price

Take the 585 legs selected under all of k=1, 2, 3 (69% of the shipped 843), fix
the tercile cutpoints on them, and change only which print is filled:

| variant | n | thin | liquid | spread | 95% CI | P(≤0) |
|---|---|---|---|---|---|---|
| k=3 | 585 | +6.32c | +3.72c | **2.60c** | [−8.06, +12.98] | 0.31 |
| k=2 | 585 | +6.23c | +3.62c | **2.61c** | [−8.29, +13.20] | 0.31 |
| k=1 | 585 | +6.34c | +3.07c | **3.28c** | [−7.32, +13.36] | 0.26 |

Flat. **Pricing explains none of the swing; selection explains all of it.** The
tercile cutpoints themselves barely move across k=1/2/3 (33rd percentile 1,077
/ 1,110 / 1,078), so "thin" means the same thing in all three.

**The fixed-population estimate is the one to believe: ~+3c thin − liquid,
P(≤0) ≈ 0.3.** Less than half the +7.17c in §2, which was partly the particular
legs k=3 happened to draw.

---

## 5. How much liquidity selection is already in the panel

`liquidity_gates.py`. There is **no volume filter anywhere in the pipeline** —
verified by grep across `stg/panel/`, `stg/io/` and `stg/events/`, and by the
panel containing legs with lifetime volume of 1 contract (83 legs under 10, 182
under 100, p5 = 30). The series curation in `registry.py` is hand-picked macro
nodes; its liquidity-flavoured decisions (`role: "target_only"` on thin CPI
subcomponents) are **never read as a gate**, and the RATECUT / TNOTED /
NASDAQ100D exclusions are "zero trades collected", not a thinness threshold.

Four gates are nonetheless implicit, and together they nearly triple the median
pre-trigger volume of the tested population:

| gate | legs | events | median pre-volume |
|---|---|---|---|
| panel rows (**G1**: ≥1 print after `t_res`) | 10,714 | 401 | 2,007 |
| **G2**: has a pre-`t_res` print (for `p0`) | 8,746 | 385 | 4,255 |
| **G3** k=1: ≥2 post prints | 7,981 | 379 | 4,872 |
| **G3** k=2: ≥3 post prints | 7,509 | 373 | 5,354 |
| **G3** k=3: ≥4 post prints (shipped) | 7,161 | 364 | **5,670** |

### G0 — the archive, which is not relaxable

Of 3,685 threshold legs listed across the 401 target events, only **2,462
(66.8%)** are in the panel, and coverage is graded by volume:

| metadata volume | legs listed | in panel |
|---|---|---|
| 0 | 769 | 0% *(correctly absent)* |
| 1–99 | 324 | **56%** |
| 100–999 | 536 | **70%** |
| ≥1000 | 2,056 | **93%** |

Excluding the 769 that never traded, the archive still misses 454 legs that
did — at 44% for thin legs against 7% for liquid ones. This is
`research_log.md` §12's convenience sample **selecting directly on the variable
under test**. The thin tercile is not "thin legs", it is "thin legs that
happened to get collected". Whether the puller's coverage is independent of
outcomes cannot be established from the archive itself.

---

## 6. Relaxing G3 does not manufacture the result

`liquidity_gates.py` §3:

| variant | n | events | net | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| k=3 median-of-3 (shipped) | 843 | 194 | +4.10c | [−1.34, +9.56] | 0.070 |
| k=2 median-of-2 | 812 | 197 | +4.80c | [−0.71, +10.54] | 0.050 |
| k=1 first print confirms | 817 | 208 | **+4.66c** | **[+0.03, +9.39]** | **0.021** |
| no confirmation, fill at 1st print | 2,981 | 295 | +1.89c | [−0.93, +4.70] | 0.100 |

And the 186 legs G3 was excluding pay **+4.55c** on their own, CI [−1.83,
+11.03]. So the gate was mildly *costing* edge, not creating it.

k=1 is the first specification in this study whose CI excludes zero. **It
should not be adopted on that basis.** §4b shows k=1 and k=3 differ by a
third of the portfolio and the differences are inside the noise; switching
because an interval cleared zero would be a sixth outcome-dependent choice on
top of the five in `strategy_spec.md` §6, and `strategy_spec.md` §10 already
names multiplicity as the binding limitation.

---

## 7. Capacity, which cuts against the edge

`liquidity.py` §5:

| tercile | net | median post-trigger volume | USD at 5% participation |
|---|---|---|---|
| thin | +8.36c | 2,859 | **$8** |
| mid | +2.74c | 4,345 | $12 |
| liquid | +1.19c | 31,102 | **$62** |

Where the edge is largest the book is thinnest. Per position, liquid legs are
worth more dollars despite being worth nearly nothing per contract. A ~5× edge
gradient offset by an ~11× capacity gradient is what an equilibrium with costly
information looks like, and it is the mechanism by which the effect can persist
without being arbitraged away. This is the right answer to give the
profitability question: not "the strategy scales", but "it does not, and the
reason it does not is the reason it exists".

---

## 8. Caveats, in order of severity

1. **Not powered.** The headline claim (§4c) has P(≤0) ≈ 0.3 and the
   best-identified version (§3, within-event) P = 0.070 on 81 events. Nothing
   here clears a conventional bar.
2. **G0 selects on the variable under test.** The archive misses 44% of
   thin traded legs against 7% of liquid ones (§5). This is the threat that
   cannot be bounded from inside the data.
3. **Liquidity is largely series identity** (§3). The within-event design
   breaks the confound but costs most of the sample.
4. **The gradient is confirmation-conditional.** It largely vanishes on the
   unfiltered signal-only set, and this sample cannot say whether confirmation
   reveals the gradient or produces it.
5. **§1's mechanism test is partly definitional** and should not be cited
   alone.
6. **In-sample only.** Nothing here touches 2026.

---

## 9. For the thesis

- The right framing is **"direction consistent with underreaction, not powered
  to measure it"** — not a finding. §3's within-event contrast is the only cell
  where the claim is identified rather than correlated, and it is the one to
  present.
- **§4 is a methods contribution in its own right.** "Four point estimates that
  looked unstable were one estimate measured four times without error bars, and
  the apparent sensitivity to a tuning parameter was portfolio churn rather
  than pricing" is a reusable diagnostic, and it belongs in the
  methods/limitations chapter alongside `arbitrage_findings.md` §1 and §3.
- **§5's G0 table is the sharpest statement of the convenience-sample problem
  the repo has.** `research_log.md` §12 asserts the archive is a convenience
  sample; this measures the bias and shows it is graded by exactly the variable
  a liquidity study needs. Any future claim conditioned on thinness inherits it.
- **§7 is the honest answer to "is it profitable?"** — the capacity gradient
  cancels the edge gradient, which is an economically coherent result rather
  than a negative one.

### Open

- `strategy_spec.md` carries three different headline figures across its
  correction header (~+4.7c), its body tables (+5.41c) and `INDEX.md` (+5.32c),
  and does not yet record the spec-v2 value (+4.10c). That reconciliation is
  outstanding and is not touched here.
- The `move ≥ 2c` confirmation threshold is unexamined against the move-size
  cross-section, which found the 2–3c bucket **negative** (−0.79c). Raising it
  would be another in-sample choice and is deliberately not made.
