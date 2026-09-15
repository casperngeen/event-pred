# Cross-market structure without forecasting: four arbitrage-shaped tests

*Written 2026-09-15. Scripts and captured output in
`analysis/arbitrage_2026_09/`. In-sample only (pre-2026). §3 **corrects a claim
made earlier the same day** in `settlement_distribution_findings.md`.*

## Why these four

Every other study in this repo is bottlenecked on the same thing — median 9
target events per ordered pair. These four are not, because a coherence
violation is **a per-observation fact, not a statistical estimate**: one ladder
and arithmetic settle it. That is the reason to look here, and it is the only
part of this project where sample size is not the binding constraint.

| # | question | answer |
|---|---|---|
| 1 | Do the MoM and YoY ladders, which price one unknown, agree? | **No — and the disagreement is correct.** It prices settlement basis risk from independent BLS rounding. Net −1.07c. |
| 2 | Is headline CPI's implied variance consistent with its components'? | **Suggestive inconsistency**: 6% of days the headline ladder is arithmetically too narrow for its own core ladder. Weakly identified. |
| 3 | Are individual ladders arbitrage-free? | **~2% of adjacent strike pairs violate monotonicity**, stable under synchronicity — real, but median 2.0c against a 2.3c cost. |
| 4 | Does implied *uncertainty* have exploitable structure? | Mild predictable decay (−14%), **no systematic variance premium**, and the cross-event effect **dies under control**. |

---

## 1. MoM vs YoY: the arbitrage that isn't

`identity_cpi.py`, `identity_arb.py`.

Kalshi lists, for the same reference month, both `CPI` ("will inflation rise
more than 0.7% in April", MoM) and `CPIYOY` ("rate of CPI inflation above 2.5%
for the year ending April", YoY). These are **one unknown under an affine map**:

```
YoY_t ≈ c_t + MoM_t ,   c_t = YoY_{t-1} − MoM_{t-12}
```

and `c_t` is already published when both ladders trade. So "YoY above K" and
"MoM above K − c_t" are the same event, two tickers, one payoff.

**The setup looked extraordinary.** 3,228 strike-matched (date, strike) pairs
across 36 reference months, and the two tickers disagree by a **mean of 13.1c**,
median 7–8c, with 58–62% of pairs more than 5c apart.

**It is not arbitrage.** The identity does not hold exactly on published prints
— only 18 of 36 months are exact, with sd 0.103pp — because BLS rounds MoM and
YoY to 0.1pp *independently* and YoY is computed off unrounded index levels. So
matched legs settle differently **12.8% of the time**, and the payoff is
savage: a break costs −100c against a gain of the price gap.

| min edge | n | break rate | gross | net | ret on capital |
|---|---|---|---|---|---|
| >0c | 3062 | 0.128 | +2.70 | **−1.07** | −0.34% |
| >2c | 2458 | — | +3.29 | **−0.71** | +0.21% |
| >5c | 1908 | — | +3.75 | −0.44 | +0.75% |

Clustered on reference month: net −1.07c, CI [−5.71, +2.73], P(≤0) = 0.68.

The confirming detail is that **the market prices the basis risk correctly**.
Break rate by distance of the matched strike from the eventual print: 18.6%
within 0.05pp, 17.4% at 0.05–0.15, **3.0%** at 0.15–0.35, **0%** beyond 0.35 —
and the price gap is largest exactly where breaks are most likely (15.4c vs
5.3c). The 13c wedge is a rounding-risk premium, not an error.

### This corrects my own earlier reading in this session

`identity_cpi.py` found the YoY ladder implying **21% more width** than the MoM
ladder (80–85% of days one-sided) and I read that as a pricing inconsistency.
It is not. If `YoY = c + MoM + e` with `e` the rounding residual, then
`sd(YoY) = √(sd(MoM)² + sd(e)²)`:

| pair | sd(MoM) | sd(residual) | predicted ratio | observed |
|---|---|---|---|---|
| CPI/CPIYOY | 0.115 | 0.103 | **1.344** | 1.210 |
| CPICORE/CPICOREYOY | 0.098 | 0.085 | **1.324** | 1.094 |

The YoY ladder is **narrower** than the rounding noise justifies. There is no
width arbitrage, in either direction.

**This is the most interesting result of the four**, because it is a case where
an apparently enormous cross-market inconsistency is fully explained by a
settlement-mechanics detail. It is also a warning for the thesis: any
cross-market "relation" between two contracts that settle off separately
rounded published figures inherits this basis risk.

---

## 2. Dispersion: headline vs components

`dispersion.py`. Headline CPI is a known weighted sum, so its variance is not
free — this is index-vs-single-stock dispersion with BLS publishing the weights.

Headline must be **wider** than core (it carries food and energy on top), so
`sd(CPI) ≥ w_core · sd(core)` under any non-negative remainder variance and
ρ ≥ 0. On 361 days across 40 months:

- head/core implied-sd ratio: p10 **0.862**, p50 1.152, p90 1.532
- **29% of days the ratio is ≤ 1** — headline no wider than core
- **6% of days it is ≤ 0.79** — below the core contribution alone, which is
  arithmetically impossible
- gating both ladders to ≥4 fresh legs (196 days) barely moves it: p50 1.116,
  33% at ratio ≤ 1

The over-determined three-way system (headline + core + gasoline, 22 days)
returns an implied correlation with median **−0.82** and 36% of days below −1
at `w_core = 0.79`. Values below −1 are impossible.

**But this is the weakest of the four**, for three honest reasons: the weights
are entered by hand (BLS relative importances are not in `data/`; this is
`research_summary.md` §3 asset 2, still outstanding), the three-way test has 22
days over 10 months, and `implied_std` comes from `recover_pdf` on ladders with
known coverage defects. Read it as **a direction worth instrumenting properly**,
not as an established inconsistency.

---

## 3. Ladder coherence: real, small, and it corrects an earlier claim

`coherence.py`. `P(X>K)` must fall as `K` rises; a violation is a locked
arbitrage (buy the low strike, sell the high — the low leg pays whenever the
high one does).

| synchronicity tier | adjacent pairs | violations | rate | mean size |
|---|---|---|---|---|
| any time same day | 34,037 | 782 | 2.30% | 4.98c |
| within 60 min | 13,491 | 269 | **1.99%** | 5.33c |
| within 5 min | 6,962 | 140 | **2.01%** | 5.99c |

**The rate is flat across tiers, so this is not staleness** — it is a real
~2% incoherence rate. But the mean is carried by outliers: the median violation
within 5 minutes is **2.0c** against a ~2.3c cost. Restricting to violations
above 3c with both legs priced 5–95c leaves **34 instances across 24 events**,
median 7c, 16 of them in `FED`. Real, tradable, and rare.

### Correction to `settlement_distribution_findings.md`

That document reported bucket ladders summing to a median of **124c** against
the required 100c, and called it a venue-level overround. **That claim is not
supported.** Measured by synchronicity:

| tier | ladder-days | median mass |
|---|---|---|
| any time same day | 531 | **111c** |
| within 60 min | 82 | **70c** |
| within 5 min | 41 | **74c** |

Loosely matched ladders sum above 100 because legs traded hours apart in a
moving market; tightly matched ones sum below 100 because few legs trade in a
5-minute window and an incomplete partition is biased down. **Neither tier
measures true overround — trade prints cannot measure it at all, only quotes
can.** The WTI wing result itself does not depend on this (it survived
mass-normalisation and the coherent-ladder subset), but the standalone
"124c overround" finding should be withdrawn.

---

## 4. The term structure of uncertainty

`vol_term.py`. The ladder prices a second moment, and
`build_daily_implied_means` has been emitting it per event per day since the
panel was built. Nothing had used it.

**Decay is real but mild.** Each event's `implied_std` relative to its own mean:
1.055 at 30–90 days → 1.029 → 0.958 → 0.930 → 0.927 → **0.904** at 0–1 day. A
~14% decline, consistent across every series, strongest in `FED` (0.758 at 2–6
days) and `GDP` (0.813).

**No systematic variance risk premium.** `sd(realised error) / mean(implied_std)`
straddles 1: `CPI` 1.151, `CPICORE` 1.149, `U3` 1.080, `GDP` 1.019 above;
`CPIYOY` 0.935, `PCECORE` 0.827, `CPICOREYOY` 0.616 below. Median ≈ 0.96. The
extreme is `FED` at 0.294 — the ladder prices 3.4x the uncertainty Fed decisions
actually carried — but n = 12 and its `mean_abs_z` of 0.884 sits near the
calibrated 0.80, so the two statistics disagree and it should not be cited.

This matters: it says the WTI variance result from
`settlement_distribution_findings.md` **does not generalise to macro ladders.**

**The cross-event effect dies under control.** Pooled, a target's `implied_std`
falls 1.1% across a foreign resolution (CI [−2.6%, +0.3%]), and the per-trigger
pattern looked coherent — CPI-family triggers all negative (CPICOREYOY −0.075,
CPISHELTER −0.053, CPIYOY −0.045), labour and energy ~0. But §1 says
`implied_std` falls anyway as an event nears its own close, and any window
straddling a foreign resolution also moves the target closer to its own close.
Matching straddling against non-straddling steps on elapsed days *and* the
target's own horizon:

| horizon | elapsed | diff | 95% CI |
|---|---|---|---|
| 0–6d | 1d | **−0.0238** | [−0.053, +0.006] |
| 0–6d | 2–3d | −0.0225 | [−0.071, +0.055] |
| 7–20d | 1d | +0.0081 | [−0.014, +0.039] |
| 21d+ | 1d | +0.0145 | [−0.016, +0.053] |

Every CI spans zero. **The pooled effect was the term-structure confound.** The
only suggestive cell is a target within 6 days of its own close, where a foreign
resolution shrinks uncertainty an extra ~2.4% — plausible, not established.

---

## 5. What this adds up to

Three negatives with mechanisms and one small positive. That is a worse
headline than "we found an arbitrage" and a better contribution than one,
because each negative identifies *why*:

1. **Settlement mechanics defeat the cleanest cross-market identity available.**
   Independent rounding of two published figures creates a 12.8% break rate that
   no amount of forecasting skill removes. Any thesis claim about relations
   between separately-settling contracts has to clear this bar.
2. **Trade prints cannot measure ladder coherence.** Loose windows are stale,
   tight windows are incomplete, and the answer flips from 111c to 70c depending
   on which you choose. This is a measurement result about the data source, and
   it retires a claim I made earlier today.
3. **The second moment is not a graph** — at least not one detectable here. The
   apparent cross-event uncertainty response is the term structure.
4. **Genuine incoherence exists at ~2%** and is concentrated in `FED`, but the
   median violation does not clear costs.

### For the thesis

- §1 and §3 belong in a **methods/limitations** chapter, not a results chapter.
  They are about what this venue's data can and cannot support, which is exactly
  the kind of contribution the structure-learning power curve also makes.
- §4's variance-premium table is a **one-table refutation** of generalising the
  WTI result, and is worth including for that alone.
- §2 is the only one worth more analysis time, and only after the BLS
  relative-importance table is actually fetched. That is an afternoon's work and
  it is already on the Phase 0 list.
- None of this changes the recommendation from the strategy discussion: the
  reframe plus the mechanical decomposition is still the best use of the
  remaining weeks. §1 is in fact the first piece of that decomposition, and it
  says the identity route is harder than it looks.
