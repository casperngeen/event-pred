# Beyond surprise: a predictive distribution over the settlement statistic

*Written 2026-09-14. A design document, not a findings document — **nothing
below has been run.** Every number quoted is a citation to an existing result
(`relations_findings.md`, `research_log.md` §13, `edge_economics.md`), never a
new measurement. Successor to `relations_study_plan.md`, which was the last
design doc for the surprise primitive.*

> **Why this document exists.** Every signal this project has built is a
> *shock* variable — `surprise = resolved_value − implied_mean`, defined
> against consensus, existing to explain instantaneous repricing. That is a
> drift model: it predicts `m_{t+δ} − m_t`. Three independent results now say
> the drift business is closed (§1). The proposal here is to change the
> **target of prediction** rather than search for a better shock measure: stop
> predicting the price path, predict the settling statistic, and hold.

**Scope reminder.** In-sample only (pre-2026). The 2026 block stays untouched,
including for feasibility checks (`splits.py::assert_no_oos()`).

---

## 1. What is actually closed, and what is not

The cost argument for holding to settlement is real and worth stating once:

| path | round-trip cost at 50c |
|---|---|
| taker in, taker out | ~3.5c |
| taker in, **hold to settlement** | ~1.75c — settlement is not a trade, so there is no exit fee, win or lose |
| maker in, hold to settlement | well under 1c, and the fill can be patient because there is no moment you must be in by |

So the hurdle drops 2–4x against `research_log.md` §13's ~2.6c floor. But
**this does not rescue the surprise signal**, and the reason is specific rather
than general. `relations_findings.md` Addendum 2 already ran the
hold-to-settlement test:

- (a) the market's own first executable quote scores Brier **0.0147** against a
  0.2486 base rate;
- (b) adding the channel signal at weight δ worsens Brier *and* log loss at
  every δ in 0.01–0.10;
- (c) a walk-forward optimiser sets **δ = 0.000 every year**;
- (e) the P&L is indistinguishable from a within-series direction permutation
  (p = 0.498).

Halving the cost does not rescue a numerator of zero. Cheaper is not the issue.

**What that null does and does not cover.** It is a null about
*surprise-derived structure* — the channel signal, evaluated as an increment to
the market's settlement forecast. It is **not** a null about settlement
forecasting in general, because no estimate built from outside the ladder was
ever tested. The distinction is what this document is for; it should be stated
that carefully in the thesis too, because the loose version ("we tested
hold-to-settlement and it failed") over-claims.

**The honest headwind, stated up front.** Brier 0.0147 is a very good
forecaster. Most of that is easy — about half of all legs trade under 10c and
settle NO. The place where the market is genuinely uncertain is the 20–80c
band, and there Addendum 2 found the **gross** edge, before any cost at all, to
be **−0.138c**, clustered CI [−8.04, +6.84]. That band is contested and this
project has already failed to beat it once. The proposal below deliberately
does not start there.

---

## 2. The object to build

Not a signal. A predictive distribution over the settlement statistic:

```
p̂ = P(X_T > K) = 1 − F̂_T(K)
```

Two estimates — central forecast `μ̂_T`, remaining uncertainty `σ̂_T` — and one
variable that organises the whole book:

```
z = (K − μ̂_T) / σ̂_T ,     p̂ = Φ(−z)
```

This is the `d₂` of Black–Scholes and it does the same job: it says how far the
strike sits in units of *what is still unknown*.

`σ_T` is **not** the historical volatility of the series. It is the standard
deviation of the terminal value conditional on everything published so far, so
it shrinks mechanically as component releases land. That gives `p̂` a built-in
convergence schedule, which is itself testable against the market's — and the
repo can already produce the market's side of that comparison:
`implied.py:527 build_daily_implied_means` emits `implied_mean` and
`implied_std` **per event per day**, so the market's `σ` path to resolution is
a column, not a modelling project.

### Where the edge sits, and the fee geometry that follows

```
∂Φ(−z)/∂σ = φ(z) · (K − μ̂)/σ̂²
```

At the money (`K = μ̂`) this is **zero**: a digital has no vega at 50c. The
probability is ½ whatever you believe about the width. Sensitivity to `σ` peaks
near `|z| = 1`, i.e. prices near 16c and 84c.

| edge in | pays near | fee there | contested? |
|---|---|---|---|
| `μ̂` | 50c | maximal, ~1.75c | yes — competing with every nowcast |
| `σ̂` | 16c / 84c | ~0.94c | much less, and longshot bias points the same way |

That is an unusually clean alignment — the less contested edge is also the
cheaper one to trade — and it is the reason to reallocate effort to `σ̂`. It is
also where this project's existing §13/Addendum-2 nulls have the least to say,
since both were measured at or near the money.

One caveat to carry: the vega expression is exact only under the Gaussian map
`p̂ = Φ(−z)`. Where an empirical predictive CDF is available (§4, weather), use
it directly and treat `z` as a reporting coordinate, not a model.

---

## 3. The prerequisite — do this before anything else

**`σ̂_market` is currently unmeasurable in this repo, and `σ̂` is the whole
proposal.** `relations_findings.md` item 1 established that mean ladder
coverage is 0.64–0.83, so 20–35% of listed strikes are absent from every
recovered distribution, and the absent ones are disproportionately
far-from-the-money because those trade least. The recovered distribution is
therefore systematically **too narrow**: `implied_std` is biased down by
construction, exactly in the wings where the `σ` edge is supposed to live.

Item 1 already names the fix and marks it upstream of everything: **price the
full listed ladder from last-known prices rather than same-day trades only.**
It was a data-prep item for the surprise measures; here it is load-bearing.

Note also that item 1's PIT result is currently read as "a finding about the
reconstruction, not about the market" — `settlement_trade.py` priced the same
bias with real traded prices and no reconstruction at all, and found the 50c
contract accurate (priced 50.9c, realised 48%). Any wing claim inferred from
the recovered pdf inherits that doubt.

### Which is why the first experiment uses no reconstruction at all

Before building any `σ̂`, ask whether there is anything in the wings to win:

> **Wing calibration, direct.** Take every leg whose entry price (first
> executable print) is in 3–20c and in 80–97c. Hold to settlement. Compare
> realised settlement rate to entry price, clustered on `target_event`, split
> by series type (macro release vs WTI price snapshot).

Zero parameters, no pdf recovery, no `μ̂`, no `σ̂` — and
`analysis/relations_2026_09/settlement_trade.py` is already the right shape to
adapt. It answers the only question that licenses the rest of the programme:
**is the crowd systematically overpaying for tails on this venue, and in which
series?** If the answer is no, the `σ̂` programme is dead before it costs a
week, and that is a publishable result in its own right (it would be a
venue-level refutation of longshot bias, against a large literature).

Two priors worth naming, because they disagree:

- WTI's PIT signature (3.5% of outcomes in the tails against 20% expected, 83%
  in the central half against 50%) is what longshot bias looks like — wings
  settle NO far more often than they are priced. It is measured through the
  suspect reconstruction, but the direction is a real prediction.
- The macro-release ladders showed the *opposite* PIT sign (mean `u` ≈
  0.65–0.91, one-sided, upper decile 25–75%), which under the same reading says
  the **upper** wing was underpriced over a 2022–24 inflation-surge sample.
  That is regime, not structure, and it is why the wing test must be split by
  year — `settlement_trade.py`'s CPI-family hit rate ran 0.58 / 0.50 / 0.69 /
  0.26 over 2022–25 and did not survive year-clustering.

---

## 4. Model families for `μ̂` and `σ̂`

In the order they are worth this project's remaining weeks:

1. **Bottom-up component assembly**, for any statistic that is a known weighted
   sum. For CPI the weights come from BLS, and several components already exist
   in the registry as `target_only` series (`CPIFOOD`, `CPISHELTER`,
   `CPIAPPAREL`). `σ̂_T` is then *computable rather than estimated* — it is the
   variance of the not-yet-published components under their known weights.
   **Zero fitted parameters**, which is the same methodological posture that
   made the channel-pooling result credible at `n = 53`, and the only posture
   this project's sample size can afford.
2. **Ensemble → empirical CDF**, for weather. Skips the normality assumption
   entirely; the predictive distribution is the ensemble spread, given rather
   than fitted. Do not impose Gaussian where the actual predictive distribution
   is published.
3. **Nowcasting / mixed-frequency** — MIDAS, dynamic factor models, bridge
   equations. This literature exists precisely to produce a distribution over a
   not-yet-released statistic from higher-frequency inputs, and central-bank
   nowcast infrastructure is public. Ranked third only because it fits
   parameters, and `n` has been the binding constraint since August.

---

## 5. Three risks this changes, not removes

**Capital-time, not just edge.** Kalshi positions are fully collateralised, so
capital is locked at full notional for the whole hold. A 2c edge held 90 days
is a different animal from 2c held a day. Rank by **edge per capital-day** from
the start, or the book fills with correct-but-slow positions and the year comes
out flat. This is a new metric — `edge_economics.md` has nothing like it.

**No exit.** The stop loss is given up deliberately. Max loss is the premium,
which is bounded, but it is realised in full every time. Sizing must reflect
uncertainty in `p̂` itself, not just its point estimate: fractional Kelly, or
explicitly shrink `p̂` toward the market price in proportion to how thin the
evidence is. At this project's sample sizes shrinkage is not optional.

**Correlated settlement.** Holding many macro contracts to resolution means one
CPI print settles a large fraction of the book at once. This is where the
correlation structure already recovered becomes directly useful — as the input
to a **portfolio constraint**, not as a trade signal. That is a better home for
the channel result than the trade it failed as.

---

## 6. Evaluation

Drop P&L-per-trade as the primary metric.

- Score `p̂` with **log loss and Brier**, benchmarked against the market price
  at entry, **clustered by underlying event** (the §13.5 lesson: two
  configurations that looked tradable were a few events wearing a large `n`).
- **Calibration plots by `z`-bucket** are the headline diagnostic: they say
  immediately whether `σ̂` is systematically too tight or too wide, which is the
  quantity the whole programme turns on. At this sample size that single plot
  is worth more than any backtest.
- Keep P&L as a secondary check with the ~1.75c / ~0.94c hold-to-settlement
  hurdles above, not the ~3.5c round-trip figure.

---

## 7. Suggested order

| # | Work | Gate |
|---|---|---|
| 1 | Wing calibration, direct (§3) — no reconstruction, year-split, clustered | If flat, stop. The rest is unfunded. |
| 2 | Full-listed-ladder pricing from last-known prices (§3 prerequisite) | Makes `implied_std` trustworthy; also fixes every surprise measure. |
| 3 | Market `σ` convergence schedule from `build_daily_implied_means` | Descriptive; establishes what a competing `σ̂` must beat. |
| 4 | Bottom-up CPI `σ̂_T` from component weights (§4.1) | Zero parameters. First real `p̂`. |
| 5 | Calibration-by-`z` scoring harness (§6) | Reused by everything after. |
| 6 | Edge-per-capital-day ranking + settlement-correlation constraint (§5) | Only once a `p̂` exists worth sizing. |

Items 1 and 3 are cheap and independent; they can run together. Item 2 is the
one that has to land before any wing result stated in `σ` units is meaningful.
