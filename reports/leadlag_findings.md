# Lead-lag to settlement, resolved by entry price

*Written 2026-09-15. Scripts and captured output in
`analysis/leadlag_2026_09/`. In-sample only (pre-2026). This study rebuilds the
lead-lag question from scratch against a settlement target and the full target
ladder, and it **partly retracts `relations_findings.md` Addendum 2** — see §5.*

## Summary

Holding a target contract from the trigger's resolution to the target's close,
across the whole ladder rather than one representative leg:

| claim | evidence |
|---|---|
| **The signal predicts settlement.** | Top signal tercile minus bottom: **+3.80pp**, clustered CI [+0.41, +7.32], **block-permutation p = 0.0005**. |
| **Its power is confined to mid-priced legs.** | Significant at 10–25c (p=0.01), 25–50c (p=0.00), 50–75c (p=0.02). **Absent above 75c**: 75–90c p=0.63, 95–99c p=0.52. |
| **Gross, it is worth ~2.4c a position.** | Gross **+2.36c**; the same rule on a block-permuted signal earns **−0.09 / −0.96 / −0.03c**, i.e. zero. |
| **Net, it does not clear costs with confidence.** | Friction 1.55c → net **+0.81c**, clustered CI [−1.29, +3.08], P(≤0) = 0.24. |
| **It does not improve the market's probability forecast.** | Neither a walk-forward `delta` tilt nor a logistic with the market price as an offset beats the market on Brier or log loss, in any price bucket. |
| **And it is decaying.** | 2023 **+6.46c**, 2024 **+1.55c**, 2025 **−0.35c**. |
| **Confirmation filtering is the one thing that helps.** | Requiring the market to have already moved the signal's way lifts net to **+3.03c walk-forward** (P(≤0) = 0.054), positive in all three years, and the filter's marginal value is stable (+6.4 / +2.5 / +2.7c) even as the base signal decays to negative. |
| **Maker execution does not rescue it.** | Adverse selection costs 1.7–3.1c against a 0.5–1c spread saving; best maker arm P(≤0) = 0.19 even at zero fee and no queue. Refutes `research_summary.md` §6.4. |
| **Learning the relations is worse than imposing them.** | Imposed sign (0 params) +0.86pp; per-channel (15) +0.47pp; **per-pair (135) −0.07pp**. 0 of 88 pairs survive BH-FDR; 3 of 14 channels do. |

The one-sentence version: **the lead-lag relation is real and it survives the
hardest null this project has, but what it buys is roughly the size of the
trading friction, and it is shrinking.**

Panel: 10,714 legs, 262 trigger events, 401 target events, 135 ordered pairs,
15 channels. Threshold targets only; same-release pairs excluded; zero fitted
parameters in the sign restriction.

---

## 1. The market baseline, by entry price

`price_structure.py`. Nothing here uses the trigger — this is the market's own
settlement forecast, scored by where the contract was trading when the trigger
resolved. Brier **0.0912** against a 0.2499 base rate, so the quote is the
benchmark and beating a base rate would mean nothing.

| band | n legs | n events | paid | realised | edge (pp) | 95% CI |
|---|---|---|---|---|---|---|
| 1–5c | 1739 | 252 | 1.97 | 0.92 | −1.05 | [−1.95, +0.33] |
| **5–10c** | 838 | 222 | 6.71 | 3.46 | **−3.24** | [−5.21, −0.82] |
| 10–25c | 1277 | 284 | 16.27 | 14.17 | −2.10 | [−7.23, +3.64] |
| 25–50c | 1308 | 285 | 36.80 | 32.80 | −4.00 | [−10.72, +3.23] |
| 50–75c | 1160 | 282 | 62.03 | 55.69 | −6.34 | [−13.69, +0.96] |
| 75–90c | 1144 | 269 | 82.86 | 83.83 | +0.97 | [−4.85, +6.08] |
| **90–95c** | 675 | 195 | 92.25 | 96.59 | **+4.34** | [+1.00, +6.87] |
| 95–99c | 1380 | 231 | 96.81 | 97.39 | +0.58 | [−1.93, +2.42] |

Two significant cells, and they point the same way: **cheap YES legs are
overpriced, expensive YES legs are underpriced.** That is longshot bias, and
keeping the buckets unfolded is what makes it visible — folding to
`min(p, 100−p)` would have averaged the two halves into nothing. Note it peaks
at the *moderate* wings (5–10c, 90–95c) and not the extremes (1–5c, 95–99c).

Two structural facts that any "where should the model be right" answer has to
net out, both from `price_structure.py` §2:

* the fee `0.07·p(1−p)` is **~5x heavier at 50c than at 5c** (1.76c vs 0.34c);
* the same net cent is worth **20x more per unit of capital at 5c than at 95c**,
  because Kalshi collateralises at full notional.

---

## 2. Does the lead-lag signal predict settlement?

`signal_model.py`. The signal is zero-parameter:

```
signal = HAWKISH[trigger] · HAWKISH[target] · z_surprise(trigger)
```

The test is whether it predicts the market's *error*, `win − p_entry`. The null
is a **block permutation**: shuffle `z_surprise` among trigger events within
each trigger series, rebuild the signal, recompute. That preserves the ladder,
the outcomes, the prices and the CPI/CPIYOY dependence, and destroys only the
trigger→target pairing — which is the lead-lag claim itself.

**Pooled: +3.80pp, clustered CI [+0.41, +7.32], block-permutation null
+0.26 ± 1.25, p = 0.0005.**

By entry price — the cut this study exists for:

| band | diff (pp) | 95% CI | perm null | **perm p** |
|---|---|---|---|---|
| 1–5c | +1.66 | [+0.07, +4.10] | 0.24 ± 0.79 | 0.04 |
| 5–10c | +3.06 | [−0.96, +7.87] | 0.09 ± 2.18 | 0.10 |
| **10–25c** | **+8.21** | [+0.40, +16.94] | 1.37 ± 3.03 | **0.01** |
| **25–50c** | **+9.43** | [−1.08, +20.58] | 0.06 ± 3.43 | **0.00** |
| **50–75c** | **+8.70** | [−2.85, +20.17] | −0.58 ± 4.31 | **0.02** |
| 75–90c | −0.56 | [−8.19, +6.82] | 0.36 ± 2.83 | 0.63 |
| 90–95c | +1.49 | [−2.72, +5.62] | −0.06 ± 1.45 | 0.14 |
| 95–99c | +0.23 | [−1.14, +1.67] | 0.27 ± 1.04 | 0.52 |

**The signal lives between 10c and 75c and is absent above 75c.** The shape is
sensible rather than convenient: a leg at 96c is already nearly resolved by the
information in the target's own price, so a trigger surprise has almost nothing
left to say about it, and the ~1pp of room left is smaller than the effect.

Reading the components: at 25–50c the *low*-signal tercile has a residual of
−7.93pp — the market badly overprices YES when the trigger surprise points the
other way — while the high-signal tercile sits near zero. So most of the content
is on the **short** side, which §4 confirms in cents.

### The falsification cell fires

`U3` and `JOBLESSCLAIMS` carry `HAWKISH = −1`, so a hawkish CPI surprise must
predict U3 prints *lower* and U3's YES legs getting *less* likely:

| direction block | feature | diff (pp) | 95% CI |
|---|---|---|---|
| +1 | aligned | +1.83 | [−2.24, +6.13] |
| −1 | aligned | **+4.48** | [+0.19, +9.08] |
| −1 | **raw z** | **−4.48** | [−9.11, −0.14] |

The raw association reverses exactly where the economics says it must, and the
alignment repairs it. The sign restriction is carrying the result, not
decorating it.

---

## 3. But it does not improve the probability forecast

This is the part that constrains the claim. Two specifications, both
walk-forward, both scored against the market quote:

* **`p̂ = clip(p + δ·signal)`**, `δ` chosen on prior years only. `δ` comes out
  nonzero (0.005, 0.015, 0.015), and Brier skill is +0.0003 / +0.0004 / −0.0001
  — a rounding error, with the 2025 sign negative.
* **`p̂ = sigmoid(logit(p) + a + c·signal)`** — the market price as a fixed
  offset, so the model can only *tilt* the quote, fit separately within each
  price bucket. Brier skill is **negative in seven of eight buckets**, and the
  only positive one (90–95c) has P(≤0) = 0.15.

A first attempt that let the coefficient on `logit(p)` float did worse still,
which is diagnostic: the market's calibration is already good, and re-sloping it
costs more than the signal adds.

**So the signal separates outcomes without improving the average forecast.**
Both are true, and they are not in conflict. Brier and log loss are dominated by
the thousands of near-certain legs where the signal is silent, and a tilt there
only adds noise; the separation lives in a few hundred mid-priced legs. You do
not have to price every leg to trade — only the ones you choose.

---

## 4. What it earns, held to close

`economics.py`. Rule fixed in advance, per price bucket: top signal tercile →
buy YES, bottom tercile → sell YES, middle → no position. Hold to settlement.
Terciles cut on prior years only.

The decomposition is the cleanest statement in this document:

| | gross | friction | net |
|---|---|---|---|
| **observed signal** | **+2.36c** | 1.55c | **+0.81c** |
| block-permuted signal | −0.09 / −0.96 / −0.03c | 1.55c | −1.71 ± 0.76c |

A permuted signal earns **zero gross** — the market is efficient with respect to
a random pairing, exactly as it should be. The real signal earns **+2.36c**, and
the permutation p-value on net P&L is **p < 0.0001**. That is the strongest
evidence of genuine cross-market predictive content this project has produced.

And then friction takes two thirds of it. Net **+0.81c**, event-clustered CI
**[−1.29, +3.08]**, P(≤0) = 0.24 — *not* significantly positive in absolute
terms, even though it is overwhelmingly significant against the permutation.
Both statements are correct and they answer different questions: the signal is
real; whether it pays is unresolved.

By price, walk-forward:

| band | n | net | P(≤0) | ret on capital |
|---|---|---|---|---|
| 1–5c | 1018 | −0.9 | 0.46 | **−43.1%** |
| 5–10c | 482 | −0.4 | 0.48 | −22.5% |
| **10–25c** | 731 | +3.5 | **0.11** | **+6.2%** |
| **25–50c** | 777 | +3.7 | 0.16 | +4.5% |
| 50–75c | 678 | +2.4 | 0.35 | +5.6% |
| 75–90c | 673 | −1.3 | 0.79 | −12.9% |
| 90–95c | 412 | +0.8 | 0.26 | −23.3% |
| 95–99c | 818 | −0.9 | 0.75 | −19.7% |

The tradable region — 10c to 75c — is the same region where §2 found the
predictive content. The 1–5c bucket is the clearest illustration of the user's
premise: the signal *is* significant there (perm p = 0.04), and the trade still
loses 43% of capital, because a ~0.6c round of fee and half-spread on a 2c
contract is a third of the notional before anything happens.

By side: **short YES** nets P(≤0) = 0.05; **long YES** P(≤0) = 0.61. Consistent
with §1's longshot bias and §2's residual asymmetry — the edge is mostly in
declining to buy, and in selling, overpriced YES legs.

### The two metrics disagree, and both are right

| metric | value |
|---|---|
| mean net cents per position | **+0.806** |
| mean return on capital | **−13.8%** |
| capital-weighted return (total net / total capital) | +1.62% |
| mean net cents, legs ≥ 10c | +1.054 |
| mean return on capital, legs ≥ 10c | −5.5% |

Net cents weights every position equally; return on capital weights by money
tied up. The cheap legs lose and tie up almost nothing, so they barely dent the
cent total and dominate the percentage. Which number is "the" answer depends on
whether the binding constraint is positions or capital — and for a fully
collateralised venue it is capital.

---

## 4b. Which relations carry it — and does learning them help?

`relations.py`. Everything above imposes the sign from economics and pools 135
ordered pairs. Two questions that leaves open: is the pooled effect carried by
relations that make sense, and would *learning* the structure beat imposing it?

Statistic per cell is the **aligned residual**, `mean(sign(signal)·(win − p))`
in pp — zero parameters, reads directly as "when the relation says the YES leg
is underpriced, how underpriced was it". Same block-permutation null, BH-FDR at
q = 0.10.

### Learning loses to imposing, monotonically in parameter count

Walk-forward, sign estimated on prior years only:

| variant | params | aligned (pp) | 95% CI | P(≤0) |
|---|---|---|---|---|
| **imposed** (HAWKISH) | **0** | **+0.86** | [−0.67, +2.52] | 0.15 |
| `learn_chan` (per channel) | 15 | +0.47 | [−0.82, +1.92] | 0.25 |
| `learn_pair` (per pair) | 135 | **−0.07** | [−1.38, +1.33] | 0.55 |
| `flip` (control) | 0 | −0.86 | [−2.52, +0.67] | 0.85 |

Every parameter spent makes it worse, and 135 of them take it to zero. The
control behaves exactly as it must (an exact mirror), so the sign is doing real
work. This is `relations_study_plan` §3.4's question answered on a settlement
target: **the block model wins, and per-pair estimation is strictly harmful at
this sample size.** Median cell has 9 target events; there is nothing to fit.

### Per pair: nothing survives

**0 of 88** testable pairs survive BH-FDR. The largest raw effects
(`U3→ADP` +19.8pp, `U3→JOBLESSCLAIMS` +18.3pp, `GDP→CPICOREYOY` +16.5pp) sit on
6–9 target events each and are exactly what a 88-cell search produces by
chance.

### Per channel: three survive, and they make sense

| channel | legs | tgt events | aligned (pp) | 95% CI | perm p | BH |
|---|---|---|---|---|---|---|
| **labour→labour** | 489 | 23 | **+6.79** | [−1.72, +16.63] | 0.00 | ✓ |
| growth→inflation | 391 | 68 | +5.15 | [+0.39, +9.88] | 0.09 | |
| **inflation→inflation** | 693 | 89 | +0.82 | [−3.82, +5.42] | 0.00 | ✓ |
| **labour→policy** | 657 | 30 | +0.64 | [−0.34, +1.47] | 0.02 | ✓ |
| labour→inflation | 2641 | 239 | +0.25 | [−1.25, +1.74] | 0.37 | |
| **inflation→policy** | 1031 | 33 | **−0.09** | [−0.76, +0.68] | 0.68 | |

The three survivors are economically coherent, and in the most defensible way:

* **labour→labour** is `JOBLESSCLAIMS→{PAYROLLS, ADP, U3}`, `U3→ADP`,
  `PAYROLLS→ADP`. Weekly jobless claims leading monthly payrolls is the
  textbook high-frequency-leads-low-frequency relation, and ADP is a private
  estimate of the official payrolls number.
* **inflation→inflation** is almost entirely **PCECORE ↔ CPI-family**. This is
  the strongest *a priori* relation in the whole grid — core PCE and CPI
  measure overlapping baskets, and PCE is partly constructed from CPI source
  data. It is close to mechanical.
* **labour→policy** is `{U3, PAYROLLS, JOBLESSCLAIMS}→FED`. The dual mandate.

**And the famous one is flat.** `inflation→policy` — CPI→FED, the single most
watched macro channel there is — comes out at −0.09pp, p = 0.68. That is not a
failure of the method; it is the method working. The coherent reading is
attention: the channel everyone trades is priced efficiently, and what is left
is in the less-watched corners (PCE↔CPI construction overlap, claims→payrolls).
That is a hypothesis this study suggests, not one it tests.

### The caveat that applies to all three

**Every BH survivor has a clustered bootstrap CI that includes zero.** The two
tests disagree because they ask different things: the permutation holds the
panel fixed and asks whether the trigger→target *pairing* is informative, while
the bootstrap asks whether the result would survive a different draw of target
events — and with 23–89 target events and near-perfect within-event dependence,
that is a wide question. Read the channels as **where the pooled effect lives**,
not as three independently established relations.

---

## 4c. Maker execution does not rescue it

`maker_fill.py`. `economics.py` assumed a taker throughout, and that assumption
costs **1.55c of a 2.36c gross edge**. `research_summary.md` §6.4 flags this as
*"the single most important execution assumption in Phase 4"* and argues for
resting orders, since the signal predicts settlement rather than immediacy. It
had never been tested.

Simulation: the trigger resolves; you rest a limit at `p0`, the target's last
pre-resolution price, on the side the signal indicates. A resting buy fills when
someone sells at or below the limit, a resting sell when someone buys at or
above it — both observable on the trade tape, so fills are measured.

**Adverse selection is severe, and it is worse than the spread it saves:**

| window | fill rate | gross if filled | gross if unfilled | selection |
|---|---|---|---|---|
| 1 hour | 0.15 | −0.55c | +2.57c | **−3.12c** |
| 1 day | 0.46 | +1.19c | +2.87c | **−1.68c** |
| to close | 0.86 | +0.05c | **+14.83c** | **−14.77c** |

The geometry is unavoidable: if the signal is right the price gaps away and you
are not filled; if it is wrong the price comes back and you are. The last row is
the cleanest statement — positions that never filled would have made +14.83c
gross, positions that filled made +0.05c.

**No maker arm beats the taker:**

| arm | n | fill rate | net | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| taker @ `p_entry` | 4582 | 1.00 | +0.40c | [−1.88, +2.81] | 0.38 |
| maker 1 hour, taker fee | 683 | 0.15 | −1.46c | [−6.64, +3.03] | 0.72 |
| maker 1 day, taker fee | 2090 | 0.46 | +0.33c | [−2.39, +2.93] | 0.40 |
| **maker 1 day, no fee** | 2090 | 0.46 | **+1.19c** | [−1.53, +3.80] | **0.19** |
| maker to close, no fee | 3946 | 0.86 | +0.05c | [−2.25, +2.34] | 0.49 |
| rest to close then cross, no fee | 4582 | — | +0.93c | [−1.25, +3.17] | 0.21 |

The best arm anywhere is P(≤0) = 0.19 — still not significant, and that assumes
**a zero maker fee and no queue**. A fill is recorded whenever any trade crossed
the limit, which ignores queue position, so these fill rates are upper bounds
and every maker arm is optimistic. It loses anyway.

**§6.4's maker hypothesis is refuted.** Resting orders save roughly 0.5–1c of
spread and give back 1.7–3.1c to selection. That closes the last open route to
profitability on this signal, and it closes it the way §6.4 asked — modelled
explicitly rather than assumed either way.

(The taker baseline reads +0.40c here against +0.81c in §4 because this study
buckets on `p0` rather than `p_entry`, so both arms decide on identical
information, and drops the 1,968 rows with no pre-resolution price.)

---

## 4d. Confirmation filtering — the one thing that helped

`move_filter.py`. §4c's counterfactual (+14.83c on never-filled positions) is
unattainable at `p0`, but decomposing it shows **+7.81c of that edge is still
there at `p_entry`**, which is transactable. The repricing takes 7.02c, not all
of it.

And what distinguishes those positions is observable at the decision point: the
market **had already moved in the signal's direction**.

| set | mean move | median | share >1c |
|---|---|---|---|
| all positions | +0.18c | **0.0c** | 0.26 |
| filled by close | −0.92c | 0.0c | 0.17 |
| **never filled** | **+7.02c** | **+4.0c** | **0.81** |

The median leg does not move at all and a thin tail gaps several cents — which
is why §4c's aggressive-limit ladder barely raised the fill rate (0.46 → 0.57
for 8c of slack). There is no smooth walk through intermediate prices to catch.
But you do not need to catch it mid-move; you can buy after it.

**Requiring confirmation before trading:**

| min move in signal's direction | n | net | 95% CI | P(≤0) |
|---|---|---|---|---|
| none (the §4 rule) | 4582 | +0.40c | [−1.88, +2.81] | 0.38 |
| ≥ 0c | 3033 | +0.90c | [−1.35, +3.24] | 0.23 |
| ≥ 1c | 1747 | +1.60c | [−1.33, +4.59] | 0.14 |
| **≥ 2c** | 1187 | **+2.30c** | [−1.50, +6.17] | **0.12** |
| ≥ 3c | 888 | +2.18c | [−2.28, +6.74] | 0.17 |

**Three controls say it is the conjunction, not either half:**

| control | net | P(≤0) |
|---|---|---|
| momentum only, signal ignored (\|move\| ≥ 3c) | **−0.27c** | 0.56 |
| market moved ≥3c but signal disagrees | **−0.19c** | 0.55 |
| filter on \|signal\| magnitude instead (top 50%) | **−0.27c** | 0.59 |

So it is not momentum, not large surprises, and not the signal alone — it is
the signal *and* the market's confirmation together. Economically that reads as
the signal identifying direction while the move confirms the information was
actually material to this target.

**It is still not significant.** Best cell is P(≤0) = 0.12 with a CI spanning
zero, the threshold was chosen in-sample, and five were tried. This is a
direction that deserves a walk-forward specification, not a result.

---

## 4e. Stability of the confirmation trade

`confirmation_stability.py`. Entry at the **second** post-news print throughout,
so the confirming print is never also the fill — the median gap between first
and second print is 58.7 minutes and the median price difference is 0.0c, so the
confirmation is genuinely observable and actionable.

**Positive in every year, and decaying:**

| year | n | events | net (k=2c) | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| 2023 | 85 | 40 | **+11.35c** | [+1.57, +19.63] | 0.01 |
| 2024 | 404 | 97 | **+4.12c** | [−3.11, +11.35] | 0.13 |
| 2025 | 607 | 105 | **+2.04c** | [−3.08, +7.13] | 0.21 |

**The walk-forward version holds.** Choosing the threshold on prior years only
(k=1 for 2024, k=2 for 2025) gives **+3.03c, CI [−0.69, +6.88], P(≤0) = 0.054**
over 1,190 positions. Nothing is fitted on the year being scored.

**Chronological halves**, which give larger clusters than calendar years:
first half +4.00c (P(≤0) = 0.09), second half +2.88c (P(≤0) = 0.15).

**The filter's own contribution is stable even as the base signal decays:**

| year | net unfiltered | net filtered | gain from the filter |
|---|---|---|---|
| 2023 | +4.96c | +11.35c | **+6.38c** |
| 2024 | +1.66c | +4.12c | **+2.46c** |
| 2025 | **−0.61c** | **+2.04c** | **+2.66c** |

This is the most important row in the study. By 2025 the unfiltered rule *loses
money*, and the confirmation filter still adds ~2.7c — the same amount it added
in 2023. The base signal is decaying; the filter is not.

### What stops this being a finding

1. **Three years, and 2023 is 40 events.** A year-clustered bootstrap on three
   clusters is degenerate — its "CI" is just the range of the three means, and
   the P(≤0) = 0.000 it reports must not be cited. The event-clustered figures
   are the honest ones.
2. **The decay is real**: 11.35 → 4.12 → 2.04. Extrapolated, 2026 is thin.
3. **In-sample only.** The specification was built by someone who has read every
   other result in this repo, and the walk-forward controls the threshold but
   not the analyst.
4. **Spread assumption.** Costs use `research_log.md` §11.2's effective spread,
   measured on liquid legs. These are mid-priced legs that just traded, so the
   assumption is probably reasonable here — but it is an assumption.

---

## 5. This partly retracts Addendum 2

`relations_findings.md` Addendum 2 concluded:

> the first executable quote is a settlement forecast the signal cannot improve
> at any delta, with a walk-forward optimiser independently setting its weight
> to zero.

On **proper scoring, this study agrees** — §3 finds no Brier or log-loss
improvement under either specification. But the broader reading of that sentence
— that the signal has no settlement content — **does not survive**. On the full
ladder the signal earns +2.36c gross against 0.00c for a permuted signal, at
p < 0.0001.

The difference is the object, not the statistics. Addendum 2 scored **one
representative leg per target event**, chosen as the most-traded; that leg sits
near the money, and the ladder's mid-priced legs where the signal actually lives
were never in the sample. It also scored only the *average* forecast, which §3
shows is the wrong instrument for a signal that is silent on most legs.

So the thesis statement in `relations_findings.md` §"What follows" item 3 should
be narrowed to: *the signal does not improve the market's average probability
forecast*. The stronger claim — that it is already in the price — is not
supported once the whole ladder is priced.

---

## 6. Caveats, in order of severity

1. **The decay.** 2023 +6.46c, 2024 +1.55c, 2025 −0.35c; year-clustered CI
   [−0.35, +6.46]. Three years is far too few to cluster on honestly, and the
   direction is the same one `settlement_distribution_findings.md` found for
   the WTI wings. Two independent effects in this venue both shrinking toward
   zero is a pattern, and the natural reading — the venue is maturing — predicts
   the 2026 holdout is weaker than anything here.
2. **Net P&L is not significant.** P(≤0) = 0.24 on the event-clustered
   bootstrap. Everything strong in this document is a statement about the
   signal's *content*, not its *profitability*.
3. **`data/trades/` is a convenience sample** (`research_log.md` §12). Entry
   requires a print after the trigger resolved, so legs that never traded post-
   resolution are absent, and they are disproportionately the illiquid ones.
4. **In-sample only**, and the sign table, the price buckets and the tercile
   rule were all specified by someone who has read this repo's previous
   results. The block permutation controls the pairing, not the analyst.
5. **Threshold targets only.** WTI is excluded on principle (§design notes);
   this is a claim about scheduled macro releases, not about the venue.

---

## 7. What follows

- **This is the headline candidate.** It is on-thesis in a way the WTI wing
  result is not: cross-market, event-driven, propagation between scheduled
  macro releases — the actual research question. It has a falsification cell
  that fires, zero fitted parameters in the sign restriction, and the hardest
  null in the repo.
- **The OOS cell should probably be spent here.** `TODO.md` owes a
  re-specification of the pre-registered cell, and there are now three
  claimants (channel pooling, WTI wings, this). This one has the best
  combination of on-thesis relevance and in-sample strength — and the decay in
  §6.1 is precisely the thing only an OOS test can settle. That is a supervisor
  decision, not a code change.
- **Narrow Addendum 2's claim** in `relations_findings.md` per §5, and note the
  correction in `research_log.md` §14's style.
- **One cheap extension**: restrict to 10–75c and re-run everything. Both §2
  and §4 say that is where the effect is, but choosing the region after seeing
  the table is a free parameter, so it needs its own walk-forward or it is not
  reportable.
