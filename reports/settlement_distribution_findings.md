# Wing calibration: the gate passes, but not where the plan predicted

*Written 2026-09-14. Findings for `settlement_distribution_plan.md` §7 item 1,
the gating experiment. Scripts and captured output in
`analysis/settlement_dist_2026_09/`. In-sample only.*

## Summary

The plan proposed moving the edge from `μ̂` to `σ̂` on the argument that a
digital has no vega at the money, so a better uncertainty estimate pays in the
wings — where the fee is roughly half and longshot bias should leave tails
overpriced. Item 1 was the gate: are the wings actually mispriced?

**They are, and the split is clean and not the one the plan assumed.**

| ladder kind | wing edge (longshot side) | trade, sell longshot 3–20c |
|---|---|---|
| threshold (CPI, U3, PAYROLLS, FED, …) | **≈ 0** — +0.20pp at 10–20c, +0.08pp at 20–35c | **−0.13c**, CI [−1.79, +1.46], P(≤0) = 0.55 |
| bucket (WTI, WTIW) | **−6.6pp** on coherent ladders, CI [−8.19, −4.93] | **+4.96c**, year-clustered CI [+2.26, +5.81], P(≤0) = 0.005 |

So: **the macro-release wings are fair, and the entire wing mispricing lives in
WTI's bucket ladders.** Both halves of that matter, and the second one is not
the result the plan was written to find.

4606 legs, 822 events, 19 series. No pdf reconstruction anywhere — entry is a
real traded price on the snapshot day, the outcome is the market's own `result`
field.

---

## 1. The macro wings are fair — this retires a premise of the plan

`wing_calibration.py`, threshold ladders:

| band | n legs | n events | paid | worth | edge (pp) | 95% CI | P(≥0) |
|---|---|---|---|---|---|---|---|
| 3–10c | 595 | 382 | 5.30 | 3.53 | −1.77 | [−3.21, −0.18] | 0.02 |
| 10–20c | 328 | 255 | 13.83 | 14.03 | **+0.20** | [−3.32, +3.87] | 0.54 |
| 20–35c | 343 | 240 | 26.45 | 26.53 | **+0.08** | [−4.12, +4.34] | 0.52 |
| 35–50c | 262 | 223 | 41.34 | 40.84 | −0.50 | [−6.37, +5.71] | 0.44 |

Only the 3–10c band is distinguishable from zero, at −1.77pp, which does not
clear its own ~1.2c cost. The pre-specified trade nets **−0.13c**, and the CPI
family alone nets −0.28c.

This is a genuine null and it costs the plan its motivating premise. The `σ̂`
programme was justified by "the crowd is already systematically overpricing
tails" — on macro releases *on this venue*, over this sample, it is not. §4's
model families were ranked for exactly those series: bottom-up CPI assembly
from BLS component weights was plan §7 item 4, and it was attractive because it
gives a zero-parameter `σ̂_T`. A zero-parameter `σ̂` is still a good object, but
there is now no evidence of a wing mispricing for it to collect on. **Item 4
should be demoted.**

It is worth stating plainly: this is a third independent way of saying the
macro side of this venue is efficient, alongside `research_log.md` §13 (no
tradable drift) and `relations_findings.md` Addendum 2 (the first quote is a
settlement forecast the signal cannot improve).

---

## 2. WTI's bucket wings are badly mispriced

`wing_calibration.py`, bucket ladders — every band, monotone, all CIs excluding
zero:

| band | n legs | n events | paid | worth | edge (pp) | 95% CI |
|---|---|---|---|---|---|---|
| 3–10c | 742 | 332 | 5.55 | 1.07 | **−4.48** | [−5.23, −3.53] |
| 10–20c | 495 | 302 | 13.22 | 4.44 | **−8.78** | [−10.49, −6.92] |
| 20–35c | 387 | 254 | 26.16 | 16.79 | **−9.37** | [−13.10, −5.46] |
| 35–50c | 254 | 201 | 41.20 | 33.85 | **−7.35** | [−12.74, −1.84] |

Selling the longshot at 3–20c and holding to settlement nets **+4.96c** per
contract after a fee charged once and half the measured spread.

### Four attempts to kill it, all failed

`wing_overround.py`, `wing_robustness.py`:

1. **Ladder overround.** Buckets are mutually exclusive and exhaustive, so
   prices must sum to 100c. At day granularity they sum to a median of **124c**,
   and 77% of ladders exceed 105c — **but see the correction below; that
   measurement does not survive a synchronicity control** — selling every leg of a 124c ladder returns 24c by
   construction. This was the obvious explanation and it is wrong. On
   mass-normalised prices the edge is **unchanged** (−5.4pp at 3–10c, −8.9pp at
   10–20c), and restricted to coherent ladders only (mass 0.95–1.10, 78 events)
   the edge is **−6.58pp**, CI [−8.19, −4.93], with the trade netting +5.36c.
   Per-event `corr(mass, net)` is +0.07.
2. **The winner missing from the sample.** We see only legs that traded, so if
   the winning bucket routinely failed to trade, observed legs would lose
   mechanically. This is the `relations_findings.md` item 1 coverage defect in
   its most damaging form. Measured: the winner is present in **95%** of bucket
   ladders, **97%** of coherent ones. Not the explanation.
3. **A few events wearing a large n** (the §13.5 failure mode). 397 events,
   **93% individually positive**, median +6.87c. Dropping the best decile still
   gives +4.15c; dropping the best *quartile*, +3.00c.
4. **One series.** WTI +5.00c (348 events) and WTIW +4.71c (49 events) agree
   independently; WTIW is the weekly contract.

Spread sensitivity: at 2×/3×/4× the measured effective spread the trade still
nets +4.25 / +3.55 / +2.85c, all CIs excluding zero.

---

> **Correction (2026-09-15).** The 124c overround quoted in §2 is withdrawn.
> `analysis/arbitrage_2026_09/coherence.py` re-measures ladder mass by how far
> apart the legs actually traded: median **111c** at day granularity, **70c**
> within 60 minutes, **74c** within 5 minutes. Loose windows are stale (legs
> priced hours apart in a moving market), tight windows are incomplete (few legs
> trade in five minutes, and a partial partition is biased down). Trade prints
> cannot measure overround; quotes can. The wing result itself is unaffected —
> it survived mass-normalisation and the coherent-ladder subset independently —
> but the standalone overround finding should not be cited. Full detail:
> `arbitrage_findings.md` §3.

## 3. What this actually is — and it is a `σ` result

The temptation is to call this a forecasting edge. It is not, and the right
reading makes it more interesting rather than less.

`edge_economics.md` §2(b) established that **WTI has no information event**: its
resolution is a price snapshot near the prevailing price, not a release. So the
terminal distribution is tight around spot, and buckets away from spot are
structurally unlikely — while the market prices them as if the distribution
were wider. Nothing is being forecast. The only quantity in play is the width.

`relations_findings.md` item 1 said exactly this from the other direction, and
was disbelieved for good reason. Its PIT found WTI *underconfident* — 3.5% of
outcomes in the tails against 20% expected, 83% in the central half against
50% — but the whole of item 1 was then read as "a finding about the
reconstruction, not about the market", because `settlement_trade.py` priced the
same bias with real prices and contradicted it on the macro side.

**For WTI, the reconstruction and the direct price test now agree.** The PIT
said the implied distribution is too wide; the wing trade collects exactly that,
at real traded prices with no reconstruction involved. That is independent
confirmation of the one half of item 1 that survived, and it is the plan's `σ̂`
thesis holding in its purest available form — the market's implied `σ` is too
large, and the wings are where you get paid for knowing it.

---

## 4. Caveats, in order of severity

1. **It is decaying.** Net by year: 2022 **+5.81c**, 2023 **+5.81c**, 2024
   **+2.43c**, 2025 −1.17c on 4 events (unreadable). 2024 is under half of
   2022–23, with the realised longshot win rate climbing 0.65% → 2.06% → 4.95%
   toward its ~8.6c price. The year-clustered CI still excludes zero, but the
   direction is monotone and the obvious reading — the venue is maturing — is
   the one that predicts this is gone by the 2026 holdout. **Any OOS test of
   this will likely be testing a weaker effect than the in-sample number.**
2. **`data/trades/` is a convenience sample** (`research_log.md` §12), and WTI
   runs at 64% event coverage. Which legs traded is not random, and wing legs
   are the least-traded ones, so the selection acts precisely on the population
   under test. Threat 2 above bounds the damage but does not remove it.
3. **In-sample only.** Nothing here touches 2026.
4. **Fills are assumed, not modelled.** Selling a 5c bucket 397 times assumes
   size is available at the last trade price. The spread sensitivity is a proxy
   for this and it survives 4×, but depth is not spread.
5. **This is one underlying.** WTI and WTIW are the same oil price. The `n = 397
   events` is not 397 independent bets on the world.

---

## 5. What follows

The plan's §7 order needs rewriting, because item 1 answered a question it was
not asked.

- **Promote: WTI variance.** This is the funded direction. The next question is
  whether the width mispricing is *estimable ex ante* — a zero-parameter `σ̂_T`
  for WTI is realised volatility over the remaining window, which needs no
  model and no BLS weights. That gives a real `z` and turns this from "sell the
  wings" into the plan's `p̂ = Φ(−z)`, testable with calibration-by-`z`.
- **Demote: bottom-up CPI `σ̂` (old item 4).** Still methodologically the
  cleanest object in the plan, but §1 says there is no wing mispricing on macro
  releases to collect.
- **Keep, and it is now more urgent: full-listed-ladder pricing (old item 2).**
  Caveat 2 is the binding threat to this result, and that item is the fix.
- **Add: a decay test with a pre-registered cut.** Caveat 1 is the thing most
  likely to make this result wrong going forward, and the honest version is to
  state the 2024 figure as the current estimate rather than the pooled one.
- **The OOS decision gets harder.** `relations_study_plan.md` §4 already owes a
  re-specification of the pre-registered OOS cell to the channel test. This is
  now a competing claimant with a stronger in-sample result and a decay problem
  that only an OOS test can settle. That is a supervisor conversation, not a
  code change.
