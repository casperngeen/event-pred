# Reconstruction-free ladder statistics, and what they overturn

*Written 2026-09-15. Module `stg_infra/stg/events/quantile.py` (+ 10 unit
tests); scripts and captured output in `analysis/quantile_2026_09/`. In-sample
only. **This document retracts `relations_findings.md` item 1's headline.***

## The change

`implied.py` recovers a pdf from the ladder and integrates it, so every moment
inherits an assumption about strikes that never traded — and item 1 measured
that 20–35% of listed strikes are absent, disproportionately far-from-the-money.
`recover_pdf` handles the gap by placing tail mass `spacing/2` past the extreme
observed strikes.

The new module never integrates. Everything is interpolated between adjacent
**traded** strikes:

| statistic | definition | touches the tails? |
|---|---|---|
| `q50` | strike where `P(X>K)` crosses 0.50 | no |
| `iqr` | `K(p=0.25) − K(p=0.75)`; `sigma_iqr = IQR/1.349` | no |
| `skew_q` | Bowley quartile skew | no |
| `ladder_pit` | `1 − P(X > resolved)`, interpolated at the outcome | no |

Where a ladder does not bracket a level the statistic is **`None`**, not an
extrapolation. A unit test confirms the headline property directly: dropping
both 5% wing legs leaves `iqr` and `q50` unchanged.

Two implementation notes. Ladders are projected onto the monotone cone by
**PAVA** rather than a running minimum, because `coherence.py` found ~2% of
adjacent pairs violate monotonicity even within five minutes, and a running
minimum would drag every later strike down after one bad print. Censoring is
reported rather than hidden.

**The cost is honest and large.** `q50` is bracketed on only **43%** of
event-days, `iqr` on 34%. On the rest the traded ladder genuinely does not
locate the median — the integrated moment was supplying an answer the data does
not contain.

---

## 1. Item 1's calibration result was a reconstruction artifact

`pit_direct.py`. Item 1 reported mean PIT of 0.65–0.91 across the macro ladders
and read it as "the market is biased low". The document already flagged the
problem — `settlement_trade.py` priced the same bias with real strikes and found
the 50c contract accurate — and left the verdict as *read item 1 as a finding
about the reconstruction, not about the market, until this is resolved.*

Resolved. Same events, same snapshot day, PIT interpolated off the traded ladder:

| series | n | **ladder PIT** | item 1 (recovered) | KS D | KS p |
|---|---|---|---|---|---|
| CPI | 37 | **0.529** | 0.722 | 0.165 | 0.244 |
| U3 | 36 | **0.511** | 0.648 | 0.075 | 0.984 |
| CPICORE | 34 | **0.493** | 0.716 | 0.142 | 0.466 |
| CPIYOY | 33 | **0.419** | 0.668 | 0.187 | 0.176 |
| FED | 31 | **0.489** | 0.906 | 0.435 | **0.000** |
| PAYROLLS | 27 | **0.487** | 0.552 | 0.098 | 0.948 |
| CPICOREYOY | 25 | **0.444** | 0.745 | 0.240 | 0.095 |
| PCECORE | 11 | **0.560** | 0.791 | 0.264 | 0.371 |

Every mean collapses to ~0.5 and **no series except FED rejects uniformity**.
Pooled, mean PIT 0.487 and share above 0.5 is 0.452.

FED's rejection is a different failure: `tail20 = 0.000`, so its PIT is
concentrated in the middle — under-dispersion, not location bias. That is
consistent with `vol_term.py` §2, where FED's implied width was 3.4x its
realised error.

**So the price test was right and item 1's headline was wrong.** The macro
ladders are not biased low; `recover_pdf`'s open tails were dragging the
recovered mean below the ladder's own median. Item 1's §"Superseded" section
anticipated exactly this, and this is the evidence it asked for.

### One thing this does *not* settle

Tail behaviour. **19.9% of outcomes landed outside the traded strike range**
entirely (CPIAPPAREL 56%, CPIGAS 36%, CPI 26%, FED 0%), and those are by
construction the extreme ones. Excluding them biases `tail20` down mechanically;
counting them all as tail events biases it up. **Width and tail calibration are
not identified from traded ladders alone** — that needs quotes. Do not read the
sub-0.20 `tail20` values as evidence the ladders are too wide.

---

## 2. Quantile vs integrated moments, on identical ladders

`build_ladder_panel.py`. Location agrees (median `q50 − implied_mean` ≈ 0 for
every series). Width does not, and the disagreement is series-specific:

| series | `sigma_iqr / implied_std` (p50) |
|---|---|
| CPIYOY | **0.849** |
| FED | 0.923 |
| CPI | 0.951 |
| U3 | 1.000 |
| CPICORE | 1.078 |
| ADP | 1.171 |

**This feeds back into `arbitrage_findings.md` §1.** That study found the YoY
ladder implying 21% more width than the MoM ladder and, after the rounding
reconciliation, concluded there was no width arbitrage. The ratio table
sharpens it: CPIYOY's integrated width is inflated 1/0.849 ≈ 18% relative to its
own IQR, against CPI's 1/0.951 ≈ 5%. So roughly **12 of the 21 points were
reconstruction**, on top of the rounding explanation. Two independent reasons,
same conclusion.

---

## 3. The response vector: location moves, width and shape do not

`response_vector.py`. A resolves; which moments of B's belief distribution move?
Sign restriction imposed as in the lead-lag study (`HAWKISH[A]·HAWKISH[B]`, zero
fitted parameters), same-release pairs excluded, clustered on B's event.

**Location — yes.** 386 straddling steps over 129 target events:

- mean `d_q50` when signal > 0: **+0.0231** IQR
- mean `d_q50` when signal < 0: **−0.0197** IQR
- aligned difference **+0.0428 IQR**, 95% CI **[+0.0121, +0.0741]**, P(≤0) = 0.003
- `corr(signal, d_q50) = +0.201`

So a hawkish-aligned surprise in A moves B's median about **4% of B's own
interquartile range**, in the predicted direction. This is the lead-lag result
reproduced on a reconstruction-free statistic and on a *belief* rather than a
tradable price — which matters, because it is an information claim, not an
economic one.

**Width — no.** Matched on elapsed days and B's own horizon, the change in
`log IQR` across a foreign resolution is −0.0024 / −0.0070 / −0.0009 with every
CI spanning zero. Same verdict as `vol_term.py` §3b on the integrated moment,
now on a statistic immune to the coverage defect, so the null is not an artifact
of `implied_std`.

**Asymmetry — no.** Aligned `d_skew` +0.0044, CI [−0.0204, +0.0309].

### Why the width null is the interesting half

A genuine information channel should shrink B's conditional variance:
conditioning on A reduces uncertainty about B. It does not, on either estimator.
Two readings, and they are not distinguishable here:

1. the channel is real but **weak** — a 0.04-IQR mean shift implies a
   second-order variance reduction far below what 129 events can detect;
2. the market treats A's print as a **level** signal about B rather than as
   information that resolves uncertainty about B.

This is identification route 2 from the strategy discussion, half-run. It came
back null, and that is a tension inside the project's own results that belongs
in the thesis rather than smoothed over.

---

## 4. A methods error this found and fixed

Three scripts (`vol_term.py` §3b, and this study's §1 and §2) compared two
groups by bootstrapping `concat([a, −b])`. That statistic is
`(Σa − Σb)/(n_a + n_b)`, which does **not** estimate `mean(a) − mean(b)` once
the groups differ in size — and here they differ badly (101 vs 616). The
reported point estimates were correct; the CIs around them were not.

Both are now on a proper clustered difference bootstrap (`cluster_boot_diff`,
resampling whole events and taking `mean(a) − mean(b)` within the resample).
Conclusions are unchanged in every case — the location CI moves from
[+0.0137, +0.0736] to [+0.0121, +0.0741], the width CIs still span zero — but
the earlier intervals should not be quoted. `signal_model.py` in the lead-lag
study already used the correct estimator and is unaffected.

---

## 5. What follows

- **`relations_findings.md` item 1 needs rewriting**, not annotating. Its
  headline ("the ladder is not calibrated, macro ladders biased low") is
  withdrawn; what survives is the *coverage defect* and the WTI
  under-dispersion signature, which was always the half that reconciled with
  the price test.
- **Anything resting on `implied_std` should be re-checked** against
  `sigma_iqr`. The ratio table says the damage is series-specific and reaches
  15% — which is the same order as several published effects in this project.
- **The response vector is the on-thesis object** and it now has a clean
  headline: *A's resolution shifts B's location by ~4% of B's IQR and leaves
  B's width unchanged.* That is a characterisation of the update, which is what
  §3.2's gap statement actually asks for.
- **Quotes remain the blocker** for everything about width and tails, and the
  orderbook archive is sports-only from late 2025, so it cannot be done
  retroactively.
