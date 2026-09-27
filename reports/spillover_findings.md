# More shocks for the macro graph: jumps, order flow, and releases Kalshi does not trade

*2026-09-27. Scripts: `analysis/spillover_2026_09/` (outputs in its `out/`,
git-ignored). In-sample (2021-10 → 2025-12); nothing from 2026 was read.*

## The question

The macro graph is starved of events: 154 Kalshi release instants, about 5
co-firing instants per edge. `recovery_test.md` puts the data a graph learner
needs at roughly 10× that. This study asks whether other shocks can supply it,
and whether they make the graph learnable. Three sources:
1. unscheduled price jumps;
2. bursts of signed order flow;
3. scheduled releases that have a consensus but no Kalshi market.

**Summary:**

| shock source | shocks | carries information? | helps learn the graph? |
|---|---|---|---|
| price jumps ≥ 5¢ away from releases | 2,198 | no (spillover ≈ 0; accuracy 0.50) | no |
| … jumps ≥ 15–20¢ | 97–181 | weakly, at +1 h in the theory direction | too few |
| order-flow bursts (top 1–5%) | 275–1,515 | little: the series' own price does not respond | no |
| **calendar releases with a consensus** | **1,036 instants (6.7×)** | **yes: significant theory-direction responses** | **at channel level only** |

With the calendar releases, a learned channel graph (15 free-sign
coefficients) recovers theory's structure (11 of 15 channels in the theory
direction, 4 significant, none against), and predicts as well as the theory
rule out of sample, but not better. At edge level (776 coefficients) learning
still fails.

## 1. Price jumps (`jumps.py`, `predictive.py`, `dose.py`)

**Jumps:** a threshold contract's 15-minute price moves ≥ 5 / 10 / 15 / 20¢,
with the following conditions:
- the move is not half-reversed within an hour;
- it is more than 24 h from any macro release or settlement;
- it is one jump per series per bar, merged within 1 h.

**Response:** the other series' lead contract, from the end of the jump bar to
+1 h … +72 h, theory-signed by sign(ΔA)·HAWKISH[A]·HAWKISH[B]. It is compared
with the same measurement at a random quiet time 3–10 days away (placebo).
All 17 series are both sources and targets.

| jump size | jumps | real − placebo at +24 h, ¢ [95% CI] | pairs surviving BH |
|---|---|---|---|
| ≥ 5¢ | 2,198 | +0.003 [−0.150, +0.163] | 0 of 89 |
| ≥ 10¢ | 425 | **+0.518 [+0.049, +0.958]** | 1 of 27 (**CPI → FED**, +1.26¢ [+0.44, +2.18], p = 0.002) |
| ≥ 15¢ | 181 | +0.485 [−0.306, +1.313] | – |
| ≥ 20¢ | 97 | +0.508 [−0.504, +1.510] | – |

- **The target rarely moves in the same bar** (co-jumps 1.0–1.9%). So jumps are
  almost never shared news; where spillover exists, it is lagged.
- **Direction learned from the data, walk-forward** (`predictive.py`): the
  direction of the target's move is learned per pair (or per channel) from
  earlier jumps and scored on later ones. It scores 0.46–0.53 balanced accuracy
  at every size and horizon, the same as at placebo times.
- **The theory direction's accuracy at +1 h rises with jump size:** 0.504 /
  0.517 / 0.582 [0.504, 0.662] / 0.650. At +4 h: 0.492 / 0.503 / 0.549 /
  0.552. That is suggestive, since large jumps are the ones likely to carry
  news, but only the 15¢ cell at +1 h clears 0.5, out of ~48 cells.
- **Weighting the jumps instead of thresholding them** (`dose.py`): the
  placebo-adjusted response does not grow with jump size (slope t between −0.4
  and +0.8 at every horizon), and the size bands are not monotone.

A tie-break bug made the target's lead contract non-deterministic when two
contracts tied on print count (`polars.unique()` keeps no order). It is fixed
in `_tape.py` by sorting the legs, and two runs now give identical output.
Figures quoted before the fix moved slightly (10¢ at +24 h was +0.44
[−0.01, +0.87]); no conclusion changed.

## 2. Signed order flow (`flow.py`)

**Shocks:** 1-hour bars in a series' top 5% (1,515) or top 1% (275) of
|net signed flow| (YES-taker minus NO-taker contracts across its threshold
legs), away from releases. Responses and placebo are as in §1.

| shocks | series' own price, +24 h | other series, +4 h | other series, +24 h | pairs surviving BH |
|---|---|---|---|---|
| top 5% | +0.24¢ [−0.21, +0.68] | +0.12 [−0.02, +0.27] | −0.02 [−0.19, +0.16] | 1 of 78 |
| top 1% | +0.38¢ [−0.84, +1.59] | **+0.41 [+0.10, +0.74]** | +0.27 [−0.10, +0.65] | 0 of 23 |

Most flow bursts are not information: the series' own price does not respond
to its own buying pressure, except +0.72¢ at +72 h for the top 5%. Without that
sanity check, the small spillover in the top 1% is hard to interpret.

## 3. Releases Kalshi does not trade (`releases.py`, `calendar_learn.py`)

**The triggers:** 34 US releases from the economic calendar with a consensus,
no Kalshi market, a clear economic direction, and a release time more than 1 h
from any Kalshi-traded release.
- Families: inflation (PPI, import prices, ISM prices, inflation
  expectations…), activity (retail sales, ISM and S&P PMIs, durable goods,
  industrial production, regional Fed surveys, housing, GDPNow…), labour
  (JOLTS, continuing claims) and sentiment.
- Surprise: (actual − consensus) / its own sd.
- The calendar is lum.id findata, cached at
  `data/external/econ_calendar_us_2021q4_2025.parquet` by
  `relations_2026_09/consensus_surprise.py`.
- Test: the theory-signed response of the Kalshi markets, against a null that
  flips each release's surprise sign at random.

That gives **1,036 release instants, against 154 for Kalshi releases.**

| | +1 h | +4 h | +24 h |
|---|---|---|---|
| all releases × all Kalshi targets | +0.19¢, p = 0.009 | **+0.31¢, p < 0.001** | **+0.30¢, p < 0.001** |
| activity → growth targets | +0.81¢, p < 0.001 | +1.56¢, p < 0.001 | +1.77¢, p < 0.001 |
| activity → policy | +0.39¢, p = 0.004 | +0.31¢, p = 0.010 | n.s. |
| labour → policy | +1.10¢, p = 0.003 | n.s. | n.s. |
| inflation → policy | +0.29¢, p = 0.07 | +0.41¢, p = 0.026 | +0.66¢, p = 0.021 |
| activity → inflation targets | n.s. | +0.44¢, p = 0.022 | +0.40¢, p = 0.016 |

- **By trigger:** 3 of 34 survive BH at +4 h (Atlanta Fed GDPNow, personal
  spending, building permits); most of the rest are positive but thin (30–80
  releases each).
- **The strongest cell, activity → growth, is partly mechanical:** GDPNow
  estimates the very GDP those contracts settle on.
- **Learned direction, walk-forward** (no theory): for the first time, a learned
  direction beats chance out of sample, and the placebo stays at chance:

  | horizon | learned per edge | learned per family | theory | flipped (control) |
  |---|---|---|---|---|
  | +1 h | 0.526 [0.496, 0.556] | **0.524 [0.503, 0.545]** | 0.546 | 0.490 / 0.491 |
  | +4 h | 0.513 [0.496, 0.531] | **0.517 [0.502, 0.530]** | 0.527 | 0.504 / 0.496 |
  | +24 h | **0.520 [0.507, 0.532]** | 0.500 | 0.523 | 0.501 / 0.506 |

- **The full-sample learned edge signs match theory on 57% of 166 edges**
  (balanced). The 4 edges that survive BH on their own all have the theory sign:
  personal spending → FED, industrial production → GDP, consumer confidence →
  FED, retail sales ex autos → GDP.

## 4. Do the extra releases make the graph learnable? (`augmented.py`, `channel.py`)

**Setup:**
- 154 Kalshi and 1,030 calendar instants, with 51 sources (17 Kalshi series, 34
  calendar releases) and 17 Kalshi targets. That gives 4,521 labelled cells: one
  (release instant, target) pair each, with the target's move to +4 h.
- Walk-forward, 8 folds. Every model is fitted on Kalshi-release cells only,
  and on Kalshi + calendar.
- Scored on the same test cells: 481 at Kalshi releases, 2,703 at calendar
  releases.

AUC, with the change from adding the calendar data and the difference from
the theory rule:

| model | params | Kalshi cells | Δ from +calendar | vs theory, Kalshi cells | calendar cells |
|---|---|---|---|---|---|
| theory rule | 0 | 0.557 | – | – | **0.528** |
| one slope | 1 | 0.558 | +0.005 [−0.010, +0.024] | +0.001 | 0.525 |
| channel, learned sign | 15 | 0.569 | −0.000 [−0.035, +0.034] | +0.012 [−0.044, +0.068] | 0.523 |
| source type × target, no edge signs | 68 | **0.580** | +0.034 [−0.014, +0.082] | +0.023 [−0.037, +0.082] | 0.495 |
| free edge graph | 776 | 0.517 | +0.017 [−0.020, +0.058] | −0.040 [−0.109, +0.029] | 0.498 |
| low-rank graph (rank 4) | – | 0.508 | −0.014 [−0.085, +0.058] | – | 0.506 |
| Bayes, soft sign (per edge) | 776 | 0.559 | +0.017 [−0.020, +0.056] | – | 0.518 |

The Bayesian fits did not all converge (max R̂ 2.96 in some fold), so read
that row loosely.

**The learned channel graph** (full sample, Kalshi + calendar, each channel's
sign free, t clustered by instant):

| channel | cells: Kalshi / calendar | coefficient | t |
|---|---|---|---|
| labour → policy | 59 / 166 | +0.32 | **+3.3** |
| inflation → policy | 60 / 269 | +0.12 | **+3.2** |
| growth → growth | 0 / 447 | +0.15 | **+3.5** |
| growth → policy | 11 / 629 | +0.06 | **+2.0** |
| the other 11 channels | | 7 positive, 4 negative | all \|t\| < 1.5 |

- **At channel granularity, learning recovers theory's structure.** 11 of 15
  channels come out in the theory direction, 4 are significant, and none is
  significantly against it. The recovered graph is a policy hub fed by labour,
  inflation and growth news.
  - With Kalshi releases alone, only labour → policy was supported
    (`graph_ablation.md` §8). Growth → policy rests almost entirely on the
    calendar releases.
- **Out of sample it ties theory, and doesn't beat it.** The channel-level
  learners are the best models on the Kalshi cells (0.569–0.580 vs 0.557), but
  their edge over the theory rule has a CI through zero, and theory still wins
  on the calendar cells.
- **Edge-level learning still fails, with or without the extra data.** The free
  edge graph, the low-rank graph and the per-edge Bayes model are all at or
  below the theory rule, and the calendar data improves none of them
  significantly.

**Caveats:**
- Within each channel the relative edge signs still come from theory (e.g. U3
  flipped relative to PAYROLLS); only the channel's direction is learned.
- The channel coefficients are in-sample.
- Responses are read from one lead contract per target.

## What this means for the thesis

1. **Kalshi macro markets price far more news than they trade.** Releases with
   no Kalshi market move Kalshi prices in the direction theory predicts. That is
   more evidence that information spreads across these markets, and it widens
   the structural claim beyond the ten traded releases.
2. **Not every extra shock is information.** Unscheduled jumps and order-flow
   bursts add events but little signal; scheduled releases with a consensus add
   both. More events help only when each event carries news.
3. **The graph can be learned at the level of detail the data supports.** With
   the calendar releases, 15 free channel coefficients recover a theory-
   consistent policy hub. 776 free edges still cannot be learned. So learned and
   theory graphs agree at channel level, which is the level these markets'
   data can resolve.
