# Relations study, items 1–4: what the exploration found

Four independent explorations of the ranked list in `relations_study_plan.md`
§4, run 2026-09-14 on the corrected panel. Scripts and captured output:
`analysis/relations_2026_09/`. Everything is in-sample only.

Each item was run standalone, so no conclusion here depends on another being
adopted. Item 4 was run twice — on `surprise` and on `s_pit` — so that it does
not inherit item 1's fate either way.

---

## Summary

| # | item | verdict |
|---|---|---|
| 1b | Stage-1 measure swap | **Caution.** No edge disappears (all four stay nominally significant, ρ falls 0.04–0.19), but BH survivors go 4 → 1 → 1 → 0. Partly a resolution artefact: the permutation floor (0.0005) sits just under BH's rank-1 bar (0.00071). `CPI→FED` is the robust edge. |
| 1 | PIT surprise + calibration | **Finding.** The ladder is *not* calibrated. 9 of 14 series land above their own implied centre significantly often; the CPI family does so 71–83% of the time. This **overturns `edge_economics.md` §5B**, whose table predates §14.1. |
| 2 | Surprisal / \|s_pit\| cross-validation | **Partially reopens the magnitude decision.** In §2's own headline cell, \|s_pit\| cross-validates at r = 0.588 where \|surprise\| gave 0.332 (0.242 as published). Across all five sibling pairs the advantage shrinks to a rank-correlation edge (0.46 vs 0.37). |
| 3 | CPI-family collapse, PAYROLLS−U3 | **Split.** The CPI collapse **does not pay** — it buys rows and loses ρ. The labour index **does**, and it rehabilitates U3 — but the rehabilitation is confounded with the sample period. |
| 4 | Channel pooling as a block model | **Strong, and the best result of the four.** data→policy: 598 rows over 53 target-event clusters, 63.2% aligned sign agreement vs a 50.0% ± 3.4% clustered null, **p = 0.0004**, zero fitted parameters. Improves on §11's 456 rows / 55.0% / p = 0.021. |

The two things worth acting on are **item 4** (which is now the strongest
in-sample structural result in the project) and **item 1's calibration finding**
(which is a standalone contribution and also a correction).

---

## Item 1 — the ladder is not calibrated

Under a calibrated market the realised value's position in the market's own
implied distribution, `u = F_implied(resolved)`, is uniform. It is not.

Per-series, gated panel, exact KS against U(0,1):

| series | n | mean u | tail20 | KS D | KS p |
|---|---|---|---|---|---|
| WTI | 198 | 0.462 | 0.035 | 0.217 | <0.001 |
| CPI | 41 | 0.722 | 0.317 | 0.387 | <0.001 |
| U3 | 38 | 0.648 | 0.263 | 0.247 | 0.016 |
| PAYROLLS | 32 | 0.552 | 0.250 | 0.143 | 0.490 |
| CPICORE | 31 | 0.716 | 0.323 | 0.438 | <0.001 |
| CPIYOY | 21 | 0.668 | 0.286 | 0.365 | 0.005 |
| CPICOREYOY | 20 | 0.745 | 0.250 | 0.412 | 0.001 |
| PCECORE | 15 | 0.791 | 0.467 | 0.503 | <0.001 |
| FED | 12 | 0.906 | 0.750 | 0.710 | <0.001 |

`tail20` is the share landing in the outer decile pair; 0.20 under calibration.
Two distinct failures, in opposite directions:

- **The macro-release ladders are biased low — and this is a location error, not
  an overconfidence one.** CPI-family, PCECORE, U3 and FED all have mean `u` ≈
  0.65–0.91. Splitting the tails shows the miss is one-sided: the *lower* decile
  holds 0–5% of outcomes against 10% expected, while the upper decile holds
  25–75%. A genuinely too-narrow distribution would overpopulate **both** tails,
  so the implied distributions are roughly the right width and simply sit too
  low. That also rules out a discretisation or open-tail artifact from
  `recover_pdf`, which would be symmetric and cannot empty one tail while
  filling the other. Prints came in above the market's central forecast far more
  often than not, over a sample dominated by the 2022-24 inflation surge — which
  is a known stylised fact about forecasters in that period, so recovering it is
  evidence the measure works rather than a surprise.
- **The asset-price ladders are underconfident.** WTI's distribution is too
  *wide*: 3.5% of outcomes in the tails against 20%, 83% in the central half
  against 50%. Consistent with `edge_economics.md` §2(b) — WTI's resolution is a
  price snapshot near the prevailing price, not an information event, so it
  lands mid-distribution by construction.

### This overturns §5B

`edge_economics.md` §5B concluded "**the ladder-recovered implied mean is a
well-calibrated forecast**", on max |t| = 1.66 across 17 series. On the current
panel that is false. Same test, same statistic:

| series | n | mean z | t | frac above centre | sign-test p | BH q=0.1 |
|---|---|---|---|---|---|---|
| CPI | 41 | +0.517 | **+2.83** | 0.780 | <0.001 | yes |
| PCECORE | 15 | +0.798 | **+3.42** | 0.933 | 0.001 | yes |
| CPICOREYOY | 20 | +0.561 | **+2.72** | 0.850 | 0.003 | yes |
| CPICORE | 31 | +0.387 | +1.95 | 0.774 | 0.003 | yes |
| FED | 12 | +0.864 | **+4.89** | 0.917 | 0.006 | yes |
| U3 | 38 | +0.436 | +1.99 | 0.711 | 0.014 | yes |

Nine of fourteen series survive BH on the sign test; five on the t-test. §5B's
table reports CPI at n=48, mean surprise −0.005, mean z −0.09 — the current
panel gives n=41, +0.057, +0.517. **The sign is flipped and the magnitude is
10×**, so this is not a power difference: §5B's table predates §14.1, which
replaced ladder-inferred resolved values with the true printed
`expiration_value` for 442 of 519 rows. Inferring the outcome from the ladder
biases it toward the ladder's centre, which is exactly what would manufacture an
unbiasedness result.

**What it does not touch.** No edge result moves. Stage 1 is a Spearman computed
*within* a pair, and a location shift common to a trigger series does not reorder
that series' events — so the 4 edges, the channel-pooling result and the
direction study all stand unchanged. The finding sits beside them rather than
underneath them.

**What it gives back.** The PIT is also a *single-series* diagnostic, needing no
edge, no target and no horizon: WTI's signature is the exact inverse of CPI's
(3.5% of outcomes in the tails against 20%, 83% in the central half against
50%), which is what a resolution that is a price snapshot near the prevailing
price looks like. That is an independent test for the trigger-inclusion
criterion, and it bears on the open "WTI in Universe A?" decision without
reference to any result.

**Consequence.** §5B is cited as closing the "hold on an ex-ante bias" path
(`edge_economics.md` §5B, path B). That path is *not* closed on this evidence —
though note the bias is a level effect over a specific macro regime, so it is
much weaker as a trading claim than as a measurement one, and a pre-registered
OOS test would be the only honest way to make it.

`analysis/relations_2026_09/out/pit_calibration.txt`.

---

## Item 1b — the 4 Stage-1 edges are measure-dependent, and the BH gate is too coarse

The full sweep, re-run four times over the same 141-pair grid changing only the
number fed in as the trigger's surprise:

| measure | nominal p<0.05 | **BH survivors** |
|---|---|---|
| `surprise` (published) | 14 | **4** |
| `s_pit` | 17 | **1** |
| `z_surprise` | 14 | **1** |
| `surprise_median` | 14 | **0** |

The baseline reproduces `adjacency_report.md` exactly, so the machinery is
confirmed. But read the survivor counts against what happened to the edges
themselves:

| edge | surprise | s_pit | z_surprise | median |
|---|---|---|---|---|
| CPI→FED/any | +0.617 | +0.580 | +0.570 | +0.567 |
| PAYROLLS→FED/any | +0.583 | +0.544 | +0.554 | +0.501 |
| PAYROLLS→FEDDECISION/hike | +0.596 | +0.483 | +0.484 | +0.449 |
| CPICOREYOY→FEDDECISION/cut | −0.683 | −0.497 | −0.685 | −0.670 |

**No edge disappears.** Every one keeps its sign, keeps a substantial ρ, and
stays *nominally* significant under all four measures (worst case p = 0.0475).
The largest fall is PAYROLLS→hike, 0.596 → 0.483. So "4 survivors → 1" is not
three edges evaporating; it is three edges drifting across a threshold.

### The threshold is the problem

BH at q = 0.10 over 141 tests requires the smallest p to clear
**0.10/141 = 0.00071**. The permutation p is computed at `n_perm = 2000`, whose
floor is **1/2001 = 0.0005**. There is no achievable p-value between the two, so
an edge can occupy BH rank 1 *only by landing exactly on the permutation floor*.

Under `surprise` the four p's are 0.0005, 0.0005, 0.0010, 0.0025 against a BH
ladder of 0.00071, 0.00142, 0.00213, 0.00284 — all four clear. Under
`surprise_median` the smallest p is 0.0010, one grid step above the floor, and
**nothing survives** — even though CPICOREYOY sits at p = 0.0020 with ρ = −0.670,
a stronger showing than the `s_pit` version that did survive in its own sweep.

That is a resolution artefact, not a measurement result.

**Two consequences.**

1. **The published "4 BH survivors" is more fragile than it reads**, and the
   fragility is partly mechanical. `n_perm` should be raised to ~20,000 (floor
   0.00005, comfortably clear of the rank-1 bar) and the sweep re-run before
   anything is concluded from survivor counts. This matters directly: the
   pre-registered OOS cell is `sign_rule` × **`bh`**, so which pairs populate
   that gate currently depends on a permutation grid nobody chose deliberately.
2. **`CPI→FED` is the robust edge** — the only one surviving under three of four
   measures, and the only one whose ρ barely moves (0.617 → 0.567). The other
   three carry a real scale dependence worth reporting as such.

### And it strengthens item 4

Note the contrast with channel pooling, which was **measure-independent**:
63.2% on `surprise`, 62.8% on `s_pit`, p ≤ 0.0004 either way. Per-pair FDR
selection is sensitive to a scaling choice that has never been justified; the
pooled channel test is not. That is an argument for pooling on grounds
independent of power.

`analysis/relations_2026_09/out/stage1_variants.txt`.

---

## Item 2 — magnitude is partly recoverable

`research_log.md` §2 rests the whole "direction, not size" design on one cell:
CPI vs CPIYOY, same BLS print, n = 37, signed surprise r = 0.686 vs |surprise|
r = 0.242. Re-run on the corrected panel with mean-independent magnitudes:

| measure | Pearson r | Spearman ρ |
|---|---|---|
| surprise (signed) | 0.654 | 0.718 |
| z_surprise (signed) | 0.646 | 0.705 |
| s_pit (signed) | 0.597 | 0.672 |
| **\|s_pit\|** | **0.588** | **0.568** |
| implied_entropy | 0.484 | 0.502 |
| surprisal | 0.450 | 0.484 |
| **\|surprise\|** | **0.332** | **0.383** |
| \|z_surprise\| | 0.367 | 0.288 |

Two things move. First, |surprise| itself is 0.332 here, not the published
0.242 — part of §2's headline was the pre-§14.1 resolved values. Second,
|s_pit| reaches 0.588, close to the signed measure's 0.654.

Across all five CPI-family sibling pairs the effect is smaller: mean Spearman
0.464 for |s_pit| vs 0.368 for |surprise| (|s_pit| wins the rank comparison in
4 of 5 pairs; on Pearson the two are level, because Pearson on |surprise| is
inflated by a handful of large events — §2's own critique). `surprisal` is the
weaker of the two new measures, at 0.431, and its discreteness shows.

**Verdict.** "Magnitude does not cross-validate" is too strong: on a
distribution-relative measure, sibling ladders agree on size at ρ ≈ 0.46–0.61.
"Direction cross-validates better than size" survives (0.62–0.68 vs 0.46–0.57).
The design decisions that follow from the *weak* form — rank statistics, a
direction label — are safe. The decision that followed from the strong form —
abandoning a magnitude target outright, including AGCRN's — was made on a bad
estimator and would be worth one re-test on `|s_pit|`.

`analysis/relations_2026_09/out/magnitude_xval.txt`.

---

## Item 3 — collapse the labour release, not the CPI family

### (a) The CPI collapse does not pay

§3.1(a) proposed collapsing all four CPI ladders to one factor. Two reasons not
to, both measured:

**The four are not one factor on a unit-free measure.** §3.1's r ≈ 0.88–0.89 is
reproduced exactly on `z` (0.880 / 0.867 / 0.893 / 0.697 / 0.694 — a clean
validation of the pipeline). On `s_pit` the picture splits by *tenor*:

| pair | r on z | r on s_pit |
|---|---|---|
| CPI ↔ CPICORE | 0.880 | **0.930** |
| CPI ↔ CPICOREYOY | 0.867 | 0.899 |
| CPICORE ↔ CPICOREYOY | 0.893 | 0.822 |
| CPI ↔ CPIYOY | 0.697 | **0.478** |
| CPICORE ↔ CPIYOY | 0.694 | **0.333** |
| CPIYOY ↔ CPICOREYOY | 0.708 | **0.181** |

Headline and core are near-duplicates *within* a tenor; month-over-month and
year-over-year are not. Collapsing all four merges two distinct signals.

**And the collapse loses more ρ than it gains rows.** Against FED/any:

| trigger | n | ρ | p (perm) |
|---|---|---|---|
| CPI alone | 36 | **0.617** | 0.0005 |
| CPI+CPICORE (MoM collapse) | 45 | 0.519 | 0.0005 |
| all four (full collapse) | 48 | 0.493 | 0.0010 |
| CPIYOY+CPICOREYOY (YoY collapse) | 32 | 0.257 | 0.150 |

25–33% more rows, a materially lower ρ, no better p. The multiplicity argument
for collapsing (~21% fewer pairs in the grid) still stands on its own, but
"collapsing buys power" does not.

The subcomponents were checked before folding in, as §3.1 asked: CPIGAS,
CPIUSEDCAR and CPISHELTER correlate with the family at |r| ≤ 0.33 on both
measures. Correctly excluded.

### (b) The labour index works, with a caveat

PAYROLLS ↔ U3 are near-orthogonal as §3.1(b) reported (r = −0.180 on z, −0.135
on s_pit, sign agreement 50–54%). The zero-parameter index
`z_labour = z_PAYROLLS − z_U3`, on the releases where both printed:

| trigger | FED/any ρ | FEDDECISION/hike ρ |
|---|---|---|
| **LABOUR_Z (index)** | **+0.610** (n=22, p=0.0035) | **+0.512** (n=19, p=0.021) |
| PAYROLLS, same releases | +0.507 (n=24, p=0.015) | +0.427 (n=20, p=0.049) |
| U3, same releases | −0.591 (n=22, p=0.0055) | −0.419 (n=19, p=0.071) |

The index beats PAYROLLS on identical rows in both cells, and edges out U3.
More striking is what happens to U3 itself, the project's weakest falsification
cell:

| U3 → FED/any | n | ρ | p (perm) |
|---|---|---|---|
| all U3 events | 35 | −0.259 | 0.138 |
| joint Employment-Situation releases | 22 | **−0.590** | **0.0055** |
| U3 events with no payrolls partner | 13 | +0.319 | 0.282 |

Correctly signed (higher unemployment is dovish) and significant, where the
pooled figure was a null.

**The caveat, and it is a real one.** The 14 U3 events with no payrolls partner
are almost all from 2022 and early 2023 — the joint subset starts 2023-04-07,
the non-joint one 2022-02-04. So "U3 read alongside payrolls" and "U3 after
2023-04" are nearly the same subsample, and this data cannot separate them.
The result is worth having, but it is not yet evidence for the release-vector
specification specifically.

`analysis/relations_2026_09/out/family_collapse.txt`.

---

## Item 4 — channel pooling, and it is the strongest result

Node types (inflation, labour, growth, energy, policy), one theory-fixed sign
per channel, hawkish +1 and dovish (U3, JOBLESSCLAIMS) −1, `FEDDECISION/cut`
carrying the opposite target sign as the falsification cell. Same-release pairs
excluded. Nothing is fitted.

Both nulls are reported: the block permutation shuffles within a trigger series
(handling CPI/CPICORE's r ≈ 0.9), and the **clustered** null flips the sign of
all rows sharing a `target_event` (handling the fact that one FOMC meeting is
the target of every trigger in the grid). §13.5 killed two headline results that
were not clustered, so the clustered column is the one to cite.

On `surprise`:

| channel | pairs | rows | clusters | sign agree | clustered null | **p (clustered)** | pooled ρ |
|---|---|---|---|---|---|---|---|
| inflation→policy | 21 | 369 | 53 | 0.627 | 0.500 ± 0.046 | **0.0038** | +0.344 |
| labour→policy | 9 | 204 | 47 | 0.661 | 0.501 ± 0.042 | **0.0004** | +0.256 |
| growth→policy | 3 | 25 | 12 | 0.417 | 0.499 ± 0.145 | 0.803 | +0.299 |
| energy→policy | 6 | 397 | 33 | 0.465 | 0.501 ± 0.039 | 0.833 | −0.128 |
| **ALL data→policy** | **33** | **598** | **53** | **0.632** | 0.500 ± 0.034 | **0.0004** | **+0.307** |

On `s_pit` the same table reads 0.629 / 0.629 / 0.628 for the three live rows,
p = 0.0044 / 0.0010 / 0.0002 — i.e. **the result does not depend on which
surprise measure is used**, which is itself the robustness line item 1 was
supposed to buy.

Three things to note:

1. **It improves substantially on §11**, which measured 456 rows, 55.0% and
   p = 0.021 on the pre-correction panel. Now 598 rows, 63.2%, p = 0.0004 — and
   clustered, which §11 was not.
2. **The effective sample is ~212, not 598.** Clustering inflates the null's
   standard deviation from a naive 0.0204 to 0.0343 (1.68x), which corresponds
   to an effective n of about 212 — the 598 rows sit across 53 target-event
   clusters, but rows within a cluster are correlated, not identical, so the
   effective figure lands between the two. Quote 598 rows / 53 clusters and the
   clustered p; do not quote a naive binomial interval on 598.
3. **Energy→policy is a clean falsification.** 397 rows — the largest channel —
   sitting *below* chance at 46.5%, pooled ρ = −0.128. The theory sign is wrong
   for WTI, which is what `edge_economics.md` §2(b) predicts for a series with no
   scheduled information event. A pooling scheme that scored everything positive
   would be suspect; this one does not.

`analysis/relations_2026_09/out/channel_pooling.txt`.

---

## What follows

- **Item 4 is ready to be promoted** out of exploration into a script + report
  pair, and it is the natural headline structural result: one channel, 53
  independent events, zero fitted parameters, a falsification cell that fires
  correctly. The OOS debt §11 records still stands — the channel definition was
  informed by Stage 1 — so the pre-registered OOS cell should be re-specified as
  the channel test *before* it is spent, per `relations_study_plan` §4.
- **Item 1's calibration result is a standalone contribution** and should be
  written up as one. It also requires a correction to `edge_economics.md` §5B
  and a re-check of anything else resting on it.
- **Item 3(a) removes a planned change**: do not collapse the CPI family for
  power. Collapse at most within a tenor, and only for the multiplicity saving.
- **Item 3(b) and item 2 both want one more test**, and it is the same test in
  both cases: a period split, to see whether these are regime effects.
