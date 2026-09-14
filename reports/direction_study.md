# Stage 2: does the estimated structure predict *direction*?

*The successor to `agcrn_study.md` — same graph, simpler learners, and the
target the post-mortem prescribed. Reproduce with
`venv/bin/python scripts/run_direction_study.py`; the generated tables are
`artifacts/direction_report.md`. In-sample only; 2026 is never read.*

> **Rewritten 2026-09-14 after the §14 corrections. The answer changed to "no".**
> This document previously reported that the estimated structure predicts
> direction out of fold (63.0%, p = 0.041) and that the edge rather than the
> surprise carried it. Neither survives. Three defects were doing the work: the
> response instrument was chosen using trades from after the decision
> (`research_log.md` §14.5), the p-values came from a normal approximation the
> data's 98% tie rate does not license (§14.4), and every re-run between
> 2026-09-08 and 2026-09-13 silently read a cached pair panel that never
> rebuilt (§14.7). The measurement design below is unchanged and still the right
> design — only the result moved.

## The question

`agcrn_postmortem.md` concluded that the AGCRN failure was a **task/horizon
mismatch, not an architecture failure**, and named the replacement task
precisely: predict at the **dormant horizon** (trigger resolution → the
target's 3rd subsequent trade, median ~0.5 h) as a **direction classifier**,
because every magnitude-weighted test in the in-sample record is null while
sign/rank statistics survive (`research_log.md` §1–2).

That prescription and `research_summary.md` §9 Phase 2 item 5 (ablation ladder
rungs 1–2: "single-pair regression on Stage-1 structure", then "multi-trigger
regression aggregating over active Stage-1 neighbours") are the same
experiment. This is it.

It also closes a gap in the Stage-1 claim. `artifacts/adjacency_report.md`
reports 77.3% aligned sign agreement **among BH survivors** — but those
survivors were selected on the same rows, so that number is a coherence check,
not a predictive claim. Here every edge is re-estimated inside each fold on
training rows only, so the structure a prediction consumes is structure that
fold could have known.

## Setup

- **Unit.** One row per (trigger event → next target event) match, over the
  same grid Stage-1 searched (141 pairs): **3,127 labelled rows**, 14 triggers,
  22 targets, 2022-01 → 2025-12 (ZIRP burn-in excluded). 2,347 are scored out of
  fold. The row count is roughly half what this document previously reported,
  because a usable row now needs a pre-trigger print on the leg actually used
  (`research_log.md` §14.5) and the surprise panel is quality-gated (§14.2).
- **Label.** `sign(response)` — the target's dormant-window move; rows with a
  flat 3rd print are dropped. 50.8% up, so the pooled task is near-balanced by
  construction. **Caveat established below:** the label is *not* independent of
  the target's price level, which is what makes the `p0c` result in Result
  point 2 a design problem rather than a finding.
- **Feature.** `z_surprise = surprise / implied_std`: the market's own forecast
  error in units of its own implied sd, all of it observable at the trigger's
  resolution.
- **Folds.** 8 expanding walk-forward folds on trigger-resolution time with the
  standard 21-day purge (`stg_infra/stg/splits.py`). Every row is predicted
  exactly once, by a model fitted only on rows resolving ≥ 21 days earlier.
- **Structure.** Per fold, per pair: Spearman(surprise, response) on training
  rows, plus BH-FDR *within that fold*. Three **coverage gates** — `all`, `p05`
  (nominally significant train edge), `bh` (survives BH in that fold).
- **Ladder.** `base_rate` → `sign_rule` (zero fitted parameters beyond each
  edge's sign, plus one calibration constant) → `edge_logit` (one fitted weight
  on ρ̂·z) → `no_structure` (**the ablation** — raw surprise and context, edge
  weight removed) → `feature_logit` → `neighbour_logit` (adds the aggregate
  signal from co-active in-neighbours, strictly earlier-resolving only).
- **Significance.** Block permutation of labels **within trigger series**,
  which preserves the CPI/CPIYOY same-print dependence that makes a naive
  per-row null ~10× too narrow (`research_log.md` §3).

The pre-specified primary cell is **`sign_rule` × `bh`** — the rung and the
gate the post-mortem named before the run. Everything else in the table is
exploratory and is read as such. `p05` is reported as the headline gate because
it is the better powered of the two structure gates; both are shown.

## Result

Pooled over all 2,347 scored rows, **nothing predicts.** The sign rule scores
50.9% (p = 0.39). The cells that do clear significance pooled —
`no_structure`, `feature_logit`, `neighbour_logit`, all at p < 0.001 — share one
feature and it is not structure; see "What the winning rung was actually
reading" below.

On the rows the structure actually claims:

| rung | subset | n | acc | perm null | majority | bal. acc | AUC | perm p |
|---|---|---|---|---|---|---|---|---|
| base_rate | p05 | 92 | 0.554 | 0.554 | 0.554 | 0.500 | 0.465 | 1.000 |
| **sign_rule** | **p05** | **92** | **0.587** | **0.541** | 0.554 | 0.572 | 0.466 | **0.232** |
| edge_logit | p05 | 92 | 0.587 | 0.541 | 0.554 | 0.572 | 0.668 | 0.230 |
| *no_structure* | *p05* | *92* | *0.685* | *0.565* | *0.554* | *0.665* | *0.641* | *0.007* |
| feature_logit | p05 | 92 | 0.609 | 0.562 | 0.554 | 0.587 | 0.707 | 0.199 |
| neighbour_logit | p05 | 92 | 0.620 | 0.563 | 0.554 | 0.595 | 0.709 | 0.147 |
| sign_rule | bh | 45 | 0.689 | 0.564 | 0.644 | 0.647 | 0.499 | 0.059 |
| sign_rule | all | 2347 | 0.509 | 0.506 | 0.508 | 0.505 | 0.492 | 0.388 |

`p05` is the headline gate: better powered than `bh`, which selects harder edges
but leaves half the rows. Full table in `artifacts/direction_report.md`.

**Reading this table.** `perm null` is what these same predictions score against
labels shuffled within trigger series; that, not `majority`, is the yardstick
for `acc`. The two diverge because always-guess-the-majority-class is a
*different strategy*: on a 64%-up subset it scores 0.644 while carrying no
directional information at all (balanced accuracy 0.500, AUC 0.370).

Three things follow, and they are not what this document said before.

**1. The estimated structure does not carry demonstrated out-of-fold directional
information.** The sign rule is +4.6 pp over its own null at `p05` (58.7% vs
54.1%) and does not clear significance: **p = 0.232**. At `bh` it is +12.5 pp
but on 45 rows, **p = 0.059**. Pooled it is nothing (p = 0.39). The point
estimates are close to what was reported before; the samples are half the size,
because a usable row now requires a pre-trigger print on the leg actually used
rather than on the leg hindsight would pick (`research_log.md` §14.5). The
earlier significance was not robust to that.

**2. What the winning rung was actually reading — the price level, not the
surprise.** `no_structure` (surprise + context, edge weight removed) now wins at
every gate, which read naively would invert the previous claim. It does not.
Ablating its features on the pooled cell, the only well-powered one (n = 2,061):

| rung | acc | perm p |
|---|---|---|
| `p0c` only | 53.5% | **0.001** |
| `p0c` + `dtc` | 54.0% | **0.000** |
| `no_structure` (full) | 54.4% | **0.000** |
| `no_structure` **minus `p0c`** | 49.2% | 0.77 |
| `z` only | 49.4% | 0.72 |
| `abs_z` only | 49.5% | 0.76 |

`p0c` is `(p0 - 50) / 50`, the target's **pre-trigger price level**. Kalshi
contracts are bounded in [0, 100], so from a low `p0` the next print is
mechanically more likely to be up, and a logit handed that feature will find it.
It is bounded-support arithmetic, not propagation — and it carries *all* of
`no_structure`'s performance. Remove it and the rung is at chance. The surprise
alone (`z`, `abs_z`) is at chance throughout.

**3. So the result is a null, and it is a null on both sides.** Neither the
estimated structure nor the surprise predicts the dormant-horizon direction on
the corrected panel. The one thing that does predict is a property of the target
contract's price, available without any macro data at all.

**This also invalidates the ladder's internal comparison.** `p0c` sits in
`CONTEXT`, so `feature_logit` and `neighbour_logit` contain it too. The
R4-minus-R3 contrast the ladder is built around — what the structure buys over
the surprise alone — is not identified while both sides carry a feature that
predicts the label mechanically. Re-running this design means either dropping
`p0c` from `CONTEXT` or re-specifying the label to be orthogonal to the price
level. That is a design decision, not a defect, and it is deliberately left open
rather than patched.

**What does survive.** Stage 1. Its four BH survivors rest on a signed rank
correlation between surprise and response with an exact permutation test, and
`p0c` plays no part in it — see `artifacts/adjacency_report.md` and
`research_log.md` §14.9. The structure appears to exist; the claim that it was
shown to *predict* does not.

## What weakens it

- **Coverage is tiny.** 92 of 2,347 scored rows (3.9%) at `p05`; 45 (1.9%) at
  `bh`. Whatever this is, it applies to a small corner of the panel — and after
  §14's corrections it is a smaller corner than before (221 / 81 previously).
- **Concentration.** At `p05`, five pairs supply 75 of 92 rows: WTI→CPICORE
  (19 rows, 47.4% acc), CPI→FED (17, 64.7%), CPI→CPICORE (15, 46.7%),
  PAYROLLS→FED (13, 76.9%), CPICORE→FED (11, 72.7%). The largest single
  contributor scores *below* chance, and two of the five are same-release.
- **Two of the five carriers are mechanical.** CPI→CPICORE and CPI→CPIYOY are
  *same-release* pairs — one print resolves both — so they belong to the
  identification problem in `research_summary.md` §5 rather than to diffusion.
  Note that `CPICORE→CPI`, previously the single strongest carrier at 84.6% on
  13 rows, now contributes **one row**: the look-ahead in instrument choice was
  what made it look strong (`research_log.md` §14.5).
- **The pooled significant cells are an artifact.** See point 2 of the Result:
  every rung that clears significance contains `p0c`, and `p0c` alone reproduces
  the whole effect. This is the most serious weakness in the design, not a
  caveat on the margin.
- **Multiplicity.** The main table is 21 rung × subset cells. One was
  pre-specified; the rest are a family, not seven independent tests. With the
  primary cell at p = 0.059 and the headline cell at p = 0.232, no correction is
  needed to conclude nothing is established.
- **Skewed base rates.** Covered rows concentrate on a few pairs and periods, so
  the majority class runs at 55.4% (`p05`) and 64.4% (`bh`). Read `acc` against
  `perm null`, not against `majority` — see the note under the results table.
- **Still in-sample.** Walk-forward *inside* the in-sample block is not the
  holdout. This remains an IS-generated hypothesis — and on the corrected panel
  it is one that failed in sample, so there is nothing here to take to the
  holdout yet.

## What, if anything, strengthens it

Two robustness cuts move the sign rule in the right direction, neither far
enough to rescue the claim.

**Excluding same-release pairs** — the cut that removes the mechanical channel —
*helps* rather than hurts, which is the opposite of the pre-correction finding:

| rung | subset | n | acc | perm null | bal. acc | perm p |
|---|---|---|---|---|---|---|
| sign_rule | p05 | 65 | 0.600 | 0.554 | 0.585 | 0.290 |
| sign_rule | bh | 37 | 0.730 | 0.590 | 0.686 | **0.042** |
| *no_structure* | *bh* | *37* | *0.784* | *0.582* | *0.763* | *0.008* |

The `bh` cell clears 0.05 on 37 rows. But `no_structure` still beats it there,
and per Result point 2 that comparison is contaminated by `p0c` — so this is not
evidence for structure, it is one small cell in a family of 21 where the
structure-free rung remains ahead.

**Excluding WTI as a trigger** (the open Universe-A question in
`data_prep_plan.md` §5) also helps and also falls short: sign_rule 60.3% at
`p05`, p = 0.116; 73.5% at `bh` on 34 rows, p = 0.088. WTI is *mass without
structure* — it supplies most of the panel's rows and no surviving edge — so the
recommendation stands unchanged: keep it as a **target**, report it separately
as a **trigger**, never let it dominate a pooled estimate.

Taken together: the sign rule's point estimate is consistently above its null
across cuts (58.7%, 60.0%, 60.3%, 68.9%, 73.0%, 73.5%), and never convincingly
so once power is accounted for. That pattern is what a weak real effect looks
like, and also what noise looks like at n ≈ 40–90. This design cannot tell them
apart, which is the honest statement of where it lands.

## Where this leaves the thesis

- **Stage 1 remains the primary deliverable, and it is intact.** Four BH
  survivors on an exact permutation test, all non-same-release, all
  macro → policy path, with theory-consistent signs. Nothing in this document's
  reversal touches it: it never used `p0c`, and the look-ahead that inflated
  Stage 2 also inflated the *mechanical* Stage-1 edge, whose removal made that
  table cleaner rather than weaker (`research_log.md` §14.5, §14.9).
- **Stage 2 no longer supplies a predictive validation.** The previous claim —
  structure predicts direction out of fold, and the edge rather than the surprise
  carries it — does not survive the corrections. What the ladder now shows is a
  null on both sides plus one mechanical regularity in the target's price level.
- **This changes the shape of the contribution, not its existence.** The thesis
  claim becomes "cross-market belief updates in event prediction markets are
  *structured*, estimated directly and FDR-controlled" — which is what
  `research_summary.md` §3.2 actually set out to establish — without the
  additional claim that the structure is *predictively exploitable*. The
  tradability work (`edge_economics.md`) already argued the second claim was
  economically empty; it now turns out to be statistically unestablished too,
  which is a more coherent overall position than the two in tension.
- The identification problem is *less* binding than before: no surviving Stage-1
  edge is same-release, and removing same-release pairs improves rather than
  degrades Stage 2.

## Next

1. **Do not spend the holdout.** There is no longer an in-sample result worth
   confirming out of sample. The pre-registered cell should be re-specified
   against Stage 1's edge table (does the estimated adjacency replicate?), not
   against a direction rule that failed in sample.
2. **Fix the ladder's design before re-running it.** Drop `p0c` from `CONTEXT`
   or re-specify the label orthogonal to the price level; until then the
   R4-minus-R3 ablation is not identified. This is the single highest-value
   change to the Stage-2 design.
3. **Raise coverage.** The binding constraint is 45–92 rows, not model class.
   Phase B canonicalisation adds triggers, and channel pooling
   (`research_log.md` §11) recovers ~5.6x of the coverage that per-pair BH
   selection discards — that remedy is unaffected by these corrections.
3. **Sports/crypto negative control** through this exact pipeline — the ladder
   should find nothing where no channel exists. Cheap now that the harness
   exists (`stg_infra/stg/direction/`, `scripts/run_direction_study.py`).
4. **Spread-aware evaluation** (`research_summary.md` §9 Phase 2 item 7): 63%
   directional accuracy on 81 events is a statistical result, not an economic
   one, until execution costs are modelled.
