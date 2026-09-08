# Stage 2: does the estimated structure predict *direction*?

*The successor to `agcrn_study.md` — same graph, simpler learners, and the
target the post-mortem prescribed. Reproduce with
`venv/bin/python scripts/run_direction_study.py`; the generated tables are
`artifacts/direction_report.md`. In-sample only; 2026 is never read.*

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
reports 75.9% aligned sign agreement **among BH survivors** — but those
survivors were selected on the same rows, so that number is a coherence check,
not a predictive claim. Here every edge is re-estimated inside each fold on
training rows only, so the structure a prediction consumes is structure that
fold could have known.

## Setup

- **Unit.** One row per (trigger event → next target event) match, over the
  same 140-pair grid Stage-1 searched: **4,816 labelled rows**, 14 triggers, 19
  targets, 2022-01 → 2025-12 (ZIRP burn-in excluded). 3,487 are scored out of
  fold.
- **Label.** `sign(response)` — the target's dormant-window move; rows with a
  flat 3rd print are dropped. 50.9% up, so the pooled task is near-balanced by
  construction.
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
exploratory and is read as such.

## Result

Pooled over all 3,487 scored rows, **nothing predicts.** The sign rule scores
50.6% (p = 0.35); the best pooled cell is `neighbour_logit` at 51.4%
(p = 0.030), a 1.4 pp edge that is one of 21 cells examined and does not
reappear on the covered subsets. This is the expected consequence of a sparse
graph: most rows have no edge, so most rows have no signal.

On the rows the structure actually claims:

| rung | subset | n | acc | perm null | majority | bal. acc | AUC | perm p |
|---|---|---|---|---|---|---|---|---|
| base_rate | bh | 81 | 0.593 | 0.593 | 0.593 | 0.500 | 0.403 | 1.000 |
| **sign_rule** | **bh** | **81** | **0.630** | **0.522** | 0.593 | **0.626** | 0.576 | **0.041** |
| edge_logit | bh | 81 | 0.617 | 0.522 | 0.593 | 0.611 | **0.680** | 0.062 |
| *no_structure* | *bh* | *81* | *0.494* | *0.515* | *0.593* | *0.473* | *0.467* | *0.746* |
| feature_logit | bh | 81 | 0.605 | 0.526 | 0.593 | 0.596 | 0.659 | 0.105 |
| neighbour_logit | bh | 81 | 0.593 | 0.529 | 0.593 | 0.580 | 0.649 | 0.171 |
| sign_rule | p05 | 221 | 0.575 | 0.501 | 0.570 | 0.573 | 0.561 | 0.022 |

**Reading this table.** `perm null` is what these same predictions score against
labels shuffled within trigger series; that, not `majority`, is the yardstick
for `acc`. The two diverge because always-guess-the-majority-class is a
*different strategy*: on a 59%-up subset it scores 0.593 while carrying no
directional information at all (balanced accuracy 0.500, AUC 0.403). So the
sign rule's margin is **+10.8 pp over its own null**, not the +3.7 pp that
comparing it to the majority rate suggests — and `no_structure`, at 0.494
against a null of 0.515, is *below* chance rather than merely unimpressive.

Three things follow.

**1. The estimated structure carries out-of-fold directional information.** The
sign rule beats its permutation null on the covered rows — 63.0% vs 52.2%
(p = 0.041, `bh`) and 57.5% vs 50.1% (p = 0.022, `p05`) — against a null that
respects the same-print dependence. This is
the first predictive — rather than descriptive — evidence for the Stage-1
adjacency, and it is not the 75.9% figure recycled: the edges were refit inside
each fold, on training rows only.

**2. What predicts is the edge, not the surprise.** The ablation is decisive:
strip the edge weight, keep the surprise and context, and accuracy on the same
81 rows falls to 49.4% (balanced 0.473, AUC 0.467, p = 0.75). The directional
content is in *which pair, with what sign* — exactly the object Stage-1
estimates, and not something a model recovers from the surprise alone.

**3. Capacity does not pay — again.** Ranked by accuracy on the covered rows:
`sign_rule` (0 fitted weights) ≥ `edge_logit` (2) ≥ `feature_logit` (7) ≥
`neighbour_logit` (8). This reproduces the AGCRN study's monotone
capacity-hurts pattern at the *bottom* of the ladder, and gives the ablation
promised in the CA report §5.3 an empirical stopping point: **zero
parameters**.

One honest qualification to (3): the logits *rank* better than they
*classify* — `edge_logit` has the best AUC (0.680) while scoring below the sign
rule at a 0.5 threshold. A fitted weight buys ordering and calibration, not
decisions. On 81 rows that gap is well inside noise; it is a reason to report
both metrics rather than to prefer either model.

## What weakens it

- **Coverage is tiny.** 81 of 3,487 scored rows (2.3%); 221 at `p05` (6.3%).
  Whatever this is, it applies to a small corner of the panel.
- **Concentration.** Three pairs supply 60 of the 81 covered rows: CPI→FED (24
  rows, 66.7%), CPICORE→FED (23, 56.5%), CPICORE→CPI (13, 84.6%).
- **The strongest carrier is mechanical.** CPICORE→CPI is a *same-release* pair
  — one print resolves both — so its 84.6% is arithmetic, not diffusion, and
  belongs to the identification problem in `research_summary.md` §5. Excluding
  same-release pairs entirely: **60.7% on 61 rows, p = 0.13.** The point
  estimate survives; the significance does not. The non-mechanical carriers are
  the two CPI→FED channels.
- **Multiplicity.** The main table is 21 rung × subset cells. One was
  pre-specified; the rest are a family, not seven independent tests.
- **Skewed base rates.** Covered rows concentrate on a few pairs and periods, so
  the majority class runs at 59.3% there. Read `acc` against `perm null`, not
  against `majority` — see the note under the results table.
- **Still in-sample.** Walk-forward *inside* the in-sample block is not the
  holdout. This remains an IS-generated hypothesis.

## What strengthens it

Excluding WTI as a **trigger** — the open Universe-A question in
`data_prep_plan.md` §5 — improves every structure rung and leaves the ablation
null:

| rung | subset | n | acc | perm null | majority | bal. acc | AUC | perm p |
|---|---|---|---|---|---|---|---|---|
| sign_rule | bh | 82 | 0.646 | 0.520 | 0.573 | 0.651 | 0.648 | **0.007** |
| edge_logit | bh | 82 | 0.646 | 0.519 | 0.573 | 0.651 | 0.687 | 0.005 |
| *no_structure* | *bh* | *82* | *0.585* | *0.529* | *0.573* | *0.576* | *0.610* | *0.137* |

WTI supplies **56% of the panel's rows** (2,705 of 4,816) and one BH edge —
WTI→JOBLESSCLAIMS — which contributes a single scored row. On this evidence it
is *mass without structure*, and the answer to "WTI in Universe A?" is: as a
**target**, yes; as a pooled **trigger**, report it separately rather than
letting it dominate a pooled estimate.

## Where this leaves the thesis

- Stage 1 (`stg_infra/stg/structure/`) remains the primary deliverable, and now
  has a predictive validation beside its hypothesis tests — edges refit per
  fold, scored forward, with an ablation isolating their contribution.
- Stage 2's answer to "how much capacity does consuming the structure need?" is
  **none beyond the sign of the edge**, argued along a ladder whose rungs run
  from 0 parameters to AGCRN's 290k. That is a finding, not an unfinished
  deliverable.
- The identification problem is unresolved and now *quantified*: remove the
  same-release pairs and the effect loses its significance. Separating
  transmission from shared information arrival is the binding constraint on the
  claim — not model class.

## Next

1. **Do not spend the holdout yet.** The OOS test should be one pre-registered
   cell — `sign_rule` × `bh`, same-release pairs excluded — run once, at the
   end (`TODO.md`; `research_log.md` §5).
2. **Raise coverage before raising capacity.** The binding constraint is 81
   rows, not model class. Bucket contracts and the Phase B canonicalisation add
   triggers, and every added trigger adds candidate edges.
3. **Sports/crypto negative control** through this exact pipeline — the ladder
   should find nothing where no channel exists. Cheap now that the harness
   exists (`stg_infra/stg/direction/`, `scripts/run_direction_study.py`).
4. **Spread-aware evaluation** (`research_summary.md` §9 Phase 2 item 7): 63%
   directional accuracy on 81 events is a statistical result, not an economic
   one, until execution costs are modelled.
