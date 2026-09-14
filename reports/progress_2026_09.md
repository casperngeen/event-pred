# Progress since the CA report — 2026-08-12 to 2026-09-14

**Ng E En Casper** · *AI in Financial Data Analytics and Trading: Analysing
Cross-Market Influence Propagation in Event Prediction Markets via
Spatial-Temporal Graphs*

*Supervisor-facing. Continues and partly supersedes `update_2026_08.md`, which
covered the first four weeks of this period and proposed seven refinements; this
document covers the whole period, records which of those refinements landed, and
reports the results obtained since. Detail for every claim is in
`research_log.md` (§-numbered), with per-topic writeups named inline.*

**Scope caveat applying to everything.** All analysis is **in-sample only**
(pre-2026). The 2026 block is held out and untouched, enforced uniformly by
`stg_infra/stg/splits.py::assert_no_oos()`. Several results below are
in-sample-generated hypotheses and are labelled as such. The out-of-sample
confirmation is reserved for a single pre-registered test at the end.

---

## 1. Summary

The period divides into three phases:

1. **Measurement** (mid-Aug). The data pipeline the CA report listed as
   outstanding was completed, and in completing it four parsing and
   classification defects were found that had been silently corrupting the
   central quantity — the market-implied belief. → `update_2026_08.md` §1,
   `research_log.md` §8.
2. **Estimation** (early Sep). AGCRN was repositioned from primary model to
   object of study and then run honestly; direct structure estimation became the
   standalone deliverable; a Stage-2 direction study and a full economic
   evaluation were built on top of it. → `agcrn_study.md`,
   `artifacts/adjacency_report.md`, `direction_study.md`, `edge_economics.md`.
3. **Correction** (2026-09-13/14). A review of the data path and the study code
   found nine defects. Fixing them changed two headline results, one of them
   into a null. → `research_log.md` §14.

**The honest one-line state of the project:** the structural claim is intact and
is now better supported than when it was first reported; the *predictive* claim
built on top of it did not survive scrutiny and has been withdrawn; the
tradability question is closed in the negative and that closure is a result
rather than a failure.

### What changed about the thesis's claims

| Claim as at the CA report / `update_2026_08.md` | Status 2026-09-14 |
|---|---|
| Inductive AGCRN is the primary model and methodological contribution | **Repositioned** to a validation study. Run honestly: **no model beats predict-zero**, and more capacity is worse. Diagnosed as a task/horizon mismatch, not an architecture failure. → `agcrn_study.md` |
| Cross-market structure is learned from a forecasting loss | **Replaced** by direct estimation with FDR control and an exact permutation test, so every edge carries a point estimate, a p-value and a falsifiable sign. → `stg_infra/stg/structure/` |
| 22 cross-market relationships (asserted, uncorrected) | **141 ordered pairs searched**, 13 nominally significant against 7.1 expected by chance, **4 survive BH-FDR at q = 0.10** |
| 8 BH survivors (reported in `update_2026_08.md` §4) | **4**, after the §14 corrections. All 4 are **non-same-release**, which is a stronger claim than the 8 — see §5 below |
| The estimated structure predicts direction out of fold (63.0%, p = 0.041) | **Withdrawn.** Clears no gate on the corrected panel (58.7%, p = 0.23 at `p05`). The rung that appeared to win was reading the target's bounded price level |
| Drift magnitude scales with surprise magnitude | **Rejected** — every magnitude-weighted test is null; only rank/sign survives. (Now re-opened as a measurement question, §8) |
| Trading strategy as the deliverable, via PortfolioBench | **Reframed** as a test of economic significance. Net **−3.15c/trade** taker, t = −4.31. Structure is economically significant and not economically exploitable |

---

## 2. The measurement layer

The CA report acknowledged submarket heterogeneity as a challenge; the pipeline
as submitted did not handle it. Work completed:

- **A PMF path for bucket ladders.** Kalshi macro markets come in three
  structurally different shapes — measured **5,169 threshold / 7,875 bucket /
  204 categorical**. The pipeline treated all of them as cumulative threshold
  ladders, which is correct for CPI/U3/GDP/payrolls and wrong for buckets, which
  price `P(a ≤ X ≤ b)` directly. WTI is **85% bucket** and was therefore absent
  from implied-mean output entirely. → `update_2026_08.md` §1
- **Four parsing defects, all failing silently.** `parse_threshold` required a
  literal `-T` prefix and so dropped whole series (JOBLESSCLAIMS 0/132, ISMPMI
  0/49); contract types were conflated; `resolved_value` hardcoded
  `spacing = 0.1`, wrong by ~6 orders of magnitude for payrolls; inclusive and
  exclusive strike conventions coexisted undetected. → `research_log.md` §8
- **The implied distributions are now checked for being distributions.**
  `coverage` and `ladder_mass` were computed, documented as the coherence
  metric, and never filtered on — a WTI ladder whose mutually exclusive buckets
  summed to **2.96** was being accepted and divided by its own total. Two-sided
  gates now apply identically to both contract paths. Cost: 799 → 519 rows and
  **no trigger series**. → §14.2
- **True settlement values.** The outcome of each event was being *inferred*
  from the ladder to a precision of `spacing / 2` — against CPI's median
  |surprise| of 0.0738 pp that error is 0.05 pp, **68% of the signal being
  measured** — while the actually-printed `expiration_value` sat unread on disk
  for 1,022 events. Now used for 442 of 519 panel rows; surprises moved by a
  median 50–67% of their own previous magnitude. → §14.1
- **One staleness rule.** The surprise panel and the node panel applied
  *opposite* rules, so 74.5% of node-feature rows were implied distributions
  assembled from legs last traded on different days. → §14.3
- **A test suite.** From zero to **110 tests**, including explicit leakage
  guards and a regression test for each defect above.

**Why this section is long.** Every defect in it failed silently rather than
raising, and each would have been caught by a basic coverage assertion. The
measurement layer is the foundation for every result in the thesis, and the
honest record is that it needed two rounds of repair. This is limitations-section
material, not something to leave in the commit log.

---

## 3. Data quality: the archive is a convenience sample

`data/trades/` was assembled by fetching one ticker at a time from a supplied
list, and that list was never complete. **350 series have no trades at all**, and
— the part that matters — several *registered* trigger series silently run on a
fraction of their events: WTIW 32%, CPIFOOD 38%, WTI 64%, CPIAPPAREL 68%,
**PAYROLLS 72%**. PAYROLLS carries two of the four surviving Stage-1 edges.

Nothing is retracted on this basis: incomplete collection widens the uncertainty
and its direction is unknown until the selection is tested. But **every `n` in
this project is a floor set by collection, not a measurement of market
activity**, and the final report needs to say so. → `research_log.md` §12

---

## 4. AGCRN: run, and the result is negative

`update_2026_08.md` §2 argued on capacity and architecture grounds that AGCRN
was better positioned as an object of study than as the primary model. That
argument was then *tested* rather than asserted.

**No model beats predict-zero out of sample.** More capacity is monotonically
worse (R² vs zero: linear baselines ≈ 0, AGCRN minimal −0.18, AGCRN default
290k-param −0.61, learned-E variants −1.04/−1.39). The learned adjacency `Ã`
does not recover the validated structure: **0 of 4 BH survivors** among its
top-ranked edges.

Diagnosed as a **task/horizon mismatch, not an architecture failure** — no
model, including oracles that leak the future, scores positive:

1. The signal is at a horizon the model cannot see (76% sign agreement at the
   dormant horizon, ~52% — coin flip — at the snapshot horizon AGCRN predicts).
2. The target is a magnitude; the signal is a sign.
3. There is no predictable central tendency between releases — a future-leaking
   per-node-mean oracle scores R² ≈ 0.006, and the apparent AR(1) predictability
   is an overlapping-window artifact (+0.48 on overlapping windows, **−0.07** on
   non-overlapping).

Re-run 2026-09-14 against the corrected Stage-1 table and rebuilt node panel;
conclusion unchanged in every particular. → `agcrn_study.md`,
`agcrn_postmortem.md`

**A negative result about a proposed method, diagnosed to a cause, is a
legitimate contribution** — and it is what motivates the direct estimator being
the deliverable instead.

---

## 5. Stage 1: the structural claim

The project's actual contribution. For each ordered (trigger, target, side)
pair, a signed rank correlation between the trigger's surprise and the target's
post-resolution price response, with BH-FDR across the full grid and an exact
permutation p.

- **141 ordered pairs searched** — reported, per `update_2026_08.md` §3, because
  reporting a hit rate without the search size is the most likely examiner
  objection.
- 13 nominally significant at p < 0.05, against **7.1 expected by chance**.
- **4 survive BH-FDR at q = 0.10**, all `CPI`/`CPICOREYOY`/`PAYROLLS` → `FED` or
  `FEDDECISION`.

| trigger | target | side | n | ρ̂ | p (perm) | same-release |
|---|---|---|---|---|---|---|
| CPI | FED | any | 36 | +0.617 | 0.0005 | no |
| PAYROLLS | FEDDECISION | hike | 26 | +0.596 | 0.0005 | no |
| PAYROLLS | FED | any | 30 | +0.583 | 0.0010 | no |
| CPICOREYOY | FEDDECISION | cut | 17 | −0.683 | 0.0025 | no |

**This is 4 survivors where `update_2026_08.md` §4 reported 8, and it should be
presented as a stronger claim, not a retreat.** Three things changed: selection
moved from an asymptotic p to an exact permutation p (the data is 98% tied, so
the t-approximation's assumptions are not met); a look-ahead in how the response
instrument was chosen was removed; and the implied distributions are now gated.
The strongest edge in the published table, `CPICORE→CPI` at ρ̂ = 0.744, was a
**same-release** pair already flagged as "arithmetic, not diffusion" — and it
fell out. The recovered structure is now *entirely* scheduled-macro-surprise →
policy path, the Kuttner (2001) / Gürkaynak–Sack–Swanson (2005) channel, with no
mechanical same-print pair carrying any of it. **§5's identification problem is
no longer load-bearing on the headline.**

**Sign restrictions pass, including the falsification cell.** Signs were fixed
from theory before the estimates were consulted. Every hawkish channel is
positive; both dovish `U3` cells flip as required. This is the check that would
have caught the estimator fitting generic co-movement, and a validation loss
cannot provide it. → `edge_economics.md` §2

**The graph is a policy hub, not a web.** Every survivor points at the policy
path. That is a more modest claim than "prediction markets form a
spatio-temporal graph", and it is the honest one — it is also the best-documented
transmission channel in macro-finance, which is what makes the signs available
*a priori*. Consequence for the modelling: with no interior node, there is
nothing for multi-hop message passing to traverse, which the mediation tests
independently confirm (every partial unchanged to within ±0.036).

---

## 6. Stage 2: the predictive claim, and its withdrawal

The AGCRN post-mortem prescribed the replacement task precisely — dormant
horizon, direction label, edges re-estimated **inside every walk-forward fold on
training rows only** so nothing consumes the published in-sample adjacency. That
machinery was built and is correct (`stg_infra/stg/direction/`, 16 tests
including planted-signal recovery and a pure-noise null).

**On 2026-09-07 it reported that the structure predicts direction out of fold:**
63.0% vs a 59.3% base on BH-covered rows, p = 0.041, with the ablation showing
the *edge* rather than the surprise carried it.

**On the corrected panel that result is a null.** The sign rule clears no gate —
58.7% at `p05` (p = 0.23, n = 92), 68.9% at `bh` (p = 0.059, n = 45). The point
estimates barely moved; **the samples collapsed** (221 → 92, 81 → 45), because a
usable row now requires a pre-trigger print on the leg actually used. Three
defects had been doing the work: the look-ahead in instrument choice, an
asymptotic p-value the 98% tie rate does not license, and — for every re-run
between 2026-09-08 and 2026-09-13 — a cached pair panel that never rebuilt
through two rounds of upstream fixes.

**The rung that now appears to win is an artifact.** `no_structure` beats every
structure rung, which would invert the earlier claim. It does not, because the
win is entirely one feature: `p0c`, the target's pre-trigger **price level**.
Kalshi prices are bounded in [0, 100], so from a low `p0` the next move is
mechanically more likely to be up, and a logit handed `p0c` will find that.
Ablating it takes the rung from 54.4% to 49.2% — chance. `z_only` and
`abs_z_only` are at chance too.

**So the honest reading is a null: on the corrected panel neither the estimated
structure nor the surprise predicts direction at the dormant horizon.**

`p0c` also sits in the shared `CONTEXT` block, so it contaminates
`feature_logit` and `neighbour_logit` equally — which means the ablation contrast
the ladder exists to compute is **not identified** while both sides carry a
feature that predicts the label mechanically. Fixing that is a design change
(drop `p0c`, or re-specify the label orthogonal to the price level), and is the
highest-value open Stage-2 item. → `direction_study.md`, `research_log.md` §14.10

---

## 7. Economics: coherent, and not exploitable

Two questions the structural result raises and does not answer: do the edges make
economic sense, and can they be traded?

**(a) They make sense.** Sign restrictions all fire as theory demands (§5
above). The one clear failure is diagnostic rather than damaging:
**`WTI→CPIGAS` is flat** (ρ̂ = +0.04, n = 80, p = 0.73) despite oil passing into
gasoline CPI by construction. That is not a power failure — **WTI has no
information event.** Its settle is public and continuously observable, so at
resolution there is no news and the "surprise" is an artefact of when the ladder
was snapped. Consequence: *a trigger needs a scheduled information release, not
merely a resolution timestamp* — a selection criterion now encoded in the
registry (`scheduled_release`) and bearing directly on the open Universe-A
decision.

**(b) They cannot be traded, and the reason is not cost.** The whole edge lives
in the gap between the last pre-resolution trade (what the estimator sees) and
the first post-resolution print (the earliest executable price):

| measured from | cents/trade | hit rate |
|---|---|---|
| `p0` — last pre-resolution trade | +0.91 | 0.587 |
| first post-resolution print — earliest executable | **+0.47** | **0.380** |

Median lag between the two: **33.6 minutes**. It is not a fast move being missed
— **no tradable price exists in between, because these books do not print.**
Costs then bury the remainder: net **−3.15c** per trade taker (t = −4.31,
n = 92), −0.77c under the most favourable maker assumption.

**(c) The edge is a jump, not a drift — and staleness did not manufacture it.**
The market reprices once, completely, at its first print after resolution, so
there is no decay curve to arrive early on. This matters because a stale
reference price was the CA report's §4.1 central threat: if staleness created the
edge, accuracy would *rise* with it. It falls monotonically (0.630 / 0.622 /
0.450 by `p0` age). **Threat checked and cleared.**

**(d) Holding longer does not help**, because gross does not grow — swept
1h/6h/1d/3d/7d/14d, gross peaks near one day at ~1–2c and then decays. And two
configurations that appeared to work were each a handful of events wearing a
large `n`: clustering on `target_event` (rows sharing a target event are
perfectly correlated at settlement) removes both.

**Read as the plan pre-committed:** economic significance, not profitability.
**The recovered structure is economically significant and not economically
exploitable** — which quantifies the limits to arbitrage that explain the
anomaly's persistence, and agrees with Angelini & De Angelis (2026) on Kalshi.
The friction that creates the anomaly is the friction that makes it
unexploitable. → `edge_economics.md`, `research_log.md` §10, §13

One tradable hypothesis remains open: **pre-resolution coherence** — trade the
disagreement between a trigger's implied distribution and the target's price
*before* resolution and hold through the jump. It is the only path positioned
before the print that carries the move, and the only one where the maker argument
survives (hours to get filled, not 17 minutes).

---

## 8. Where this leaves the design

A design document written 2026-09-14 (`relations_study_plan.md`) proposes the
next round, on the principle that `n` is the binding constraint and so only
changes that *add* power are worth making:

- **Surprise is a first moment of a distribution that is recovered and then
  discarded**, and it is not unit-free, which is what stops channel pooling from
  being more than a sign test. Proposed: a PIT/quantile surprise, which is
  unit-free, reads the whole distribution, and carries a **standalone
  calibration study** of the Kalshi macro ladder as a by-product.
- **The decision to abandon magnitude rests on one number** (`|surprise|`
  cross-validating at r = 0.242 where signed surprise gives 0.686). That number
  was computed with an estimator inheriting all of the mean's error. Re-testing
  it with an information-theoretic surprisal is cheap and has a binary outcome.
- **The CPI family is one factor, not five** — standardised surprises run
  r = 0.88–0.89 at 93–95% sign agreement — so it should be collapsed, which
  raises events per node and cuts the pair grid ~21%.
- **PAYROLLS and U3 are a genuine double trigger**: same instant, same print,
  r = −0.18, sign agreement at exactly chance. A zero-parameter labour index
  `z_PAYROLLS − z_U3` is properly identified and should dominate either
  component — which would also rehabilitate `U3`, the weakest falsification cell,
  which has only ever been tested in isolation.
- **Channel pooling as a block model** — 141 free pair parameters collapsed to
  ~4 theory-signed channel coefficients, tested at 456 rows vs 81, p = 0.021,
  **zero fitted parameters**.

---

## 9. Open decisions for discussion

1. **Sign-off on the AGCRN pivot.** The negative result is now measured and
   diagnosed rather than argued. Does it stand as a contribution in its own
   right, and how much of the thesis should it occupy?
2. **Profitability as goal or as test.** The project has answered this in
   practice — economic significance, with the negative tradability result as the
   finding. Worth confirming this is the right framing for examiners.
3. **WTI in the trigger universe.** It is 56% of the panel's rows and contributes
   effectively nothing, and §7(a) gives a principled rather than results-driven
   reason to exclude it. Proposed: keep as a target, report separately as a
   trigger.
4. **Sports as a second arm.** ~100× the events, already local, games *are*
   scheduled information reveals, and — the real argument — the graph is
   **mechanically known** (game → season total → division → championship), so
   edges are verifiable rather than guessed. The macro universe cannot support
   the thesis's multi-hop graph claim because it has no depth; sports can. This
   is a scope decision and needs a view before the October freeze.
5. **What the single out-of-sample test is spent on.** Currently specified as
   `sign_rule` × `bh` gate, same-release excluded, dormant horizon. Given §6, the
   hypothesis that cell was specified to confirm no longer has in-sample support,
   so the cell should be **re-pre-registered before it is spent**, not silently
   swapped.

---

## 10. Risk register for the final report

| risk | current state |
|---|---|
| Specification search | Documented explicitly (magnitude → signed → sign-only). Must be disclosed; the OOS block is the only clean test |
| Multiplicity | Controlled — 141 pairs searched, reported, BH-FDR at q = 0.10, exact permutation p |
| Identification | **Improved but not solved.** All 4 survivors are now non-same-release, so the headline no longer rests on it. Nothing yet separates propagation from two contracts reading one number in the general case |
| Data completeness | Convenience sample; several triggers on 32–72% of their events (§3) |
| Small `n` | The binding constraint everywhere. 519 usable trigger events, of which 198 are WTI |
| Execution realism | Measured, not assumed — order-book spreads 8× wider than trade-based estimates; spreads roughly double at release time; median 1 strike trading in the 30 min after a resolution |
| Reproducibility | Three nondeterminism bugs found and fixed; artifacts built before 2026-09-08 carry the old behaviour |

---

## 11. Document and code map

| where | what |
|---|---|
| `reports/research_log.md` | Running findings, §1–§14. Most-cited document. §14 is the correction round |
| `reports/agcrn_study.md` / `agcrn_postmortem.md` | The AGCRN negative result and its diagnosis |
| `reports/direction_study.md` | Stage 2 — the null, and why the design is still right |
| `reports/edge_economics.md` | Sign restrictions, frictions, the trade ledger |
| `reports/relations_study_plan.md` | Design doc for the next round (§8 above) |
| `reports/graph_definition.md` | The authoritative node/edge/snapshot/label definition |
| `reports/TODO.md` | Consolidated task list and open decisions |
| `stg_infra/stg/structure/` | Stage 1 — the direct estimator |
| `stg_infra/stg/direction/` | Stage 2 — folds, learners, tradability |
| `stg_infra/stg/panel/` | The measurement layer — surprise, nodes, targets, registry |
| `artifacts/` | Generated tables; regenerate with the `scripts/run_*.py` that names them |

110 tests pass. Working tree clean as of 654de46.
