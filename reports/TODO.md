# TODO

Consolidated view. Detail lives in `data_prep_plan.md` (data), `research_log.md`
(findings + measurement problems), `research_summary.md` (the standing plan).
Submission target November 2026; analysis should freeze early October.

Last updated 2026-09-07.

---

## Done (2026-09-14)

- [x] **Nine defects found and fixed across the data path and the study code**,
      each with a regression test (110 tests pass). → `research_log.md` §14
      - `resolved_value` was *inferred* to ±spacing/2 when the true printed
        value (`expiration_value`) was on disk for 847 IS events — 68% of CPI's
        median |surprise| was measurement error. Now used, 442/519 rows. (§14.1)
      - `coverage` / `ladder_mass` were recorded and never enforced; the bucket
        path accepted a ladder summing to 2.96. Two-sided gates, 799 → 519 rows,
        no trigger series lost. (§14.2)
      - The node and surprise panels applied *opposite* staleness rules; 74.5%
        of node rows mixed legs last traded on different days. (§14.3)
      - Spearman ranks ignored ties on data that is 98% tied (max |Δρ| = 0.238);
        the p-value used a normal where a t belongs; BH controlled the wrong
        p-value; permutation p could be exactly 0; one shared mutable RNG made
        every p depend on call order. (§14.4)
      - The response instrument was chosen using post-decision trades. (§14.5)
      - `neighbour_signal` split simultaneous triggers by row order. (§14.6)
      - The pair-panel cache had no staleness check and served an 8-day-old
        panel through two rounds of fixes. (§14.7)
      - Rolling windows counted rows not days (60-69% of node features wrong);
        `clearance_days` was disabled by a literal `if False`;
        `aggregate_to_event_level` z-scored against the future. (§14.8)
- [x] **Stage 1 re-run: 4 BH survivors, not 8** — all non-same-release, all
      macro → policy path, selected on an exact permutation p.
      → `artifacts/adjacency_report.md`, `research_log.md` §14.9
- [x] **AGCRN re-run** against the corrected table and rebuilt node panel —
      conclusion unchanged in every particular. → `agcrn_study.md`
- [x] **`p05` adopted as the headline gate** (better powered than `bh`), in the
      direction study and the ledger.

---

## Done (2026-09-07)

- [x] **Stage-1 structure estimation** — 141 pairs, BH-FDR q=0.1, **4 survivors**,
      selected on the permutation p, with term structure and mediation
      → `artifacts/adjacency_report.md`
      *Was 8 survivors selected on an asymptotic p; corrected 2026-09-13, see
      `research_log.md` §14. All 4 are non-same-release — the mechanical
      `CPICORE→CPI` pair fell out when the look-ahead in instrument choice was
      removed, which makes this a stronger claim than the 8 it replaces.*
- [x] **AGCRN run and post-mortem** — 0 models beat predict-zero; diagnosed as a
      target/horizon mismatch, not an architecture failure → `agcrn_study.md`,
      `agcrn_postmortem.md`
- [x] **Stage-2 direction ladder built** (`stg/direction/`,
      `scripts/run_direction_study.py`) — the post-mortem's prescription:
      dormant horizon, direction label, edges refit per fold. The *machinery* is
      done and correct. **Its result is now a null**: corrected 2026-09-14, the
      sign rule reaches 58.7% vs a 54.1% null at `p05` (p=0.232) and does not
      clear any gate; the `no_structure` rung that appears to win is reading the
      target's bounded price level (`p0c`), not the surprise. Neither structure
      nor surprise predicts direction on the corrected panel.
      → `direction_study.md`, `research_log.md` §14.10
- [x] **§4.1 staleness threat cleared** — and it survives the §14 corrections.
      Accuracy still *rises* with a fresher reference price (0.630 / 0.622 /
      0.450 by p0 age), corr(staleness, |jump|) = +0.12 — stale prices dilute
      rather than create. → `edge_economics.md` §4
- [x] **Implied means are unbiased** — no series shows an ex-ante bias (max
      |t| = 1.66 over 17, uncorrected). Validates `implied.py`'s PDF recovery
      against an economic criterion, and closes the naive pre-emptive-hold path.
      → `edge_economics.md` §5B
- [x] **Edge economics + tradability** (`stg/direction/tradability.py`,
      `scripts/run_edge_economics.py`) — sign restrictions all fire as theory
      predicts, incl. the U3 dovish-flip falsification cell; WTI→CPIGAS is flat
      (WTI has no information event); the signal is fully consumed by the first
      post-resolution print (+0.91c from p0 → +0.47c from the executable entry,
      hit rate 58.7% → 38.0%), so net is **−3.15c**/trade taker (t = −4.31,
      n = 92), −0.77c maker bound. Re-priced 2026-09-14; conclusion unchanged
      and firmer, but read it as an upper bound on this *class* of signal since
      the underlying rule no longer has demonstrated skill.
      → `edge_economics.md`, `research_log.md` §14.11
- [x] **Effective-spread estimator made durable** — §4.2.1's taker-direction
      method promoted from the ad-hoc script into `direction/tradability.py`
      with the window-invariance check; cached in
      `artifacts/effective_spreads.parquet`.
- [x] **Reproducibility bug in `panel/targets.py`** — `representative_tickers`
      broke trade-count ties via an unordered `group_by`, and `response_panel`
      broke close-time ties by list order, so identical rebuilds differed by a
      row. Fixed with lexicographic tiebreaks + a test.
- [x] **Tests for the new module** — `tests/test_direction.py`, 16 cases: fold
      purge/disjointness, train-only edge fitting, causal neighbour feature,
      planted-signal recovery, pure-noise null.

## Done (2026-08-20)

- [x] **A1/A2/A3 pulls landed** in `data/` — see `data/MANIFEST_new_pulls.md`
- [x] **IS/OOS separation by directory** — `_OOS_DO_NOT_USE_*` prefixes
- [x] **`implied.py` bugs 1–3** — threshold parsing (0/132 → 132/132 on JOBLESSCLAIMS),
      contract classification, per-event spacing, inclusive/exclusive conventions
- [x] **Backfill merged** into `data/trades/` (39,387,030 rows; zero `trade_id` overlap)
- [x] **`stg/splits.py`** — single source of truth for the OOS wall, with `assert_no_oos()`
- [x] **`build_daily` forward-fill** — now emits `close_raw` / `is_filled` / `stale_days`
      alongside the filled series instead of silently replacing it

---

## Data prep — remaining

- [x] ~~**Bucket-contract PMF path.**~~ **Done** — `panel/surprise.py::_bucket_surprise`
      dispatches on `registry.SPECS[...].kind == "bucket"`; the surprise panel now
      carries 414 bucket events (369 WTI, 45 WTIW) against 385 threshold events,
      and WTI/WTIW appear as usable triggers. Verified 2026-09-07.
      → `research_log.md` §1(b), §8
- [ ] **Phase B — canonicalisation.** Series alias table (`CPISHELTER`→`KXSHELTERCPI`,
      `JOBLESS`→`KXJOBLESSCLAIMS`, PROLLS/PAYROLLS split), decided from
      `rules_primary` not title similarity. Plus the event calendar table.
      → `data_prep_plan.md` Phase B
      - [x] `expiration_value` — **done** 2026-09-12, no re-pull needed. 847 IS events
            from `markets_api_pull_raw.jsonl`; `surprise_panel.resolved_value` prefers it
            over the ladder midpoint (442 of 519 rows). → `research_log.md` §14.1
      - [ ] resolution timestamps: keep `settlement_ts` as metadata only. It is a median
            4.65 h *after* `close_time` — administrative settlement, not the release. Do
            not substitute it for `close_time`, which already sits 5 min pre-print.
- [ ] **Phase C — durable panels.** Four panels (ticker-day w/ staleness,
      trade-time, event-day implied distribution, quote panel). Everything this
      session rebuilt these ad hoc in scratch scripts. → `data_prep_plan.md` Phase C
- [ ] **Phase D — repartition** `data/trades/` by series/year/month. Currently 3,891
      flat files named by row-index, so every query scans 1.8 GB to touch a few
      macro series. → `data_prep_plan.md` Phase D
- [ ] **Consume `stale_days` downstream.** The columns now exist but nothing filters
      on them; `build_daily_implied_means` still takes `close` unconditionally.
      Only ~12% of CPI event-days have every ladder leg fresh. → `research_log.md` §7
- [ ] **Coherence-violation counter.** `np.clip` in `recover_pdf` silently discards
      exactly the monotonicity violations §5 wants to measure as a standalone
      contribution. Count before clipping. → `research_summary.md` §5, Phase 3 item 10
- [ ] **Tests — extend, no longer absent.** `stg_infra/tests/` now covers splits,
      implied, panel, structure, models and direction (the last with explicit
      leakage guards). Still missing: the `parse_threshold`-returns-None-for-a-
      whole-series class of silent failure that motivated this item, i.e. a
      coverage assertion per series at panel-build time.
      → `research_summary.md` §2.4
- [ ] *(cosmetic)* 18 files in `data/trades/` have a `(1)` suffix — verified exact
      duplicates of existing files, invisible to `*.parquet` globs. Harmless, but
      delete when convenient.

---

## Analysis — next

- [ ] **Channel pooling — the highest-value next step.** Replace 140 per-pair
      free parameters with ~4 theory-signed channel coefficients. Feasibility
      tested: data→policy channel, 456 rows (vs 81), 55.0% aligned sign
      agreement, block-perm null 50.1%±2.3%, **p=0.021**, zero fitted
      parameters. Needs the walk-forward rung + an OOS-owed note (the channel
      definition was informed by Stage-1). → `research_log.md` §11
- [ ] **Promote sports to a first-class arm.** ~100x events (NFL multi-game
      132,094; single-game 31,291), already local, games *are* scheduled
      information reveals, and the graph is mechanically known
      (game → season total → division → championship) so edges are verifiable
      rather than guessed. → `research_log.md` §11
- [ ] **Generalise `representative_tickers` to the whole ladder** — but only as
      part of the pre-resolution strategy. 4-7 usable strikes/event is a
      *lifetime* count; within 30 min of a resolution the median event has **one**
      strike trading, so there is no capacity multiplier for the dormant trade.
      Even then it multiplies capacity, not statistical power (strikes on one
      event are one bet in larger size). Still worth it for measuring at the
      near-the-money strike and for the moneyness axis.
      → `research_log.md` §11.1, §11.2
- [ ] **Charge a release-conditional spread in the ledger.** It currently uses
      the unconditional per-series figure (1.19c); the book widens near a
      release, which would push net below the current −3.15c.
      → `research_log.md` §11.2
- [ ] **Fix the Stage-2 ladder's design before re-running it.** `p0c` (the
      target's price level) sits in `CONTEXT`, predicts the label mechanically
      via bounded support, and therefore contaminates `no_structure`,
      `feature_logit` and `neighbour_logit` alike — so the R4-minus-R3 ablation
      the ladder exists to compute is not identified. Either drop `p0c` or
      re-specify the label orthogonal to the price level. **Highest-value change
      to Stage 2.** → `research_log.md` §14.10
- [ ] **Raise edge coverage.** The Stage-2 binding constraint is 92 covered rows
      at `p05` out of 2,347 scored (45 at `bh`), five pairs supplying 75 of
      them. Bucket contracts + Phase B canonicalisation add triggers, and
      channel pooling (§11) recovers ~5.6x — unaffected by the §14 corrections.
      → `direction_study.md` §Next
- [ ] **Sports/crypto negative control through the direction ladder** — the
      harness now exists, so this is a re-run with a different panel. The ladder
      must find nothing where no channel exists.
- [x] ~~**Spread-aware evaluation of the direction result**~~ **Done** →
      `edge_economics.md` §4. Answer: not tradable, and for a sharper reason
      than cost — the edge is gone by the first executable print.
- [ ] **Pre-resolution coherence strategy (the remaining tradable hypothesis).**
      Trade the disagreement between a trigger's implied distribution and the
      target's price *before* resolution, hold through the jump. The only path
      that is positioned before the print carrying the move, and the only one
      where §6.4's maker argument survives (hours to get filled, not 17 min).
      **Needs Phase C panels — this moves Phase C onto the critical path.**
      → `edge_economics.md` §5C, `research_summary.md` §5 / Phase 3 item 10
- [ ] **Limit-order / fill-probability study.** The ledger prices one specific
      execution (cross at the 1st print, cross at the 3rd). §6.4's maker
      argument does not survive a ~13-minute window, but the fill-probability
      version is untested. `analysis/exploratory_2026_08/fill_prob.py` started
      one. → `edge_economics.md` §4
- [ ] **Trigger inclusion criterion.** WTI→CPIGAS being flat says a trigger
      needs a *scheduled information release*, not merely a resolution
      timestamp. Apply this before re-running the sweep — it may remove WTI/WTIW
      from the trigger set on principle rather than on results.
      → `edge_economics.md` §2(b)
- [ ] **Re-run the pairwise sweep** once bucket contracts are handled — the WTI
      nulls (the highest-powered cells) are currently invalid, not informative.
- [ ] **Multiple-testing correction on the CA report's 22 relationships**, and
      report how many pairs were searched to find them. Most likely examiner
      question. → `research_summary.md` Phase 1 item 1
- [ ] **Identification (§5).** Nothing so far separates "information propagated
      through the market" from "both contracts respond to the same public
      release". A sign test cannot address this. Needs release-vs-resolution
      timestamps, or the CPI-subcomponent mechanical-weight decomposition.
- [ ] **Sports/crypto controls** — negative control (does the pipeline invent
      drift where no channel exists) and NBA positive control (does it recover a
      known effect). `kalshi_orderbooks.jsonl` has real depth for NFL/NCAAF/NBA,
      and 37,844 of 88,808 sports/crypto markets sit in macro's 1–10 trades/day
      band, so density-matching is feasible.
- [ ] **OOS test — one pre-registered cell, once, at the end.** Now specified:
      `sign_rule` x `bh` gate, same-release pairs excluded, dormant horizon.
      Still an in-sample-generated hypothesis, not a finding.
      → `research_log.md` §5, `direction_study.md`

---

## Decisions still open

- [ ] **Supervisor sign-off on the AGCRN pivot** → `research_summary.md` §10.1
- [ ] **Profitability as goal vs. test** → `research_summary.md` §6.4, §10.2
- [ ] **WTI in Universe A?** 824 events dwarfs everything else and would dominate
      any pooled estimate. → `data_prep_plan.md` §5
      *Evidence now in:* WTI is 56% of the Stage-2 panel's rows and contributes
      one BH edge worth a single scored row; dropping it as a trigger takes the
      sign rule from p=0.041 to p=0.007. Proposed answer: keep as target, report
      separately as trigger. → `direction_study.md`
      *Reinforced:* WTI has no scheduled information event, so its "surprise" is
      a snapshot artefact — WTI→CPIGAS, the most mechanically certain link in
      the grid, is flat (rho=+0.04, n=80). → `edge_economics.md` §2(b)
- [ ] **Read Diercks et al. and Angelini & De Angelis in full** before finalising
      the gap statement → `research_summary.md` §10.3
