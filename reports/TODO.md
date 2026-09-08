# TODO

Consolidated view. Detail lives in `data_prep_plan.md` (data), `research_log.md`
(findings + measurement problems), `research_summary.md` (the standing plan).
Submission target November 2026; analysis should freeze early October.

Last updated 2026-09-07.

---

## Done (2026-09-07)

- [x] **Stage-1 structure estimation** — 141 pairs, BH-FDR q=0.1, 8 survivors,
      permutation-checked, with term structure and mediation → `artifacts/adjacency_report.md`
- [x] **AGCRN run and post-mortem** — 0 models beat predict-zero; diagnosed as a
      target/horizon mismatch, not an architecture failure → `agcrn_study.md`,
      `agcrn_postmortem.md`
- [x] **Stage-2 direction ladder** (`stg/direction/`, `scripts/run_direction_study.py`)
      — the post-mortem's prescription: dormant horizon, direction label, edges
      refit per fold. Sign rule 63.0% vs 59.3% base on structure-covered rows
      (p=0.041); the edge, not the surprise, carries it; capacity does not pay.
      → `direction_study.md`
- [x] **§4.1 staleness threat cleared.** The plan's "do this first — if drift
      does not survive, the thesis needs to change in August" check. Accuracy
      *rises* with a fresher reference price (0.690 / 0.615 / 0.538 by p0 age),
      corr(staleness, |jump|) = +0.09 — stale prices dilute the finding rather
      than create it. → `edge_economics.md` §4
- [x] **Implied means are unbiased** — no series shows an ex-ante bias (max
      |t| = 1.66 over 17, uncorrected). Validates `implied.py`'s PDF recovery
      against an economic criterion, and closes the naive pre-emptive-hold path.
      → `edge_economics.md` §5B
- [x] **Edge economics + tradability** (`stg/direction/tradability.py`,
      `scripts/run_edge_economics.py`) — sign restrictions all fire as theory
      predicts, incl. the U3 dovish-flip falsification cell; WTI→CPIGAS is flat
      (WTI has no information event); the signal is fully consumed by the first
      post-resolution print (+0.86c from p0 → +0.01c from the executable entry,
      hit rate 63.0% → 39.5%), so net is −4.25c/trade taker, −1.50c maker bound.
      → `edge_economics.md`
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
      `rules_primary` not title similarity. Plus the event calendar table
      (resolution timestamps, `expiration_value`). → `data_prep_plan.md` Phase B
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
      the unconditional per-series figure (1.38c); the book widens to a 2.0c
      median within 0.5 h of a release, which moves net from −4.25c to ≈ −4.9c.
      → `research_log.md` §11.2
- [ ] **Raise edge coverage.** The Stage-2 binding constraint is 81 covered rows
      out of 3,487 scored, three pairs supplying 60 of them. Bucket contracts +
      Phase B canonicalisation add triggers, and every trigger adds candidate
      edges. This beats any further modelling. → `direction_study.md` §Next
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
      nulls (n≈221, the highest-powered cells) are currently invalid, not
      informative.
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
