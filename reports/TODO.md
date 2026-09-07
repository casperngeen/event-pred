# TODO

Consolidated view. Detail lives in `data_prep_plan.md` (data), `research_log.md`
(findings + measurement problems), `research_summary.md` (the standing plan).
Submission target November 2026; analysis should freeze early October.

Last updated 2026-08-20.

---

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

- [ ] **Bucket-contract PMF path.** `parse_bucket()` exists but nothing calls it.
      Bucket contracts price `P(a ≤ X ≤ b)` directly — already a pmf, needs
      normalisation not `recover_pdf`'s differencing. **WTI is 85% bucket (7,818/9,222)
      and is currently absent from implied-mean output entirely.** It was the
      highest-n trigger in the pairwise sweep before the parse bug invalidated
      those results. → `research_log.md` §1(b), §8
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
- [ ] **Tests.** Still zero. The three bugs fixed today were all silent-failure
      cases a single assertion would have caught (`parse_threshold` returning None
      for an entire series). Start there. → `research_summary.md` §2.4
- [ ] *(cosmetic)* 18 files in `data/trades/` have a `(1)` suffix — verified exact
      duplicates of existing files, invisible to `*.parquet` globs. Harmless, but
      delete when convenient.

---

## Analysis — next

- [ ] **Stage-1 structure estimation, sign/rank-based**, FDR-corrected across the
      full pair grid, IS only. The sign result gives the first reason to think
      this will find something. → `research_log.md` §6
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
- [ ] **OOS test of the sign result** — once, at the end. It is an
      in-sample-generated hypothesis, not yet a finding. → `research_log.md` §5

---

## Decisions still open

- [ ] **Supervisor sign-off on the AGCRN pivot** → `research_summary.md` §10.1
- [ ] **Profitability as goal vs. test** → `research_summary.md` §6.4, §10.2
- [ ] **WTI in Universe A?** 824 events dwarfs everything else and would dominate
      any pooled estimate. → `data_prep_plan.md` §5
- [ ] **Read Diercks et al. and Angelini & De Angelis in full** before finalising
      the gap statement → `research_summary.md` §10.3
