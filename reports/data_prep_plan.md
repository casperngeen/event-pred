# Data Preparation Plan

*Companion to `research_summary.md`. API probes run live against Kalshi on 2026-08-20.*

---

## 0. What "the archive" means here

Three assets under `data/`, all verified directly on 2026-08-20:

| Asset | Size | Contents |
|---|---|---|
| `data/trades/` | 1.8 GB, 3,908 parquet | 39,363,600 trades, **2021-06-30 → 2025-12-30** |
| `data/markets/` | 136 MB, 427 parquet | 4,252,570 rows / 4,252,445 tickers; `_fetched_at` spans **2025-11-23 → 2026-03-31** |
| `data/kalshi_markets_1y_clean.jsonl` | 1.2 GB, 544,853 lines | Raw unnormalised market JSON, `created_time` **2023-08 → 2026-02** |

Two corrections to `research_summary.md` §3:

- **`data/markets/` is not a single point-in-time crawl.** The summary says all rows were
  fetched within ~50 seconds on 2025-11-23. There are later passes running through
  2026-03-31. Still snapshots rather than a time series, so the "no historical quotes"
  conclusion stands — but there is more than one observation per market to work with.
- **The strike fields are already on disk.** §2.3 says `normalize_markets` discards
  `cap_strike`/`floor_strike`/`strike_type` at ingest. True of the parquet path — but the
  raw JSONL retains them, along with `expiration_value`, `settlement_ts`, `rules_primary`
  and `yes_sub_title`. Of 1,323 macro rows in it, **1,294 carry strike information and all
  1,323 carry `expiration_value`**. Its macro coverage is mostly 2025-04 → 2026-02 (it is a
  "1y" file), so it shrinks the metadata re-pull rather than removing it — but it means the
  fragile ticker-regex in `implied.py` can be validated against on-disk truth today, before
  any network call.

---

## 0b. Headline findings

1. **The archive ends 2025-12-30. Today is 2026-08-20.** You are missing ~7.7 months
   in the densest part of the sample — and it is fully recoverable.
2. **Full trade history is still retrievable, back to 2021.** The `/historical/trades`
   endpoint your `scripts/fetch_kalshi_data.py` already uses serves 2022-era tickers fine.
   Nothing about the historical archive is at risk.
3. **`/historical/markets?event_ticker=...` works for delisted events** (verified on
   `CPI-22JUN`, `CPIYOY-23JUN`). §2.3 of the summary said pre-KX tickers 404 — that is
   true only for `/markets/{ticker}`. The event-filtered historical endpoint is the fix.
4. **Candlestick retention is eroding week by week.** On 2026-08-14 the summary recorded
   `KXCPIYOY-26MAY` as retrievable. Today it returns zero candles. This is the only
   genuinely time-sensitive item in the plan.
5. **The 2026 re-pull is a natural out-of-sample set** — data that did not exist when the
   methodology was designed. This is a much stronger OOS claim than an arbitrary date cut,
   and it is the organising idea of §3 below.

---

## 1. What more can be pulled

### 1.1 New data in existing series (2026-01-01 → 2026-08-20)

Event counts per calendar year, enumerated live from `/events`:

| Series | 2021 | 2022 | 2023 | 2024 | 2025 | **2026 (→Aug)** |
|---|---|---|---|---|---|---|
| KXCPI | 7 | 12 | 12 | 12 | 12 | **11** |
| KXCPIYOY | 0 | 2 | 12 | 12 | 12 | **10** |
| KXCPICORE | 0 | 7 | 12 | 12 | 12 | **11** |
| KXCPICOREYOY | 0 | 1 | 12 | 12 | 12 | **10** |
| KXFEDDECISION | 0 | 0 | 6 | 8 | 8 | **8** |
| KXFED | 3 | 8 | 8 | 8 | 8 | **8** |
| KXPAYROLLS | 0 | 0 | **10** | 12 | 12 | **11** |
| KXU3 | 6 | 12 | 12 | 12 | 12 | **11** |
| KXGDP | 3 | 3 | 3 | 4 | 4 | **4** |
| KXPCECORE | 0 | 2 | 1 | 2 | **12** | **11** |

Two corrections to the summary fall out of this:

- **PAYROLLS starts 2023, not 2024-02.** There are ~10 events in 2023 the archive simply
  did not capture. That materially eases the "thin core trigger" problem.
- **PCECORE has 12 events in 2025**, against the 58 tickers / 3.5K trades in the archive.
  The archive's coverage of it is badly incomplete, not genuinely sparse.

### 1.2 The actual gap: archive vs. live API

Distinct event periods, counted directly from `data/trades/` and from `/events`:

| Series | In archive | On API | **Missing** | Nature of the gap |
|---|---|---|---|---|
| KXJOBLESSCLAIMS | 16 | 52 | **36** | Archive trades stop 2025-10-02. Weekly cadence — the §3-asset-1 small-N fix. |
| KXPAYROLLS | 21 | 45 | **24** | Archive starts 2024-02; **API has 10 events in 2023** the pull never captured. |
| KXPCECORE | 13 | 28 | **15** | Present since 2022 but only 3,480 trades / 58 tickers — genuinely thin *and* incomplete. |
| KXU3 | 50 | 65 | 15 | Well covered (36,884 trades); the summary's §3 table omits it entirely. |
| KXCPICOREYOY | 34 | 47 | 13 | |
| KXCPI / KXCPICORE | 54 / 42 | 66 / 54 | 12 each | |
| KXADP | 9 | 21 | 12 | |
| KXCPIYOY | 37 | 48 | 11 | |
| KXFEDDECISION | 31 | 39 | 8 | |
| KXGDP / KXISMPMI | 19 / 7 | 25 / 13 | 6 each | |
| KXFED | 47 | 46 | ~0 | Pre-KX naming; effectively complete. |

**≈169 missing event periods across the core 13 series.**

*Correction to my first draft: JOBLESSCLAIMS, ADP and ISMPMI are **not** absent from the
archive — they are partially present (2,034 / 2,123 / 463 trades). The gap is truncation at
2025-12-30, not absence.*

Genuinely absent, because they are 2026-only listings:

| Series | Title | Events | Why it matters |
|---|---|---|---|
| **KXSHELTERCPI / KXUSGASCPI / KXUSEDCARCPI / KXAIRFARECPI** | CPI subcomponents | 6 each | Successors to `CPISHELTER`/`CPIGAS`/`CPIUSEDCAR`, whose archive coverage dies in 2025-03/04. Keeps the **BLS-basket-weight ground-truth validation** (§3 asset 2) alive — it would otherwise have expired. |
| **KXUSPPIYOY / KXUSPPI** | US PPI | 6 | PPI is named as an event type in the CA report and is **absent from the archive entirely**. |
| KXUSRETAIL, KXUSDURABLE, KXUSMICHCSP, KXUSISMSERV, KXCPINDEX, KXUSNFP | misc macro | 1–7 | Thin. Capture, but do not build on. |

### 1.2b How much of the metadata re-pull is already local

Checked event-by-event against the full API universe (1,422 events across 26 core series,
598 ex-WTI).

**Event coverage:**

| Source | Events covered (of 1,422) | Carries strike fields |
|---|---|---|
| `data/markets/` parquet | **1,136 (80%)** | No |
| `kalshi_markets_1y_clean.jsonl` | 148 (10%) | **Yes — all 148** |
| Union | 1,152 (81%) | — |
| **Neither** | **270 (19%)** | — |

Ex-WTI: 448 of 598 events covered, 150 missing.

The JSONL adds only **16 events** beyond the parquet, so as a *coverage* asset it is nearly
redundant. Its entire value is the field set on the 148 events it does hold.

**But the parquet turns out to solve §2.3 anyway.** It carries `yes_sub_title` with **zero
nulls**, and that field parses to a numeric threshold for **99% of core markets** — against
**90% for `implied.py`'s ticker regex**:

| Series | Markets | Ticker regex | `yes_sub_title` |
|---|---|---|---|
| KXCPICOREYOY / KXFED / KXU3 / KXCPIYOY / KXCPI / KXPAYROLLS / KXCPICORE / KXGDP / KXPCECORE | 2,988 | 100% | 100% |
| KXADP | 150 | 95% | 100% |
| **KXFEDDECISION** | 195 | **0%** | 86% |
| **KXJOBLESSCLAIMS** | 132 | **0%** | 100% |
| **KXISMPMI** | 49 | **0%** | 100% |
| **TOTAL** | **3,914** | **90%** | **99%** |

### 1.2c Two bugs this surfaces

**(a) `implied.py:parse_threshold` silently drops three series.** `_THRESHOLD_RE` requires a
`-T` prefix. Jobless claims tickers are `KXJOBLESSCLAIMS-25AUG21-225000` — no `T`. So the
regex returns `None`, the market is dropped, and **the entire claims series vanishes from
any implied-distribution calculation without raising anything.** Same for ISMPMI (49
markets). That is the series §3-asset-1 nominates as the small-N fix, so this must be fixed
before it can be used at all.

FEDDECISION's 0% is *correct* behaviour, not a bug — `H0`/`C25` are categorical outcomes,
not a threshold ladder, so it should not go through the numeric PDF path at all. But it
should be **excluded explicitly** rather than by silent parse failure, or the two failure
modes are indistinguishable in the output.

**(b) Inclusive vs. exclusive threshold conventions coexist.** Where both parsers succeed
they agree on 3,516 of 3,531 markets. All 15 disagreements are off-by-one and *both sides
are right*: `PAYROLLS-24FEB-T299999` has subtitle `"300,000 or above"`. The ticker encodes
the exclusive bound, the subtitle the inclusive one. Phrasing varies by series — `"Above
4.3%"` (CPI, exclusive) vs `"At least 225000"` (claims, inclusive) vs `"216,000 or above"`
(early payrolls).

`recover_pdf` takes successive differences of P(X > threshold). Mixing conventions shifts
bin edges by one tick. Harmless for payrolls (one job); **material for CPI, where thresholds
sit 0.1pp apart.** Normalise to one convention at parse time and assert it.

**(c) `resolved_value()` hardcodes `spacing = 0.1`.** In the one-sided cases (all-yes or
all-no ladders) it returns `threshold ± 0.05`. That is right for CPI/U3/GDP, where strikes
sit 0.1 apart — and wrong by orders of magnitude for **PAYROLLS** (strikes ~50,000 apart)
and **JOBLESSCLAIMS** (~5,000 apart). Spacing must be inferred per event from the observed
threshold ladder, not assumed. It also inherits bug (a), so claims and ISMPMI return `None`
here regardless.

**This matters because `resolved_value` is your surprise-factor input.** `data/markets/`
carries `result` (yes/no) per strike but **not** `expiration_value`, so locally you recover
the realised macro print by bracketing it between the highest `yes` and lowest `no` strike —
i.e. to within half a bin. Good enough for most purposes, and it means the surprise factor
(§6.2) is computable from data already on disk. `expiration_value` from the re-pull gives it
exactly, which is worth having but is a refinement, not a blocker.

**Practical consequence:** run the threshold parser against the local parquet *today*, before
any network call. It validates the fix on 3,914 markets at zero cost, and it is a
prerequisite for Phase 1.3 (wiring `implied.py`) regardless of what the re-pull adds.

### 1.3 Metadata upgrades available on re-pull

`/historical/markets` returns fields the current `normalize_markets` discards:

- **`yes_sub_title`** — e.g. `"Above 0.9%"`. A *robust* threshold source; replaces the
  fragile `-T3.2` ticker regex in `implied.py:parse_threshold` (§2.3).
- **`expiration_value`** — e.g. `"1.3"` for `CPI-22JUN`, `"Fed maintains rate"` for
  FEDDECISION. **This is the realised macro print, straight from Kalshi.** It gives you
  the surprise-factor ground truth (§6.2) without needing FRED at all.
- `settlement_ts`, `settlement_value_dollars` — exact resolution timestamps, needed for
  the §5 identification argument (official release vs. Kalshi resolution).
- `rules_primary`, `price_level_structure`, `price_ranges`, `yes_bid_size_fp`.

Note: `historical/markets` does **not** carry `cap_strike`/`floor_strike`/`strike_type` —
those live only on the live `/markets` endpoint. So preserve them going forward from
`/markets`, and parse `yes_sub_title` for the historical backfill.

### 1.4 Candlesticks — harvest this week

`GET /markets/candlesticks` returns OHLC for **both `yes_bid` and `yes_ask`**, plus
`open_interest_fp`, and **quotes update on zero-volume days** — the direct fix for §4.1.

Availability tracks exactly whether the market is still listed on `/markets`. Currently
listed across the 21 core series: **2,560 markets**.

| Series | Live markets | Event periods retrievable |
|---|---|---|
| KXFEDDECISION | 70 | **14** (26JUN … 28JAN) |
| KXWTI | 1,363 | 52 |
| KXCPIYOY | 110 | 5 (26JUN … 26NOV) |
| KXFED | 109 | 8 |
| KXJOBLESSCLAIMS | 99 | 10 |
| KXPAYROLLS / KXU3 | 80 each | 6 each |
| KXCPICOREYOY | 76 | 5 |
| KXUSGASCPI | 75 | 3 |
| KXCPICORE / KXCPI | 66 / 59 | 6 each |
| KXCPINDEX / KXUSPPIYOY | 63 / 58 | 3 each |
| KXGDP | 51 | 6 |
| KXPCECORE | 40 | 7 |
| KXADP | 42 | 6 |
| others | ~120 | 3–4 each |

The 14-period FEDDECISION curve confirms §6.5 — the term structure exists *now* and this
harvest is the only way to get a descriptive figure of it.

**Erosion is measurable:** `KXCPIYOY-26MAY` was listed as retrievable on 2026-08-14 and is
gone on 2026-08-20. Roughly one event period per series per month is disappearing.

---

## 2. Preparation pipeline

### Phase A — Acquisition (do A1 first, today)

**A1. Candlestick harvest** — *time-sensitive, ~2 hours*
- Enumerate live markets via `/markets?series_ticker=` across the 21 core series (2,560).
- Pull `period_interval=1440` (daily) for all; `60` (hourly) for all; `1` (minute) only for
  macro tickers within ±3 days of a scheduled release.
- Batch up to 100 tickers / 10,000 candles per request.
- Land at `data/candles/{series}/{period}/`. Append-only, keyed on `(ticker, end_period_ts)`
  so weekly re-runs self-heal gaps.
- Re-run weekly, opportunistically. No cron needed — candlesticks are historical.

**A2. Market metadata re-pull** — *~1 hour, and smaller than first thought*

Scope it down. `data/markets/` already covers 80% of events and its `yes_sub_title` solves
the strike problem at 99% (§1.2b). What the re-pull is genuinely still needed for:
- the **270 uncovered events (19%)** — chiefly 2026, plus the 2026-only series;
- **`expiration_value` and `settlement_ts`**, absent from both local parquet and (outside its
  148 events) the JSONL. Exact realised values and resolution timestamps — the latter is what
  §5's identification argument needs, and it has no local substitute.

- For every event in the universe (§2.1 below), `GET /historical/markets?event_ticker=`.
  This is far cheaper than the current per-ticker `_fetch_single_market` loop and works on
  delisted events.
- Extend `normalize_markets` to keep `yes_sub_title`, `expiration_value`, `settlement_ts`,
  `settlement_value_dollars`, `rules_primary`, and (from `/markets` for live ones)
  `cap_strike`/`floor_strike`/`strike_type`.
- **Check the JSONL first.** For macro events with `created_time` in 2023-08 → 2026-02 the
  strike/settlement fields are already local (§0). Re-pull only what it does not cover —
  chiefly pre-2023-08 events and everything after 2026-02.
- This supersedes `data/markets/` for quote purposes: those columns are per-fetch snapshots
  (`_fetched_at` 2025-11-23 → 2026-03-31), not a time series.

**A3. Trade backfill** — *a few hours, resumable*
- Reuse `fetch_all_econ_trades` as-is; its checkpointing already handles this.
- Ticker list = every market from A2 not already in the archive, i.e.:
  - all 2026 events across every core series,
  - KXJOBLESSCLAIMS / KXADP / KXISMPMI / KXUSPPIYOY / the new CPI subcomponents (all years),
  - KXPAYROLLS 2023 and KXPCECORE 2022–2025 (archive gaps).
- Estimated ~10–14K tickers at ~15 markets/event. Do **not** re-pull tickers already in the
  archive; verify overlap on `trade_id` for a sample instead.

### Phase B — Canonicalisation

**B1. Series alias table.** Kalshi renamed things mid-sample and the archive straddles it.
Needs an explicit mapping, not a `str.replace("^KX","")`:
`CPISHELTER → KXSHELTERCPI`, `CPIGAS → KXUSGASCPI`, `CPIUSEDCAR → KXUSEDCARCPI`,
`JOBLESS → KXJOBLESSCLAIMS`, plus the blanket `KX` prefix. Treat a rename as a **series
break**, not a continuation, until you have checked the contract rules match.

**B2. Strike extraction.** Parse `yes_sub_title` (`"Above 4.3%"` → `4.3`, `greater`), fall
back to the ticker regex, and **assert agreement** where both are available. This is the
§2.3 fix and it must land before `implied.py` is wired.

**B3. Event calendar.** One row per event: `event_ticker`, series, resolution timestamp
(`settlement_ts`), `expiration_value`, official release timestamp (BLS/BEA — deferrable),
n markets, total volume. This is the spine everything else joins to.

### Phase C — Panels

Build four, all keyed on `event_ticker`:

- **C1 — ticker-day OHLCV.** Refactor `io/kalshi.py:build_daily` so forward-fill is *not*
  baked in: emit `close_raw` (null on no-trade days), `close_ffill`, and `is_filled`.
  Today the fill is unconditional and `volume==0` is the only trace of it, which makes the
  §4.1 diagnostic awkward. Also emit VWAP alongside last-trade close (bid-ask bounce, §4.2).
- **C2 — trade-time panel.** Sequence-indexed rather than date-indexed: next-N-trades
  horizons. Needed for the §4.1 robustness check, which the plan says to run *first*.
- **C3 — event-day implied distribution.** Route through `implied.py:build_daily_implied_means`
  → `implied_mean/std/skew/kurtosis/entropy`. This replaces the VWAP-across-strikes collapse
  in `aggregate_to_event_level`, which volume-weights prices across thresholds that mean
  different things.
- **C4 — quote panel** from A1 candlesticks: true mid, effective spread, open interest.
  2026 only; used as the §4.1/Phase-3 robustness subsample.

### Phase D — Storage

Repartition from 3,908 flat files to `data/trades/series={S}/year={Y}/month={M}/`.
Every analysis currently scans all 1.8 GB to touch a handful of macro series. One-off cost,
pays back immediately.

---

## 3. IS / OOS split

### 3.1 Proposed split — chronological, event-level

Split on **trigger event resolution date**, never on rows. An event contributes 15–25
markets and 30+ days of panel; all of it must sit on one side.

| Split | Window | Approx. core events | Role |
|---|---|---|---|
| *Burn-in* | 2021-06 → 2021-12 | ~19 | **Excluded from estimation.** ZIRP regime, 86K trades all year, most series absent. Keep for descriptive stats only. |
| **Train (IS)** | 2022-01 → 2025-03 | ~250 | Stage-1 structure estimation, feature construction, all normalisation statistics. |
| **Validation** | 2025-04 → 2025-12 | ~105 | Ablation-ladder model selection, hyperparameters, threshold tuning. Touch freely. |
| **Test (OOS)** | 2026-01 → 2026-08 | ~145 | **Touch once.** Everything the re-pull adds. |

~533 core macro events total; OOS ≈ 27% of the usable sample.

### 3.2 Why this cut, specifically

The OOS boundary is not arbitrary — it is **the date your archive was collected**. Every
methodological choice in the CA report and in `research_summary.md` was made looking at
data ending 2025-12-30. The 2026 block is therefore genuinely unseen, in the strong sense
that no specification search could have touched it. That is a claim worth making explicitly
in the write-up, and it is rare enough in applied ML finance to be worth a sentence of its own.

It also happens to be where the new series live: KXJOBLESSCLAIMS (33 events), the CPI
subcomponents, ADP, PPI. Which cuts both ways — see 3.5.

### 3.3 Leakage controls

- **Purge gap.** The event study runs [−3, +14] days, so a trigger resolving within 14 days
  of a boundary has its label window straddling the split. Drop triggers in a **21-day
  purge band** either side of each boundary (14-day window + slack). Costs ~3 events per
  boundary; cheap insurance.
- **Normalisation fit on train only.** `aggregate_to_event_level` z-scores per event, which
  is contained. But any *cross-event* standardisation, any edge-weight estimate, and the
  Stage-1 effect sizes themselves must be fit on train and applied frozen to val/test.
- **Multiple-testing correction (§9 Phase 1.1) is fit on train.** The FDR threshold and the
  22 surviving pairs are a *training-set* result. Re-running discovery on the pooled sample
  and then reporting OOS performance is the classic version of this mistake.
- **Semantic edges.** `KalshiSemanticTopicEdges` embeds market titles with a
  sentence-transformer pretrained on a corpus with no date boundary. Not fixable, but state
  it — it is a mild, disclosable form of lookahead.
- **Alias mapping is a leakage surface.** Deciding `CPISHELTER ≡ KXSHELTERCPI` because the
  series look correlated in 2026 would be fitting the split boundary. Decide it from
  contract rules (`rules_primary`), on train, before looking at 2026.

### 3.4 Two honest caveats

**Regime confound.** The OOS block is a single contiguous 8-month macro regime. If 2026
differs structurally from 2022–2025 (rate path, inflation level, Kalshi's own volume growth),
OOS degradation confounds regime change with overfitting. Mitigate by *also* reporting a
**rolling-origin evaluation** on the train+val period — expanding-window refits, e.g.
6-month test folds from 2023-01 — as the primary generalisation estimate, with the 2026
block as the single clean holdout on top. Report both; they answer different questions.

**Universe drift.** Several series exist only in the OOS block. A model trained without
jobless claims cannot be tested on them. Resolve it by defining **two universes**:

- *Universe A (balanced)* — series present across the whole 2022→2026 span: CPI, CPIYOY,
  CPICORE, CPICOREYOY, FED, FEDDECISION, PAYROLLS, U3, GDP, WTI. This is the split above,
  and the headline result.
- *Universe B (extended)* — adds JOBLESSCLAIMS, PCECORE, ADP, ISMPMI, PPI, CPI subcomponents.
  2025-08 onward only, so it gets its own shorter split (train 2025-08→2026-03, test
  2026-04→2026-08) and is reported as a **secondary, small-sample** result. Weekly claims
  make it worthwhile despite the short span; do not let it carry the headline.

---

## 4. Suggested order

1. **A1 candlestick harvest** — today. Losing ~1 event period/series/month.
2. **A2 metadata re-pull + B2 strike extraction** — unblocks `implied.py` wiring (Phase 1.3).
3. **A3 trade backfill** — runs unattended; start it and move on.
4. **D repartition**, then **C1 + C2 panels**.
5. **§9 Phase 1.2, the trade-time robustness check, on train only.** The summary is right
   that this comes first among analyses: if the drift does not survive removal of
   forward-filled rows, the thesis changes — and it is now August.
6. Freeze the split definition to a config file *before* step 5, and do not revisit it.

## 5. Open questions for you

- **Do you want WTI in Universe A?** 824 events dwarfs everything else and would dominate
  any pooled estimate. It is a genuine trigger (WTI→CPI is one of your 22 pairs), but it
  probably needs its own weighting or a cap on events per series.
- **Is the 2026 regime acceptable as the holdout,** or would you rather hold out
  2025-07→2025-12 and use 2026 as a second, later test? The former is cleaner; the latter
  is more conservative if you expect 2026 to be structurally odd.
