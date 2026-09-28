# Rerun on the backfilled archive (2026-09-28)

*The key structural, lead-lag and graph-model studies, rerun unchanged on the
in-sample archive after the 2026-09-28 backfill. Nothing in the code or
specification changed; only the data did. In-sample only (pre-2026); no OOS
data was read. This is the reference for every "rerun 2026-09-28" note in the
other reports.*

## What changed in the data

`scripts/fetch_is_backfill.py` asked Kalshi for every in-sample leg on the
registered macro series (except WTI, WTIW and the equity indices) that traded
but was missing or short in `data/trades/`:

| | |
|---|---|
| legs fetched | 220 (176 with no local trades, 44 partial) |
| new trades | **5,573** (`data/trades/trades_backfill_2026_09.parquet`), created 2022-03 → 2025-04 |
| by series | PAYROLLS 5,085 (12 events, 2022–23) · CPIFOOD 261 · CPIAPPAREL 152 · CPIUSEDCAR 38 · JOBLESSCLAIMS 27 · CPISHELTER 10 |
| new market metadata | 39 legs absent from `data/markets/` (`markets_backfill_2026_09.parquet`) |
| settlement values | Kalshi's printed `expiration_value` for **172 more events** (CPI components, legacy JOBLESS, 2022 PAYROLLS), read by `load_settlement_values`; agrees with the earlier pull on all 356 shared events |

Three effects follow, and the rerun does not separate them:
1. **More PAYROLLS triggers and targets**, which feed the labour → policy
   channel, the strongest channel in the project.
2. **True resolved values replacing ladder-inferred ones** for 172 events,
   which moves those events' surprises.
3. **CPIFOOD and CPIAPPAREL now clear the 10-event trigger minimum**, so every
   grid gains pairs.

The gated surprise panel went from **519 to 538 rows**. The node panel now has
6,502 rows over 933 dates. `artifacts/panels/MANIFEST.md` has the per-series
gate table.

## Results, before → after

### Structure (Stage 1, channels, consensus)

| study | before | after |
|---|---|---|
| Stage-1 grid (`run_structure_estimation.py`) | 141 pairs, 13 nominal (7.1 expected) | **157 pairs, 14 nominal (7.9 expected)** |
| BH survivors, q = 0.10 | 4 | **the same 4** |
| CPI → FED | n 36, ρ +0.617, p 0.0005 | unchanged |
| **PAYROLLS → FED** | n 30, ρ +0.583, p 0.0010 | **n 33, ρ +0.596, p 0.0005** |
| PAYROLLS → FEDDECISION (hike) | n 26, ρ +0.596, p 0.0005 | unchanged |
| CPICOREYOY → FEDDECISION (cut) | n 17, ρ −0.683, p 0.0025 | unchanged |
| sign agreement among survivors | 77.3% vs null 52.7%, p 0.0003 | 78.2% vs 53.1%, p 0.0003 |
| nominal edges lost | – | PCECORE → CPICOREYOY (ρ −0.74 → −0.38, p 0.25) |
| nominal edges gained | – | PAYROLLS → CPIFOOD (ρ −0.77, n 10; **against** the theory sign) |
| mediation | every partial within ±0.04 | unchanged (PAYROLLS→FED \| U3: −0.007) |
| channel pooling, data → policy (`channel_pooling.py`) | 598 rows, 63.2%, p 0.0004 | **672 rows, 61.0%, p 0.0006** |
| — labour → policy | 204 rows, 66.1%, p 0.0004 | 207 rows, 66.9%, p 0.0002 |
| — inflation → policy | 369 rows, 62.7%, p 0.0038 | 440 rows, 59.0%, p 0.015 (the new CPI components dilute it) |
| consensus surprise (`consensus_surprise.py`) | 93 pairs; corr(ρ mkt, ρ cons) +0.904; BH 3 / 1 | 96 pairs; **+0.909**; BH 3 / 1 (PAYROLLS→FED survives both) |

**Reading:** the structural claim is unchanged, and slightly stronger on the
labour → Fed edge that got most of the new data. Adding the two thin CPI
components grows the search and dilutes the inflation → policy pool without
adding an edge.

### Lead-lag to settlement (`analysis/leadlag_2026_09/`)

| | before | after |
|---|---|---|
| panel | 10,714 legs, 262 trigger / 401 target events, 135 pairs | **11,884 legs, 293 / 429, 149 pairs** |
| signal predicts settlement (top − bottom tercile) | +3.80pp, CI [+0.41, +7.32], p 0.0005 | **+3.71pp, CI [+0.61, +6.89], p 0.0020** |
| region with content | 10–75c | 10–75c (plus 1–5c, +1.22pp, p < 0.01) |
| falsification cell (−1 block, raw z) | −4.48pp, CI [−9.11, −0.14] | −3.90pp, CI [−8.26, +0.14] |
| walk-forward net, all legs (`economics.py`) | gross +2.36c, net +0.81c, CI [−1.29, +3.08] | gross ≈ +2.5c, **net +0.90c, CI [−0.96, +2.87]** |
| — by year | 2023 +6.46 · 2024 +1.55 · 2025 −0.35c | **2023 +3.26 · 2024 +2.00 · 2025 −0.04c** |
| learned vs imposed (`relations.py`) | imposed +0.86 / channel +0.47 / pair −0.07pp | **+0.69 / +0.38 / −0.17pp** (same ordering) |
| BH channels | labour→labour, inflation→inflation, labour→policy | same three |
| **BH pairs** | 0 of 88 | **1 of 99: PAYROLLS → FED**, +1.79pp, CI [+0.57, +3.10], p 0.0005 |
| maker execution (`maker_fill.py`) | no maker arm beats the taker | unchanged (best arm P(≤0) 0.14, zero fee) |
| confirmation filter, k = 2c, by year | 2023 +11.35 · 2024 +4.12 · 2025 +2.04c | **2023 +3.59** · 2024 +4.69 · 2025 +2.06c |
| filter's marginal value by year | +6.38 / +2.46 / +2.66c | +1.01 / +2.62 / +2.51c |
| confirmation, walk-forward threshold | +3.03c, P(≤0) 0.054, n 1,190 | +3.30c, P(≤0) 0.087, n 933 |

### The frozen specification (`strategy_spec.md`)

| tape / tie-break | before | after |
|---|---|---|
| committed panel (`ablation.py`, `blotter.py`) | +5.41c, n 889, CI [+0.59, +10.21] | **+4.45c, n 988, CI [−0.20, +8.90]** |
| deduped, last fill at the instant (`tie_handling.py`) | +4.71c, CI [+0.37, +9.12] | **+3.43c, CI [−1.10, +7.78]** |
| deduped, VWAP of the instant | +4.66c, CI [+0.03, +9.39] | **+4.39c, CI [+0.02, +8.54]** |
| VWAP + median-of-3 confirmation (`spec_v2.py`) | +4.10c, CI [−1.34, +9.56] | +3.72c, CI [−1.51, +8.91] |
| permuted-signal null through the whole spec | +1.65c | +0.65c |
| **signal's marginal contribution** | **+3.76c** | **+3.79c** |
| perfect-foresight ceiling | +16.50c | +16.73c |

**Reading:** the signal is worth what it was: its marginal contribution over
the apparatus is unchanged. The apparatus itself earns less, because the
filters' null fell from +1.65c to +0.65c. The ~+4.7c headline was already down
to "~+4.7c under two tie-break rules". It is now **+3.4c to +4.4c**
depending on the rule, and only the VWAP rule's CI excludes zero, by 0.02c.
Most of the loss is in **2023**, which gained the most rows: the confirmation
trade's 2023 result falls from +11.35c to +3.59c, so the "positive in every
year, and decaying" shape becomes roughly flat at +2 to +5c.

The pre-registered OOS expectation in `strategy_spec.md` §9 ("lower than
~+4.7c; +2 to +4c with a CI including zero would be consistent") is not
changed. The in-sample reference it was anchored to is now +3.4 to +4.4c. The
OOS test has not been run.

### Propagation (`analysis/propagation_2026_09/`)

| | before | after |
|---|---|---|
| panel | 31,514 rows, 222 trigger events | 32,230 rows, 225 trigger events |
| labour → policy, d = 0, next / 2–3 / 3–4 / 4–6 months | +0.3 / +3.0 / +2.6 / +3.5pp | **+0.5 / +4.2 / +3.8 / +5.2pp** (p 0.15 / 0.00 / 0.02 / 0.01) |
| 2–3m, 10–75c legs, d = 0 / 1 / 7 / 30 | +14.3 / +11.8 / +4.0 / −0.4pp | **+16.9 / +15.0 / +5.7 / +2.2pp** |
| with all three controls | +11.7pp, CI [+1.5, +20.7] | **+14.0pp, CI [+3.3, +23.7]** |
| far-meeting rule, net (`economics.py`) | +6.24c, CI [+0.81, +13.27], perm p 0.008 | +6.24c (unchanged), perm p **0.022** |
| always-YES control | +6.30c | +6.30c (unchanged) |

**Reading:** the horizon profile is stronger, and still gone within a week.
The costed rule selects the same positions, so its P&L is identical. Only the
permutation null moved, because the pool of surprises it shuffles changed. The
rule still does not beat always-YES.

### Graph models on the event-time panel (`analysis/event_time_2026_09/`)

| | before | after |
|---|---|---|
| panel | 154 instants × 17 series, 142 edges | **157 instants**, 158 edges (27 BH-channel) |
| zero-param rule, labour → policy (imm / settle) | 0.704 / 0.661 | unchanged |
| zero-param rule, 3 BH channels (imm) | 0.634 | 0.637 |
| best linear, R² vs zero (imm / settle) | +0.005 / +0.008 (econ signal) | +0.007 / +0.006 (econ signal) |
| best AGCRN, R² vs zero (imm / settle) | −0.013 / −0.000 (frozen BH) | −0.027 / **+0.002** (adaptive / frozen BH) |
| any graph model beats linear on any metric (`metrics.py`) | no | **no**; several AGCRN ablation arms now significantly *below* linear on balanced accuracy |
| STG vs its parts (`ablation.py`) | no part helps beyond noise | unchanged |
| unsigned AGCRN, balanced sign accuracy (all edges) | 0.50–0.53 | 0.48–0.58 |
| gross, linear own state (`returns.py`) | +5.20c | +5.08c |
| Bayes hierarchical, hard sign, R² (imm) (`bayes.py`) | +0.0071 vs linear one-slope +0.0051 | +0.0079 vs +0.0066 |
| Bayes, edges supported (P ≥ 0.95) | 3 of 142: PAYROLLS→FED, CPICORE→FED, CPIUSEDCAR→PAYROLLS | **4 of 158**: PAYROLLS→FED, CPICORE→FED, CPI→FED, PCECORE→GDP (≈ 8 expected by chance) |
| Bayes channels with 90% CI above 0 | labour → policy | labour → policy (+0.199), inflation → growth (+0.141, lower bound +0.007) |

**Reading:** the STG answer is unchanged on every axis. With the extra data
the Bayesian graph's supported edges are now all but one on the policy hub.

## Not rerun

Of the studies cited in `update_2026_09.md`, these were **not** rerun on the
backfilled archive:
- `stage1_variants.py` (item 1b): about 40 min;
- `magnitude_xval.py`, `family_collapse.py`, `pit_calibration.py`;
- `quantile_2026_09/`, `arbitrage_2026_09/`, `settlement_dist_2026_09/`;
- `event_time_2026_09/` `backtest.py`, `exits.py`, `nonlinear.py`,
  `scoped.py`, `inversion.py`, `temporal.py`;
- `spillover_2026_09/`, `intraday_2026_09/`, `recovery_2026_09/`;
- the leadlag `liquidity*`, `exit_rules`, `sizing` and `leakage_audit` scripts.

Their reports still describe the pre-backfill archive. The backfill added no
WTI data and only thin CPI-component data, so the WTI-wing, arbitrage and
quantile results should be unaffected. The rest would move by about as much as
the studies above.

## Reproduce

    venv/bin/python scripts/build_panels.py
    venv/bin/python scripts/run_structure_estimation.py
    # then each study's README run order: leadlag_2026_09, relations_2026_09
    # (channel_pooling, consensus_surprise), propagation_2026_09,
    # event_time_2026_09 (build_panel, models, ablation, metrics, returns, bayes)

Captures are in each study's `out/`, which is untracked (`analysis/README.md`).
