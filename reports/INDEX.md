# Reports index

Every hand-written report and analysis document for the FYP lives in this
directory. Generated outputs (written by scripts) stay next to their `.parquet`
siblings in `artifacts/` and are linked below.

**Path convention.** Bare `.md` names refer to siblings in this directory.
Every other path is relative to the **repo root** (`event-pred/`, the parent of
this directory), e.g. `stg_infra/stg/structure/`, `analysis/exploratory_2026_08/`.
The older documents additionally use package-relative shorthand for code
(`implied.py`, `events/config.py`, `pairs/config.py`) — those are relative to
the `stg` package at `stg_infra/stg/`. The CA report PDFs stay outside the repo
at the FYP root.

Last reorganised 2026-09-07 — `reports/` and `analysis/` moved inside the repo,
so everything is version-controlled under one root. `direction_study.md` added
the same day.

Submission target November 2026; analysis freeze early October.

---

## Start here

| Document | What it is |
|---|---|
| [TODO.md](TODO.md) | Consolidated task list and open decisions. The current state of play. |
| [research_summary.md](research_summary.md) | The standing plan — research question, four components, phases. The master document. |
| [update_2026_08.md](update_2026_08.md) | Supervisor-facing: seven proposed refinements to the CA report design, with motivation and supporting measurement. |
| [progress_2026_09.md](progress_2026_09.md) | **Supervisor-facing progress report, CA submission (2026-08-12) → 2026-09-14.** What changed about each of the thesis's claims, what was retracted and why, the open decisions, and a risk register for the final report. Start here for the current state; `update_2026_08.md` covers only the first four weeks of that period. |

## Findings and measurement

| Document | What it is |
|---|---|
| [research_log.md](research_log.md) | Running findings and measurement problems, by section (§1 magnitude nulls, §2 sign survives, §5 first-print decay, §6 structure estimation, §7 staleness, §8 bucket parse bug, **§12 data-quality caveat**, **§13 horizon/cost and the tradability ceiling**, **§14 corrections: true settlement values, five statistics bugs, a look-ahead in instrument choice**). Most-cited document in the set. |
| [agcrn_study.md](agcrn_study.md) | **Why AGCRN did not work** — thesis-facing writeup: question, setup, result table, five reasons, conclusion. |
| [agcrn_postmortem.md](agcrn_postmortem.md) | The diagnostic evidence behind those five reasons — horizon comparison, oracle R² ladder, the overlapping-window artifact. *(Was `analysis/agcrn_diagnostics_2026_09/FINDINGS.md`.)* |
| [edge_economics.md](edge_economics.md) | **Do the edges make sense, and can they be traded?** Sign restrictions (incl. the U3 dovish-flip falsification cell), the WTI-has-no-information-event finding, the jump-not-drift decomposition (which clears §4.1's staleness threat), and why pre-resolution coherence is the one remaining tradable path. |
| [direction_study.md](direction_study.md) | **The simple-learning successor** — dormant-horizon direction prediction on the Stage-1 structure. **Result is a null** (rewritten 2026-09-14): the sign rule clears no gate (58.7%, p=0.232 at `p05`), and the rung that appears to win is reading the target's bounded price level, not the surprise. The earlier 63.0% / p=0.041 claim did not survive the §14 corrections. |
| [leadlag_findings.md](leadlag_findings.md) | **Lead-lag to settlement, resolved by entry price (2026-09-15).** From-scratch rebuild against a settlement target and the **full target ladder**. The signal predicts settlement (+3.80pp top-vs-bottom tercile, block-permutation **p = 0.0005**), the content is confined to **10-75c** and absent above 75c, and it earns **+2.36c gross against 0.00c for a permuted signal** — but friction is 1.55c, net +0.81c has a clustered CI straddling zero, and it decays 2023 -> 2025. **Learning the relations is worse than imposing them** — per-pair (135 params) scores −0.07pp against +0.86pp for the zero-parameter economic sign, and 0 of 88 pairs survive BH; the 3 surviving channels (labour→labour, inflation→inflation, labour→policy) are economically coherent while CPI→FED is flat. **Partly retracts Addendum 2** (§5). |
| [quantile_findings.md](quantile_findings.md) | **Reconstruction-free ladder statistics (2026-09-15)** — quantile moments read off the traded ladder instead of integrating a recovered pdf. **Retracts `relations_findings.md` item 1's headline**: the ladder PIT is uniform (mean 0.42-0.56 against item 1's 0.65-0.91), so the "macro ladders biased low" result was `recover_pdf`'s open tails, and `settlement_trade.py` was right. Also gives the response vector a clean form: A's resolution shifts B's median by **+0.0428 IQR** (CI [+0.0121, +0.0741]) and leaves B's **width and skew unchanged**. Fixes a CI error in three earlier group-difference bootstraps. |
| [arbitrage_findings.md](arbitrage_findings.md) | **Four arbitrage-shaped tests (2026-09-15)** — cross-market structure without forecasting, chosen because a coherence violation is a per-observation fact rather than an `n`-limited estimate. The MoM/YoY identity is **not** arbitrage (12.8% break rate from independent BLS rounding; net −1.07c); ladder monotonicity fails on **~2%** of adjacent pairs, flat across synchronicity tiers but median 2.0c against a 2.3c cost; implied uncertainty decays ~14% into a release with **no systematic variance premium**; the cross-event uncertainty effect dies under a term-structure control. **Withdraws the 124c bucket-overround claim** in `settlement_distribution_findings.md`. |
| [settlement_distribution_findings.md](settlement_distribution_findings.md) | **Wing calibration — the gate result (2026-09-14).** The macro-release wings are **fair** (−0.13c, P(≤0)=0.55), which retires the plan's motivating premise and is a third independent efficiency result. The entire wing mispricing is in **WTI's bucket ladders** (−6.6pp, +4.96c, 93% of 397 events positive) and survives overround, missing-winner, concentration and single-series kills — but it is **decaying** (2022 +5.81c → 2024 +2.43c). Reads as a σ result, not a forecasting one, and independently confirms the WTI half of `relations_findings.md` item 1. |

## Design and data

| Document | What it is |
|---|---|
| [graph_definition.md](graph_definition.md) | The single authoritative STG definition — node, edge, snapshot, label. Settled 2026-09-02. |
| [data_prep_plan.md](data_prep_plan.md) | Data pipeline plan, Phases A–D, source-by-source. Phases B/C/D still partly open. |
| [settlement_distribution_plan.md](settlement_distribution_plan.md) | **Design doc, not findings (2026-09-14)** — the successor direction: retire surprise as the primitive and predict the *settlement statistic* instead. A predictive distribution `p̂ = Φ(−z)`, `z = (K − μ̂)/σ̂`; why the σ edge pays in the wings where fees are half and longshot bias points the same way; the ladder-coverage defect that must be fixed before any σ claim is meaningful; and a gating first experiment that needs no pdf reconstruction at all. |
| [relations_study_plan.md](relations_study_plan.md) | **Design doc, not findings (2026-09-14)** — better surprise measures (PIT/quantile surprise, surprisal), richer edge metrics (order-flow and distributional responses, uncertainty and attention channels, state-conditional ρ), and structure beyond the pair (release-vector triggers, the PAYROLLS−U3 double trigger, CPI-family collapse, channel pooling as a block model). Ends with a suggested order and where it cuts across `TODO.md`. |

---

## Generated artifacts (not moved — scripts write these)

Regenerate by running the script from the repo root:

| Artifact | Produced by |
|---|---|
| `artifacts/adjacency_report.md` | `scripts/run_structure_estimation.py` — Stage-1 edges, 4 BH-FDR survivors (was 8 before the §14 corrections; all 4 are non-same-release) |
| `artifacts/agcrn_report.md` | `scripts/train_agcrn.py` — walk-forward results, 0 models beat predict-zero |
| `artifacts/direction_report.md` | `scripts/run_direction_study.py` — Stage-2 ladder, per-fold edges, robustness cuts |
| `artifacts/edge_economics.md` | `scripts/run_edge_economics.py` — sign restrictions, effective spreads, the trade ledger |
| `artifacts/adjacency_comparison.md` | `scripts/compare_adjacency.py` — learned Ã vs Stage-1 adjacency |
| `artifacts/panels/MANIFEST.md` | `scripts/build_panels.py` — panel provenance |

## Code-adjacent documents (left with their code)

| Document | Why it stays |
|---|---|
| `README.md` | The repo's own README. |
| `data/MANIFEST_new_pulls.md` | Manifest describing the files it sits beside. |
| `analysis/exploratory_2026_08/README.md` | Index of the exploratory scripts in that directory. |
| `analysis/agcrn_diagnostics_2026_09/` | `postmortem.py` + captured output; its writeup is `agcrn_postmortem.md` here. |
| `analysis/data_quality_2026_09/` | `coverage_audit.py` + captured output; its writeup is research_log.md §12. |
| `analysis/quantile_2026_09/` | Quantile-moment scripts + captured output; its writeup is `quantile_findings.md`. Module is `stg/events/quantile.py`. |
| `analysis/arbitrage_2026_09/` | Identity, coherence, vol-term-structure and dispersion scripts + captured output; its writeup is `arbitrage_findings.md`. Has its own README. |
| `analysis/leadlag_2026_09/` | Lead-lag-to-settlement scripts + captured output; its writeup is `leadlag_findings.md`. Has its own README with run order. |
| `analysis/settlement_dist_2026_09/` | Wing-calibration scripts + captured output; its writeup is `settlement_distribution_findings.md`. Has its own README with run order. |
| `analysis/horizon_2026_09/` | Horizon/cost scripts + captured output; its writeup is research_log.md §13. Has its own README with run order. |

---

## Tradability: settled

`research_log.md` §13 closes the question §6.4 left open. The recovered
structure is **economically significant and not economically exploitable**: the
repricing completes at the target's first post-resolution print (median 6.2 min),
leaving ~1c against a ~2.6c cost floor. Holding longer does not help — there is
no drift to hold. Two configurations that appeared to work (PAYROLLS→FED, and
buying against the odds) were each a few events wearing a large `n`; **§13.5
clusters on `target_event` and neither survives.** Cite clustered figures only.

---

## Data-quality caveat

`data/trades/` is a **convenience sample**, not a complete archive — it was
fetched one ticker at a time from a list that was never complete, so 350 series
have no trades at all and several *registered* triggers run on a third to
two-thirds of their events (WTIW 32%, CPIFOOD 38%, WTI 64%, CPIAPPAREL 68%,
PAYROLLS 72%). A small `n` anywhere in these reports is a floor set by
collection, not a measurement of market activity. Full diagnosis, affected
results and remedies: **research_log.md §12**. Earmarked for a limitations
section in the final report.

---

## Scope reminder

All analysis is **in-sample only** (pre-2026). The 2026 block is held out and
untouched — including for exploratory or feasibility checks. See
`research_summary.md` and `stg_infra/stg/splits.py` (`assert_no_oos()`).
