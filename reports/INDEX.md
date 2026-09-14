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

## Findings and measurement

| Document | What it is |
|---|---|
| [research_log.md](research_log.md) | Running findings and measurement problems, by section (§1 magnitude nulls, §2 sign survives, §5 first-print decay, §6 structure estimation, §7 staleness, §8 bucket parse bug, **§12 data-quality caveat**, **§13 horizon/cost and the tradability ceiling**, **§14 corrections: true settlement values, five statistics bugs, a look-ahead in instrument choice**). Most-cited document in the set. |
| [agcrn_study.md](agcrn_study.md) | **Why AGCRN did not work** — thesis-facing writeup: question, setup, result table, five reasons, conclusion. |
| [agcrn_postmortem.md](agcrn_postmortem.md) | The diagnostic evidence behind those five reasons — horizon comparison, oracle R² ladder, the overlapping-window artifact. *(Was `analysis/agcrn_diagnostics_2026_09/FINDINGS.md`.)* |
| [edge_economics.md](edge_economics.md) | **Do the edges make sense, and can they be traded?** Sign restrictions (incl. the U3 dovish-flip falsification cell), the WTI-has-no-information-event finding, the jump-not-drift decomposition (which clears §4.1's staleness threat), and why pre-resolution coherence is the one remaining tradable path. |
| [direction_study.md](direction_study.md) | **The simple-learning successor** — dormant-horizon direction prediction on the Stage-1 structure. **Result is a null** (rewritten 2026-09-14): the sign rule clears no gate (58.7%, p=0.232 at `p05`), and the rung that appears to win is reading the target's bounded price level, not the surprise. The earlier 63.0% / p=0.041 claim did not survive the §14 corrections. |

## Design and data

| Document | What it is |
|---|---|
| [graph_definition.md](graph_definition.md) | The single authoritative STG definition — node, edge, snapshot, label. Settled 2026-09-02. |
| [data_prep_plan.md](data_prep_plan.md) | Data pipeline plan, Phases A–D, source-by-source. Phases B/C/D still partly open. |
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
