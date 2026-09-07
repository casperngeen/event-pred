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
so everything is version-controlled under one root.

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
| [research_log.md](research_log.md) | Running findings and measurement problems, by section (§1 magnitude nulls, §2 sign survives, §5 first-print decay, §6 structure estimation, §7 staleness, §8 bucket parse bug). Most-cited document in the set. |
| [agcrn_study.md](agcrn_study.md) | **Why AGCRN did not work** — thesis-facing writeup: question, setup, result table, five reasons, conclusion. |
| [agcrn_postmortem.md](agcrn_postmortem.md) | The diagnostic evidence behind those five reasons — horizon comparison, oracle R² ladder, the overlapping-window artifact. *(Was `analysis/agcrn_diagnostics_2026_09/FINDINGS.md`.)* |

## Design and data

| Document | What it is |
|---|---|
| [graph_definition.md](graph_definition.md) | The single authoritative STG definition — node, edge, snapshot, label. Settled 2026-09-02. |
| [data_prep_plan.md](data_prep_plan.md) | Data pipeline plan, Phases A–D, source-by-source. Phases B/C/D still partly open. |

---

## Generated artifacts (not moved — scripts write these)

Regenerate by running the script from the repo root:

| Artifact | Produced by |
|---|---|
| `artifacts/adjacency_report.md` | `scripts/run_structure_estimation.py` — Stage-1 edges, 8 BH-FDR survivors |
| `artifacts/agcrn_report.md` | `scripts/train_agcrn.py` — walk-forward results, 0 models beat predict-zero |
| `artifacts/adjacency_comparison.md` | `scripts/compare_adjacency.py` — learned Ã vs Stage-1 adjacency |
| `artifacts/panels/MANIFEST.md` | `scripts/build_panels.py` — panel provenance |

## Code-adjacent documents (left with their code)

| Document | Why it stays |
|---|---|
| `README.md` | The repo's own README. |
| `data/MANIFEST_new_pulls.md` | Manifest describing the files it sits beside. |
| `analysis/exploratory_2026_08/README.md` | Index of the exploratory scripts in that directory. |
| `analysis/agcrn_diagnostics_2026_09/` | `postmortem.py` + captured output; its writeup is `agcrn_postmortem.md` here. |

---

## Scope reminder

All analysis is **in-sample only** (pre-2026). The 2026 block is held out and
untouched — including for exploratory or feasibility checks. See
`research_summary.md` and `stg_infra/stg/splits.py` (`assert_no_oos()`).
