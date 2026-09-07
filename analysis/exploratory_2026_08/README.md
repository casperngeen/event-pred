# exploratory_2026_08 — frozen

These scripts are the **research record** for the August 2026 measurement work
(the sign result, the spread/friction measurements, the AGCRN complexity
accounting). They are kept verbatim and are **not maintained** — several import
from a scratchpad directory that no longer exists.

The validated logic has been promoted into the package and is what new work
should build on:

| exploratory script | promoted to |
|---|---|
| `liquid_window.py::pdf_surprise` | `stg.panel.surprise.build_surprise_panel` (threshold path) |
| `bucket_surprise.py::build_bucket_surprise` | `stg.panel.surprise.build_surprise_panel` (bucket path) |
| `sign_diagnostics.py::reps_for` | `stg.panel.targets.representative_tickers` |
| `sign_diagnostics.py` / `structure_discovery.py` response windows | `stg.panel.targets.response_panel` |
| `structure_discovery.py` §1–2 (grid + BH + permutation) | `stg.structure.estimate_adjacency` |
| `structure_discovery.py` §3 (mediation) | `stg.structure.test_mediation` |
| `sign_diagnostics.py` block-permutation null | `stg.structure.stats.block_permutation_sign_p` |
| `agcrn_complexity.py` T1 (param count) | `stg.models.capacity.agcrn_param_count` (exact, supersedes the approximation) |
| `agcrn_complexity.py` T2 (ICC / effective n) | `stg.models.capacity.within_snapshot_icc` |
| `agcrn_complexity.py` T5 (linear complexity ladder) | `stg.models.baselines` + `stg.models.train.run_linear` |

Reproduce from `data/` alone:

```
cd event-pred
venv/bin/python scripts/build_panels.py
venv/bin/python scripts/run_structure_estimation.py   # -> artifacts/adjacency_report.md
venv/bin/python scripts/train_agcrn.py                # -> artifacts/agcrn_report.md
venv/bin/python scripts/compare_adjacency.py          # -> artifacts/adjacency_comparison.md
```
