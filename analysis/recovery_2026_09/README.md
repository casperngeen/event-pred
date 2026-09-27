# Recovery test, 2026-09

Plants the imposed economic graph (25 signed, directed edges on the 3 BH
channels) in semi-synthetic data built from the real event-time panel. Then
asks each learner to find it, as the calendar lengthens from the real 154
instants to 30×. Writeup: `reports/recovery_test.md`. Needs
`analysis/event_time_2026_09/out/event_nodes.parquet`.

    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 1 3 --reps 10 --tag a   # ~15 min
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 10 --reps 4 --tag b     # ~1.5 h
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 30 --reps 2 --tag c     # ~3 h
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 3 10 --rhos 0.6 --reps 3 --truth pos --tag pos
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 1 3 10 --rhos 0 0.4 0.6 --reps 3 \
        --learners agcrn_z signed_z --no-baselines --tag zonly
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --tmult 1 3 10 --rhos 0.4 0.6 --reps 3 \
        --learners lowrank agcrn signed agcrn_prior agcrn_frozen --no-baselines --tag signs   # ~30 min; sign ablation
    venv/bin/python -W ignore analysis/recovery_2026_09/recovery.py --summarise > analysis/recovery_2026_09/out/recovery.txt

The times are wall-clock with the runs in parallel on 8 cores. Each run writes
`out/runs_<tag>.parquet`; `--summarise` reads all of them. `out/v1/` holds a
first pass that is superseded: lr 1e-3, a 200-epoch cap that the 30× runs hit,
and no signed or low-rank learners.

| script | what it does |
|---|---|
| `recovery.py` | Semi-synthetic generator: bootstrapped real calendar and features, fresh surprises with the real cross-series correlation, resampled real labels as noise, and the planted signal at per-edge ρ. Learners: Stage-1 pairwise + BH, lasso, a rank-4 signed low-rank graph, AGCRN, AGCRN with a signed directed adjacency, and z-only ablations. Scores AUROC, sign accuracy and *balanced* sign accuracy (19 of 25 true edges are positive, so raw sign accuracy of an all-positive read-out is 0.76), precision@25, power/FDP and R² against the oracle. |

Results (AUROC of true vs false edges; 0.5 = chance):

| | 1× (real) | 10× | 30× |
|---|---|---|---|
| pairwise, ρ = 0.4 | 0.53 | 0.81 | 0.90 |
| low-rank graph, ρ = 0.4 | 0.51 | **0.85** | **0.90** |
| AGCRN ∂ŷ/∂z, ρ = 0.6 | 0.52 | 0.51 | 0.59 |
| AGCRN Ã, any ρ, incl. null | 0.35–0.37 | 0.37–0.39 | 0.37–0.39 |
| AGCRN signed, z-only input, ρ = 0.6 | 0.52 | **0.86** | – |
