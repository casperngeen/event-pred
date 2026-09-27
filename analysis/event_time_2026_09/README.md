# Event-time graph panel, 2026-09

The daily node panel collapses a release and its repricing into one end-of-day
close, so AGCRN never sees the lead-lag (`reports/agcrn_checklist.md` §3b). This
study moves the clock to the release itself. One step is one release instant,
and node state is read from the trade tape just before it. Writeup:
`reports/agcrn_checklist.md` §3c. In-sample only.

    venv/bin/python analysis/event_time_2026_09/build_panel.py   # ~15 s, writes out/event_nodes.parquet
    venv/bin/python analysis/event_time_2026_09/models.py        # ~5 min, reads it; writes out/oof_settle.parquet
    venv/bin/python analysis/event_time_2026_09/backtest.py      # ~10 min (permutations), reads both
    venv/bin/python analysis/event_time_2026_09/exits.py         # ~5 s, reads both
    venv/bin/python -W ignore analysis/event_time_2026_09/nonlinear.py   # ~15 min (permutations), reads the panel
    venv/bin/python -W ignore analysis/event_time_2026_09/ablation.py    # ~8 min; STG vs its parts, sign read-outs
    venv/bin/python -W ignore analysis/event_time_2026_09/metrics.py     # ~10 min; every model on every metric
    venv/bin/python -W ignore analysis/event_time_2026_09/returns.py     # ~1 min; gross returns, no costs
    venv/bin/python -W ignore analysis/event_time_2026_09/scoped.py      # ~25 min; hub / BH / walk-forward scopes
    venv/bin/python -W ignore analysis/event_time_2026_09/scoped.py --scopes t2 t2wf   # ~5 min; t ≥ 2 channels

| script | what it does |
|---|---|
| `build_panel.py` | 154 release instants (usable-trigger events grouped by minute) × 17 `HAWKISH` threshold series. State at t⁻ from each leg's last pre-release print (≤14 d old, VWAP over ties) via `quantile_moments`. Two labels on the node's lead contract (most prints before t): **imm** = price after ≤3 post-release prints within 24 h minus the last pre-release print; **settle** = 100·[YES] − first post-release print. Labels are masked on releasing and same-release nodes. |
| `models.py` | A: the zero-parameter economic signal Σ HAWKISH[a]·HAWKISH[b]·z_a against each label, with a CI from a bootstrap over release instants. B: 8-fold walk-forward, purged on each window's label end (`win["label_end"]`), with linear and AGCRN rungs, scored on all cells and on the cells where the BH-channel signal fires. Saves out-of-fold settle predictions. |
| `backtest.py` | Take the lead contract at its first post-release print, hold to settlement, pay taker fee + half spread (`leadlag_2026_09/economics.py`). Zero-parameter rules, walk-forward model predictions, and the controls that matter here: always-NO, an intercept-only walk-forward baseline, the favourite. Clustered CIs and a surprise-shuffling permutation null. |
| `exits.py` | Buy, then sell once the surprise is captured (3rd post-release print, ≤24 h) or at 24 h, against holding to settlement; taker and maker on each leg (maker fills measured from the tape, fee bounds 0 / 0.0175 / 0.07, strict-through variant), for the aggregate strategies and per channel, with a permutation p and BH q per channel. |
| `nonlinear.py` | Is the signal non-linear? The theory-signed response by quintile of s, then walk-forward one-parameter shapes against linear, with and without an intercept: sign, tanh, asymmetric, price-scaled 4p(1−p), width-scaled, parent-mean. Also a small gradient-boosted model with and without s, against a permutation null. Writeup: `reports/recovery_test.md`. |
| `_panel.py` | Not a script. The shared tensor construction (features, masks, labels, economic graphs, windows, ridge rungs, AGCRN factory), imported by `models.py` and `ablation.py` so they see identical data. `models.py` output is unchanged by the move. |
| `ablation.py` | (A) STG vs temporal-only (no graph) vs spatial-only (no history) vs neither, for AGCRN and linear. (B) Effective-edge sign read-out (∂ŷ_b/∂z_a) against the theory sign, with no prior, an unsigned structural prior and the signed prior, scored by *balanced* sign accuracy. Writes `oof_ablation.parquet` and `ablation_edges.parquet`. |
| `metrics.py` | Every out-of-fold model on R², accuracy, balanced accuracy, F1 (up and macro) and AUC, with ΔAUC / Δbalanced accuracy vs the best linear rung (bootstrap over instants). |
| `returns.py` | Gross P&L before costs of each model's side, held to settlement, against always-YES/NO and the zero-parameter rule, plus the excess over a random side with the same long/short mix. |
| `scoped.py` | Scope each model to a set of relations and train and score only on the cells they can explain (a target's label at an instant where an in-scope source released). Scopes: labour/inflation→FED hub, the BH channels, walk-forward-selected edges, and channels with theory-signed t ≥ 2 (full sample or per fold). Compares the zero-parameter rule, linear (sign imposed / free) and AGCRN (adaptive / signed / frozen), plus the all-cells models on the same cells. |

Results (from `out/models.txt`):

| | imm | settle |
|---|---|---|
| zero-param signal, all 142 edges | 0.512 [0.470, 0.556] | 0.505 [0.472, 0.539] |
| zero-param signal, 3 BH channels | **0.634 [0.552, 0.713]** | 0.544 [0.464, 0.625] |
| labour→policy | **0.704 [0.574, 0.815]** | **0.661 [0.542, 0.780]** |
| inflation→policy | **0.648 [0.519, 0.778]** | 0.576 [0.455, 0.697] |
| best learned model, R² vs zero | +0.005 (linear, econ signal only) | +0.008 (linear, econ signal only) |
| best AGCRN, R² vs zero | −0.013 (frozen BH graph) | −0.000 (frozen BH graph) |
| **net ¢/contract, capture exits** | – | 71/78 channel × plan cells negative; the 7 positive all have CIs spanning 0 |
| **hold to settlement, per channel** | – | only labour→policy beats shuffled surprises (p = 0.007), BH q = 0.088 across 13 channels |
| **net ¢/contract, walk-forward test folds** | – | econ ridge +2.51 vs intercept-only +3.00; a-priori rule −5.77; labour→policy rule +6.03 (p = 0.015) |

`out/` is untracked; see `analysis/README.md`.
