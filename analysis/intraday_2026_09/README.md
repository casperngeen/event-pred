# Intraday post-release paths, 2026-09

This study follows every target market's price on a grid of bars after each release. It
measures how fast and in what order news reaches each market, and whether a temporal
model can learn the path. It moves the STG's time axis inside a release, because the
across-release version found nothing to learn (`reports/graph_ablation.md` §9).

- Plan: `reports/intraday_path_plan.md`.
- Writeup: `reports/intraday_path_findings.md`.
- In-sample only: no path is read or forward-filled past 2026-01-01.

    venv/bin/python -W ignore analysis/intraday_2026_09/build_paths.py > analysis/intraday_2026_09/out/build_paths.txt   # ~30 s
    venv/bin/python -W ignore analysis/intraday_2026_09/curves.py > analysis/intraday_2026_09/out/curves.txt             # ~6 min, reads the panel
    venv/bin/python -W ignore analysis/intraday_2026_09/order.py > analysis/intraday_2026_09/out/order.txt               # ~5 s
    venv/bin/python -W ignore analysis/intraday_2026_09/models.py > analysis/intraday_2026_09/out/models.txt             # ~25 min
    venv/bin/python -W ignore analysis/intraday_2026_09/models.py --price mean2 > analysis/intraday_2026_09/out/models_mean2.txt   # ~25 min, bounce check
    for g in trees mlp bayes; do                                                                                       # ~25 / 5 / 40 min per price, in parallel
        venv/bin/python -W ignore analysis/intraday_2026_09/models_extra.py --group $g > analysis/intraday_2026_09/out/models_extra_$g.txt & done; wait
    venv/bin/python -W ignore analysis/intraday_2026_09/models_extra.py --group score > analysis/intraday_2026_09/out/models_extra.txt   # add --price mean2 to all four for the bounce check
    venv/bin/python -W ignore analysis/intraday_2026_09/torch_check.py > analysis/intraday_2026_09/out/torch_check.txt   # ~40 min, imports models.py

Inputs:
- the trade tape, through `spillover_2026_09/_tape.py`;
- the event-time panel's release instants (`event_time_2026_09/out/event_nodes.parquet`,
  from `event_time_2026_09/build_panel.py`);
- the economic-calendar cache `data/external/econ_calendar_us_2021q4_2025.parquet`.
  It supplies the true release minute for the Kalshi instants and the non-Kalshi
  calendar triggers.

| file | what it does |
|---|---|
| `build_paths.py` | Builds the panel. There are three trigger arms: 139 Kalshi release minutes with z, 1,030 calendar release minutes with (actual − consensus)/sd, and a placebo copy of each, moved 3–10 days away from any release. For each arm, it takes each of the 17 series' lead contract at the release and records p₀, the last pre-release print. It then records the price at every bar: −60/−30/−15 min, 5-min bars to +2 h, 15-min to +6 h, hourly to +24 h, daily to +7 d. Each bar also has the prints and flow since the release and two truncation flags. The k-th post-release print is stored for trade time. Writes `out/cells.parquet` (one row per release × target) and `out/paths.parquet` (one row per release × target × bar). |
| `_common.py` | Not a script. Loads the panel and holds the channel families, the (cells × τ) response matrix, and a batched weighted least-squares fit. That fit serves the bootstrap over release instants and the per-instant sign-flip null. |
| `curves.py` | Q1 and Q2, absorption curves. For each channel (source family → target type), the own series' next contract, and pooled liquid / thin targets, it computes β(τ) of the move on the theory-signed surprise, with β(τ)/β(24 h), half-life, a bootstrap CI, a sign-flip p and BH. It adds robustness rows (2-print price, truncation at any release), the pre-release drift, the placebo arms, and trade time: the slope at the k-th post-release print against the 24 h slope. Writes `out/curves.parquet`. |
| `order.py` | Q3, order. Regresses the target's remaining move after τ = 15 min / 1 h / 4 h on the surprise, the source contract's move so far, its own move so far and its flow. SEs are clustered by release, followed by a walk-forward ΔR² of the nested specs. |
| `models.py` | Q4, temporal models. Predicts the move over the next ~1 h and to +24 h at every bar, using: 1 pooled absorption curve, 2 channel × τ-bucket ridge, 3 linear path state, 4 GRU per target, 5 AGCRN over the bars (adaptive and frozen economic graph). Walk-forward over 8 folds by release. Scored on R², balanced accuracy and AUC, on all bars and on bars where the target trades, by arm, with a bootstrap Δ vs the best linear rung. Adds rungs 3b and 4b, the linear and GRU rungs fitted on traded bars only. Also prints the full-sample path-state coefficients. `--price mean2` reruns everything on the 2-print mean price, as the bid-ask bounce check. Writes `out/oof_models[_mean2].parquet` and `out/models_scores[_mean2].parquet`. Importable, since the tensors build at import and the run sits in `main()`. |
| `models_extra.py` | Q4 continued: the standard-ML and Bayesian-graph rungs. 6a/6b LightGBM, 6c random forest, 6d/6e per-bar MLP, 7a Bayes graph (hard sign, edges × τ-bucket pooled by channel), 7b Bayes STG (soft sign plus per-node path state pooled across nodes), fitted by Gibbs on sufficient statistics. Run as `--group trees|mlp|bayes`, one process each (LightGBM and torch OpenMP runtimes crash in one process on macOS; LightGBM must be imported before torch), writing `out/oof_extra_<group>[_mean2].parquet`. Then `--group score` scores them jointly with `models.py`'s OOF predictions, writing `out/models_extra_scores[_mean2].parquet`. Needs `pip install lightgbm`. |
| `torch_check.py` | Checks whether the sequence rungs are under-trained. On two folds it runs the GRU and AGCRN under several batch sizes, learning rates and loss masks, and reports training and early-stop loss against predict-zero. |
