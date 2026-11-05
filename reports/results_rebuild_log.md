# Results rebuild log (Section 6.5), started 2026-10-30

Chunk notes, in order. Final sizing 20x everywhere (CLAUDE.md).

## Chunk 1: data and inclusion (done)
- Principle: include any game whose inputs can be reconstructed to mimic the live run; exclude only far deviations. 26 NBA preseason games excluded.
- MLB 849820 (12 Oct), 849813 (13 Oct): last feed doc still said In Progress; final score verified vs statsapi (7-3, 3-1); marked Final (`marked_final.json`, `.orig` backups).
- MLB 849817 (16 Oct): rain-delayed, never started that day -> excluded.
- NBA 20 Oct 088/089/090 fidelity control (b59): rebuilt inputs (contracts+candles from replay_hist, WS trades from book) vs real inputs: same orders/fills (one order differs in 089), net within ~1.5 c per game -> backfill accepted.
- NBA 23 Oct (9 games, no live contract list) backfilled with b59 `apply` and replayed.
## Chunk 2: guard / late-block arms (b57 variants)
- noguard: 83 games done; nolate: 46 NBA games.
## Chunk 3: baselines (b60)
- Rules: sym / micro-price / trade-flow; clock vs model timing; 1 contract per order; forecaster stubbed.

## Chunk 1b: remaining exclusions (decided 2026-10-30 against the inclusion principle)
- NBA 401909836 (21 Oct): book file re-pulled from the sandbox (gzip OK); included at next refresh.
- MLB 849818 (15 Oct), 849817 (18 Oct): no contract list and no recorded book at all (only feed and trades) -> a maker-fill simulation would be invented; excluded.
- MLB 849816 (17 Oct): live runner traded the wrong contracts after the rain-out; excluded. MLB 849815 (19 Oct): only REST book snapshots about every 30 s per contract (no WebSocket book), too coarse for queue simulation; excluded.
- NFL 401873008 (19 Oct): feed only (no arrivals, trades, candles, book); excluded.
- NBA 30 Oct (ten games 401909277-286): recorded after this run began; enter at the refresh.
## Baseline design notes (b60/b61/b62)
- Six arms in one replay (sym/micro/flow x clock/model timing); forecaster stubbed so only the book stream and execution cost time (~5.5 min per game).
- Size 1 contract per baseline order; compare cents/contract and Sharpe, not dollars.
- Tables built by b61_make_tables.py -> thesis/generated/results_tables.tex.

## Chunk 2 results (2026-10-30, 109 scored games: MLB 10, NBA 70, NFL 29; provisional until the ~1 Nov refresh)
- Pooled: net $69.03 on 1,568 contracts, 4.40 c/contract, fees $13.70 of $82.73 gross; pooled annualised Sharpe 6.35 (per-league MLB 3.2, NBA 4.0, NFL 6.7, wide bootstrap intervals: only NFL's excludes zero).
- Spreads earn in every league; totals lose in MLB/NBA, earn in NFL.
- Guard on vs off (net $): MLB 2.0 vs 29.5 (guard HURTS in simulation, 10 games), NBA 34.9 vs -3.5 (guard helps), NFL 32.1 vs 29.5 (about equal). Report as is.
- NBA totals late block on vs off: -9.0 vs -55.7 net (block helps).
- 23 Oct NBA games: backfilled but recorder started mid-game -> 0-1 orders each; kept in (principle: mimics live), count as zero-P&L games and dilute Sharpe slightly.
## Chunk 3 notes
- Baselines: six arms per game take 25-60 min for NBA, >1 h for NFL (serial preflight, per-arm execution of the whole book) -> NFL not replayed, NBA every 6th game, MLB all. Stated in the table footnote and text.
- Thesis: section Forward-Test Results rewritten; tables and numbers generated (thesis/generated/results_tables_*.tex, results_macros.tex); tab:sim now generated (print-rule column dropped); tab:fade removed; 6.3 gate text corrected (gate ON). New bib entries: sharpe1994ratio, lo2002statistics, bailey2014deflated, stoikov2018micro, cont2014price, spooner2018market.
- Refresh: ./refresh_results.sh (full) or ./refresh_results.sh tables.
- Open: two sentences of baseline/guard interpretation in the thesis (todo), after baselines finish.

## Chunk 4 (2026-10-31): refresh, baselines complete, speed
- Main run refreshed 31 Oct (121 inventory entries; 4 of the 30 Oct NBA games, 401909278/280/282/284, had truncated sandbox downloads: retried by the queued full refresh).
- Baselines (b60) finished on baseball 10/10, basketball all scored games at that time, football 8 of 29 (every 4th). The queued `refresh_results.sh` (log in session scratchpad, `MIN_FREE=10 WORKERS=2`) then reruns the main pull, guard/late variants and baselines for the new games and rebuilds the tables.
- Sharpe: tables now show per-game Sharpe; annualised only in the risk table (values were misleading, e.g. 24 or -80).
- MLB guard attribution: the 15-60 s skip window accounts for the extra baseball profit without the guard (101 fills, +$31.5 before settlement; ball-in-play wait 19 fills, +$1.1). Window origin: journal rows 2026-10-08 (read off training folds, walk-forward check not done).
- Speed: loader time is mostly EventStream.add/finalize and the replay loop, not JSON parsing; b64_fast_stream (orjson, identical stream verified on two games) saves 18-30% of load time and is imported by b57/b60. A bigger gain would need a vectorised loader (not done).
- Disk is tight (11 GiB, swap uses disk): superseded outputs b49_firstpass_dup_connections, b49_event_replay, b49_1x_sizing_run (1.25 GB) await approval to delete.

## Baseball guard analysis (b65, 31 Oct)
Guard-off run, 10 games, settled-inclusive net by the main-run skip reason of each order:
- window skipped, model up to date (no pitch missed): 92 fills, 322.8 contracts, +$31.14, +9.65 c/contract
- window skipped, a pitch had been missed: 9 fills, 10.6 contracts, +$0.16, +1.5 c/contract
- ball-in-play skipped: 19 fills, +$1.13, +4.8 c/contract
- not skipped in main (runs diverge): 48 fills, 163.5 contracts, -$2.90 (37.6 contracts held to settlement, -$20.11)
So the 15-60 s window almost never caught a missed pitch (9 of 101 skipped fills) and skipped profitable trades; the guard-off run's extra settlement losses are in the diverging orders, not the window. Clean test: b57 VARIANT=nowindow (window off, in-play wait on) -> out/b57_nowindow, started 31 Oct.

## Baseball 15-60 s window removed (31 Oct, user decision)
- Deleted from scripts/live_trade.py (SKIP_AGE, Engine._skip, the skip branch in _decide, the cancel_rule "window" mode) and from live_rerun.py's "window" cancel option. Not deployed to the live hosts yet (live runner still carries the old copy).
- Consequences: b49's rule_hash changed (it digests live_trade.py). Baseball replays were rerun (out/b55_main_20x MLB entries now window-off; the previous window-on results are kept as out/b57_withwindow for the "earlier setup" row of the guard table). Basketball and football results are unchanged: verified identical on NBA 401909090 (59 orders) and NFL 401872994 (19 orders) by a fresh replay (out/b67_verify), then re-keyed with b68_rehash.py (main, noguard, nolate, baselines; never baseball). Baseball guard-off arm and baseball baselines rerun by the refresh.
- Baseball guard comparison is now ball-in-play wait on vs off.
- Thesis: chapter 4 baseball bullet says the forward-test configuration drops the rule; forward-test section explains the drop. Chapters on backtests (1845, 1891) still describe the window as part of the backtested design: reread.
- Live host still needs the code deployed (edit deploy copy) if the live runner will be used again.

## Window check, same classifier on backtest and replay (b69, 31 Oct)
Classifier: last known pitch = last archived pitch end <= t-10 s; window = age 15-60 s; "pitch missed" = a pitch ended after the last known one and by t. No-window fills:
- Backtest dev folds (762 games, 20x): in window/missed 2414 fills 1.12 c/unit; in window/current 2484 fills 2.69; outside/missed 265 fills 3.62; outside/current 1707 fills 6.22.
- Replay (10 games, settlement incl.): in window/missed 56 fills 8.99 c; in window/current 59 fills 9.87; outside/missed 11 fills 16.14; outside/current 42 fills -6.45.
Reading: the share of window fills with a missed pitch is the same (49% vs 49%), so late arrivals do not explain the gap. Backtest window trades earn less than outside (1-3 c vs 4-6 c) and missed-pitch ones least; replay shows no such gradient, on 115 window fills (about 2-3 standard errors from the backtest values). The baseball season is over, so no more games to settle it.

## Window discussion removed from the thesis (31 Oct, user decision)
Deleted the 15-60 s window from chapter 4 (baseball bullet), chapter 5 (feed-delay sensitivity and the sentence listing the window) and chapter 6 (guard text, "earlier setup" table row, comparison paragraph, the one-line note). Chapter 6 discusses only the feed-timing guards (baseball ball-in-play wait; basketball/football stall check and, for basketball, review wait and real-time-stamp gate). Caveat: the saved historical baseball fills (chapter 5, appendix sizing recalculation) were produced with the window rule; the thesis no longer says so. out/b57_withwindow and the b69 check remain only as working notes.

## 31 Oct: skill tables at the live feed delays (NBA, NFL 60 s; MLB 10 s)
- NBA d60 fold forecasts pulled from the cluster (`results/ingame/ens5/*d60*.npz`); NFL d60 final model trained on the cluster (5 seeds) and its holdout forecasts scored once (user approved the unseal): `NFL_fold99_gru_d60_..._ens5_seed0.npz`.
- `thesis_nba_skill_stats.py` (NBA_DELAY, default 60) and `thesis_holdout_skill_stats.py` (NFL_DELAY, default 60; MLB stays d10, reproduces the old MLB numbers exactly). Old d10 outputs kept as `out/*_d10.csv`.
- tab:skill rebuilt from the d60 jsons, row-weighted over folds (this weighting reproduces the old d10 row exactly): NBA 5.62/4.48, NFL 6.26/7.93.
- NFL holdout at 60 s (120 s horizon): spread 1.5% (t 2.9), total -0.2% (t -0.2): no total skill on the holdout (d10 was 1.0%, t 1.1).
- Thesis: both tables and 5.2.2 text updated, captions state the delays.

## 1 Nov: baselines on every game; unrestricted NBA totals
- Football baselines now cover all 29 scored NFL games (b62 STEP = 1 everywhere). Timing: a typical football game takes about 3 minutes for all six arms (1.1 GB RSS); the earlier ">1 h" estimate was wrong. The one 4 GB uncompressed book (NFL 401872994) was gzipped (3995 -> 225 MB) and the inventories repointed.
- Recorder files over 20 MB under data/sports_live were gzipped in place (readers fall back to name.gz; originals removed only after gzip -t passed). Deleted: MLB and NFL panel_v3, replay_nfl_s5, replay_nfl_sbx_k5. Kept contract_steps (b5_fills reads them).
- b68_all_totals_reporting.py --refresh: replays 57 NBA games (guard off with the late-game block off) and 53 NBA games (model-time baselines, because switching the block off changed order times); no NFL/MLB reruns. Chain after it: refresh_results.sh tables, then latexmk (log: session scratchpad after_refresh.log).
- To check after the refresh: prose in 6.x claims the model has the highest per-unit return, profit factor and Sharpe in each league; confirm against the final Table 6.4.

## 1 Nov: Table 5.4 without signal-strength selection
- User requested replay without knowledge of signal strength. `thesis_unfiltered_signals.py` uses forecast direction, `MIN_SIGNAL=0`, and one contract per fill as an explicitly unsized comparison arm. Retained: five-second decisions, one order/position per contract, one-cent strictly-through maker entry, 120-second exits, original fees, MLB 10-second and NBA/NFL 60-second delays, and MLB's 15--60-second post-pitch exclusion. Signal magnitude and width are used only for grouping afterward; final strategy sizing is unchanged.
- Reconstructed the deleted MLB/NFL panel row identities from retained contract_steps, using b4_export's filtering/sorting. Every saved forecast's step, kind and anchor was checked. Replaying the original 5-cent minimum exactly reproduced all original fills in all 15 development folds (MLB 5,203; NBA 22,018; NFL 3,397), including direction, signal, entry/exit prices, exit method and fill time. No holdout reads or retraining.
- Unfiltered replay: MLB 52,536 fills across 762 games; NBA 114,287 across 790; NFL 21,910 across 162. The <=5-cent band has spread/total net returns of -3.60/-3.55 cents (MLB; 20,579/30,693 fills), -2.64/-2.86 (NBA; 56,521/55,319), and -2.83/-2.62 (NFL; 12,464/9,019). One exactly-zero-signal MLB fill follows the original simulator's short-side tie convention; all other fills have nonzero signals.
- All bands were regenerated because weak-signal orders can displace later strong-signal orders. All width bands are now negative; >=10-cent width beats <5-cent width in only three groups. Removed the previous claim that the table generally supports increasing width weights. Highest-signal NFL spread/total cells have just 11/7 fills, now noted in the text.
- Outputs: `out/thesis_unfiltered_signals/{MLB,NBA,NFL}_fold[1-5].parquet`, per-fold and per-league audit JSON, and `validation.json`; the original table/CSV/provenance were preserved in `original_min5/`. `thesis_sizing_evidence.py` now reads the unfiltered logs and writes the existing table and CSV, including counts. Verified finite returns, execution variant, no overlapping positions and complete band counts; `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passed.
- Memory: projected and partitioned source trades one file at a time, then replayed one game at a time. Measured maximum RSS using Python's resource counter: MLB full run 798 MB, NBA final-fold run 885 MB, NFL full run 588 MB (MLB three-game check 921 MB). `/usr/bin/time -l` completed the scripts but its extra sysctl metric is unavailable in the sandbox.
- Reproduce: `POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 venv/bin/python -W ignore analysis/graph_2026_10/thesis_unfiltered_signals.py LEAGUE` for each of MLB/NBA/NFL, then `venv/bin/python analysis/graph_2026_10/thesis_sizing_evidence.py`. Completed folds are cached in the output directory.

## 1 Nov: live-result adapter and daily pull (built, not yet usable for the 20x headline)
- `analysis/graph_2026_10/b73_live_adapter.py`: rebuilds lots from the live runner's own exchange-simulator log (`execution.jsonl`, same events as the replay) and writes replay-shaped result files for b61. Validated: lots rebuilt from the events are identical to `lots.json` on 60 of 60 replay games. `./live_pull.sh` pulls only the small files from the backup VM (execution, orders, decisions, summary, feed; never books/trades/candles/forecasts/published), builds `out/b73_live_vm`, and prints live vs replay on common games.
- First run: 17 VM games with a finished exchange-sim log (the VM flag has been on since 30 Oct). Live vs replay is not comparable yet: the deployed runner has `CONTRACTS_K = None` (weights 0-1, i.e. 1x sizing), the replay is at 20x (3 vs 58 contracts on the NBA spreads in common). Per-contract returns are on tiny samples.
- To make live results the headline source: redeploy `live_trade.py` (CONTRACTS_K 20, window removed, late block off for NBA totals if the report keeps unrestricted totals) to the hosts, outside game hours. Comparison arms (guard off, block off, baselines) still need replays, or extra engines on the host (memory test first).

## 1 Nov: NBA spreads fell from +5.7 to +1.3 c/contract (debugged)
- Symptom: after the 53/57-game block-off and guard-off replays, NBA spread net on the 60 block-on games was $7.40 (main run $39.26). A totals-only switch should not move spreads.
- Cause: every game is recorded on both hosts (VM, sandbox) with different arrival stamps. b49's inventory picks, per game, the root with the most feed files; my live pull (rsync into data/sports_live/vm) added files to some VM folders, so 16 of 60 block-off, 20 of 106 guard-off, 26 of 110 combined games, and 23 of 123 baseline games were replayed from the other host's copy. Game 401909877: other copy gave different signals and sizes ($31.13 vs $3.24); game 401909893: VM feed with the sandbox's 1.3M-event book gave zero orders.
- Fix: b74_pin_inputs.py (imported from b64_fast_stream) replaces every wrapper replay's root and book with the ones recorded in b55_main_20x's result.json. It is not hashed into b49's rule_hash or b60's code hash, so cached results stay valid. Deleted the mismatched variant (57) and baseline (23 + overlay) results and re-ran them. live_pull.sh now writes to data/sports_live_pull/vm, outside the replay inventory.
- Check: on the 60 block-on games block-off spread P&L equals main exactly ($39.26, 707.4 contracts, no game differs).
- Result: NBA spread +$40.32 (5.59 c/contract), NBA total -$91.55 (-6.55) over 88 games with unrestricted totals (headline per user, 1 Nov). Guard NBA -51.23 on vs -101.85 off.

## 1 Nov: Chapter 5 NBA/NFL at the 60 s delay
- Chapter 5 generators now use each league's feed delay (`b33_ablation_table.V.at(league)`: NBA/NFL component and input versions at 60 s, MLB 10 s). Old 10 s summaries kept as `out/thesis_sizing20x/*_summary_d10.csv`.
- New inputs: NFL holdout 60 s trade logs (ens5, five seeds, original 60 s grid; user approved the unseal for the 60 s backtest), 30 s-hold logs for NBA/NFL dev folds at 60 s, NFL dev panel_v3 rebuilt (`b4_export.py NFL prints`; deleted earlier in the cleanup).
- Results (net c/contract, full model, 120 s): dev NBA spread 5.9 (was 7.5), NBA total 2.3 (3.9), NFL spread 5.2 (6.5), NFL total 6.9 (8.0); NFL holdout spread 2.6 (4.6), total 0.1 (0.1).
- Ablation table: NBA/NFL cells show a dash until the 60 s ablations are trained on the cluster (`NBA_NFL_DELAY=60 NBA_NFL_ONLY=1 STEPS="A1 A2 A3 B1 B2 B3 B4 B5" cluster/submit_ablations.sh`, then `ABL_D60=1 cluster/average_ablations.sh`). The ablation prose (message passing, play memory, inputs) quotes old NBA/NFL numbers and must be re-read after the rebuild.

## 1 Nov: MLB backtest without the 15-60 s skip window (user decision)
- Chapter 5 MLB now uses the rule the live runner has: every 5 s, one order at a time, no skip window (`b33_ablation_table.V.at`, `thesis_unfiltered_signals.py`, `thesis_sizing20x.py` holdout rule). New inputs: dev MLB 30 s logs, MLB dev panel_v3 rebuilt (peak 2.5 GB), MLB holdout logs (ens5 + 5 seeds x 5 parts, peak 4.0-4.9 GB), unfiltered-signal replay (parity exact on 5 folds). Old versions kept: `out/thesis_unfiltered_signals_skip1560/`, `*_summary_skip1560_d60.csv`.
- Effect (net c/contract, 120 s): dev MLB spread 9.8 -> 3.2, total 7.0 -> 2.3; holdout MLB spread 5.4 -> -0.2 (t -0.3), total 5.0 -> 0.6 (t 1.0). The skip window carried most of the historical MLB profit.
- Thesis: disclosure of the dropped rule with both sets of numbers (Section 3 dated-holdout paragraph, limitations), Summary of Results, conclusions, horizon and sizing prose updated. The 13 Oct post-pitch diagnostic (49% / 5%, 7.0 -> 10.1) is labelled as earlier and not rerun.
- Open: MLB ablation versions need their forecasts copied back from the cluster to be re-traded without the skip; the ablation prose (all leagues) is rewritten after the 60 s NBA/NFL runs finish.

## 1 Nov (later): reverted to the 15-60 s MLB skip window (user decision)
- The window is back in `scripts/live_trade.py` (SKIP_AGE, `_skip`, the skip branch in `_decide`, cancel_rule "window") and `scripts/live_rerun.py`; the sizing changes (CONTRACTS_K 20, SIZE_X) are kept. Reason: the window passed the independent holdout (MLB holdout +5.4/+5.0 vs -0.2/+0.6 c/contract without it; development 9.8/7.0 vs 3.2/2.3); the 31 Oct removal rested on 10 forward games.
- Chapter 5 baseball is back on the skip-window logs (`b33_ablation_table.V.at`, `thesis_unfiltered_signals.py` and `thesis_sizing_evidence.py` restored; the no-skip outputs stay in `out/thesis_unfiltered_signals_noskip/` and `*_summary` copies as comparison). Thesis: window explained in the Chapter 5 decision-rule list; limitations say it was read off the development folds and fixed before the holdout; guard text says the window applies in both arms (the guard comparison is unchanged: guard on vs off only).
- Verification before re-keying cached NBA/NFL replays to the new rule hash: a fresh NBA (401909896) and NFL (401872995) replay reproduced the cached lots exactly (`out/b76_verify`). Baseball is recomputed with the window (main, guard-off arm, model-time baselines).
- Draft ablation table (for the supervisor draft): NBA/NFL ablation versions use the earlier 10 s runs and are compared with the 10 s full model (`full model (10 s reference)` rows in b33 / thesis_sizing20x); once the 60 s ablations are averaged and copied back, `V.at` picks them up automatically (logs `b33_ablation_trades.sh NBA gru 60 <tag>`), and the note disappears.

## 1 Nov (late): forward results with the skip window restored
- Replays done (main MLB with window, guard-off arm, baselines on all 88 NBA games and 29 NFL, MLB model-time baselines). A first compile failed with a corrupt `main.out` (latexmk rc 12); deleting the aux files fixed it.
- Table 6.1 (net, c/contract): MLB spread +$4.01 (6.58), total -$1.98 (-1.46); NBA spread +$40.32 (5.59), total -$91.55 (-6.55); NFL spread +$16.19 (8.80), total +$15.91 (5.89). 127 games, net -$17.10 pooled (-0.62 c/contract). With the window MLB places 74 guard-on fills (158 without).
- Guard on vs off (window on in both): MLB $2.03 vs $6.15; NBA -$51.23 vs -$101.85; NFL $32.10 vs $29.49.
- Baselines (c/unit): the model is best in five of six groups; NBA totals are the exception (model -6.55 vs baselines -5.33 to -5.50). Every baseline loses.
- Prose updated: totals earned in football only; forward returns positive in four of six groups; NBA-totals baseline figures.

## 3 Nov: 60 s NBA/NFL ablations in Chapter 5
- Cluster runs (8 versions x 2 leagues x 5 folds, five-seed averaged) copied to `results/ingame/ens5/`, traded locally (`b7_maker_trades.py`, DECIDE_S 5, ONE_AT_A_TIME, d60; 80 fold logs, 0 errors), forecasts deleted afterwards (the `.json` summaries stay; recopy from the cluster to retrade).
- Per-seed scores no longer exist for these versions (averaged on the cluster), so NBA/NFL skill changes are differences of five-seed ensemble scores against the 60 s full model's ensemble score (table note says so); the "Full model" skill row stays the mean of seeds like Table 5.1.
- Table caption no longer "(Provisional)"; ablation prose rewritten: message passing -1.4 to -4.2 skill points, profit significant in three of six; play memory only helps baseball; structural reference -1.9 to -7.4 points, significant in four of six; football trades 54-63% less without the reference or win probability; "preliminary" removed from abstract, introduction and conclusion.
