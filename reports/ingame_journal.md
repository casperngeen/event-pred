# In-game sports model: decisions and design journal

Running record of the in-game (play-by-play → Kalshi prices) work, started 2026-10-04. Each entry
gives the decision, why it was made, and where the code and results live. Paths are relative to the
repo root. Numbers are on the in-sample working block (everything before the 2026-07-01 holdout).

Terms used here:
- **Contract:** one Kalshi yes/no market paying $1.
- **Winner contract:** "Will team X win?" One per team.
- **Spread contract:** "Will team X win by more than k points (runs)?" A ladder of k values per game.
- **Total contract:** "Will the two teams score more than k combined?" A ladder per game.
- **Futures:** season-long contracts: champion, conference, division, playoffs.
- **Usable price:** a two-sided quote with a gap of 10c or less, mid between 3c and 97c, and quotes
  no older than 5 minutes.
- **Feed delay:** how long after a play happens a trader's live feed delivers it.

---

## 1. Goal and focus

| Date | Decision | Why |
|---|---|---|
| 2026-10-04 | Build a model that learns how in-game events move prices, at a horizon of seconds to minutes. Leagues: NFL, NBA, MLB. | Dense steps and fast dynamics suit a spatio-temporal graph better than between-game data does. |
| 2026-10-04 | No order-book data will be collected. | Not worth the effort. So market-making can't be backtested properly; taker trading can, using trades and 1-min quotes. |
| 2026-10-04 | Initially framed as explanatory: how events influence each other's pricing. | The thesis contribution. |
| 2026-10-05 | **Focus switched to trading.** The model is a pure forecaster: every input is known at the step. | The user's choice. |
| 2026-10-05 | **Archived:** an explanatory "response mode" that conditions on what happens inside the horizon. | Not tradable. Parked in `reports/graph_learning_proposal.md` ("Parked"), with the reasoning and comparison numbers. |
| 2026-10-04 | Player props deferred. | Do the game-level graph first. Props come later, with learned teammate and part-of-whole links and player embeddings. |

## 2. Data facts that shaped the design

- **Fees:** a taker pays about 0.07·p(1−p) dollars per contract (base rate unverified; see
  `kalshi-fees`). A typical in-game round trip on a winner contract costs 3.3–4c: about 1.3–1.9c
  spread plus two fees. The median 3-minute move is only 2–2.5c, so trading every minute loses.
- **Usable prices during play:** winner contracts 83–86% of minutes; spread and total contracts
  42–57% (8–11 usable at any moment). Team totals and half-game markets are mostly dead.
- **Liveness is decided per contract at each moment, not per market type.** In a blowout the winner
  contract sits at 99c, but spread and total contracts near the current score are still live.
- **Spread and total contracts lose their quotes straight after big plays** (NFL: 35% usable a minute
  before → 27% a minute after). They are kept, not skipped. They're trained on the range their
  quotes allow (interval loss), with quote quality as an input, and trades happen only at prices
  that were actually quoted.
- **Spread and total contracts don't exist before October 2025 for the NBA and MLB on Kalshi**
  (checked live and archived via the API on 2026-10-05). The 2024-25 NBA playoffs and the 2025 MLB
  regular season are winner-only.
- **Seasons:** each league has only one season of full graph data (winner + spread + total). There's
  no multi-season validation in-sample. The NFL holdout (a few weeks of 2026) is the only
  cross-season test; MLB's holdout is the second half of 2026; the NBA has none until 2026-27 starts.

## 3. Event timing

| Date | Decision | Evidence |
|---|---|---|
| 2026-10-05 | **NFL plays are timed at the end of the play** (nflverse `end_clock_time`, matched to ESPN by snap). Without an end time: the snap plus the median snap-to-end time for that play type, from earlier games only. Stoppages keep their own time. | The snap is the start of the play; field goals end about 3.7 s later and passes about 5.1 s. Timed from the end, half of a big price move happens within about 3 s. Missing end times are 4% of runs and passes, mostly special teams, 99.5% of extra points. |
| 2026-10-05 | **MLB reactions are measured from the pitch start.** | The logged play end comes after the play (home runs after the trot), which made moves look instant. |
| 2026-10-05 | **NBA uses possessions, not raw rows** (`analysis/graph_2026_10/b1_nba_events.py`). Free-throw trips are timed at the foul. | 485 rows per game shrink to 203 possessions. The corr of win-prob change with the 30 s price move rises from 0.47 to 0.64. A per-game clock shift barely helps. |
| 2026-10-05 | **Model inputs enter when the play's result is known:** NFL play end, MLB plate-appearance end (home runs: pitch start + 7 s), NBA possession end. | Pitch-start inputs leaked MLB outcomes (median 8 s early) and gave a fake 60 s R² of 21%. Job 911452 results are invalid (`results/ingame/leaky_mlb_nba_v2/`). |
| 2026-10-05 | **Feed delay is applied at load time** (`--delay`), not baked into the data. | The model originally saw plays the instant they happened, which a real trader can't. One panel then serves every delay. |

Code: `stg_infra/stg/sports/loaders.py` (`t_event`, `nflverse_home_wp_post`).

## 4. Inputs

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | **Use published win probabilities; don't train our own.** NFL: nflverse, with ESPN as a check (corr 0.95). NBA: ESPN. MLB: statsapi, per plate appearance. | Simpler. An as-of check found no hindsight: the market beats ESPN's Brier score in every period, and the feed's WP doesn't predict the next market move. The WP only measures how big a play was. |
| 2026-10-05 | The raw game state (score margin, time left, possession) is a context input. | Needed for "a touchdown matters to the 7.5 spread only near a margin of 7". |
| 2026-10-05 | Quote-quality inputs: bid, ask, which sides are missing, gap, quote age. | Tells the model how much to trust each contract's own price. |
| 2026-10-05 | **Recent trading activity replaces "season phase"** (winner trades in the last 5 min). | The apparent late-season speed-up is not a seasonal cycle. Across two NBA and MLB season boundaries, speed tracks trading activity (MLB 2025 speed 0.04 at about 3 trades/min, 2026 0.39–0.67 at 45–82/min), and busy games are faster within the same month. |

## 5. Preliminary findings that set the model's structure

- **Winner contracts absorb plays within 30–60 s.** NFL from the play's end: 49% done by 5 s, 67% by
  10 s, 90% by 60 s.
- **Spread and total contracts respond most when their threshold is near the current score.**
- **Spread contracts trail the winner price by a few seconds:** the best-fitting delay is 2–5 s for
  the NFL, 1 s for the NBA, 0 s for MLB. The first trade after a big play comes about 10 s later in
  the NFL and MLB, already showing about half the move.
- **Futures follow the game over minutes.** NFL champion/conference: about 50% by 2 min, 70% by 5.
  Division/playoffs: about 35% by 2 min, 60% by 5. During the game they price only 0.13–0.36 of the
  eventual result's effect, then catch up.
- **Structural baseline** (no learning): a normal curve for the final margin and total, anchored on
  the winner price (`b1_baseline.py`).
  - Fitted margin sd: NFL 13 points, NBA 14.5, MLB 4 runs.
  - Spreads: mean abs error 2.9–4.6c; it explains 46–54% of each minute's change.
  - Totals: weak at explaining minute-to-minute changes.
  - Same misses in every league: the last 10% of a game, contracts just past their threshold, and
    blowouts. The market prices those 4–11c above the baseline.
- **Slow drift after 60 s** (+0.2–0.6c in quotes) is not confirmed in trades for the NFL/NBA. It is
  possibly real for MLB spreads.
- **Artefacts found and fixed:**
  - Stale 1-min candles made moves look earlier than they were.
  - Dropping near-settled one-sided quotes faked a reversal; they're now valued halfway to 0 or 100.
  - MLB's recorded pitch end made moves look instant.

Scripts and outputs: `analysis/graph_2026_10/b1_*.py`, `b2_futures_ingame.py`, `out/`.

## 6. Splits

| Date | Decision |
|---|---|
| 2026-10-05 | Walk-forward folds by date over games that have both spread and total ladders (spring training excluded): NFL 282, NBA 1,316, MLB 1,289. The first 40% is fold 0, training only. Five contiguous test folds follow, cut at day boundaries, each with expanding training. The last 15% of each training set is used for early stopping. (`b1_splits.py` → `out/b1_folds_<L>.parquet`) |
| 2026-10-05 | The holdout (on or after 2026-07-01) stays sealed. The export refuses to write anything that reaches it. |

## 7. Model

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | **Two versions sharing everything except the play memory.** v1 is a GRU; v2, later, a small causal transformer. | A clean comparison. A GRU rather than a plain RNN because its gates keep early-game context over about 180 plays; it's smaller than an LSTM, which suits this little data. |
| 2026-10-05 | **Contract graph:** message passing between contracts by kind (winner / spread / total), with a weight for each (receiving kind, sending kind) pair, 2 rounds. | Known relations. Weights are shared across all games and rungs. |
| 2026-10-05 | **Outputs:** the change in each contract's value at each horizon, plus a correction to the market's log-odds of its final result. | Forecasting, plus a direct mispricing test. |
| 2026-10-05 | **The model learns play sequences** (scoring runs and the like) itself, rather than from a pre-specified list. | The user's choice. Validated by ablation (no memory) on later folds, with any discovered pattern confirmed once on unseen data. |
| 2026-10-05 | **Loss:** a Huber penalty on how far the predicted price falls outside the bid–ask range at each horizon, plus 0.1 × log loss on the final result. | Range targets suit thin, one-sided quotes. |
| 2026-10-05 | **Train on 30 s – 2 min only** (`--horizons 30s,60s,120s`). | Equal-weighted 5–10 min horizons dominated the loss and are neither predictable nor tradable. |
| 2026-10-05 | **Train on traded prices** (`panel_v3`, `b4_export.py <L> prints`). The starting price is the last trade (if no more than 2 min old, else the quote); targets are the range and volume-weighted average of trades in the 15 s after each horizon; last-trade, flow and volume inputs are added; quotes are kept as inputs. `panel_v2` is kept for comparison. | The quote-based targets partly taught the model that old prices catch up. Only about 26% of NFL contract-steps have a trade in the target window. |
| 2026-10-05 | **Add futures as graph nodes and targets** (`panel_v3f`, `b4_export.py <L> prints+futures`): each team's championship, conference, division and playoff contracts open during the game, as four new contract types with their own message weights. Quotes come from 1-min futures candles plus trades (current for up to a day). Targets are trades over 120 s windows. A 30 min horizon is added. No final-result target. | Futures react to games over minutes, so the graph can learn how game contracts feed into them. They settle months later, some after the holdout start. |
| 2026-10-06 | **Drop the transformer version; keep the GRU.** | On NFL folds 1–3 (traded prices, 30 s – 2 min), a 2-layer causal transformer lost to the GRU on every contract type and horizon, typically by 0.02–0.05 R² (60 s fold 1: spread 0.10 vs 0.13, total 0.14 vs 0.19), and a second seed agreed. The GRU itself beats no memory (`results/ingame/v1`). Outputs in `results/ingame/tf_test/`; the `--memory transformer` option stays in the code. |
| 2026-10-07 | **Simplify the model: drop the futures contracts from the graph and train on `panel_v3` (game contracts only) in all three leagues. Replaces "Add futures as graph nodes and targets".** (`b14_futures_ablation.py`, `b14_nofutures_pnl.py`; 5 seeds each, 10 s delay, 30 s – 30 min horizons) | The futures made the game-contract forecasts worse: they added noise rather than information. The harm grows with how many futures rows a league has: NFL (most futures) 2–4 R² points worse on spreads and totals at 30 s – 2 min (60 s total 13.3 vs 16.7%, game-clustered t −7 to −11, every seed agrees); NBA (few) 0.1–0.5 worse; MLB (none) within 0.25, which is the size of the difference the setup alone makes. In the frozen maker strategy (5-seed average forecasts), NFL totals net 7.4c per contract without futures vs 3.2c with (t 3.0 vs 1.5, 5/5 folds positive; per-game gain t 3.1, each seed alone t 2.0–3.2); NBA spreads, NBA totals and MLB totals unchanged (5.7 / 2.5 / 7.2c). NFL spreads turn from −1.9 to +2.5c (t 1.2) but stay dropped: weak, and picking them now would be choosing after looking. Candidates and sizing stay frozen; strategy levels are still from the folds they were chosen on. |
| later | Pool leagues with a league embedding; pre-train the play encoder on winner-only games (MLB 2025, NBA 2024-25 playoffs). | More data, especially for the NFL. |

Code: `stg_infra/stg/models/ingame.py`, `cluster/train_ingame.py`.

## 8. Evaluation and trading

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | Sweep the feed delay (0–20 s) at load time. | The NFL edge in the first taker backtest needed a delay of 10 s or less. |
| 2026-10-05 | **Don't trust taker P&L from rebuilt quotes; price every test from trade prints** (`b5_fills.py`, `b5_longshort.py`). | The profitable entries were mostly prices the market had already left. Entries confirmed by a later print net −0.4 to −2.7c. |
| 2026-10-05 | **Drop taking as a strategy, for game contracts and for futures.** | The signal is real at traded prices (NBA t 7–20) but moves only 1–4c, against taker costs of about 4–6c. Futures prints arrive 2–5 min after a play, already priced (`b6_futures_taker.py`). |
| 2026-10-05 | **Pursue making: post only on the side the model favours** (`b6_maker.py`). | A no-signal maker loses 0.5–1.5c per fill to adverse selection. Signal-filtered orders make +1 to +3c per fill (NBA t 5–17). Not yet validated: real exits, queue position, the real book, maker fees. |
| 2026-10-05 | **With fills requiring a trade-through and real exits, making only works on the strongest signals (≥8c), and only with limit orders on both legs** (`b6_maker_exit.py`). | Gross −1 to −2.6c otherwise. At ≥8c: NBA spread/total +2.0/+1.7c net (t 2.8/2.1), NFL total (0 s delay) +4.9c (t 2.5). Taking the exit roughly halves it. Few trades, and the 8c cut-off was chosen after looking. |
| 2026-10-06 | **Read the trade-price model's spread/total skill as partly stale prints catching up, not only forecasting** (runs on panel_v3f: traded-price targets plus futures, 0 and 10 s delay, all three leagues). | It explains 8–18% of the 30–60 s change on spread/total contracts in every fold, and nothing on winner contracts. But the last print lies outside the rebuilt bid/ask in about half of rows, and a no-model "next print lands at the mid" rule alone explains 2–12%. The 30-min horizon is worse than "no change" everywhere. |
| 2026-10-06 | **Drop futures from in-game trading.** | The model almost never predicts a 60 s futures move of 3c or more (none for NFL champion, conference or division), and when it leans one way the move is about 1c, well under the ~4c a resting order loses to adverse selection. |
| 2026-10-06 | **Keep the fixed exit (rest 1c past the last print); don't chase the mid** (`b6_maker_mid.py`). | An exit order that follows the estimated mid, with a 30/60/120 s deadline, fills more often but at a worse price: 0.2–1c less net. It also doesn't rescue the 10–25% of trades with no later print at all. |
| 2026-10-06 | **Make NBA spreads, ≥8c signal, 120 s hold the one candidate; drop NBA and MLB totals and NFL** (`b7_maker_robust.py`, 10 s delay). | NBA spreads: +2.7c net (t 3.2), positive in all 5 test folds, net rises smoothly with the cut-off, better with a 2c offset, survives twice the maker fee, median trade +3.2c. Weak points: about 1c lower if fills need a print 2c past our price, and a third of the profit comes from the 10 best games. NBA totals faded over the season (negative Feb–Apr folds). MLB totals' median trade is negative (a few big wins), with only 2 months tested. NFL is negative in this setup. |
| 2026-10-06 | **Size each trade by signal and width instead of hard cut-offs: size = ((\|signal\| − 5)/10)² × width/10, each part capped at 1, width part = 1 when width is missing (one side has no print in the last 120 s)** (`b9_sizing.py`, `b9_cutoff_grid_all.py`). | Net rises smoothly with both signal and width, with no real threshold at 8c or 9c (those were picked after looking). On NBA spreads this beats the best hard rule (≥8c, width ≥9c) at every delay and in every test fold: t 6.2 vs 5.0 at 10 s, about 27% more P&L for the same total size. Also best or near best on NBA, MLB and NFL totals without retuning. Missing width counts as wide: those moments do at least as well as width ≥10c in every contract type, and 0.5 instead of 1 changes net by only 0–1c. The squared shape and missing = 1 were chosen on the test folds; frozen from here. |
| 2026-10-06 | **Candidates, all at 120 s with the sizing rule above: NBA spreads (main), MLB totals, NBA totals, NFL totals. Drop MLB and NFL spreads and all winner contracts. Replaces "NBA spreads is the one candidate".** (`b9_cutoff_grid_all.py`, `b9_league_profile.py`) | Net per contract (t) at 0 / 10 s delay: NBA spreads 6.2 / 6.2 (7.1 / 6.2); MLB totals 5.8 / 4.8 (3.8 / 3.1); NBA totals 4.1 / 2.3 (4.4 / 2.5), thin at 10 s; NFL totals 5.8 / 4.7 (3.2 / 2.4), one season. Spreads outside NBA are about 0 or negative under every rule: MLB fills come at large discounts but the price moves back ~7c within 2 min, and NFL spreads are too tight (median 5c) to cover costs. MLB totals at 60 s instead of 120 s trade nearly the same entries (96%) with the same P&L per game, so one horizon for all. All chosen on test folds; the holdout (MLB Jul–Sep 2026, NBA/NFL 2026-27) decides. |
| 2026-10-07 | **Freeze the holdout spec** (`reports/holdout_spec.md`): the no-futures model trained on every pre-holdout game (same early stopping as the folds), 5-seed average, the four candidates and the frozen sizing, **both exits reported as main results** (fixed exit and hold to settlement), MLB postseason included and shown separately, no pre-set pass rule (the user interprets the results for deployment). Part 1 = everything pulled by 2026-10-06. | Holding to settlement earned more per contract on all four candidates in the test folds (MLB totals 15.7 vs 7.2c, paired t 4.1; NBA spreads 8.4 vs 5.7c but t 3.7 vs 5.6 and 3× the drawdown; `b16_exit_vs_hold.py`). It was found on the same folds, so both go to the holdout rather than choosing now. Win rates are only 50–60%: the profit comes from the large-signal trades the sizing favours, and the fixed exit's large losses come from the 14–28% of trades with no exit print. The NBA playoffs traded no worse than the regular season (spreads 7.9 vs 5.5c, tighter markets). |
| 2026-10-07 | **Fix a play-order look-ahead before the holdout run:** sort each game's plays by when they became known (`b4_export.py`) and count a play as known only once every earlier play has happened (`known_state`); rebuild the NFL / NBA panels and retrain their final models (MLB unchanged). Re-pull the holdout's missing games (NFL week 4, MLB postseason) with their play-by-play. | The holdout integrity check (`b18_holdout_integrity.py`) found plays out of time order in 281 of 282 NFL games (NBA 157 of 1,316, MLB none); the model's binary search then read a later play up to ~1 min early at 0.16% (NFL) / 0.05% (NBA) of steps, including scoring plays. Only 2–15 test-fold trades per candidate were affected and removing them moved net by ≤ 0.1c, so the frozen candidates stand. Logged in the spec (section 9). |
| 2026-10-08 | **MLB: the model sees every pitch as a play, not only the end of each plate appearance** (`PLAY_UNIT=pitch`, `ingame/MLB_pitch/panel_v3`; runs tagged `_tp_pitch`). | Same 5 seeds × 5 test folds. Forecasts improve on every spread / total horizon (60 s R²: totals 8.8% → 11.5%, spreads 9.8% → 13.1%; `b13_seed_ensemble_gru_d10_h30-60-120-1800_tp_pitch.txt`), and the maker strategy earns more in every MLB row under every decision rule (`b19_decision_timing_tp_pitch_ens5.txt`). A pitch enters when its result is known; the win probability stays per plate appearance, so it only changes on the last pitch. |
| 2026-10-08 | **Don't trust a single once-a-minute decision grid; compare rules on all 12 grids (second 0, 5, …, 55 of each minute).** | Which second of the minute the decisions fall on moves MLB results by about ±1c on totals and ±2c on spreads (`mlb_decision_offsets.csv`); the original grid happened to be the best of 12 for MLB totals and near the worst for spreads. Trading all 12 grids at 1/12 size each ("scaling in", `b24_scale_in.py`) gives the highest t on most contracts, but it is not adopted: the MLB rule below beats it, and NBA / NFL keep the frozen rule. |
| 2026-10-08 | **Keep one minimum signal (5c) for every league and contract; don't set it per league.** (`b22_league_thresholds.py`, `b23_min_signal_one.py`) | Picked walk-forward (each fold's minimum chosen on earlier folds only, from 5 / 8 / 10 / 12c), per-league and per-contract minimums lose money out of sample with the frozen once-a-minute rule (−$0.8 and −$3.6 on the fixed exit vs 5c, folds 2–5): the sizing already gives weak signals tiny orders, and the chosen minimums jump between folds. Under "every 5 s, one at a time" a single higher minimum (10–12c) does help, because a weak order no longer blocks a stronger one (NBA t 8.3 → about 10.5), but that rule is not used for NBA / NFL. |
| 2026-10-08 | **MLB's problem with deciding every 5 s is the feed delay: the model is often one pitch behind.** (`b25_mlb_pitch_bins.py`, `b26_pitch_detect.py`) | When no pitch has been thrown since the last pitch the model knows, trades earn about 10c per contract and carry 70–90% of the profit under every rule. Deciding every 5 s acts at the first moment a gap between forecast and price appears, which is usually just after a new pitch the model hasn't seen yet (79–81% of its contracts); those earn about 0c. Price activity can't tell these moments apart (the game's contracts print in almost every 5-s window). Time since the last known pitch can: a new pitch has usually been thrown 15–60 s after it, rarely before 15 s or during long breaks (≥ 60 s). |
| 2026-10-08 | **MLB rule: decide every 5 s, one order or position per contract at a time, and skip decisions 15–60 s after the last known pitch** (`SKIP_AGE=15,60` in `b5_longshort.py`). | Test folds, fixed exit: MLB totals 2.3c (t 3.2) → 7.0c (t 5.1), spreads 3.2c (t 3.1) → 9.8c (t 5.1); about 73% of trades are then made with the model up to date, against about 20%. More net dollars than every other MLB rule with the fixed exit. Skipping beats trading a smaller size in the window: a small order still occupies the contract and blocks the better moment seconds later. The 15 / 60 s window was read off these folds, so the holdout is its test. |
| 2026-10-08 | **Revise the holdout spec before any holdout result was read: all six spread / total contract types are candidates, the fixed exit is the main result (holding is descriptive), MLB uses the every-pitch model and the rule above; NBA / NFL keep the once-a-minute rule.** Delete the NFL holdout trade logs made under the old rules and rerun. | The user's choices. MLB and NFL holdout results had not been looked at (confirmed). Spec section 9 records each change; the 5 final MLB models are retrained on the every-pitch panel. NFL spreads are the weakest candidate on the test folds (3.3c, t 1.7). |
| 2026-10-08 | **Holdout part 1 read** (`b17_holdout_report.txt`, fixed exit): MLB spreads 5.4c (t 3.9), MLB totals 5.0c (t 5.1), both positive in every month and every seed; NFL under every 60 s: spreads −2.1c (t −0.5), totals +4.0c (t 1.2). | MLB holds up below its test-fold level, as expected. NFL is about 60 games; its forecasts fell well below every test fold (60 s R²: spreads 4.0% vs 12.0%, totals 1.3% vs about 17%), while MLB's held (11.3% / 9.1% vs 13.1% / 11.5%). Cause not yet known: much busier 2026 markets (integrity flag) or early-season football (no test fold covered September–October). |
| 2026-10-08 | **NBA and NFL switch to every 5 s, one at a time; NFL part 1 rerun under it.** | The user's intended rule, which the written spec revision had missed (it kept every 60 s). Logged in spec section 9 as a post-result correction for NFL part 1; clean for NBA and NFL part 2. |
| 2026-10-08 | **Read NFL part 1 as a forecasting failure, not an execution one; no change to the frozen spec** (`b27_nfl_total_diag.py`, `b28_mlb_postseason.py`, diagnostic only). | NFL 60 s R² fell on both contract types, totals much more (16.7% → 1.3%) than spreads (12.0% → 4.0%), in every quarter of the game; forecasts became too large (slope of the actual move on the forecast 1.10 → 0.57 on totals, 0.97 → 0.78 on spreads). The no-model "price moves to the middle of the bid and ask" rule fell too (totals 9.3% → 0.8%): in busier 2026 markets stale prints no longer catch up. Fills and exits behaved as on the test folds, so totals lost their edge before costs (10–15c signals 8.3c → −1.3c). Spreads kept part of theirs (10–15c signals still 7.0c) and fewer of their trades rode to settlement (8% vs 17%). MLB playoff markets were about 3× busier and their totals forecasts weaker (R² 4.3% vs 9.2% in the regular season); 19 games, too few to judge P&L. Any fix (recalibrating, retraining on 2026 data) needs part 2 or paper trading. |
| 2026-10-14 | **Live paper trading: compare live fills and signals with the backtest from 13 Oct (ET) onward; the two 12 Oct MLB games (LAD @ MIL, CLE @ TB) are reported but left out of that comparison.** | On 12 Oct the runner polled Kalshi's REST trade endpoint, hit about 2,700 rate-limit errors (HTTP 429) and saw trades a median 31–35 s late, so 69–78% of its 5-s decisions had no trade in the last 60 s (even on a line trading every ~2 s) and it posted only 13–15 orders a game. From 13 Oct the runner streams trades over the WebSocket (median 0.1 s, no 429s; 25% without a recent trade, mostly on thin lines; 40 orders). The rule itself is unchanged. |

Results files: `analysis/graph_2026_10/out/b4_*`, `b5_*`, `b6_*`, `b7_*` (`b7_maker_robust_d10.txt`), `b8_*`, `b9_*` (`b9_sizing.txt`, `b9_cutoff_grid_all.txt`), `b13_seed_ensemble*.txt`, `b14_futures_ablation.txt`, `b14_nofutures_pnl.txt`.

## 9. Infrastructure

| Step | Code | Output |
|---|---|---|
| Training table (5 s steps, plays, contract-steps, targets) | `analysis/graph_2026_10/b3_table.py <L>` | `data/sports/derived/ingame/<L>/` |
| Table checks | `b3_table_check.py` | `out/b3_table_check_<L>.txt` |
| Model-ready panel (float16, streamed, holdout-checked) | `b4_export.py <L>` | `.../<L>/panel_v2/` |
| Train / evaluate one fold | `cluster/train_ingame.py` | `results/ingame/<L>_fold<k>_<memory>_d<delay>[_h…]_seed<s>.json` (+ `.npz`) |
| Grid, locally | `cluster/run_ingame_local.sh` | |
| Grid, on the cluster | `cluster/train_ingame.sbatch` (`LEAGUES`, `DELAYS`, `PACK`, `HORIZONS`) | |
| Summaries | `b4_collect.py` | |
| Taker backtest | `b5_backtest.py` | `out/b5_backtest_<L>.parquet` |
| Maker backtests (fixed exit; mid-following exit) | `b6_maker_exit.py`, `b6_maker_mid.py` (`RUN_TAG`, panel dir) | `out/b6_maker_exit_*`, `out/b6_maker_mid_*` |
| Maker per-trade log; robustness checks | `b7_maker_trades.py <res> <L> <delay> <panel>`; `b7_maker_robust.py <delay>` | `out/b7_maker_trades_*.parquet`, `out/b7_maker_robust_d<delay>.txt` |

The feed delay is a training setting (applied when batches are built), not part of the panel: a new
delay needs only new training runs, not a new panel.

**Cluster rules (SoC Slurm):**
- Copy only `stg_infra/` and `cluster/` plus the exported panels, never the repo root: it holds
  `.git`, derived data and `kalshi_api_key.txt`.
- Limits: MaxSubmit 32 and MaxJobs 16, with each array task counting as one job. Use `PACK` to run
  several trainings per task.
- pip needs `TMPDIR=~/tmp`. Don't import polars or torch on the login node.
- See `cluster/README.md`.

## 10. Open items

- Capacity: how many contracts the prints that would have filled us carried, to set the maximum size.
- Save model weights and add a "train on everything before the holdout" option to `cluster/train_ingame.py`.
- Test the frozen candidates and sizing rule untouched on the holdout: MLB and NFL part 1 now (locally), NBA and the rest of the NFL in 2026-27, or in paper trading.
- Not in the holdout spec, for part 2 or paper trading: NBA / NFL deciding every 5 s, one at a time, with a 10–12c minimum (NBA test-fold t about 10.5 vs 5.0 for the frozen rule; `b23_min_signal_one.txt`); scaling in (`b24_scale_in.txt`); a walk-forward check of the MLB 15–60 s window.
- Record the live order book and paper trade the maker rules.
- Signal strength and extra filters (`b21_signal_sizing.py`): every trade counted once, net after fees rises steeply with |signal| (below ~8c it loses with the fixed exit, >= 15c earns 10-19c in NBA / NFL); sizing by signal alone is 0.4-2c worse than signal x width, so width earns its place. Under one order per contract at a time, weak 5-8c orders (1-9% of full size) still occupy the contract and can block a stronger signal seconds later: test a minimum signal (8c / 10c) for placing an order. Longer term, replace the hand-made sizing with a small trade-decision model (signal, width, liquidity, time since the last play -> net P&L, trained on test-fold trades). Width as a forecaster input is possible (the model sees quotes and the last print, not the side-split width) but would not make the forecast cost-aware.
- Time stop for fixed-exit trades with no exit print (14–28% ride to settlement and carry the large losses; `b16_exit_vs_hold.py`): test in paper trading with the live order book, since prints can't price it. Not part of the holdout spec.
- Measure the real live-feed delay; verify Kalshi taker and maker fees.
- Later: no-memory ablation on the traded-price panel and more seeds; a better totals baseline.
