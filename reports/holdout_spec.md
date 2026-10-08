# In-game model: frozen spec for the holdout test

Written 2026-10-07, **before** any holdout data has been read. Everything below is fixed. The test
runs once. If a bug is found after seeing results, fix it, log it in `ingame_journal.md`, and report
both the before-fix and after-fix numbers.

**Status: FROZEN on 2026-10-07** at code commit `eb92644`. **Revised 2026-10-08, before any holdout result was read** (section 9): MLB moves to the every-pitch model with its own decision rule, all six spread / total contract types are candidates, and the fixed exit is the main result. The holdout runs code commit `cb255a4`.

## 1. What is being tested

Whether the in-game maker strategy, chosen and sized on the 2025–26 test folds, still makes money
after fees on games it has never seen.

- **Holdout:** every game whose first play is on or after 2026-07-01. Its data sit in
  `data/sports/_HOLDOUT_DO_NOT_USE/` and have never been read by any model or analysis script.
- **What exists now** (pulled up to about 2026-10-06):
  - MLB: July to September plus the postseason so far.
  - NFL: about the first 5 weeks of the 2026 season.
  - NBA: no games yet (the season starts late October).

## 2. The model (frozen)

| Setting | Value |
|---|---|
| Inputs | `panel_v3`: game contracts only (winner, spread, total), traded-price targets, no futures |
| What counts as a play | NFL a play, NBA a possession; **MLB every pitch** (`PLAY_UNIT=pitch`, panel `ingame/MLB_pitch/panel_v3`; section 9). A pitch enters when its result is known; the last pitch of a plate appearance keeps the plate-appearance rules |
| Play memory | GRU (a recurrent network that reads the plays so far), hidden size 64 |
| Contract graph | message passing between winner / spread / total contracts, 2 rounds |
| Feed delay | 10 s (a play reaches the model 10 s after it happens) |
| Horizons trained | 30 s, 60 s, 120 s, 30 min |
| Loss | as in all test-fold runs: Huber 5c on the traded-price range, + 0.1 × log loss on the result |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001, dropout 0.1, gradient clip 1.0 |
| Batches | 16 games per batch, 256 sampled steps per game |
| Training games | every game before 2026-07-01 that the test folds used (all folds, 0–5), per league |
| Early stopping | as in every test-fold run: the most recent 15% of those games (by date) are held back to pick the epoch (maximum 40 epochs, stop after 5 without improvement); they are not trained on |
| Seeds | 0, 1, 2, 3, 4; **the forecast is the average of the 5 seeds' forecasts** |

Code: `cluster/train_ingame.py --final --save-model` (via `cluster/train_final.sbatch`). The holdout
panel is built with the same code and the same fixed scales (`b3_table.py`, `b4_export.py`, nothing
fitted on data). `b15_holdout_predict.py` refuses to run if the panel's features, scales or horizons
differ from the training panel's.

## 3. The strategy (frozen)

The same rules as the test-fold results (`b7_maker_trades.py`, `b9_sizing.py`):

1. **When:** for each contract of the candidate type (section 4) that has a trade print in the last
   60 s. That last print is the **reference price**.
   - **Every league:** every 5 s (every model step), **one order or position per contract at a time** (a
     contract is busy from its order until the order expires unfilled or the trade is closed).
   - **MLB also:** **no decision while 15–60 s have passed since the last pitch the model knows** (after
     the 10 s feed delay). In that
     window a newer pitch has usually been thrown that the model hasn't seen, so its forecast is behind
     the market (`b26_pitch_detect.py`).
2. **Signal:** the model's forecast price 120 s ahead minus the reference price (cents).
3. **Entry:** post on the side the model favours only. Buy limit at reference − 1c if the signal is
   positive; sell limit at reference + 1c if negative. The order rests for 60 s. It counts as filled
   only if a later print trades strictly through our price (so queue position doesn't matter).
4. **Exit: the fixed exit is the main result** (section 9). Holding to settlement is reported only
   as a description, on the same entries and sizes:
   - **Fixed exit:** 120 s after the fill, rest a limit 1c past the last print for up to 120 s,
     filled only by a print through it. If not filled, take the next opposite-side print within a
     further 120 s. If there is none, hold to settlement.
   - **Hold to settlement:** keep the contract to its result; P&L = side × (100 × result − entry),
     minus the maker fee at entry, no exit fee.

   On the test folds holding earned more per contract on all four candidates (e.g. MLB totals 15.7
   vs 7.2c, NBA spreads 8.4 vs 5.7c) but with larger swings on NBA spreads (game-level t 3.7 vs 5.6,
   drawdown 3×) (`b16_exit_vs_hold.py`). It was found on the same folds, so the holdout is its test.
5. **Size** (contracts, 0 to 1 per trade): ((|signal| − 5) / 10)², capped at 1, × (width / 10),
   capped at 1, where width = last YES-taker price − last NO-taker price, each within 120 s (a stand-in for the
   bid–ask spread). If either side is missing, the width part is 1. Below 5c of signal the size is 0.
6. **Fees:** maker 1.75 × p(1 − p) cents on each limit fill, taker 7 × p(1 − p) on a taken exit, none
   at settlement (p = price / 100). The maker rate is not yet verified against Kalshi's schedule.

## 4. Candidates (frozen)

| Candidate | Decision rule | Test folds, fixed exit: net c/contract (game-level t) | Hold to settlement (descriptive) |
|---|---|---|---|
| NBA spreads, 120 s | every 5 s, one at a time | 7.47 (t 7.7) | 12.40 (t 6.7) |
| NBA totals, 120 s | every 5 s, one at a time | 3.89 (t 4.3) | 6.79 (t 4.8) |
| MLB spreads, 120 s | every 5 s, one at a time, skip 15–60 s | 9.83 (t 5.1) | 15.37 (t 5.2) |
| MLB totals, 120 s | every 5 s, one at a time, skip 15–60 s | 7.02 (t 5.1) | 14.11 (t 5.8) |
| NFL spreads, 120 s | every 5 s, one at a time | 6.50 (t 3.7) | 9.25 (t 2.1) |
| NFL totals, 120 s | every 5 s, one at a time | 8.03 (t 4.1) | 13.63 (t 4.0) |

All six are candidates on equal footing (no "main" one). Test-fold numbers from the current models' trade
logs (`b24_scale_in.py`, `out/b24_scale_in.txt`); they replace the 2026-10-07 numbers, which came from an
earlier run (NBA spreads were 5.7c there).

Not tested, and not to be added after seeing results: winner contracts, futures, other holds, other
decision rules.

Deferred to paper trading: a time stop for trades the fixed exit can't close (14–28% of trades find no
exit print within 4 minutes and ride to settlement, where the large losses sit; `b16_exit_vs_hold.py`).
The backtest can't price it: those trades have no print to exit against, and rebuilt quotes are not
trusted for taking. It needs the recorded live order book.

All of these levels were chosen on the same test folds they are measured on, so the holdout is
expected to come in lower.

## 5. What is reported, and in what order

For each candidate, with the fixed exit (section 3), using the 5-seed average forecasts:

1. **P&L before fees** (cents per contract), then **after fees** at the rates above, then after
   fees at **twice** those rates.
2. Contracts traded, total net P&L ($ at a 1-contract maximum per trade), game-level t (each game
   is one unit), share of game days with a profit, largest drawdown, and Sharpe ratio from daily
   P&L. The Sharpe ratio assumes unlimited capacity, so it is only a relative number.
3. Win rate (all trades, and weighted by size), average winning and losing trade, and the share that
   found no exit print and rode to settlement.
4. Each single seed's net, to show how much the result depends on the seed.
5. MLB split into regular season and postseason; NBA (part 2) into regular season and playoffs.

Descriptive only: holding to settlement on the same entries; forecast R² against "no change" on the
holdout; a 0 s feed delay.

## 6. How results are judged

There is no pre-set pass/fail rule: the results are interpreted when deciding on deployment. What is
fixed in advance is everything above: the model, the rules, the candidates and the numbers reported
(section 5), so the reading can't be steered by choosing them after seeing the holdout. Small samples
(NFL part 1 has about 75 games, NBA none yet) should be read as such.

## 7. Order of work

1. Settle section 8; mark this spec FROZEN and record the git commit of the code.
2. Cluster: train the final models (`cluster/train_final.sbatch`, 15 tasks) and copy the weights back.
   MLB every-pitch model: `LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 sbatch --array=0-4 cluster/train_final.sbatch`
   (panel `panels/MLB_pitch_v3` = local `ingame/MLB_pitch/panel_v3`).
3. Laptop, with the holdout unsealed (`STG_SPORTS_UNSEAL=final-test`): build the holdout as-of layer,
   the training table and the panel (MLB with `PLAY_UNIT=pitch`, NFL), then run the integrity checks.
4. Predict (`b15_holdout_predict.py`, or `cluster/predict_holdout.py` for MLB), run the strategy
   (`b7_maker_trades.py` with `ONE_AT_A_TIME=1 DECIDE_S=5 ROW_KINDS=spread,total`; MLB also
   `SKIP_AGE=15,60`, in parts with `GAME_PART=i/n` if memory runs short), report (`b17_holdout_report.py`).
5. NBA, and NFL beyond week 5: rerun steps 3–4 after the season's data are pulled, with the same
   frozen weights. These count as a second holdout part under the same spec.

## 8. Choices made at freezing (2026-10-07)

| # | Choice | Decision |
|---|---|---|
| 1 | Exit | Both the fixed exit and holding to settlement, reported side by side as main results. **Changed 2026-10-08: the fixed exit is the main result; holding is descriptive (section 9)** |
| 2 | Final training | The same procedure as the test folds: the most recent 15% of games held back for early stopping (not trained on) |
| 3 | MLB postseason | Included, and reported separately from the regular season (section 5) |
| 4 | Holdout part 1 | Every game pulled by 2026-10-06; later pulls (NBA, the rest of the NFL season) are part 2 under this spec |

Postseason check, on pre-holdout data only (no holdout read):
- MLB: the 2025 postseason (36 games, 2025-09-30 to 11-01) sits in the training-only fold 0, so the
  final model has seen postseason games, but no test fold had any. The holdout is the first test on them.
- NBA, the closest analogue: the 2026 playoffs (81 games, test fold 5) traded fine with the fixed exit,
  NBA spreads 7.9c (t 2.4) vs 5.5c (t 5.1) in the regular season, totals 5.5c (t 1.7) vs 2.4c (t 2.5).
  Playoff markets were tighter (median width 4–5c vs 7c), and totals traded fewer contracts per game.
  Nothing suggests the playoffs need different rules.

Code commit: `eb92644` ("Add the holdout test path"). The test runs this code unchanged.

## 9. Changes after freezing, before any holdout result was seen

| Date | Change | Why |
|---|---|---|
| 2026-10-07 | **Plays sorted by the time they became known** (`b4_export.py`), and the model counts a play as known only once it and every earlier play have happened (`known_state`). NFL and NBA training panels rebuilt and their final models retrained; MLB unchanged (its plays were already in time order: 0 of 97,824 moved, same weights). | Found by the holdout integrity checks: in NFL (2% of plays) and NBA (0.1%) a play logged earlier than the one before it let the model read a later play up to ~1 min early, at 0.16% / 0.05% of steps (NFL scrimmage plays, NBA free-throw trips). On the test folds only 2–15 trades per candidate fell on an affected step, and removing them changed net per contract by ≤ 0.1c, so the test-fold results in section 4 stand. |
| 2026-10-07 | **Boundary games excluded:** 9 MLB games that started late on 2026-06-30 (UTC) sit in the holdout folder but start before the holdout; they are in neither set. | Section 1 defines the holdout by first play. |
| 2026-10-07 | **Part 1 completed:** the play-by-play pull had stopped on 2026-09-27 (MLB) / 09-29 (NFL), so NFL week 4 and the MLB postseason had markets but no plays, and their winner-contract trades were pulled mid-game. Re-pulled listing, trades and candles for every MLB / NFL winner, spread and total market closing from 2026-09-26, and the play-by-play. Part 1 = every game final at that re-pull (2026-10-07). Training-side files were restored byte for byte. | Section 8 choice 4 (everything up to 2026-10-06) assumed complete data; it wasn't. |
| 2026-10-08 | **A rescheduled doubleheader re-matched, not excluded:** BAL–NYY, 2026-09-25. Kalshi split each game across two listings (winner on the original Friday / Saturday listing, spreads and totals on new G1 / G2 listings whose winner legs never traded), and the matcher put the Saturday listing (really game 1: last trade 22:54:49, settled BAL at 22:55 as game 1 ended 10–2) on game 2. A documented override in `fetch_pbp.py` (`MANUAL_MATCH`) fixes it; both games are in, each with its own winner, spread and total contracts. Games with two *traded* contracts on the same key would still be left out (none now). One MLB feed that stops a run early (LAA–SF, 2026-07-24) is kept: results come from Kalshi. | Found by the integrity check (duplicate contract keys; game 1 had no winner price). The training data has no such case. |
| 2026-10-08 | **MLB: the every-pitch model and its own decision rule.** (1) MLB's model sees every pitch as a play (`PLAY_UNIT=pitch`), not only the end of each plate appearance; 5 final MLB models retrained. (2) MLB decides every 5 s, one order or position per contract at a time, and makes no decision 15–60 s after the last pitch the model knows. | Test folds only. The every-pitch model forecasts better (60 s R²: totals 11.5% vs 8.8%, spreads 13.1% vs 9.8%; `b13_seed_ensemble_gru_d10_h30-60-120-1800_tp_pitch.txt`). Deciding every 5 s lost money in MLB because ~80% of its trades came while a newer pitch had been thrown that the model hadn't seen (the 10 s feed delay); those trades earned ~0c vs ~10c when the model was up to date (`b25_mlb_pitch_bins.txt`). The time since the last known pitch predicts this (`b26_pitch_detect.txt`); skipping 15–60 s raised MLB totals from 2.3c (t 3.2) to 7.0c (t 5.1) and spreads from 3.2c (t 3.1) to 9.8c (t 5.1). Skipping beat trading a smaller size in that window, because a small order still blocks the contract. The 15 / 60 s window was read off the same test folds (pitch detection, not P&L), so the holdout is its test. |
| 2026-10-08 | **All six spread / total contract types are candidates:** MLB and NFL spreads added. | The user's choice: every contract type tradable. On the test folds NFL spreads are weak (3.3c, t 1.7) and are expected to be the least reliable. |
| 2026-10-08 | **The fixed exit is the main result for every candidate;** holding to settlement is descriptive. | The user's choice. Holding earns more per contract on the test folds, but each position carries the game's full risk until it ends. |
| 2026-10-08 | **Section 4 reference numbers updated** to the current models' test-fold trade logs. | The 2026-10-07 numbers came from an earlier run; the holdout is compared against what the current models did. |
| 2026-10-08 | **NFL holdout trade logs deleted before anyone read them** and rerun under these rules; the NFL forecasts (same model) are kept. The MLB plate-appearance forecast (`MLB_fold99_..._tp_seed0`) is no longer used. | Logs had been generated under the old rules, but no NFL or MLB holdout number had been looked at (confirmed by the user). |
| 2026-10-08 | **NBA and NFL decide every 5 s, one order or position per contract at a time** (no skip window), replacing every 60 s. **NFL part 1 had already been run and read under every 60 s** (spreads −2.1c, t −0.5; totals +4.0c, t 1.2), so for NFL part 1 this is a correction made after seeing results, and both are reported (the every-60-s result as descriptive). NBA has no holdout games yet, so for NBA, and for NFL part 2, it is fixed before any result. | The user's intended rule. The written revision kept "the other current rules" (every 60 s) for NBA / NFL, which differed from the user's intent. On the test folds every 5 s, one at a time had the best t for NBA spreads (t 7.7 vs 5.0), NBA totals (4.3 vs 2.6), NFL spreads (3.7 vs 1.7) and NFL totals (4.1 vs 2.8) (`b24_scale_in.txt`). Read NFL part 1 under this rule with care: the rule was confirmed after the every-60-s result was seen. |
