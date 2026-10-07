# In-game model: frozen spec for the holdout test

Written 2026-10-07, **before** any holdout data has been read. Everything below is fixed. The test
runs once. If a bug is found after seeing results, fix it, log it in `ingame_journal.md`, and report
both the before-fix and after-fix numbers.

**Status: FROZEN on 2026-10-07** at code commit `eb92644`.

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

1. **When:** every 60 s of game time, for each contract of the candidate type (section 4) that has a
   trade print in the last 60 s. That last print is the **reference price**.
2. **Signal:** the model's forecast price 120 s ahead minus the reference price (cents).
3. **Entry:** post on the side the model favours only. Buy limit at reference − 1c if the signal is
   positive; sell limit at reference + 1c if negative. The order rests for 60 s. It counts as filled
   only if a later print trades strictly through our price (so queue position doesn't matter).
4. **Exit: two versions, both reported as main results**, on the same entries and sizes:
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

| Candidate | Role | Test folds, fixed exit: net c/contract (game-level t) | Hold to settlement |
|---|---|---|---|
| NBA spreads, 120 s | main | 5.7 (t 5.6) | 8.4 (t 3.7) |
| MLB totals, 120 s | candidate | 7.2 (t 4.3) | 15.7 (t 5.5) |
| NBA totals, 120 s | candidate | 2.5 (t 2.7) | 4.7 (t 2.5) |
| NFL totals, 120 s | candidate | 7.4 (t 3.0) | 12.8 (t 3.2) |

Not tested, and not to be added after seeing results: MLB and NFL spreads, winner contracts, futures,
other holds.

Deferred to paper trading: a time stop for trades the fixed exit can't close (14–28% of trades find no
exit print within 4 minutes and ride to settlement, where the large losses sit; `b16_exit_vs_hold.py`).
The backtest can't price it: those trades have no print to exit against, and rebuilt quotes are not
trusted for taking. It needs the recorded live order book.

All of these levels were chosen on the same test folds they are measured on, so the holdout is
expected to come in lower.

## 5. What is reported, and in what order

For each candidate and each exit (section 3), using the 5-seed average forecasts:

1. **P&L before fees** (cents per contract), then **after fees** at the rates above, then after
   fees at **twice** those rates.
2. Contracts traded, total net P&L ($ at a 1-contract maximum per trade), game-level t (each game
   is one unit), share of game days with a profit, largest drawdown, and Sharpe ratio from daily
   P&L. The Sharpe ratio assumes unlimited capacity, so it is only a relative number.
3. Win rate (all trades, and weighted by size), average winning and losing trade, and for the
   fixed exit the share that found no exit print and rode to settlement.
4. Each single seed's net, to show how much the result depends on the seed.
5. MLB split into regular season and postseason; NBA (part 2) into regular season and playoffs.

Descriptive only: forecast R² against "no change" on the holdout; a 0 s feed delay.

## 6. How results are judged

There is no pre-set pass/fail rule: the results are interpreted when deciding on deployment. What is
fixed in advance is everything above: the model, the rules, the candidates and the numbers reported
(section 5), so the reading can't be steered by choosing them after seeing the holdout. Small samples
(NFL part 1 has about 75 games, NBA none yet) should be read as such.

## 7. Order of work

1. Settle section 8; mark this spec FROZEN and record the git commit of the code.
2. Cluster: train the final models (`cluster/train_final.sbatch`, 15 tasks) and copy the weights back.
3. Laptop, with the holdout unsealed (`STG_SPORTS_UNSEAL=final-test`): build the holdout as-of layer,
   the training table and the panel (MLB, NFL), then run the integrity checks.
4. Predict (`b15_holdout_predict.py`), run the strategy (`b7_maker_trades.py`), report (section 5).
5. NBA, and NFL beyond week 5: rerun steps 3–4 after the season's data are pulled, with the same
   frozen weights. These count as a second holdout part under the same spec.

## 8. Choices made at freezing (2026-10-07)

| # | Choice | Decision |
|---|---|---|
| 1 | Exit | Both the fixed exit and holding to settlement, reported side by side as main results |
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
