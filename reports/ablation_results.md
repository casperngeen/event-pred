# In-game model ablations: results (2026-10-23)

What each part of the in-game model contributes, measured by removing it, retraining, and trading the result under
the frozen rule. Written so that sections can be lifted into the report. How the runs were made is in
`ablation_runsheet.md`.

**Outputs**
- Full table: `analysis/graph_2026_10/out/b33_ablation_table.txt` (`.csv` alongside). Built by `b33_ablation_table.py`.
- Figures (`analysis/graph_2026_10/out/`), built by `b33_ablation_figures.py`:
  - `b33_fig_components.png`: model parts and inputs;
  - `b33_fig_delay.png`: feed delay.
- Per-version trade logs: `analysis/graph_2026_10/out/b7_maker_trades_*_ens5*_fold*.parquet`.

---

## 1. Method (for the report's methods section)

An **ablation** removes one part of the model, retrains it from scratch the same way, and measures what is lost.
Every version here differs from the frozen model in exactly one respect. Everything else stays the same: the
training data, the 5 walk-forward test folds, the 4 forecast horizons (30 s, 1 min, 2 min, 30 min), the 10-s feed
delay and the trading rule.

- **Seeds.** Each version is trained 5 times with different random seeds, and the 5 forecasts are averaged. With a
  single seed, the seed alone moved profit about as much as removing a part did (the earlier single-seed ablations
  are superseded for that reason).
- **Data.** Test folds only. The holdout (games from 1 July 2026) was not touched.
- **Forecast skill.** R² of the predicted 2-minute change in a contract's traded price, against predicting no change.
  It is reported as the mean over the 5 seeds, with folds weighted by their number of rows.
- **Trading result.** The averaged forecasts are traded under the frozen rule:
  - one decision every 5 s, and at most one order or position per contract at a time;
  - in MLB, no decisions 15–60 s after the last known pitch;
  - a limit order 1c inside the reference price, which fills only if the market trades strictly through it;
  - exit after 2 minutes, with the frozen position sizing.

  Profit is in dollars after Kalshi fees, over all test-fold games.
- **Comparison with the full model.** For each game, take the version's profit minus the full model's profit, over
  every game either one traded. Report the sum and a paired t-statistic (mean difference / its standard error).
  Negative means the removed part was earning money.
- **Multiple comparisons.** There are 76 version-vs-full comparisons. At |t| ≥ 2, about 4 would pass by chance; 41
  do (18 at |t| ≥ 3). Effects with |t| ≥ 3, or ones repeated across contract types, are the ones to rely on.

## 2. Results tables

**Forecast skill**: R² (%) of the 2-minute price change, with the change from the full model in brackets.

| Version | MLB spread | MLB total | NBA spread | NBA total | NFL spread | NFL total |
|---|---|---|---|---|---|---|
| **Full model (frozen)** | **7.47** | **6.68** | **6.74** | **5.76** | **7.10** | **10.92** |
| *A. Model parts* | | | | | | |
| No graph (no message passing) | 4.78 (−2.7) | 3.53 (−3.1) | 5.34 (−1.4) | 2.37 (−3.4) | 5.49 (−1.6) | 7.92 (−3.0) |
| No play memory | 6.94 (−0.5) | 5.24 (−1.4) | 6.20 (−0.5) | 5.16 (−0.6) | 7.26 (+0.2) | 10.36 (−0.6) |
| Neither | 4.43 (−3.0) | 1.86 (−4.8) | 5.02 (−1.7) | 1.88 (−3.9) | 5.95 (−1.1) | 7.13 (−3.8) |
| *B. Inputs* | | | | | | |
| No play types | 7.32 (−0.1) | 6.30 (−0.4) | 6.72 (−0.0) | 5.55 (−0.2) | 6.79 (−0.3) | 10.98 (+0.1) |
| No win probability (and the baseline built from it) | 4.73 (−2.7) | 4.23 (−2.4) | 3.84 (−2.9) | 2.27 (−3.5) | 1.12 (−6.0) | 2.26 (−8.7) |
| No structural baseline | 2.50 (−5.0) | 2.24 (−4.4) | 2.55 (−4.2) | 3.42 (−2.3) | 0.73 (−6.4) | 2.13 (−8.8) |
| No recent trades / order flow | 7.07 (−0.4) | 6.07 (−0.6) | 6.18 (−0.6) | 5.02 (−0.7) | 6.30 (−0.8) | 10.45 (−0.5) |
| No quotes (bid / ask) | 6.44 (−1.0) | 6.11 (−0.6) | 6.01 (−0.7) | 5.21 (−0.5) | 5.63 (−1.5) | 9.45 (−1.5) |
| *C. Feed delay (frozen: 10 s)* | | | | | | |
| 0 s | 8.50 (+1.0) | 8.09 (+1.4) | 8.77 (+2.0) | 8.15 (+2.4) | 7.25 (+0.2) | 12.35 (+1.4) |
| 5 s | 7.53 (+0.1) | 6.72 (+0.0) | – | – | – | – |
| 15 s | 6.96 (−0.5) | 5.94 (−0.7) | – | – | – | – |
| 30 s | – | – | 5.69 (−1.0) | 4.32 (−1.4) | 6.43 (−0.7) | 8.79 (−2.1) |
| 45 s (ESPN's live delay) | – | – | 5.61 (−1.1) | 4.33 (−1.4) | 6.12 (−1.0) | 8.18 (−2.7) |
| 60 s | – | – | 5.62 (−1.1) | 4.48 (−1.3) | 6.26 (−0.8) | 7.93 (−3.0) |
| *D. MLB play unit* | | | | | | |
| Every pitch, no skip window | 7.47 | 6.68 | – | – | – | – |
| Plate appearances, no skip window | 6.93 (−0.5) | 5.66 (−1.0) | – | – | – | – |
| Double plays coded correctly | 7.15 (−0.3) | 6.10 (−0.6) | – | – | – | – |

"Neither" = no graph and no play memory: only the current game state and each contract's own features.

**Trading result**: test-fold profit after fees. The full model's row shows profit, net cents per contract, and the
per-game t. Other rows show profit, with the paired t against the full model in brackets.

| Version | MLB spread | MLB total | NBA spread | NBA total | NFL spread | NFL total |
|---|---|---|---|---|---|---|
| **Full model (frozen)** | **$23, 9.8c, t 5.1** | **$26, 7.0c, t 5.1** | **$56, 7.5c, t 7.7** | **$44, 3.9c, t 4.3** | **$12, 6.5c, t 3.7** | **$16, 8.0c, t 4.1** |
| No graph | $10 (−3.8) | $12 (−3.7) | $52 (−0.6) | $26 (−2.2) | $7 (−2.1) | $8 (−2.5) |
| No play memory | $18 (−2.6) | $22 (−1.2) | $51 (−1.0) | $50 (+1.0) | $11 (−0.6) | $12 (−2.1) |
| Neither | $8 (−4.6) | $11 (−3.7) | $55 (−0.1) | $21 (−2.9) | $7 (−2.3) | $5 (−3.3) |
| No play types | $23 (−0.3) | $23 (−1.1) | $51 (−1.3) | $55 (+1.9) | $11 (−1.5) | $14 (−1.4) |
| No win probability | $13 (−2.6) | $9 (−4.5) | $37 (−2.4) | $9 (−3.5) | $5 (−2.5) | $1 (−4.0) |
| No structural baseline | $3 (−4.9) | −$1 (−6.3) | $21 (−4.3) | $10 (−3.5) | $5 (−2.4) | $1 (−4.0) |
| No recent trades / flow | $21 (−1.2) | $23 (−1.5) | $53 (−0.5) | $43 (−0.3) | $10 (−1.2) | $14 (−1.1) |
| No quotes | $20 (−1.3) | $25 (−0.5) | $45 (−2.2) | $49 (+0.8) | $9 (−1.7) | $12 (−1.8) |
| Delay 0 s | $35 (+3.0) | $59 (+6.5) | $58 (+0.4) | $63 (+2.4) | $12 (+0.2) | $21 (+2.6) |
| Delay 5 s | $26 (+1.2) | $33 (+2.3) | – | – | – | – |
| Delay 15 s | $6 (−4.3) | $6 (−5.1) | – | – | – | – |
| Delay 30 s | – | – | $44 (−2.2) | $31 (−1.6) | $11 (−0.7) | $11 (−2.5) |
| Delay 45 s | – | – | $40 (−2.7) | $30 (−1.8) | $10 (−1.5) | $10 (−2.6) |
| Delay 60 s | – | – | $44 (−2.1) | $24 (−2.5) | $9 (−1.8) | $11 (−2.0) |
| Every pitch, no skip window | $19 (−1.0) | $19 (−1.4) | – | – | – | – |
| Plate appearances, no skip window | $6 (−3.4) | $13 (−2.3) | – | – | – | – |
| Double plays coded correctly | $21 (−1.7) | $25 (−0.2) | – | – | – | – |

Every delay version stays profitable on its own, with per-game t from 2.3 to 8.8 (`t_game` in the full table).

## 3. Findings, section by section

### 3.1 Model parts (graph, play memory)

- **The graph matters most.** Without message passing between a game's contracts, R² falls in all six contract
  types, by 1.4–3.4 points, a 20–60% loss. Profit falls significantly in five of six (paired t −2.1 to −3.8).
- **NBA spreads are the exception.** R² still drops 1.4 points, but profit is unchanged (t −0.6).
- **Without the graph, the model trades less.** The number of contracts traded drops by 30–50% in MLB and NFL (MLB
  spreads 236 → 168, NFL spreads 188 → 111). Except for MLB spreads (9.8c → 6.0c), profit per contract changes
  little, so most of the loss is fewer trades: the graph mainly lets the model find more trades it is confident in.
- **Play memory is second-order.** Replacing the GRU's summary of past plays with zeros costs 0.5–1.4 R² points (NFL
  spreads: +0.2).
  Profit changes are mixed: significant for MLB spreads (t −2.6) and NFL totals (t −2.1), and noise elsewhere (NBA
  totals +1.0).
- **"Neither" is about as bad as "no graph", or slightly worse.** Most of the loss comes from removing the graph, not
  the memory.

### 3.2 Inputs

- **The structural baseline drives the model.** The baseline is each contract's fair price implied by the game state:
  win probability, score, time left and the contract's threshold. Removing it costs the most of any ablation:
  R² falls by 2.3–8.8 points, and NFL R² goes from 7–11% to about 1–2%. Profit falls in every contract type
  (t −2.4 to −6.3). MLB totals turn to a loss (−$1).
- **Removing win probability hurts almost as much.** That version also removes the baseline built from it, so the
  two cannot be fully separated.
- **What happened on the play adds almost nothing.** Removing play types (home run, strikeout, turnover, …), while
  keeping score, time and win probability, changes R² by at most 0.4 points and profit by noise. The model reads a
  play through its effect on the game state, not through its type.
- **Recent trades and quotes matter little individually.** Removing each costs 0.4–1.5 R² points, and profit by t of
  at most 2.2. **Interpretation caveat:** neither ablation hides the contract's reference price (its last traded
  price), which every version still sees together with the gap between the baseline and that price. The two groups
  also carry overlapping information about where the market is. So these rows do **not** show that market prices
  are unimportant. They show the model does not need the extra detail (bid/ask, trade age, flow, volume) beyond
  the reference price. A version without the reference price was not run.

### 3.3 Feed delay

Each delay is a separate model, trained and tested with plays arriving that many seconds late.

- **NBA and NFL: the edge survives at ESPN's live delay.** At 45 s, about the median delay of the free ESPN feed,
  the model keeps roughly two thirds to four fifths of the frozen model's profit:

  | Contract | Profit at 45 s | Net per contract | Per-game t |
  |---|---|---|---|
  | NBA spread | $40 | 5.5c | 6.1 |
  | NBA total | $30 | 2.8c | 3.1 |
  | NFL spread | $10 | 5.8c | 3.0 |
  | NFL total | $10 | 5.9c | 2.8 |

  The loss against 10 s is significant for some contract types (t −1.5 to −2.7) but far from total.
- **NBA/NFL: the curve flattens after 30 s.** Skill and profit fall from 0 s to 30 s, then hardly change through
  60 s. What the model loses with delay is the reaction to the most recent play; what remains does not depend on
  fresh play-by-play.
- **MLB: speed matters more.** Going from 10 s to 0 s adds 1.0–1.4 R² points and much profit (totals $26 → $59,
  t +6.5). Going to 15 s cuts profit by three quarters (t −4.3 and −5.1).
- **Caveat on MLB 15 s:** the frozen skip window confounds part of that drop. At 15 s, every pitch becomes known
  exactly when its 15–60-s skip window starts, so the model can almost never act on a fresh pitch. Contracts traded
  fall from 236 to 86 (spreads). The R² drop (−0.5 to −0.7) is the cleaner measure of what the model itself loses.
  Similarly, part of the gain at 0–5 s is extra trading time before the skip window.
- **Live MLB feed.** The live MLB feed arrives 7–15 s after the pitch (about 7 s with the cache-busting query
  parameter, about 15 s without). The frozen 10-s model sits inside that range, so keeping the feed fast is worth
  real money.

### 3.4 MLB play unit

- **Modelling every pitch beats modelling plate appearances.** R² is higher by 0.5–1.0 points. Without the skip
  window, profit is $19 vs $6 (spreads) and $19 vs $13 (totals).
- **The skip window raises profit per contract a lot.** It goes from 3.2c to 9.8c (spreads) and from 2.3c to 7.0c
  (totals), by avoiding orders that get filled just after a pitch, when better-informed traders pick them off. Total
  profit rises only slightly, because there are fewer trades.
- **Coding double and triple plays correctly** changes nothing beyond noise (decided 14 Oct, not deployed).

## 4. Key insights for the report

1. **The graph earns its place, unlike in the macro study.**
   - In the earlier macro-release work (`graph_ablation.md`), no graph model beat a linear one, and removing the
     graph never hurt.
   - Here, message passing between the contracts of the same game is the most valuable part of the architecture.
     It raises forecast skill in all six contract types, and profit in five.
   - The difference is structural. A game's spreads and totals are tightly linked by construction: same game, same
     score, neighbouring thresholds. So information in one contract's price really does tell you about another's.
   - This contrast is a good thesis-level conclusion: graphs help when the edges are mechanical, and not when they
     are loose economic relations.
2. **The model's edge is a fair-value gap, not a news-reaction edge.** Three results point the same way:
   - removing the structural fair-price baseline destroys skill and profit;
   - removing the play type costs nothing;
   - in NBA/NFL, most of the profit survives a 45–60 s delay.

   The model makes money mainly by comparing where a contract should be priced, given the game state, with where it
   last traded, and trading the gap. This matches the live finding (`forward_test_log.md`, stale-print analysis)
   that signals often come from a contract's last trade lagging the market.
3. **That same mechanism is the main risk to live profit.** A gap against a stale last trade is exactly where the
   backtest's fill assumption is weakest. Live MLB paper trading showed limit orders priced off a stale reference
   fill mainly when the price reverses. The comparisons between versions here are fair, since they all share the
   fill model, but the absolute profit levels are an upper bound. Say this explicitly next to any profit figure.
4. **Feed latency matters differently by sport.**
   - MLB profit is steeply sensitive to the delay. The free statsapi feed is good enough when cache-busted (median
     about 7 s in a live test), but at about 15 s, its delay without cache-busting, most of the edge is gone.
   - NBA and NFL lose some edge by 30 s but are still profitable in backtest at ESPN's 45 s.
   - The case for a paid low-latency feed is therefore strongest for MLB, and weaker than assumed for NBA/NFL. The
     NBA/NFL edge at 45 s is still unproven live.
5. **What happened matters through the game state, not the play label.** Play types add nothing. Play memory adds a
   little. A simpler model without play-type inputs would lose essentially nothing. This is useful for the design
   discussion and for any "future work: simpler model" point.
6. **Forecast skill and profit usually agree, but not always.** For example:
   - NBA spreads lose R² without the graph but keep their profit;
   - removing quotes raises MLB spreads' profit per contract while lowering R².

   Profit depends on the few confident, fillable signals, not on the average forecast, so the report should show
   both measures.
7. **Averaging seeds helps every version.** The 5-seed average's R² is 0.3–1.3 points above the mean single seed in
   every row (the `r2_ens` column). This supports the frozen model's use of a 5-seed average.

## 5. Limitations to state

- **Test folds, not the holdout.** The trading rule and sizing were chosen on these folds, which flatters every
  version equally. Differences between versions are the result here; absolute levels belong to the holdout section.
- **Optimistic fills.** Fills are simulated as "strictly through the limit price" on historical trades. Live, with
  orders priced off a stale reference, fills were rarer and adverse (see 4.3).
- **Small amounts.** Frozen sizing gives at most about $60 per contract type over the whole test period. The result is a
  statistical edge, not a sizeable one.
- **Games are treated as independent** in the t-statistics. Games on the same night share market conditions, so
  the true uncertainty is somewhat larger.
- **One rule for all versions.** The rule was tuned on the 10-s model. That is the fair way to isolate the model's
  contribution, but versions that would prefer a different rule are disadvantaged (most clearly MLB at 15 s, 3.3).
- **Groups were removed one at a time.** Overlapping information, such as quotes and recent trades, or win
  probability and the baseline, is not separated. Possible follow-ups:
  - remove trades and quotes together;
  - remove the reference price itself;
  - rerun MLB 15 s with the skip window measured from when the pitch becomes known. Its averaged forecasts are
    still on the cluster under `results/ingame/ens5/`.

## 6. Suggested figure and table captions

- **Table: ablation, forecast skill and profit.** "Each row retrains the in-game model with one part removed (5
  seeds averaged; walk-forward test folds; frozen trading rule). R² of the 2-minute traded-price change; profit
  after fees with the paired per-game t against the full model."
- **`b33_fig_components.png`.** "Change in forecast skill (left) and paired per-game t of profit (right) when each
  part of the model or group of inputs is removed. Red: removing it hurts. Bold: |t| ≥ 2."
- **`b33_fig_delay.png`.** "Forecast skill (top) and test-fold profit (bottom) of models trained and tested with
  plays arriving 0–60 s late. Red band: the live feed's measured delay. Dotted: the frozen model (10 s)."
