# In-game MLB forward test: game log

A running record of every MLB game in the live paper test, one row per game. Decisions go in `ingame_journal.md`;
this file holds only the results and what happened each night.

**What runs:**
- The live runner (`scripts/live_trade.py`, on the lum.id sandbox) trades the frozen MLB rule on paper, on spread
  and total contracts. It publishes the totals signals to Lumid.
- From 13 Oct (US Eastern date) it receives trades over Kalshi's WebSocket and decides at pitch end + 10 s, the
  timing the model was trained on. The 12 Oct games used slower REST polling and are kept out of comparisons (journal,
  2026-10-14).

**How the columns are worked out:**
- **Live:** the runner's own paper orders.
- **Backtest replay:** the same game re-run through the same engine (`scripts/live_rerun.py`, scenario
  `d10,tape,tape,prints`) under the backtest's assumptions:
  - each pitch is known exactly 10 s after it ends, taken from the statsapi archive feed;
  - every trade is seen the moment it prints;
  - a resting order fills only when a later trade prints through its price.
- **P&L:** in cents, after the maker fee, and sized by the frozen sizing rule. One full-size order is 1 contract.
  Positions still open at the end are valued at the contract's result.
- **Per contract:** the sized P&L divided by the sized contracts, so it measures edge whatever the scale.

| ET date | Game (final) | Live: orders / fills | Live: sized contracts | Live: P&L | Backtest replay: orders / fills | Backtest replay: contracts | Backtest replay: P&L | Notes |
|---|---|---|---|---|---|---|---|---|
| 12 Oct | LAD @ MIL | 13 / 1 | 0.000 | 0.0c | 58 / 8 | 0.181 | −3.4c (−19.0c/contract) | REST polling; left out of comparisons |
| 12 Oct | CLE @ TB | 15 / 4 | 0.353 | +1.0c | 34 / 9 | 2.212 | +21.1c (9.5c/contract) | REST polling; the backtest's gain is mostly one full-size sell of the over 9 at 66 (+25c) that the slow live feed never placed |
| 13 Oct | CLE @ TB | 40 / 4 | 0.404 | +7.8c (19.3c/contract) | 41 / 6 | 0.442 | +7.8c (17.7c/contract) | Live matched the backtest replay |
| 14 Oct | MIL @ LAD (MIL 6–2) | 19 / 3 | 0.403 | +11.1c (27.5c/contract) | 21 / 4 | 0.009 | +0.15c | Live P&L is one fill: a sell of the over 9 at 33 in the bottom of the 9th, filled on a buying burst during a ball in play, which turned into a lineout (+27.5c × 0.40). The replay didn't place that order, so the difference comes from one batted ball, not an edge |
| 15 Oct | TB @ CLE (CLE 3–2) | — | — | — | 40 / 5 | 0.013 | +0.13c (9.9c/contract) | **Missed live:** every runner died silently after the 12:31 UTC start. Replay built from public data (`data/sports_live/sandbox/2026-10-15/missed/`). All 5 fills were sells of the over near the 5c minimum signal; unsized they made +41.6c, one bet that no more runs would score. The two strong signals (+14c, buy TB −3.5 at 7 and CLE −2.5 at 18) didn't fill |
| 15 Oct | MIL @ LAD (MIL 6–4) | 18 / 7 (from 02:15 UTC) | 0.128 | +0.59c (4.7c/contract) | 44 / 9 | 0.194 | +2.25c (11.6c/contract) | Runners restarted by hand at 02:15 UTC (6th inning). A duplicate set ran 02:15–02:19 (its 3 orders kept once, none filled) and the sandbox was recreated at 02:20. **Missed part (before 02:15), from the replay:** 23 orders, 2 fills, +3.24c, almost all from a buy of the over 13 at 27 in the top of the 5th, before MIL tied it 4–4 (+22c × 0.15). Six orders of size 0.86–1.0 (signals 14–21c) didn't fill. **From 02:15:** live +0.59c against the replay's −0.99c, both from 7 fills. Live got a sell of the over 11 at 77 (size 0.10, +9.3c) that the replay didn't place. Both lost on a sell of the over 10 at 50 in the 8th, which MIL's 9th-inning run settled yes (−50c; size 0.01 live, 0.02 replay). The sizeable live orders (sell over 14 at 34, signal −14, size 0.84; sell over 13 at 29, size 0.37) didn't fill. Replay before 02:15 plus live after: +3.8c |
| 16 Oct | TB @ CLE (postponed, rain) | 0 / 0 | — | — | — | — | — | ALCS Game 4 postponed for rain before the first pitch; the runner waited with no plays and placed nothing |
| 17 Oct | TB @ CLE, Game 4 (TB 7–3) | 137 / 7 (**invalid**) | — | — | 66 / 15 | 2.644 | +16.4c (6.2c/contract) | **Traded the wrong game's contracts.** Kalshi re-pointed its events by game number after the rain-out: its 26OCT16 contracts settled on this game, its 26OCT17 contracts on Sunday's Game 5. The runner matched by date and took 26OCT17, so it priced Sunday's game off Saturday's plays. Excluded. **Replay** (public data, Kalshi's 26OCT16 contracts, `data/sports_live/sandbox/2026-10-17/missed/`): almost all from one full-size sell of the over at signal 21 (+23.9c), offset by a full-size spread sell at signal 18.5 (−10.8c); spreads −10.4c, totals +26.8c |
| 18 Oct | TB @ CLE, Game 5 (TB 3–1) | 0 / 0 | — | — | 85 / 9 | 0.608 | +9.1c (15.0c/contract) | **Missed: found 0 contracts.** The 17 Oct "fix" looked under the 16 Oct date (MLB's rescheduled-from date), whose contracts Kalshi had already settled on Saturday's game. Feed fine (median 10.1 s). **Replay** (26OCT17 contracts, `data/sports_live/sandbox/2026-10-18/missed/`): mostly one spread sell at signal 12.5, size 0.56 (+8.5c); totals filled only near the 5c minimum (0.02 contracts) |
| 19 Oct | CLE @ TB, Game 6 (TB 7–3) | — | — | — | 36 / 6 | 0.013 | −0.05c (−4.0c/contract) | **Missed: no runner.** The lum.id gateway failed (502s) and the sandbox was stuck shutting down from ~16:30 UTC, before the game (see incidents). Contracts were right ("Game 6" event). **Replay** (public data, `data/sports_live/sandbox/2026-10-19/missed/`): effectively flat. All 6 fills came on 5–7c signals with near-zero size; the 7 orders on 10c+ signals all went unfilled. The local order book (innings 1 to top 4) shows why for the 3 strong signals in that stretch: each was measured against an off-market last print, not the book (a stale 31 after the book had already jumped to 42/44 on a scoring play; two thin sweeps at 50 and 89 with the book at 32/33 and 75/76), so the limit sat 10–15c from the book and could not fill. The "move 120 s" column is large for the same reason (it is measured from the last print), not because the model foresaw the moves. Local backup recording (books every 15 s) in `data/sports_live/local/`, with the books missing 01:24–02:58 UTC (Mac lid closed) |

**From 13 Oct, live games only (13 Oct, 14 Oct, and 15 Oct MIL @ LAD from 02:15):** 77 orders, 14 fills, 0.94 sized contracts, +19.5c (20.8c/contract).

## Analyses

### Signal strength vs result, and the sizing rule (16 Oct, live games to 15 Oct MIL @ LAD in the 9th)

Script: `livebt/sizing.py` in the session scratchpad (live orders through `paper.py`). Backtest: MLB test folds,
frozen rule and frozen sizing, spreads and totals.

**A stronger signal is followed by a bigger move, live as in the backtest.** The move is the price 120 s after each
order against the last trade the order was priced off, filled or not.

| \|Signal\| | Live orders (from 13 Oct) | Average 120 s move in our favour | Fill rate |
|---|---|---|---|
| 5–7c | 33 | +4.0c | 20% |
| 7–10c | 17 | +8.6c | 20% |
| 10–15c | 21 | +13.5c | 10% |
| 15c+ | 1 | +19.0c | 0% |

Live correlation +0.34 (72 orders, p < 0.01); backtest +0.34 over 102,000 signal steps.

**Per fill, a stronger signal makes more.** The backtest decides this; live points the same way but is far too small.

| \|Signal\| | Backtest fills | Backtest per contract, unsized | Live fills | Live per contract, unsized |
|---|---|---|---|---|
| 5–7c | 3,338 | −4.4c | 7 | +2.9c |
| 7–10c | 1,064 | −2.8c | 3 | +11.9c |
| 10–15c | 465 | +5.9c | 2 | +24.4c |
| 15c+ | 336 | +11.7c | 0 | — |

Correlation of signal with P&L per fill: backtest +0.17 (5,203 fills, p < 0.01); live +0.33 (12 fills, p 0.29).

**Frozen sizing compared with flat sizing on the same fills:**

| | Backtest | Live, 13 Oct onward (12 fills) |
|---|---|---|
| Frozen: contracts / P&L / per contract | — / +$49 / +8.1c | 0.93 / +19.5c / +20.8c |
| Flat, 1 contract each | −$109 / −2.1c per contract | 12 / +104.7c / +8.7c |
| Frozen scaled to the same 12 contracts | — | about +250c |

- Flat made more live only because it traded about 13× the contracts.
  - The extra came from 7 weak-signal fills (+20.2c in total, ranging from −49c to +21c), which lose on average in
    the backtest.
  - At equal exposure the frozen sizing made more.
- Scaling the frozen sizing up has two catches:
  - It makes single orders large (the biggest live fill would be about 5 contracts), but the paper fills assume
    any size fills on one print through the price.
  - About 90% of the P&L comes from the 2 strong-signal fills.

**What limits P&L is filling strong signals, not the sizing.** Live, 22 orders had a signal of 10c or more and 2
filled; those 2 made 17.2c of the 19.5c. A big signal usually means the last trade jumped away from the model's value.
The limit, 1c inside that trade, fills only if the jump continues through it, and usually the price reverts first. Open
follow-ups:
- an entry rule for 10c+ signals (price nearer the model's value, or take the best price), decided on the test folds;
- how often a 3–5 contract order would fill in full.

### Paper-trading analytics, live games from 13 Oct (after the 15 Oct MIL @ LAD final)

Script: `livebt/paper.py` in the session scratchpad. It covers 3 games: 13 Oct, 14 Oct, and 15 Oct from 02:15 UTC.

**Totals:** 77 orders, 14 fills (18%), 0.93 sized contracts, +19.5c (20.8c/contract).

**15 Oct MIL @ LAD:**
- 18 orders, 7 fills (39%, the highest yet), +0.6c.
- Almost all the fills were small sells of the over on 5–7c signals, made while the over prices were falling late
  in the game.
- Over its 44 decision steps with a signal of 5c or more, the price moved +8.4c our way in 120 s. On 14 Oct it moved
  −5.2c.

**Order outcomes against the backtest:**
- Live fills 17–19% of orders, against 11–12% in the backtest.
- The full trade tape never traded through an order the runner missed.
- Unfilled orders are mostly "price never reached": 69–78% live, 57–61% in the backtest. Live has fewer "no trades
  in 60 s", because live orders sit on the busier lines.

**By signal size (sized per contract):**

| \|Signal\| | Live | Backtest |
|---|---|---|
| 5–7c | −10.3c (0.03 contracts) | −1.5c |
| 7–10c | +11.8c | −1.1c |
| 10–15c | +24.9c (2 fills) | +5.9c |
| 15c+ | no fills | +12.4c |

**Exits:**

| How the trade closed | Fills | Average per fill |
|---|---|---|
| Exit limit filled | 9 | +17.1c |
| At the next opposite trade | 3 | +16.1c |
| Held to settlement | 2 | −32.3c |

The two settled fills are the sell of the over 10 on 15 Oct and a buy of the over 10 on 14 Oct.

**How the 15 Oct over-10 loss happened** (an example of the pick-off pattern):
- **02:56:25:** a pitch was put in play (it became a groundout). The market had been 30 bid, 32 ask.
- **02:56:27.1:** one buyer swept the thin offers above 33, about 9 contracts in total, with the last trade at 49.
  The quote was left at 29–30 bid and 47–50 ask.
- **02:56:27.7:** the runner priced off that 49. With the bid at 30 and the ask at 46–47 (mid about 38), the sell at
  50 sat behind the best ask; it was not a fill waiting to happen.
- **Why the runner wasn't stopped:** the live feed didn't yet show the ball in play, and the last known pitch had
  ended only 4 s earlier, outside the 15–60 s skip window. The model's value also jumped briefly to 43 (from about
  29).
- **The order:** sell the over 10 at 50, signal −6, size 0.01. Within 5 s the ask fell back to 40, then to 28–30 by
  02:56:52, so the order sat about 20c above the market.
- **The fill:** the order filled only at 02:57:12, when MIL hit a home run and buyers swept the over 10 from 28 to
  95 in 0.2 s. It was held to settlement and lost 50c.
- **The lesson:** a resting order far from the market fills only on a big event, and those events go against it.
  The weak-signal size kept the cost to 0.5c.

**Signal steps, live against the backtest:** these cover every decision step with a signal of 5c or more, including
the 12 Oct games, and give the average 120 s move by signal size.

| \|Signal\| | Live | Backtest |
|---|---|---|
| 5–7c | +0.8c | +3.8c |
| 7–10c | +5.1c | +7.0c |
| 10–15c | +7.7c | +11.0c |
| 15c+ (19 steps) | +3.5c | +18.9c |

The moves still rise with the signal, but they are weaker than in the backtest.

### Should a resting order be cancelled once the market leaves it behind? (16 Oct, after the 15 Oct over-10 loss)

Scripts: `strand/stranded.py` (test folds, based on the 14 Oct order-management simulator), `strand/report.py` and
`strand/live_strand.py`, all in the session scratchpad.

**The rule tested:** cancel an entry as soon as the market is X c or more away from its limit, on the side where it
can't fill (for a sell at 50, a trade at 40 or below when X is 10c).
- The distance is judged on the last trade, or on the median of the last 3 trades.
- The cancel takes effect 0.5 s after the trade that triggers it; a fill before then still counts.
- It is checked on every trade. Everything else is the frozen rule and sizing.

**How often it happens:**
- **Live (5 games, 19 fills):** only one fill had been left 10c or more behind before it filled: the 15 Oct over-10
  sell (23c behind; −50c a contract, −0.5c sized). It was also the only one 10c+ behind the order book's best price.
  - 52 of the 86 unfilled live orders were left behind like this and simply expired.
- **Backtest (test folds):** 368 spread fills and 479 total fills (19% and 15% of all fills) had been left 10c or more
  behind (median of 3) before they filled.

**Result: cancelling loses money in every version.**

| Rule | Spreads vs frozen | Paired game t | Totals vs frozen | Paired game t |
|---|---|---|---|---|
| Last trade, 5c | −$9.6 | −3.6 | −$6.8 | −2.2 |
| Last trade, 10c | −$9.2 | −3.5 | −$7.0 | −2.3 |
| Last trade, 20c | −$6.5 | −2.7 | −$7.7 | −2.7 |
| Median of 3, 10c | −$8.2 | −3.2 | −$4.7 | −1.7 |
| Median of 3, 15c | −$6.5 | −2.7 | −$5.8 | −2.2 |

The frozen rule made $23.2 on spreads and $25.7 on totals.

**Why:** a stranded order that does fill is usually catching a burst of one-sided trading that then reverts, and
fading it pays.
- The stranded fills removed at 10c (median of 3) made +2.7c (spreads) and +2.9c (totals) a fill unsized, and $9.1
  and $10.2 sized. All fills together average −1.5c and −2.4c unsized.
- Those that were held to settlement do lose (−5.7c spreads, −13.0c totals a fill), which is the 15 Oct case. But
  they are outweighed by those exited at the limit or the next opposite trade (+4c to +9c a fill).
- This matches the earlier finding that big one-sided bursts of trading are mostly uninformed.

**Not tested:** a cancel judged on the order book's quote (the folds have no book). Live, it would have flagged the
same single fill.

**Runner incidents:**
- 15 Oct: every runner was gone soon after the 12:31 UTC start, with no error logged. Since 16 Oct, an hourly
  watchdog (`deploy/lumid/watchdog.sh`) recreates the sandbox and restarts the day if any runner is down.
- 16 Oct: TB @ CLE rained out. The runner looks up contracts by the date it runs (event tag `26OCT18`), so on
  18 Oct it would have found none. The 17 Oct fix (look under statsapi's `rescheduledFromDate`) was wrong: Kalshi
  re-points postseason events by game number, not by MLB's game id, so Saturday's Game 4 runner traded Sunday's
  contracts and Sunday's found none. Fixed 19 Oct: `live_record.contracts` takes the open game-winner event titled
  "Game N" (statsapi `seriesGameNumber`), else the date; checked on the weekend's listings and on Game 6.
- 18 Oct: with 0 contracts, the MLB runner's `trades.jsonl` grew to 1 GB (the REST trades poll apparently ran
  without a ticker filter). Harmless, but a runner should stop when it finds no contracts.
- 19 Oct: the sandbox was created at 12:30 UTC, but the lum.id gateway answered about half of SSH and API calls
  with 502, so `start_day.sh` never ran. The watchdog's 13:35 retry also hit 502s. Started by hand at ~13:55
  UTC (Game 6, 849815: 18 contracts, event "Game 6", checked). The watchdog now skips an hour when the API listing
  fails and retries SSH 8 times, instead of recreating the sandbox (which would kill healthy runners). Backup:
  a local `live_record.py` for 849815 (REST books every 15 s, trades, feed) in `data/sports_live/local/`.
  Then the sandbox died anyway: from ~16:30 UTC (before the game) lum.id showed it "Terminating" and it stayed so
  all night, so the watchdog's recreates were refused ("still shutting down") and it hit its 3-a-day limit. No
  MLB orders or Lumid signals, no NFL dry run, no WebSocket books. Only the local backup recorded the game; its
  books stop 01:24–02:58 UTC, when the Mac went to sleep on battery with the lid closed (caffeinate cannot stop
  that).
- 15 Oct, MIL @ LAD restart side effects:
  - Each of the three starts re-fetched the game's earlier trades over REST, so `trades.jsonl` holds 524,740 rows
    for 218,887 unique trades. Trading is unaffected; de-duplicate by `trade_id`.
  - The order-book and trade stream (`books_ws`, `ws_trades.csv.gz`) only covers 02:20 UTC onward.
  - The logged feed delay (median 2,628 s) counts the replayed earlier plays and is not a real delay.
