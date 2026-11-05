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
- 23 Oct (first night for the NBA and NFL 60 s models): the NBA and NFL runners crashed on every start because
  the 60 s model files (`results/ingame/deploy/*_d60_*`) were missing on the sandbox, even though a copy had been
  checked there earlier that day. The folder was last changed 12 Oct, so it is unclear whether that copy ever
  landed. After 3 watchdog restarts (13:16, 14:19, 15:23 UTC) it hit its daily limit. No NFL games that night.
  12 NBA games went untraded and the live ESPN feed wasn't recorded. The WebSocket order books with trades were
  recorded for all 12 NBA games (1–2.5 GB each before gzip), so the night can be replayed against the final ESPN
  play-by-play. The replay can't test the live-feed guards (feed delay, review, placeholder rows). Models recopied
  24 Oct 03:50 UTC: 20 files checksum-identical, and both load on the sandbox.
  MLB 849805 (TB @ MIL) ran normally: 263 pitches, feed delay median 12.5 s, 10 orders, 0 fills.
  NBA runner restarted by hand inside the same sandbox at 04:28 UTC, so the order-book stream kept running. Three
  games were still live (MEM @ GSW 401909850, LAC @ LAL 401909851, POR @ SAC 401909852), each from about the fourth
  quarter, and the first order was placed by 04:34. These games' logged feed delay (median ~4,000 s) counts the plays
  replayed at the restart and isn't a real delay. The other 9 games had already finished: 0 open contracts.
  Result of the late start: 2 orders, 0 fills. LAC @ LAL sold 0.24 of the LAL spread at 27c (04:34). MEM @ GSW sold
  1 of the total-over-244 at 40c (04:47). Both expired unfilled after 60 s; POR @ SAC placed no order. The runners
  logged plays, forecasts and orders normally.

### 24 Oct NBA night on the backup server: why the live run and the backtest replay differ (25 Oct)

**What was done:** the 8 NBA games (dry run) were re-run through the same engine, from the files the runner recorded, in two ways:
live inputs (each ESPN version when it arrived, trades and candles when received) and the backtest's assumptions (every play known
exactly 60 s after it ended, trades seen the moment they print). 20–23 Oct's 27 games were also built from the public APIs
(`scripts/build_nba_replay.py`, folders in `data/sports_live/replay_hist/`) and replayed under the backtest's assumptions only.
The tool is `scripts/live_rerun_nba.py`. The ESPN feed on 21–23 Oct was not recorded live, so those nights cannot show feed problems.

**The replay is trustworthy:** replaying the live inputs reproduces the live orders (160 vs 159 orders, 64 vs 64 fills, −687c vs −701c).

**Live against the backtest assumptions, 24 Oct (cents, spreads / totals):** live −105 / −595; backtest assumptions +148 / −955.
- Spreads: about 200c of the 250c gap is one game (BKN @ CLE). The runner bought the Cleveland +6 spread (5 contracts at 40c, closed at 5c, −177c) and the +7 (−28c) on a +22c forecast; the replay on real times forecast +4c and did not trade.
- Totals: the backtest assumptions are worse than live, so the totals loss is not an execution effect.

**What the ESPN feed did in that game:** it was normal until 392 s before the order, then one document version arrived in 252 s while the game went on, then the missed plays came in clumps (delays of 90–410 s). Other games polled in the same minutes kept updating, so it was this game's document (ESPN's data or its cache), not our polling. Across the 8 games, 7.6% of plays arrive more than 120 s late (137 episodes, about 17 a game); half are a single old play added late, the rest are clumps after a quiet gap. After the last late play, a normal play arrives within a median of 30 s, five take 90 s.

**Where the losses fall:** 56 orders were placed within 180 s of a late play (21 fills, −585c). The loss is in the 60–120 s after it (11 fills, −566c, three trades make 394c), not the first 60 s (5 fills, about −25c). Orders after a quiet gap lost −29c a fill (20 fills); after an isolated late play, −3c (1 fill).

**The old guard let it through:** it paused trading while the median delay of the last 5 new plays was above 120 s; the median fell back under 120 s as the clump arrived, with late plays still among the last 5.

**Hypothesis (partly supported):** the engine stamps a play that arrived late with its arrival time, so a clump of late plays is bunched in the model's play history, which training never contains. This kept the forecast off for a minute or more after the feed looked healthy.
- Test: forecast a second time with every arrived play stamped at its real end time (what the backtest sees). On all 249,000 NBA decision rows with a later print, the two disagree on 20% of trade-sized signals (5c or more): 57% within 60 s of a late play, 26% at 60–120 s, 8% at 120–180 s, 4% at 3–5 min, 0.3% after.
- Not shown: that the real-stamp forecast is better. When they disagree, the live side's price moved +1.1c over the next 120 s and the real-stamp side −1.4c; in the 60–120 s group the live side moved −0.5c against +3.5c when they agree (217 rows, the same contract repeated every 5 s, so far fewer independent cases). Skill against the 120 s price move is near zero for both (spreads R² −1.7% vs −0.6%, totals −3.6% vs −4.9%).
- So the distortion is real and fades over about 3 minutes, but that it is what costs money is not established.

**Trials (live inputs, 8 games, no number tuned):**

| Rule | Orders / fills | Total | Spreads | Totals |
|---|---|---|---|---|
| Old: median of last 5 > 120 s | 159 / 64 | −701c | −106c | −595c |
| Any of last 5 > 120 s | 140 / 58 | −496c | +98c | −594c |
| Plus: skip unless the real-stamp forecast agrees | 138 / 56 | −394c | +200c | −594c |

A fixed 180 s pause after any late play (109 / 46, −175c) was also tried and dropped: its length was chosen after seeing this night, and a time-based pause only worked because it outlasted the distortion.

**Backtest against this season (replay of 20–24 Oct, 35 games, backtest assumptions, closed trades net per contract):** spreads +2.2c (53 closed fills; test folds +5.5c), totals −1.0c (103 closed; test folds +2.8c). Forecast slope of the actual 60 s move on the forecast: spreads 0.83, totals 0.57 (24 Oct alone: spreads 0.64, totals 0.32). R² here is not defined as in the folds (spreads 5.1%, totals 1.1% against fold means 10.2% and 7.4%), so compare slopes, not levels. A like-for-like check on last season's games has not been run.

**Status:** the guard change and the shadow forecast are in `scripts/live_trade.py` (uncommitted); nothing is deployed to the sandbox or the backup server. Next: run the shadow forecast on later nights before trading on it.

### 24 Oct sandbox night: NBA and NFL runners crashed on missing 60 s models (25 Oct)

**What happened:** on the sandbox, the NBA and NFL runners died at start with `FileNotFoundError ... d60 ... seed0.pt`, as on 23 Oct. The
watchdog recreated the sandbox three times (21:45, 00:55, 02:04 UTC) with the same error, then hit its limit of 3. The sandbox traded no
NBA games that night (MLB TB @ MIL and the `books_ws` recorder ran normally).

**Cause:** `/home/casperngeen/app/results` is a symlink into the container image (`/app/results`), not the persistent home folder. The
24 Oct 03:50 UTC recopy of the d60 models went into the running sandbox's image folder and was lost when the sandbox was recreated. The
Dockerfile only bakes in the 10 s MLB/NFL models. The same thing happened on 23 Oct.

**Fix (25 Oct):** the 10 d60 weights (+ json) are in `/home/casperngeen/models` (checksum-identical to `results/ingame/deploy`),
`start_day.sh` passes `--weights $H/models` to the NFL and NBA runners and warns if fewer than 5 seeds exist per league. Sandbox recreated
10:09 UTC (`ingame-1025`); both models load from the new path. The MLB runner exited at once because there is no MLB game on 25 Oct (checked against the schedule); NFL has 12 games (first 13:30 UTC) and NBA 7 on that date.

### NBA totals: why the trades under-perform (25 Oct, numbers redone 26 Oct)

**Correction:** the first version of this section was computed on replay files that were regenerated afterwards (the pregame-total fix of 25 Oct, files rewritten 22:59–23:11); it also gave the fourth-quarter share of signals wrongly (63% / 85%). The numbers below are on the current files; the conclusions did not change.

**Data:** every filled total trade from (a) the 8-game live dry run of 24 Oct (36 fills) and (b) the backtest-assumption replay of 20–23 Oct (25 games, 105 fills). Trades held to the result are settled against the final score here (the engine's own summary leaves them out); fees as in the engine. Scratch scripts in the session scratchpad, not committed.

**Result, net per contract (all fills, settled trades included, size-weighted):** replay +0.55c (90% interval over games −6.4 to +6.4; −4.5c without the single best game); live night −9.3c (−22 to +0.7). All 33 games together: −1.9c (−7.9 to +3.6). Totals are not shown to lose in the replay, the live night is 8 games of mostly one kind of bet (below), and the two together cannot tell break-even from a small loss.

**What is robust (both sources, spreads too):** the signal is right but the fills miss it.
- Orders that never filled moved 14c (replay) and 21c (live) in the signal's direction within 60 s of the order; orders that filled moved −0.2c and −7.2c. Averaged over all orders the signal is worth ~7c; the passive entry only fills when the price comes back (median 14 s wait).
- Spreads show the same split (13.5c vs −1.3c replay), so this is not what separates totals from spreads. Spreads keep a positive result because their forecast has more skill left over (slope of the 60 s move on the signal, signals of 5c or more: spreads 0.81 replay / 0.42 live; totals 0.47 replay / 0.04 live).

**What is specific to totals:**
- Skill is lower and weakest where most trades are: 70% (replay) and 75% (live) of totals signals of 5c or more fall in the 4th quarter. Live Q4 slope was −0.02 (all 8 games), versus +0.48 in the replay; in the last 4 minutes (the live night's 1,574 signals) ~0 live, +0.3 to +0.5 replay.
- Signals in the 2nd quarter are also weak in the replay (slope 0.11).
- Larger signals are not better in the live night (12c and above: slope −0.02, 20c and above: −0.45 on 38 rows); in the replay they rise with size only above 20c.

**What drove the live night:** 27 of 36 fills were shorts of the Over; they lost −15.4c a contract (longs +3.0c; replay: shorts +2.3c, longs −2.5c). The worst were cheap Overs sold at 18–26c on BKN @ CLE and WAS @ CHI (both finished far above the line), settled at −74 to −82c. Fills that cost 70c or more (selling an Over priced under 30c) lost −23c a fill live (11 fills, −669c); in the replay the same bucket made +1.6c (24 fills). Trades held to the result lost −13.6c live (−5.4c replay) against +0.4c replay / −6.7c live for exits filled on the limit. Two shorts on NOP @ DEN (a 276-point game) were taken at −61c and −43c. Three high-scoring games made most of the live loss.

**Risk cap / direction filter (counterfactual on the same fills, in-sample, 33 games):** skipping all shorts: replay −1.5c, live +10.0c; skipping fills that can lose more than 60 / 70 / 80c: replay −4.8 / −1.3 / +0.6c, live −0.6 / +0.8 / −8.1c; sizing down to 50c of risk: replay 0.0c, live −7.7c. No rule helps both sources, so none is adopted.

**Not shown / next:** (1) whether the fall in skill is real: the fold means were R² 7.4%, the replay −0.4%, no like-for-like check on last season's games, and the test-fold files are on the cluster. (2) How often cheap-Over shorts reach the result unclosed (the exit sits 1c off the fill and waits 4 min).

### 25 Oct NBA: strict fill rule vs book-queue fill rule (26 Oct)

**What was done:** the first 5 NBA games of 25 Oct (401909861–865, including IND @ BKN, ORL @ NYK, TOR @ MIN and LAC @ OKC; the last two games were still in play) were replayed on the VM's recorded inputs with the runner's rule (a print strictly through the limit) and with price-time priority from the recorded order book (`live_rerun_nba.py --queue prop`, book events extracted on the VM). The strict replay reproduces the live orders (16/5, 4/2, 15/6, 4/2, 16/8 orders/fills).

**Entries:** 23 strict fills against 24 queue fills. Orders are identical. The one extra fill is an ORL @ NYK Over 202 bid at 71c that the queue rule fills at once; two other fills come 3–6 s earlier. None of the other 20 differ.
**Exits:** 3 exits on LAC @ OKC post at a price that crosses the book under the queue rule and are taken at the opposite best price (taker fee) instead of waiting for a print: net +0.9 / +1.7 / +8.9c against −2.8 / −6.9 / +8.2c.
**Closed P&L over the 5 games:** −35.9c strict, −23.0c queue; the same trades are held to the result either way, apart from the one extra fill.

So the two rules give nearly the same picture on this night, as on 24 Oct (130 of 140 orders alike). Not changed in the runners; to be rescored for the last two games and each later night.

### What the NBA signal is made of: model against "fade the last 60 s move" (26 Oct)

**Question:** the model sees plays 60 s after they happen and the market prices them in 5–20 s, so what does the signal (model value minus last print) predict? **Test:** for signals of 5c or more (replay 20–23 Oct, 25 games; live 24 Oct, 8 games) the signal was split into A = (price 60 s before the decision − last print), i.e. the market's own move over the last minute with no model in it, and the rest. R² of the next 60 s price move (to the last print at +60 s, relative to the last print at the decision):

| | signal | A only (fade the last 60 s) |
|---|---|---|
| Spreads, replay | 14.7% | 1.4% |
| Spreads, live | 7.9% | 1.9% |
| Totals, replay | 6.1% | 3.5% |
| Totals, live | 0.6% | 0.9% |

Spreads: the model adds a lot beyond fading the last move. Totals: more than half the replay skill is the last-move reversal, and live the model is no better than that baseline. The R² are on decision rows 5 s apart (heavily overlapping), so they are descriptive, not tests. Not shown: that the spread skill turns into profit (fills only come on reversals, see the 25 Oct entries above).

### No-model baseline through the same replay engine (26 Oct)

**What was done:** `live_rerun_nba.py --baseline fade` replaces the model value by the last print 60 s earlier (so the signal is the market's own move over the last minute, reversed). Everything else is the real engine: the 5c minimum, the sizing, the guards, the strict fill rule, 120 s hold and exit, settlement against the final score. 41 games: 28 from 20–23 Oct (backtest assumptions), the 8 live games of 24 Oct and the first 5 of 25 Oct (live inputs). Net per contract, size-weighted, held trades settled:

| | Spreads: model | Spreads: fade | Totals: model | Totals: fade |
|---|---|---|---|---|
| Replay 20–23 Oct | +2.0c (49 fills) | −4.2c (1,387) | +0.55c (105) | −4.7c (1,380) |
| Live 24 Oct | −2.5c (28) | −5.5c (332) | −9.3c (36) | −7.2c (248) |
| Live 25 Oct, 5 games | −12.7c (10) | −4.9c (224) | −12.2c (13) | −5.3c (150) |
| All | −1.1c (87) | −4.5c (1,943) | −3.1c (154) | −5.1c (1,778) |

The baseline trades 15–20 times as often (the model's signal is 5c or more far less often) and loses 4–5c on every contract in every source, so the signal is not just reversion of the last minute: the model beats it by about 5c in the replay. The model's live results on 24–25 Oct are no better than the baseline's on totals and worse on 25 Oct spreads (10 fills, including the IND @ BKN spread at −62c). Not shown: that the model makes money; its interval includes zero in the replay and live is negative on a small sample.

### 25 Oct, full night on the backup server (dry run, closed and settled; 26 Oct)

Trades held to the result settled against the final score (fees as in the engine); cents, all contracts.

| NBA | Orders | Fills | Closed / held | Spread | Total | Net |
|---|---|---|---|---|---|---|
| LAL @ UTA 132–123 | 16 | 5 | 0 / 5 | −32 | +81 | +49 |
| IND @ BKN 136–109 | 4 | 2 | 1 / 1 | −62 | −61 | −123 |
| ORL @ NYK 111–102 | 15 | 6 | 1 / 5 | −69 | −413 | −482 |
| TOR @ MIN 94–117 | 4 | 2 | 0 / 2 | 0 | +33 | +33 |
| LAC @ OKC 101–108 | 16 | 8 | 5 / 3 | +24 | −67 | −43 |
| DET @ PHI 106–93 | 3 | 1 | 0 / 1 | 0 | +16 | +16 |
| MEM @ SAC 123–112 | 28 | 9 | 9 / 0 | −67 | −9 | −76 |
| **Night** | 86 | 33 | 16 / 17 | −206 | −420 | **−626** |

NBA spreads: closed −128c (6), held −79c (6). NBA totals: closed +16c (10), held −435c (11); shorts of the Over −245c (10 fills, −10.7c a contract), longs −175c (11 fills, −7.3c). ORL @ NYK made most of it: three 5-contract shorts of the Over at 206–212 on −23 to −25c signals, with the game finishing at 213 points, settled at −44 to −58c each (a fourth, at 214, made +32c). The first 25 Oct NBA fills are the same as in the strict/queue entry above.

| NFL | Orders | Fills | Closed / held | Spread | Total | Net |
|---|---|---|---|---|---|---|
| PIT @ NO 16–10 | 6 | 0 | – | 0 | 0 | 0 (runner was up before kickoff; killed and restarted at ~16:30 UTC, see below) |
| CIN @ BAL 35–13 | 6 | 1 | 1 / 0 | 0 | −5 | −5 |
| CLE @ TEN 27–24 | 16 | 2 | 1 / 1 | +27 | +32 | +59 |
| IND @ MIN 10–16 | 13 | 1 | 1 / 0 | +121 | 0 | +121 |
| MIA @ NYJ 28–25 | 13 | 2 | 2 / 0 | −10 | +5 | −5 |
| NYG @ HOU 27–21 | 41 | 14 | 11 / 3 | +442 | −12 | +431 |
| SF @ ATL 20–29 | 23 | 3 | 2 / 1 | 0 | +26 | +26 |
| TB @ CAR 43–13 | 11 | 1 | 1 / 0 | 0 | +23 | +23 |
| DEN @ ARI 30–16 | 23 | 3 | 3 / 0 | 0 | +94 | +94 |
| GB @ DET 17–20 | 19 | 3 | 3 / 0 | +11 | +102 | +113 |
| LAR @ LV 35–21 | 13 | 2 | 1 / 1 | +13 | −2 | +11 |
| KC @ SEA 3–20 | 7 | 0 | – | 0 | 0 | 0 |
| **Night** | 191 | 32 | 26 / 6 | +604 | +263 | **+868** |

NFL: one game (NYG @ HOU, 14 fills) makes 431c; without it +437c over 11 games. Only 6 NFL fills were held to the result. The runners on the Lumid sandbox differ slightly (e.g. NBA 865 5 vs 8 fills); the VM files are used here.

### 25 Oct NFL, PIT @ NO: the 16:30 restart, and a replay without it (26 Oct)

**What happened:** the earlier note "runner started after the kickoff" was wrong. On the backup VM the NFL runner was up at 12:18 UTC ("waiting 1.2 h") and traded PIT @ NO from the start: 6 orders, 0 fills. At about 16:30 UTC, in the 4th quarter, both the NFL and NBA runners received SIGTERM (`Terminated`, then `restart in 60 s`); what sent it is not known (not checked on the VM). The restarted runner re-read the whole game: all 162 earlier plays were logged again as received at 16:31, so their delay showed as a median of 5,742 s instead of the original 41 s (max 93 s, none over 120 s). That tripped the stale-feed guard (any of the last 5 plays over 120 s late): 180 decision rows are marked "skip: feed 209 s behind", about one minute of decisions at the end of the game (the last row is 16:32:40). The restart also overwrote `summary.json`, which says 0 orders; the 6 orders are only in `orders.jsonl`.

**Replay with no restart:** `live_rerun_nba.py --scenario live --delay 60` over all 230 recorded feed snapshots in one engine, so each play keeps its first-seen time. It reproduces the 6 live orders (one size 2 against 1 and a signal of −10.3 against −10.5, not chased) and adds 2:

| | Live (restarted) | Replay (no restart) |
|---|---|---|
| Orders / fills | 6 / 0 | 8 / 1 |
| Rows skipped for a late feed | 180 | 0 |
| Feed delay median | 5,742 s | 42 s |

New orders: **16:32:00 short the Over 29.5, 5 contracts, limit 41, signal −16.2c**, filled at 41 at 16:32:11 and held to the result; **16:32:20 short the PIT −10 spread, 1 contract, limit 27**, expired. The game ended 16–10 (26 points), so the short wins 41c a contract: +205c less about 2c maker fee, **about +203c** (settled by hand; the engine reports "to settlement" with no P&L). On this basis PIT @ NO is +203c and the night about +1,071c. The live log keeps 0c / +868c, since the runner did not place it.

**Why it is not evidence of an edge:**
- The game was in the last minute (4th quarter, 1:04 left, 16–10; the Over needed 4 more points). At 16:31:47 a 37-yard field goal was ruled good, then nullified by a penalty. The Over jumped from about 13c to 40c, and the fill at 41 was in the tail of that spike. By 16:32:20 it was back at 2–13c and by 16:32:40 at 1c; two kneel-downs ended the game.
- The model could not have known about the penalty: with `--delay 60` and ESPN's 40–50 s lag, the play was not in its inputs at 16:32:00. The −16.2c signal was its value (about 25c) against a stale spike print (41c), the same fade of the last print seen in the NBA baseline entries. It paid because the spike reverted.
- The trade ran to settlement only because of a bug: the exit limit is the last print at fill + 120 s (1c) minus 1c = 0c, and the code posts an exit only for 0 < limit < 100, so no exit order was placed (the log line "exit posted, price 0.0" is misleading). A 1c buy-back would have made about +40c a contract.
- Strict fill rule: real prints went through the limit (100 and 107 contracts at 43c at 16:32:11), but a real queue position was not tested.

**Not shown:** whether the NFL model adds anything over fading the last print (all 32 fills of the night could be tested); whether the SIGTERM came from a manual restart or deploy on the VM.

**Fixed 26 Oct (`scripts/live_trade.py`, in the working tree, not yet committed or deployed):** a restarted runner now resumes from the game's own folder instead of starting blank. `Engine(resume=True)` (used by `main`) reloads each play's first-seen time and delay from `plays_seen.jsonl`, rebuilds the paper orders from `orders.jsonl` (stage, resting entries, held positions, posted exits; the contract stays busy, so nothing is placed twice), and does not log "first play" again; all logs are appended to, never rewritten. An NBA review outcome already settled before the restart no longer starts a new decision hold. The exit limit is clamped to 1–99c (it was last print ± 1c, so 0c for a short at a 1c market, which cannot be posted; the position then goes to the "taken" step after 120 s instead of being skipped). Test (`PIT @ NO`, forced restart in one engine with the earlier trades and current feed document handed back, as the recorder does): restarts at 15:26 (3 orders expired), 16:32:15 (order 6 held) and 16:34:30 (order 6 in its exit) give the same plays seen (168), feed delay (median 42 s, none over 120 s) and the same filled order held to settlement as the run with no restart, with no repeated order ids. One difference: the restart at 16:32:15 loses the 1-contract PIT −10 spread order of 16:32:20 (its signal is −7.6c instead of −8.7c and rounds to 0 contracts; the unfilled order expired in the original too), because the carried home-win price is not restored. A hold is not re-checked against a print that happened during the downtime beyond what the recorder re-fetches. Pushed 26 Oct ~09:20 UTC: `live_trade.py` (md5 5c4f48eb) copied to the sandbox's persistent `/home/casperngeen/app/scripts/` (so every recreated sandbox uses it; the image's copy is stale) and to the VM `~/fyp/app/scripts/`; the old files are kept as `live_trade.py.bak-1026`; both import cleanly. The running runners keep the old code until the next start (sandbox: the 12:30 UTC daily recreate; VM: the 12:00 UTC timer). The sandbox's `live_trade.py` was last changed at 16:30 UTC on 25 Oct, the minute the runners were terminated, so the SIGTERM was probably a code push or manual restart, not a fault; not confirmed.

### NBA: the test folds already decay through the season (26 Oct)

**Question:** is the poor NBA spread/total result a shift the backtest missed, or an input/execution effect? **What was done:** read the per-fold results of the walk-forward test folds (expanding training, five contiguous folds by date; `results/ingame/ens5`, d60 json) and the quoted width in the d10 fold forecasts. No new training.

| Fold (dates) | Games | Spread R² 60 s | Total R² 60 s | Mean bid-ask width, spread / total (c) |
|---|---|---|---|---|
| 1 (6–27 Jan) | 163 | 16.5% | 13.7% | 5.2 / 7.3 |
| 2 (28 Jan–22 Feb) | 154 | 13.3% | 9.2% | 4.3 / 5.3 |
| 3 (22 Feb–15 Mar) | 160 | 11.4% | 5.6% | 5.1 / 4.7 |
| 4 (15 Mar–4 Apr) | 157 | 7.8% | 4.4% | 3.6 / 3.4 |
| 5 (5 Apr–14 Jun) | 156 | 2.5% | 3.4% | 2.3 / 2.4 |

Each fold is trained on all earlier games, so the fall is not a stale model. The 10.2% / 7.4% fold means quoted in the plan average over this fall; the live replay (5.1% / 1.1%) sits at the level of folds 4–5. Unsized, one-at-a-time trades from the same folds (d10, 120 s hold, not the frozen rule) fall the same way: spreads −0.1, −0.7, −0.4, −1.6, −2.6c a contract; totals +0.4, −0.4, −2.0, −1.7, −1.3c.
**Not shown:** why the width narrows (more market makers, volume, playoffs) or whether it carries into the 2026-27 season; fold 5 mixes April regular season and the playoffs.

### World Series games 1 and 2, TB @ MIL (26 Oct)

Both were missing from the table above. **Game 1** (23 Oct ET, 849805, MIL 3–2): sandbox run, 263 pitches, feed delay median 12.5 s, 10 orders, 0 fills, 0c. **Game 2** (24 Oct ET, 849802, MIL 5–4): dry run on the backup VM, 341 pitches, feed delay median 7.9 s (p90 10.8 s), 14 orders, 6 fills, all 6 closed on exits (none held to the result), **+153.0c over 12 contracts (12.8c/contract)**. Order sizes on the VM run are 1–5 contracts, not the fractional sizes of the sandbox rows, so the cents are not comparable with the table above; compare per contract (the engine's own 15.6c is the plain mean over the 6 fills, not weighted by size). Fills: one buy of the TB +4 spread at 27 (+26.3c, size 5) and five sells of the over (+16, +11, +57 (size 3), +19, +23c; the last bought back at 1). No backtest replay of Game 2 yet.

### MLB P&L reworked for whole-contract sizing (26 Oct)

**Rule:** from 24 Oct an order is round(5 x the frozen 0-1 weight) whole contracts, half up; a weight under 0.1 places nothing (`--contracts 5`, journal 2026-10-24). The earlier rows used the fractional weight.

**Why the replay first did not match the log, and the fix:** `live_rerun.py` cuts the last feed version the runner recorded that night, whose pitch times are provisional; the logged replays used the finished game's statsapi archive feed (`--doc`). Re-pulling the archive feed for 13 and 14 Oct (feed + winProbability, the recorder's format) reproduces the logged rows (13 Oct 6 fills, 0.442 contracts, +7.81c; 14 Oct 4 fills, 0.009, +0.15c). The folders under `missed/` already hold the archive feed and reproduced the log (19 Oct differs slightly: 5 fills, 0.003 contracts, 0.0c against 6 fills, +0.013, -0.05c). Use `--doc` for any live-recorded game; the default is wrong for replays.

**Live games (the runner's own fills, re-sized, no re-simulation; a dropped order that would have blocked a contract is not undone):**

| Game | Fractional (logged) | Whole contracts |
|---|---|---|
| 13 Oct CLE @ TB | 4 fills, 0.404 contracts, +7.8c | 1 fill, 1 contract, +21.4c |
| 14 Oct MIL @ LAD | 3 fills, 0.403, +11.1c | 1 fill, 2 contracts, +55.0c |
| 15 Oct MIL @ LAD (from 02:15 UTC) | 7 fills, 0.128, +0.59c | 1 fill, 1 contract, +9.3c |
| **Total** | 14 fills, 0.94, +19.5c (20.8c/contract) | **3 fills, 4 contracts, +85.7c (21.4c/contract)** |

Eleven of the 14 fills were weak signals that round to 0 contracts. The result is three trades.

**Backtest replays (`d10,tape,tape,prints`, archive feed, re-simulated: orders that round to 0 are not placed and no longer block the contract):**

| Game | Fractional: fills, contracts, P&L | Whole contracts: fills, contracts, P&L |
|---|---|---|
| 13 Oct CLE @ TB | 6, 0.442, +7.8c | 3, 3, +13.3c (1 held to the result) |
| 14 Oct MIL @ LAD | 4, 0.009, +0.15c | 1, 2, +46.6c |
| 15 Oct TB @ CLE | 5, 0.013, +0.13c | 0 |
| 15 Oct MIL @ LAD | 9, 0.194, +2.2c | 1, 1, +22.2c |
| 17 Oct TB @ CLE G4 | 14, 2.644, +16.4c | 6, 22, +105.7c (spreads -8.4, totals +114.0) |
| 18 Oct TB @ CLE G5 | 9, 0.609, +9.2c | 1, 3, +45.5c |
| 19 Oct CLE @ TB G6 | 5, 0.003, 0.0c | 0 |
| **Total** | 52 fills, 3.9 contracts, +35.9c (9.2c/contract) | **12 fills, 31 contracts, +233.2c (7.5c/contract)** |

Per contract the edge is unchanged; the dollars are larger because the same few strong signals now trade 1-5 contracts instead of 0.1-1. Most of the whole-contract P&L is two trades (17 Oct G4: +114c on totals; 14 Oct: +46.6c).

### NBA late-game totals block: night by night (28 Oct)

**What this tracks:** what the `--late-block` rule (new totals entries skipped in Q4/OT inside the lead-based window, `live_trade.LATE_RULE`) saved or cost, each night. Every NBA game on the backup server is replayed from its recorded feed (`live_rerun_nba.py --scenario live --delay 60`, fractional 0.01 sizing, block off) and every fill scored at settlement, in cents at 1 contract per full weight. "Inside" = the fill's decision was tagged `late`. The replay is close to the live run but not identical (a fill that is not taken changes which contract is busy), so "with block" is the no-block total minus the inside totals. 24 and 25 Oct were the games the rule was designed on; 26 Oct onward are the first out-of-sample nights (the block was live from 26 Oct).

| Night | Games | Totals inside window (fills, weight, P&L) | Totals outside | Spreads | No block, total | With block |
|---|---|---|---|---|---|---|
| 24 Oct (design) | 8 | 45, 8.22, -118.7c | +1.2c | -19.5c | -137.0c | -18.3c |
| 25 Oct (design) | 7 | 24, 3.05, -2.3c | +2.9c | -41.7c | -41.1c | -38.8c |
| 26 Oct | 9 | 33, 7.27, -65.9c | -41.4c | +134.4c | +27.1c | +93.0c |
| 27 Oct | 4 | 22, 6.96, -132.2c | -0.7c | +132.4c | -0.5c | +131.7c |

Live P&L on 27 Oct (the block on): +131.7c on 53 fills (spreads +133.9c, totals -2.3c).

**Reading it:** the two out-of-sample nights lose -198c on 55 late totals, the two design nights -121c on 69. The loss is not evenly spread: 25 Oct is about flat (-2.3c), and on 27 Oct 103c of the 132c is BKN @ BOS (overtime keeps the window open). Shorts on the over are most of it (27 Oct: 13 fills -122.7c, 9 longs -9.5c). **Not seen in the backtest:** on the walk-forward test folds (`b35_nba_late_totals.py`) totals inside the same window earn +3.5c per contract (90% interval +1.1 to +5.9) against +4.2c outside, falling from +8.3c early to +2.3c in the last 2 minutes but not negative. The block therefore rests on live results (and was chosen after seeing them); candidate causes (ESPN feed lateness late in games, stalled feeds, season decay, chance) are untested. Rule of thumb for later nights: keep adding a row; if the inside totals stop losing, the block can be dropped.
