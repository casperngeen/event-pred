# Sports arm: data sources

Every external source the sports arm reads, what it feeds, and what a live trading system would
need in its place. Keep this current: a model input without a live equivalent cannot be traded.

Rule for every row (the as-of principle): an input is used at time t only if it was obtainable
at t. Each source below says how its `known_at` is set.

## Sources

| # | Source | What we take | Endpoint(s) | Auth / limits | Script → output | `known_at` |
|---|---|---|---|---|---|---|
| 1 | **Kalshi**: market listing | every MLB / NBA / NFL market: ticker, event, series, rules, open / close times, strike; at settlement, result and volume | `GET /trade-api/v2/series?category=Sports`, `/markets`, `/historical/markets` (base `https://api.elections.kalshi.com`) | public; signed requests get 20 req/s (Basic tier). Key id in env `KALSHI_KEY_ID`, RSA key `kalshi_api_key.txt` (git-ignored) | `scripts/fetch_sports.py list` → `data/sports/markets.parquet` | listing fields at `open_time`; results, close and volume at settlement |
| 2 | **Kalshi**: trades | every trade: price, size, taker side | `/markets/trades` (after `/historical/cutoff`), `/historical/trades` (before) | as above | `fetch_sports.py trades` → `data/sports/trades/<series>/` | the trade's `created_time` |
| 3 | **Kalshi**: candlesticks | top-of-book bid / ask (OHLC; we use the close), volume, open interest. **1-minute for game-level legs, hourly for season-long futures** | `/series/{s}/markets/{t}/candlesticks`, `/historical/markets/{t}/candlesticks`, batch `/markets/candlesticks` (≤100 tickers, legs settled after the cutoff only) | candle endpoints: ~4 req/s unsigned, 20 signed | `fetch_sports.py candles` → `data/sports/candles/<series>/` | end of the candle's period |
| 4 | **MLB statsapi**: play-by-play | every pitch (UTC start, ms), score, count, outs, and the per-plate-appearance home win probability; the **final pitch is the result time** | `statsapi.mlb.com/api/v1.1/game/{pk}/feed/live`, `/api/v1/game/{pk}/winProbability` | public, no key | `scripts/fetch_pbp.py` → `data/sports/pbp/events_MLB.parquet` | pitch: its end time; other events: their time |
| 5 | **MLB statsapi**: schedule + scores | full regular-season schedule, official final scores, first pitch, duration (fills MLB games before Kalshi listed game markets on 2025-04-16; official end and score replace an incomplete feed, 5 games) | `statsapi.mlb.com/api/v1/schedule?sportId=1&season=…&gameType=R&hydrate=team,linescore,gameInfo` | public | `scripts/fetch_mlb_schedule.py` → `data/sports/pbp/mlb_schedule.parquet` | schedule: known in advance; result: first pitch + duration + delays |
| 6 | **ESPN site API**: NBA / NFL play-by-play and schedules | every play with wallclock (1 s), score, period / clock, down / distance / field position, ESPN home win probability; schedules for matching; the **'End Game' / 'End of Game' event is the result time** | `site.api.espn.com/apis/site/v2/sports/{basketball/nba, football/nfl}/summary?event={id}`, `/scoreboard?dates=…` | **unofficial**, no key; we cap ourselves at 4 req/s | `fetch_pbp.py` → `events_NBA.parquet`, `events_NFL.parquet`, `games.parquet` | NBA: the play's time; NFL: the next play's snap (last play: the end-of-game event) |
| 7 | **nflverse**: NFL play-by-play files | snap time to the ms, EPA, win probability before / after each play (refines source 6) | `github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet` | public | `fetch_pbp.py pull` → `data/sports/pbp/raw/NFL/nflverse_pbp_<season>.parquet` | as source 6 |

Derived (no external source): the as-of layer `data/sports/derived/` (quotes, contracts,
outcomes, edges) is built from 1–7 by `scripts/build_sports_layer.py`. Team codes: Kalshi's and
statsapi's MLB codes match; ESPN differs (map in `fetch_pbp.py`, `TO_SRC`).

## What each model input needs, historically and live

| Model input | Historical source | Live equivalent | Latency needed | Status |
|---|---|---|---|---|
| contract bid / ask, mids, spreads | 3 (+2 between candles) | 1-min candles by REST (batch `/markets/candlesticks`, every 60 s, finished minutes only), as in training. Trades from Kalshi's WebSocket `trade` channel (signed with the Kalshi key; arrive ~0.14 s after they print), with the REST trades poll every 60 s as a backup, kept once by trade_id (since 2026-10-13: polling every ticker every 5 s hit HTTP 429s all night on 2026-10-12 and missed 24% of MLB trades). Every order-book change and trade is recorded from the WebSocket by `scripts/live_books_ws.py` (analysis only, not a model input) | seconds | **built 2026-10-12**, trades streamed 2026-10-13 (`scripts/live_record.py`) |
| trades, order flow | 2 | Kalshi REST `/markets/trades` per contract every 5 s (new trades since the last seen) | seconds | **built 2026-10-12** |
| which markets exist, rules, strikes, relistings | 1 | poll `/markets` / `/events` (e.g. every few minutes); new listings matter (e.g. NFL wins relisted as `-25B`) | minutes | not built |
| pre-game win probability p_pre (surprise z) | 3 (game-winner legs at first pitch / tip-off) | the same live quotes, read at the start | seconds | not built |
| game start, live score, game state | 4, 6, 7 | MLB: statsapi `/feed/live/timestamps` every 2 s, full feed + `/winProbability` on a new timecode, every request with a random query parameter to get past the CDN (`max-age=10` + stale-while-revalidate; cached, the median delay was 15.5 s after the pitch's end on 2026-10-12, uncached ~7 s in a live test). Note that a ball in play's last pitch "end" is when the play is over (10–12 s after contact, 38 s for a home-run trot), and Kalshi's game-winner price had made half its move 2–8 s after the pitch on that night's scoring plays. NFL: ESPN `/summary` every 3 s (incl. the drive in progress). nflverse is **not live** (updated after games): live NFL times a play by when ESPN first delivered it (training: nflverse play end) and uses ESPN's win probability (training: nflverse's; 120 s forecast change differs by median 0.02c on a past game). Every play's real feed delay is logged (`plays_seen.jsonl`) | seconds | **built 2026-10-12** (`scripts/live_record.py`, `live_model.py`, `live_trade.py`) |
| result and when it was known | the league's end-of-game time (4, 6; 5 for incomplete MLB feeds) + `RESULT_LAG` (1 min). **Kalshi's close is not used**: it can precede kickoff (GB-DAL tie) or lag a year (13 MLB markets of 2025-04-18) | the final play / end event in the live feed; log its arrival time to calibrate `RESULT_LAG` | seconds | not built |
| standings, streaks, rest days | results from 4, 6 + 5 | league schedule / scores endpoints (statsapi schedule; ESPN scoreboard), refreshed after each game | minutes | not built |
| remaining schedule (season simulator) | 5 (MLB), Kalshi game listings (NBA / NFL) | statsapi schedule; ESPN / league schedules (postponements and makeups change it) | daily | not built |

## Known gaps and quirks (affect a live feed too)

- **Futures quotes are hourly** in our history (source 3; the pull's choice, not a Kalshi limit:
  1-min candles over a futures life cost ~70-90 requests per leg). Sub-hour reactions of futures
  are only visible through trades; a live order-book feed would see them. Between-game samples
  therefore decide at the first hourly close after the night's last result
  (`stg/sports/samples.py`). **Open item:** a targeted 1-min re-pull of the graphs' futures legs
  around each night (~140k requests, ~2 h at 20 req/s, ~6 GB; about half if only legs priced
  3-97c), needed only for sub-hour futures dynamics or in-game propagation into futures.
- **Kalshi listing errors**: a team listed in both conferences (DAL, NBA 2025-26); markets
  relisted (NFL wins `-25` → `-25B`); closes before kickoff (GB-DAL tie) or a year late. The
  layer guards against each (`stg/sports/edges.py`, `graph.py`, `loaders.py`); a live system
  needs the same guards.
- **ESPN's API is unofficial** and can change or rate-limit without notice; plan a fallback
  (official league feeds or a paid provider) before trading on NBA / NFL game state.
- **Holdout**: anything closing (or, for results, known) on or after 2026-07-01 lives under
  `data/sports/_HOLDOUT_DO_NOT_USE/` and is read once, for the final test.

## Trading costs (Kalshi fees)

Per series, from `GET /trade-api/v2/series/{ticker}` (fields `fee_type`, `fee_multiplier`;
checked 2026-10-04; this is the current schedule, and fees during 2025-26 may have differed):

| Series | fee_type | multiplier |
|---|---|---|
| KXMLBGAME | quadratic_with_maker_fees | 0.5 |
| KXNBAGAME, KXNFLGAME, KXNFLSPREAD | quadratic_with_maker_fees | 1 |
| KXMLB, KXNBA, KXSB (champ) | quadratic_with_maker_fees | 1 |
| KXMLBALEAST (division), KXNBAPLAYOFF, KXNFLPLAYOFF | quadratic (taker only) | 1 |

Quadratic: fee per order = round_up(rate x multiplier x contracts x P x (1 - P)) dollars, P the price
in dollars. The base rates (taker ~0.07; maker smaller, ~0.0175) are from Kalshi's fee-schedule
document, **not verified here**: confirm before any P&L. Settlement itself carries no trading fee.
Live trading needs these per series (query the endpoint), not one formula.
