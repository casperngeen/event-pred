# Kalshi sports: how information gets into prices

*Research plan (2026-09-29). Branch `casper/sports-graph`. Supersedes the
diffusion-only design in `sports_study_plan.md`; the MLB pilot it describes
(`sports_study.md`) feeds §7 and §8. Nothing below is built yet unless marked.*

## 1. The question

**How, how fast and how accurately does information get into Kalshi sports
prices, and can a graph model learn the structure that carries it?**

Every study shares one design: a **timestamped news event**, and the
**mid-price response** of some set of markets, using only data known at the
time.

| | Same market | Related markets |
|---|---|---|
| **Before the game** (no news) | calibration, biases, quiet-period martingale tests | same-game consistency |
| **During the game** (plays) | reaction to plays | same-game contracts (in-game model) |
| **After the game** (the result) | the market settles | diffusion to futures (between-game model) |

- **Core:** data layer → calibration and biases → reaction to plays →
  cross-market structure and graph models → strategy evaluation.
- **Stretch:** co-attention graphs; player-level in-game features; props in the
  in-game model.
- **Parked:** a second venue, and the popular-team bias (both need data we
  don't have yet).

## 2. Data

| Source | Contents | Status |
|---|---|---|
| Kalshi listing | ~1.17M MLB/NBA/NFL markets, 2025 onward (`scripts/fetch_sports.py`) | done |
| Kalshi trades | full history per market, with taker side | ~94% of markets; the largest are pulled last |
| Kalshi candles | 1-min bid/ask: winner (`…GAME`) markets over their whole life, other game markets the last 12 h; hourly for season-long markets | queued after trades |
| Play-by-play | MLB Stats API (per pitch, ms), ESPN NBA/NFL (per play, 1 s), nflverse (NFL snaps, ms) (`scripts/fetch_pbp.py`) | done: 3,917 MLB / 1,403 NBA / 283 NFL working-block games, preseason dropped; winners agree with Kalshi settlement on all 5,589 settled games |

- **Working block:** markets closing before 2026-07-01. That covers MLB 2025
  (plus 2026 games, but no 2026 futures), the NBA 2025 playoffs and 2025-26,
  and NFL 2025.
- **Holdout:** markets closing on or after 2026-07-01, under
  `data/sports/_HOLDOUT_DO_NOT_USE/`. Never read until the final test.

## 3. Rules for every study

1. **As-of only.** Every row carries a `known_at` time, and features use only
   rows with `known_at` at or before the snapshot. A model with training cutoff
   C uses only samples whose label ended by C, and statistics (scalers, fill
   values, graphs) fitted on data up to C.
2. **Mid prices for labels:** two-sided quotes only, spread at most about 5c,
   changes measured in log-odds.
3. **Gross first:** report mid-to-mid (signal quality) and executable gross
   (ask to buy, bid to sell, no fees). Net of fees only on request.
4. **Weight by game, and count each outcome once:** complementary markets are
   merged, and a spread or total ladder is flagged as one outcome.
5. **Walk-forward evaluation** within and across seasons, with standard errors
   clustered by game or day.

## 4. Phase 0: data layer and safeguards (new `stg/sports/`)

- **Loaders** for the listing, trades, candles and play-by-play. They never
  read the holdout folder. The sports holdout boundary goes in `stg/splits.py`;
  the macro one is unchanged.
- **`known_at` times:**

  | Data | Known at |
  |---|---|
  | trade | `created_time` |
  | candle | end of its period |
  | MLB pitch | its end time |
  | NBA play | its clock time |
  | NFL play | the next snap |
  | game result | final play / close |
  | settlement | `settlement_ts` |
  | listing fields known at listing | `open_time` |
  | final listing fields (result, volume, …) | `settlement_ts` |

- **An as-of view**, `SportsData.as_of(t)`: markets listed and still open at
  t. It's built from precomputed per-market series indexed by `known_at`, so
  lookups are by position rather than re-filtering 300M+ trades.
- **Mid series** (cents and log-odds), and **executable prices** from
  taker-side trades and candle bid/ask.
- **An outcome table** (complements merged, ladders flagged), **pre-game
  snapshots** (the mid 24 h, 1 h and 0 h before the actual start), and
  **contract rules** parsed from `rules_primary`.
- **Features** as in §9, all as-of.
- **The edge table** (§8).
- **Checks:**
  - the truncation test: build to C, and build from the full data; samples
    with label end at or before C must be identical;
  - an assertion that no feature's `known_at` is after its snapshot;
  - a rule that feature code can't read the listing's final-value columns.
- **Port the MLB pilot** through the as-of view. This fixes three leaks: the
  season-wide fill value (`graphs.py:200`), the season-wide label scale
  (`graphs.py:213`), and first pitch detected by looking back from the game's
  end (`build_mlb.py`). Rerunning the pilot tables shows how much they
  mattered.

**Status (2026-09-29): built** in `stg_infra/stg/sports/`, on the data on
disk. Build it with `venv/bin/python scripts/build_sports_layer.py` (24 s; rerun
after each pull). Tests are in `stg_infra/tests/test_sports.py`, including
the truncation test at C = 2025-07-01, which is shown to catch a planted
season-wide fill. Pilot rerun: `sports_study.md`, "As-of rerun" (no
conclusion moves).
- **Quotes come from taker-side trades until the candles land.** The
  build picks up candles automatically. Until then a valid pre-game mid is
  sparse: MLB winners have one at 0 h for 48% of games, at 1 h for 19%, and at
  24 h for 1%.
- **`season`** is the league year containing `open_time` (MLB from Dec 1, NBA
  from Jul 1, NFL from Feb 11).
- **Anchoring pre-game snapshots to the actual start** puts a little hindsight
  into the 24 h and 1 h rows; `anchor="scheduled"` avoids it.
- **Not built yet:** bracket edges; player-of, co-attention and semantic
  edges; player-level features.

## 5. Phase 1: calibration and biases

- **Calibration** of pre-game snapshots against outcomes, split by league,
  market type, time before the game, liquidity and season phase.
- **Biases:**
  - favourite–longshot (first hint: trades around 7c settled YES 5.3% of the
    time);
  - recency (after streaks and blowouts, is the next game's price too high?).
- **Strategies:** hold to settlement, fitted walk-forward, reported as
  executable gross.

## 6. Phase 2: dynamic efficiency and plays

- **Martingale tests** on mids in quiet periods (pre-game, and between plays),
  split by time before resolution. Mids only: trade prices alternate between
  bid and ask.
- **Event studies around plays:** home runs, touchdowns, scoring runs, big
  win-probability swings. Does the move continue (under-reaction) or reverse
  (over-reaction)?
- **Stale trades:** the share of trades executed at the pre-play price in the
  seconds after a play.
- **Strategies:** follow or fade, reported mid-to-mid and as executable gross.
- **First league:** MLB (pitch-level timing, most games).

**Early check** (2026-09-29; 35 MLB games, 224 scoring plays, trades only):
- The winner market moves +10.9c at a scoring play, in the right direction 95%
  of the time.
- Spread and total markets move +12–15c, right direction 84%, but trade only
  about once a minute, so their apparent ~50 s lag is mostly a
  trade-frequency artifact.
- At their first post-play trade, a median 63–75% of the move is already
  priced; about 30% of first trades are still near the old price.
- **Open:** do quotes actually lag, or is that the bid/ask gap? The 1-minute
  candles will settle it.

## 7. Phase 3: cross-market structure

- **3a. Same-game consistency.** An implied joint distribution of margin and
  total from the winner, spread and total ladders (reusing
  `stg/events/implied.py`). Deviations are relative-value trades held to
  settlement.
- **3b. Diffusion to futures (between-game model, §8).** The pilot found a
  lag: a third of the move in a team's division contract comes after the first
  post-game trade (t = 6), and rivals move the other way (t = −4). Extend to
  the NBA and NFL, including multi-hop chains (game → series → conference →
  title; game → division → conference → Super Bowl).
- **3c. Same-game contracts (in-game model, §8).** The graph-learning testbed,
  because the signal per event is strong.
- **3d. Correlation and co-attention graphs.** The correlation graph for
  sizing. Do co-attention links predict shared errors from Phases 1–2?
  (Stretch.)
- **3e. Regimes:** early season / trade deadline / playoffs, as a split across
  all of the above.

## 8. Models

### Two settings, one architecture

| | Between-game | In-game |
|---|---|---|
| Steps (event clock) | game endings, grouped by close time | plays, at `known_at` (optionally thinned to significant plays) |
| Nodes | **team × futures kind** (`NYK-series`, `NYK-conference`, `NYK-title`, …; ~120), with kinds not yet live masked | **one game's winner / spread / total / half contracts**, complements merged; props later (~20–30 per game) |
| Labels | futures mid change, first quote → +1 h / +24 h | contract mid change, +1 / +5 min |
| Signal per event | weak (ρ ≈ 0.2 own team, ~0.04 rivals, from the pilot) | strong (10–15c moves) |
| Play-by-play role | per-game summaries at the game-end step | the event stream itself |

- **Why team × kind:** it puts the hierarchy (series → conference → title) on
  graph edges, where the recovery test can score it. With plain team nodes it
  would sit inside the output layer, where it can't be scored. Plain team
  nodes stay as an ablation, which also reproduces the pilot.
- **In-game nodes are described by features,** not identities: type, side, and
  **strike relative to the current state** (spread line minus current margin,
  total line minus runs so far). That makes a node in one game comparable to
  one in another.

### No team-specific parameters

The models learn general patterns only:
- node embeddings come from features (`embedding="shared_mlp"`);
- weights are shared across nodes (`weights="shared"`);
- team differences enter through as-of state (prices, standings, contention),
  not memorised identities.

Identity appears in only two places:
1. **Rung 1** (below), which learns per-pair edges by design;
2. **one diagnostic ablation** with learned identity embeddings. If it beats
   the feature-only model on a later period, find the general feature that
   explains the gap rather than keeping identities.

### Edge table (contract level; team-level graphs derived from it)

Columns: `src`, `dst`, `edge_type`, `sign_prior` (+ / − / unknown), `strength`
(exact / learned), `valid_from`, `valid_to`, `known_at`, `provenance`. Always
cause → effect, matching `agcrn.py`'s `[i, j]` = influence of i on j.

| Group | Types | Role |
|---|---|---|
| Exact constraints | complement, ladder, part-of-whole | same-game consistency; ground truth for the in-game graph |
| Competition structure | result → own future (+), result → opponent's future (−), rival (−), bracket, hierarchy | ground truth for the between-game graph; sign priors |
| Participation | plays-in, same slate, player-of | graphs that change each step |
| Statistical (as-of, trailing windows) | co-movement, co-attention, semantic | baselines and inputs, never truth |

### Three levels of graph learning ("can the graph learn itself?")

| Rung | Graph source | Expectation |
|---|---|---|
| 1. Free | learned from node identities (AGCRN adaptive / signed, MTGNN-style one-way) | fails between games (weak signal); may succeed in-game |
| 2. Attribute-learned | weight = g(neutral pair attributes: same division/conference, standings gap, games remaining, contention, scheduled meeting), shared across pairs and leagues | the key test: does it rediscover the structure and its state-dependence? |
| 3. Supplied | one graph per edge type from the edge table | upper bound (the pilot got 89% of rival signs) |

Rung 2 gets only neutral pair attributes, not team stats, so a recovered
structure can't be credited to the stats.

### Architecture

AGCRN extended into a relation-typed, event-driven recurrent graph network.
Every new option defaults off, so the macro study's results and
`test_models.py` are unchanged.

1. **Several graphs at once:** `[self, G_1 … G_K]` replaces `K_SUPPORT = 2`
   (as in DCRNN and Graph WaveNet).
2. **Graphs that change each step:** matchups, the bracket.
3. **A pluggable graph builder:** free / attribute-learned (edge-conditioned,
   as in Simonovsky & Komodakis 2017) / supplied. Normalised by number of
   neighbours, not softmax, so negative edges survive.
4. **One shared weight per relation type** (R-GCN-style) by default.
5. **One output per target** with a (B, N, T) label mask, and an optional
   sign-aware loss.
6. **In-game: run through each game once,** predicting at every play. Memory
   is updated per play and reset per game; truncated backpropagation through
   64-step chunks.
7. **Event clock:** steps are events, with Δt as an input (TGN-like memory
   updates).
8. **A no-memory ablation** (`L = 1`), mainly for the NFL.
9. **A shared graph readout:** the Jacobian ∂ŷ_j/∂z_i, moved from
   `analysis/sports_2026_09/graphs.py` into the library, so every model is
   scored the same way on recovery.

### Comparisons (same data, trainer and scoring)

- the mechanical rule (no parameters);
- **network autoregression** (Zhu et al. 2017): linear, one weight per
  relation type;
- an **STID-style no-graph model** (Shao et al. 2022);
- an **MTGNN-style** one-way learned graph (rung 1);
- the pilot's rank-6 and pairwise graphs;
- optionally **TGN** (Rossi et al. 2020) for the in-game model.

### Training

- Purged walk-forward on label end, with a time-ordered early-stopping split.
- Scalers and graphs fitted up to the cutoff.
- Device-agnostic (CPU / MPS / CUDA), fixed seeds, per-fold data files on disk.
- Rolling retraining (e.g. monthly) in evaluation, since the market may become
  more efficient over time.

## 9. Features

The model predicts price moves and mispricing, not who wins, and the market
price already reflects public stats. So features are added in small groups
tied to a hypothesis, and a group is kept only if it beats the model without
it in walk-forward evaluation. Everything is as-of: season-to-date stats count
only games finished before the snapshot.

| Group | Examples | Source | Priority |
|---|---|---|---|
| Market state | own mid, spread, time to close, pre-game win probability | candles / trades | core |
| Game state (in-game) | score difference, time left, count/outs/bases, down/distance/field position | play-by-play | core |
| Standings state | games behind, games remaining, playoff position, contender flag | results + futures prices | core |
| Recent form | streak, last margin, last-10 record | play-by-play results | ablation (recency test) |
| Schedule | rest days, back-to-backs, home/away, travel | schedules (known in advance) | ablation |
| Team strength | runs/points for and against, ratings, EPA | derived from play-by-play, or MLB API / ESPN / nflverse | hypothesis test only |
| Player-level | starting pitcher, pitch count, injuries, lineups | MLB API / ESPN | stretch |

## 10. Generalisation

- **What carries over:** relation-type weights, the attribute-learned graph,
  the shared recurrent and output weights, and feature-based node embeddings.
- **What doesn't:** anything tied to team identity, which is why there is none
  outside rung 1 and the diagnostic.
- **Comparable inputs:** log-odds rather than cents, strikes relative to the
  current state, games remaining as a fraction of the season.
- **Tests, from easiest to hardest:**
  1. later in the same season (walk-forward folds);
  2. the next season (NBA 2025 playoffs → 2026 playoffs; 2024-25 → 2025-26);
  3. a league not trained on (e.g. train MLB + NBA, test NFL: only structural
     parameters can transfer);
  4. the holdout.
- **Risks:**
  - the market becoming more efficient as volume grows (track performance over
    time; retrain on a rolling schedule);
  - few seasons per league;
  - new contract kinds, which need a new relation type in the edge table.

## 11. Phase 4: strategy evaluation

Combine the Phase 1–3 signals, sized with the correlation graph. Enter at the
next taker trade as-of; hold to settlement, or exit at +1 h / +24 h against
the bid/ask. Report gross first (mid-to-mid and executable), net on request.
The final one-shot test is on the holdout.

## 12. Compute

- **Laptop (M2):** between-game runs take minutes; the full grid runs
  overnight.
- **SoC cluster** (Slurm, `cluster/`): A100 40GB via `--gres=gpu:a100-40:1`,
  verified with `cluster/smoke_gpu.sbatch`.
  - Between-game: ~0.03 s per epoch.
  - In-game: ~2 s per epoch with the per-game design, so ~100 runs in under
    2 hours.
  - Use large batches; at batch 256 the A100 is overhead-bound.
- **Only prepared working-block tensors go to the cluster.** Raw data and the
  holdout stay on the laptop.
- The real time sink is building the as-of data layer, not training.

## 13. Order of work

1. **Phase 0.** Starts now on data already on disk.
2. **Phase 1**, once the trades and candles pulls finish.
3. **Phase 2**, including the candle check on in-game lags.
4. **Phase 3:** 3b and 3c with the models, then 3a, then 3d–3e.
5. **Phase 4**, then the holdout test.

## 14. Decisions

| Decision | Choice |
|---|---|
| Jan–Jun 2026 in-sample (sports holdout from 2026-07-01) | yes (default; recorded in `stg/splits.py`) |
| Between-game nodes | team × futures kind; plain team nodes as an ablation |
| In-game nodes | winner / spread / total / half contracts, complements merged; props later |
| Team-specific parameters | none, except rung 1 and one diagnostic ablation |
| First league for Phase 2 | MLB |
| Second venue, popular-team bias | parked |
