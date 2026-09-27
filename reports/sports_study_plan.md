# Sports as a second arm: a graph with known structure and ~10× the events

*Design doc (2026-09-27). **Steps 1–3 have been run on MLB: see `sports_study.md`.** Nothing here has been run. Numbers are
from a read-only scan of `data/markets/` and `data/trades/` (in-sample folders
only; the `_OOS_DO_NOT_USE_*` folders were not read).*

## Why sports

Every macro result ends at the same limit: at 154 release instants there are
about 5 co-firing instants per edge. `recovery_test.md` puts the data an STG
needs at about 10× that, even for a graph learner of the right shape.

Sports offers two things macro cannot:

1. **Volume.** In the archive, 23.9M of 39.4M trades are sports, 99.97% of them
   in 2025. There are thousands of resolved games, against 263 macro releases.
2. **A known graph.** A game between T and U changes the probability that T,
   and not U, wins its series, conference and championship. That relation is
   mechanical: directed, signed (+ for T, − for U) and sparse. So every learned
   or constructed graph can be scored against a ground truth *on real data*.
   That is the recovery test without the synthetic part.

## What the data holds (2025)

| league | trigger markets (games) | futures markets (targets) | futures liquidity |
|---|---|---|---|
| MLB | `KXMLBGAME`, 2,204 games (Apr–Nov), 4.2M trades | `KXMLB` (30 teams, championship), `KXMLBAL`/`KXMLBNL` (pennants), `KXMLBSERIES` (8 playoff series) | Thin until June (≈1k trades/month across 30 teams), dense July–Nov (12k–150k/month) |
| NBA | `KXNBAGAME`, 272 games (Apr–Jun playoffs; Oct–Nov regular season), 5.6M trades | `KXNBASERIES` (12 series), `KXNBAEAST`/`WEST`, `KXNBA` (2025 and 2026 Finals) | Dense in the 2025 playoffs (24k–139k/month), thin Oct–Nov |
| NCAAF | `KXNCAAFGAME`, 502 games (Aug–Dec), 7.8M trades | `KXNCAAF` (137 teams), conference championships | Thin (4k–14k/month across 137 teams) |
| Soccer, tennis, F1 | EPL / La Liga / MLS / … ≈ 700 games; ATP 1,069 matches; F1 18 races | Few or no linked futures in the archive | – |

Two practical facts:
- **Linking is by ticker, not title.** Team codes are consistent across series:
  `KXMLBGAME-25OCT31LADTOR-TOR` ↔ `KXMLB-25-TOR`, `KXMLBSERIES-25DETSEA-SEA`,
  `KXNBA-26-UTA`. The "which markets are about the same team" graph is exact.
- **Gaps and exclusions.** Some series have trades but no market metadata in
  the crawl (`KXFIFAGAME`, `KXEUROLEAGUEGAME`, several soccer leagues);
  `markets_api_pull/` may fill them. The combinatorial NFL parlays
  (`KXMVE…`, 132k one-off events) are excluded: they are not reusable nodes.

## The event: game resolution *is* a release

The macro design carries over almost unchanged.

| macro | sports |
|---|---|
| Release instant (CPI at 08:30) | Game end, when the game market resolves |
| Surprise z = (actual − implied mean) / implied sd | Surprise = 1[T won] − p_pre(T), from the game market's last pre-game (or last pre-final) price. The market's own expectation, as with the macro ladders |
| Target: other series' ladders | Target: T's and U's futures (series, pennant or conference, championship) |
| `imm` label: first post-release prints | Futures repricing in the first prints after the game ends |
| `settle` label | Futures settlement residual (only for futures that settle inside 2025) |
| HAWKISH sign | Mechanical sign: + for the winner's futures, − for the loser's |

**The live-game alternative (your "live updates").** During a game, the game
market's price moves on every play. Those moves can be treated as a
continuous stream of small surprises, and the futures as responding to them
at the minute scale. That gives orders of magnitude more trigger events, but
two problems:
- Futures trade too rarely for minute-level responses to be observed outside
  high-attention games.
- An in-game move is partly anticipated noise, not news.

I'd treat it as a phase-2 extension on the NBA and MLB playoffs, where futures
are dense, not the primary design.

## Graph construction: five graphs, scored against the known one

The question for the thesis is not only "does the model do better with more
data" but **"which way of building the graph recovers a structure we know is
there"**. Build them all on the same nodes and score each against (A):

| graph | how | role |
|---|---|---|
| **(A) Mechanical** | Game(T, U) → T's futures (+), U's futures (−), from ticker team codes and the tournament bracket | Ground truth; also the imposed-prior model, as HAWKISH was |
| **(B) Entity-from-title** | Parse team or player names from `title` / `yes_sub_title` (alias tables per league) and link markets sharing an entity; sign from the entity's role (the YES subject vs the opponent) | The structured version of "title similarity". It should recover (A) almost exactly, and tests whether titles alone suffice where ticker codes don't exist (soccer, tennis) |
| **(C) Title-embedding similarity** | Sentence-transformer cosine between market titles, top-k or thresholded (the repo already has `KalshiSemanticTopicEdges`) | Your suggestion as stated, and the analogue of the news-co-occurrence graph in `STG_stocks.pdf`. Unsigned and symmetric, so it links "Toronto vs LA" to *both* teams' futures with the same sign. It measures how far pure semantics gets |
| **(D) Price-correlation** | Walk-forward correlation of futures and game price changes | The data-driven baseline the STG papers compare against |
| **(E) Learned** | AGCRN adaptive, signed low-rank, and the free linear graph | Can the model learn (A) from ~10× the data? The recovery test predicts the low-rank graph should, and AGCRN may not |

Scoring: AUROC of each graph's edge weights for true vs false edges, and
balanced sign accuracy, against (A). These are the same metrics as
`recovery_test.md`, now on real data.

## Models and evaluation

The macro pipeline carries over:
- event-time panel per league;
- 8-fold walk-forward within 2025, purged on label end;
- the zero-parameter mechanical rule, linear rungs, the low-rank graph, and
  the AGCRN family (adaptive, signed, frozen (A), surprise-only);
- every-metric scoring, sign read-outs, gross returns vs a random side, and
  the scoped variant.

The headline comparison is the one that failed on macro, repeated at 10–20×
the events:

> does a learned graph (E) recover (A), and does an STG beat the mechanical
> rule and the linear rung once the data is there?

A second question comes free: **how fast does a game result reach the
futures?** A game result is public and unambiguous, so any delay is pure
diffusion. That is a clean version of the thesis's information-diffusion
question.

## Scale check

- **MLB, July–Oct:** ≈1,000 games × 2 teams × 2–3 futures ≈ 4,000–6,000 labelled
  cells, against 955 settlement cells in the macro panel.
- **NBA 2025 playoffs:** about 80 games, with the full hierarchy
  (game → series → conference → championship).
- **NCAAF:** 502 games; futures are thin.

In total that is roughly 5–15× the macro panel's labelled cells, which reaches
the range where the recovery test found graph learners start to recover
structure. It may still fall short for full AGCRN, which needed more than 30×
the macro data unless fed only the surprise.

## Risks

1. **Only 2025.** One season per league. Walk-forward folds are within-season,
   and team strength drifts through a season.
2. **Futures are thin early in each season.** The response goes unobserved
   exactly when the most games happen (MLB Apr–Jun). This is the liquidity gate
   from `liquidity_findings.md`, and the same "archive misses thin legs" bias.
3. **The mechanics may be priced instantly.** If market makers update
   futures within seconds of a result, the `imm` label is ~0 and there is no
   lag to learn. That would itself be a finding (efficient diffusion), but it
   would leave the STG nothing to predict.
4. **Simultaneous games** (MLB ≈ 15 a day) move many futures at once. Unlike the
   CPI family, they involve different teams, so identification is clean, but a
   team's championship price also depends on *other* teams' results. That is
   genuinely multi-hop, which is the depth the macro graph lacked.
5. **Metadata holes** in some leagues, and name aliases for (B).
6. **The OOS wall.** 2026 sports data exists under `_OOS_DO_NOT_USE_*`. The
   same one-shot rule applies.

## Suggested order

1. **Pilot on MLB July–Oct 2025.** It has the densest futures, the most games,
   and exact ticker linking.
   - Build the trigger table: game end time, pre-game price, result, surprise.
   - Build the target panel: team futures state before each game end, plus
     `imm` and `settle` labels.
   - Check risk 3 first. If futures reprice within the first print, decide
     whether the study becomes a diffusion-speed study before building models.
2. Graph (A) and the zero-parameter mechanical rule, then (D) and (E), scored
   against (A). This is the core result.
3. Graphs (B) and (C) from titles, including a league with no ticker codes
   (EPL or ATP), to test whether text alone finds the structure.
4. Extend to the NBA playoffs (hierarchy) and NCAAF (volume); then optionally
   the live-game clock.

## Decisions needed

- **Pilot league:** MLB (recommended: volume and dense futures) or the NBA
  playoffs (cleanest hierarchy, few games).
- **Clock:** game resolution (recommended) or live in-game moves.
- **Role of title similarity:** a baseline graph (recommended) or the primary
  graph. It cannot represent the sign of an edge, which the mechanical graph
  gets for free.
