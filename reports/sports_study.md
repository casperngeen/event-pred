# Sports pilot: MLB 2025, a graph with known structure

*2026-09-27. Branch `casper/sports-graph`, kept separate from the macro study.
Scripts: `analysis/sports_2026_09/` (`build_mlb.py`, `diffusion.py`,
`graphs.py`, `signfix.py`, `predict.py`; outputs in its `out/`). Design:
`sports_study_plan.md`. In-sample 2025 data only; the `_OOS_DO_NOT_USE_*`
folders were not read.*

## The question

The macro study ended on a sample-size limit: about 5 co-firing instants per
edge, where `recovery_test.md` says a graph learner needs roughly 10× that.
Sports has two things macro lacks:
- **many more events per edge:** each MLB team plays ~160 regular-season
  games, so each team edge co-fires ~160 times;
- **a graph whose structure is known:** a team's win raises its own
  division-title odds (+) and lowers its division rivals' (−).

So the recovery question can be asked on *real* data. Which way of building
the graph finds the known structure: text from the market titles, price
co-movement, or a model learned from surprise → response?

## Data (`build_mlb.py`)

**Games:**
- 2,193 MLB games (2025-04-16 → 11-01) with both team markets settled, so
  4,386 team-games.
- Each game market closes when the game ends.
- First pitch is detected from the tape (the in-play trading spike) for
  99.5% of games.

**Surprise:**
- z = 1[won] − the pre-game win probability, from both team markets in the
  hour before first pitch.
- sd 0.49. The market's own expectation, as with the macro ladders.

**Targets:**
- Each team's championship, pennant, playoffs and division contracts:
  431k trades on 115 contracts.
- Team codes differ across series (A/ATH, ARI/AZ, KAN/KC, FLA/MIA, WAS/WSH)
  and are mapped to one code.
- Division and league membership are read from the division and pennant
  tickers.

## 1. The futures absorb a result with a lag (`diffusion.py`)

Response of the futures price to the game surprise, in cents per unit z (t
clustered by game), at checkpoints after the game ends:

| target | during the game | first print after | +24 h | drift, first print → +24 h |
|---|---|---|---|---|
| own division contract (median 13¢) | +0.76 (17%) | +3.02 (68%) | +4.44 | **+1.42 (t = 6.0)** |
| own pennant (7¢) | +0.41 (38%) | +0.84 (78%) | +1.07 | +0.23 (t = 2.6) |
| own championship (3¢) | +0.32 (49%) | +0.57 (89%) | +0.64 | +0.07 (n.s.) |
| division rivals' division contracts | −0.03 | −0.50 (60%) | −0.83 | **−0.33 (t = −4.1)** |
| division rivals' championship | 0.00 | 0.00 | −0.02 | n.s. |

**About a third of a result reaches the team's division market only after the
first post-game trade, and the rivals' markets take the sign the mechanics
predict, also with a lag.** That is measurable cross-market information
diffusion in a graph whose edges are known. It clears the plan's risk 3: the
lag exists, so there is something to learn.

The championship contracts barely move (most teams trade at ~3¢), so the
division contract is the informative target.

## 2. Which graph recovers the known structure? (`graphs.py`)

**Setup:**
- 30 team nodes; 166 regular-season slate days.
- Trigger: the team's summed game surprise that day.
- Response: its division contract's change over the day, where it traded
  (2,136 labelled team-days).
- Truth (A): self +, division rival −. That is 150 true edges among 900
  ordered pairs.

**Scoring:**
- AUROC of |edge| for true vs false, and the same on the 870 off-diagonal
  pairs only (the rival, cross-market part).
- Balanced sign accuracy on the true edges.

Full season:

| graph | AUROC | rival AUROC | sign (bal.) | built from |
|---|---|---|---|---|
| (A) mechanical | 1.00 | 1.00 | 1.00 | the competition structure |
| (A) frozen into AGCRN | 0.95 | 0.95 | 0.90 | check: training keeps a supplied graph |
| **(C2) schedule from game titles** | **0.81** | **0.999** | unsigned | how often two teams appear in the same game title |
| (C1) title embedding | 0.66 | 0.58 | unsigned | sentence-embedding cosine of the team futures titles |
| (B) entity match on titles | 0.60 | 0.50 | self only | team strings matched across series |
| (E) diagonal + rank-6 graph | 0.61 | 0.57 | 0.61 | surprise → response |
| (E) AGCRN signed | 0.57 | 0.49 | 0.60 | surprise → response |
| (E) AGCRN adaptive | 0.57 | 0.48 | 0.45 (rival hit 0.00) | surprise → response |
| (E) pairwise Spearman | 0.55 | 0.51 | 0.61 | surprise → response |
| (E) free linear graph (ridge) | 0.53 | 0.51 | 0.60 | surprise → response |
| (E) pairwise, BH-selected | 0.51 | 0.50 | – | almost nothing selected |
| (D) price-change correlation | 0.46 | 0.53 | – | co-movement of the division prices |

AUROC against the number of slate days fitted:

| graph | 30 | 61 | 91 | 122 | 166 |
|---|---|---|---|---|---|
| (C2) schedule from titles | 0.60 | 0.65 | 0.72 | 0.74 | 0.81 |
| (E) AGCRN signed | 0.57 | 0.57 | 0.56 | 0.58 | 0.57 |
| (E) rank-6 graph | 0.57 | 0.55 | 0.44 | 0.55 | 0.61 |
| (E) pairwise Spearman | 0.49 | 0.51 | 0.52 | 0.51 | 0.55 |

- **The metadata graph finds the structure; the price-learned graphs don't.**
  - Reading the schedule off the game titles identifies division rivals
    essentially perfectly (rival AUROC 0.999), and improves as the season
    accumulates games.
  - Every graph learned from surprise → response, from pairwise correlation up
    to AGCRN, stays at 0.53–0.61 overall and about chance on the rival edges.
    It barely improves with five times the days.
- **Why the learning fails despite ~160 events per edge:**
  - The per-edge signal is weak. A team's own edge has Spearman ≈ +0.2 on its
    labelled days; the daily division price also moves with every other game
    in the division.
  - The rival edges are about five times weaker again.
  - About a sixth of teams sit at 1¢ all season, where nothing moves.

  At ρ ≈ 0.2 with 30–120 labelled days per edge, `recovery_test.md` puts
  per-edge recovery near chance. The sports data has more events, but not
  enough signal per event.
- **The standard AGCRN cannot represent the rival edges at all.** Its softmax
  adjacency is non-negative, so it recovers the self edges (hit 0.90) and gets
  none of the 120 negative rival edges (hit 0.00). The signed variant gets 44%
  of them, i.e. no better than chance.
- **Pure text semantics (C1) is weak.** Embedding the futures titles mostly
  measures the similarity of city names ("New York M" / "New York Y",
  "Chicago C" / "Chicago W"), which cuts across leagues.

## 2b. Fixing the negative-edge failure (`signfix.py`)

The standard AGCRN gets none of the 120 rival edges' signs.

**Why:** each node combines its own input (identity support) with a weighted
average of its neighbours (the learned adjacency Ã). Ã is non-negative, but
the conv weight on it can be negative, so "rivals push you down" is
representable in principle. The problem is that the learned Ã also aggregates
each node's *own* surprise (the softmax includes self). With one weight per
support, a negative weight would flip the team's own effect too, so the model
keeps it positive.

**Two fixes tested,** both supplying the neighbourhood from metadata (the
schedule read from game titles, fitted on the training days only, no
self-loops):

| AGCRN variant | rival sign hit | balanced sign | AUROC | rival AUROC |
|---|---|---|---|---|
| standard adaptive | 0.00 | 0.45 | 0.57 | 0.48 |
| signed (E_dst E_srcᵀ), unrestricted | 0.44 | 0.60 | 0.57 | 0.49 |
| signed, restricted to schedule edges | 0.43 | 0.60 | 0.60 | 0.52 |
| **schedule graph as a frozen, unsigned support** | **0.89** | **0.90** | **0.83** | **0.81** |
| true signed graph frozen in (reference) | 0.89 | 0.90 | 0.95 | 0.95 |

(Full season, 166 days; at 91 days the schedule support gives 0.88 / 0.89 /
0.77.)

- **Supplying the neighbourhood as a separate, unsigned support fixes it.**
  The identity support carries the self effect (+). The schedule support
  carries only rivals, and its conv weights learn a negative sign from the
  data. Sign recovery then matches having the true *signed* graph frozen in,
  although no sign was supplied.
- **Freeing the sign per edge does not help,** even restricted to the right
  edges. A signed adjacency still has to learn 120 separate signs from a weak
  per-edge signal (0.43). The working version shares one sign across a node's
  rivals, which is a handful of parameters, not one per edge.
- **In the library** this needs no code change: pass a non-negative adjacency
  with a zero diagonal through the existing `adjacency="stage1"` path. The
  general recipe matches the rest of the study: structure from metadata or
  theory, sign and size from the data.

## 3. Trading the lag with each graph (`predict.py`)

Each graph is fitted on slate days up to a cut, and trades the next month's
games on its 150 strongest edges (walk-forward, four folds, May–September).
The trade: take the target's division contract at its first print after the
game, on side sign(W·z), and exit at +24 h. Gross cents, no costs; CI over
games.

| graph | trades | share that are true edges | gross ¢/trade [95% CI] |
|---|---|---|---|
| **(A) mechanical** | 8,025 | 1.00 | **+0.32 [+0.22, +0.42]** |
| … own contracts only | 1,612 | | +0.73 [+0.47, +0.98] |
| … division rivals only | 6,413 | | +0.22 [+0.13, +0.31] |
| (E) free linear graph | 12,516 | 0.18 | +0.19 [+0.10, +0.28] |
| (E) AGCRN signed | 10,863 | 0.24 | +0.15 [+0.05, +0.24] |
| (C2) schedule, read as "rivals move against you" | 8,032 | 0.42 | +0.11 [+0.02, +0.20] |
| (E) pairwise Spearman | 8,445 | 0.17 | +0.10 [−0.00, +0.20] |
| (E) AGCRN signed, surprise-only | 9,243 | 0.29 | +0.10 [+0.01, +0.18] |
| (E) rank-6 graph | 9,320 | 0.20 | +0.06 [−0.05, +0.19] |

- **The mechanical graph captures the lag best**, and the learned graphs
  capture it in proportion to how many true edges they hold.
- **It isn't tradeable.** +0.32¢ gross is a fraction of a taker fee on these
  contracts. The lag is a diffusion finding, not a strategy.

## What this adds to the thesis

1. **Information diffusion is directly measurable where the graph is known.**
   About a third of a game result reaches the team's division market after the
   first post-game trade, and the rivals' markets move the opposite way with
   the same lag. That is the thesis's diffusion question answered in a setting
   with no identification problem.
2. **More events alone don't make the graph learnable.** With ~30× the events
   per edge of the macro panel, learned graphs still don't recover a structure
   that is certainly there, because the signal per event is weak and
   state-dependent. The binding constraint is signal-to-noise per edge, not
   only the count. That sharpens `recovery_test.md`'s conclusion.
3. **Structure comes from metadata, not prices.** A graph read from the market
   titles (who plays whom) recovers the cross-team structure essentially
   perfectly. This is the sports analogue of `STG_stocks.pdf`'s
   news-co-occurrence graph beating the correlation graph, and here it can be
   checked against a known truth. Price co-movement (D) is no better than
   chance.
4. **The practical recipe is the same as for macro:** supply the graph (from
   theory or metadata) and let the data estimate the sign and size, rather than
   asking an STG to discover the graph from prices. §2b shows it working inside
   AGCRN: given the unsigned schedule graph as a support, it learns the negative
   rival sign that it cannot find alone (0.00 → 0.89).

## Caveats and next steps

- **One season, in-sample; the walk-forward is within 2025.** Team strength
  drifts through a season.
- **Daily slate labels** mix a team's own game with its rivals' games the same
  day. The per-game drift in §1 and §3 avoids this.
- **The rival effect should be state-dependent:** it matters only for
  contenders, not teams at 1¢. Restricting to contender days (division price
  5–95¢) is the obvious next test of whether per-edge learning can work at all.
- **Not yet done:**
  - the NBA playoff hierarchy (game → series → conference → championship: the
    multi-hop test);
  - NCAAF;
  - the live in-game clock;
  - (B)/(C) on a league without ticker codes.
