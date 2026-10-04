# Graph learning: proposed changes (October 2026)

This proposal builds on the between-game work: node values, the (node, night) samples, the 4a/4b
linear models, the trade tests, the markouts and design studies 1-7. Every number below comes from
`analysis/graph_2026_10/out/`, cited by file. It respects the decisions in `between-game-design`
(live set, cents with log-odds as a robustness check, decision at the first hourly close, league
result times), the strict as-of rule, and chunk-by-chunk work.

## Summary (by expected impact)

1. **Weights are a function of the state, not a constant per edge.** Build rung 2 (the
   attribute-learned weight g) around the leverage form that 4b recovered,
   w = g(p(1-p), season left, ...). It is the largest out-of-sample gain so far and the main thesis
   result.
2. **Re-key rivalry by competition proximity, not by event membership.** Keep and strengthen
   division-rival edges into the division, conference and playoffs contracts. Drop or shrink the
   wide "same event" rival edges (champ, other-conference teams), which carry almost nothing.
3. **Make the shock two-dimensional for NBA/NFL (result z + margin surprise).** Fix the MLB margin
   gap with a price-implied fallback.
4. **Add directed hierarchy and lead-lag edges within a team** (champ/division → the other kinds of
   the same team), plus memory over horizons. MLB/NBA absorb slowly; the NFL does not.
5. **Learn in log-odds, or with a saturating message,** so the edge effect vanishes at 0/100. That
   is where the linear-in-cents form extrapolates badly.
6. **Add scheduled-meeting edge features** (a rival met within 7 days). Small gain, cheap.
7. **Do not add** a league-wide common-factor node or asymmetric win/loss messages. The evidence
   doesn't support either (the NFL/NBA wins nodes are the exception, below).
8. **On the trading side,** use the graph for maker risk control (pull quotes on contracts whose
   team or rivals are live), not as a taker signal.

## Part A: thesis-core graph learning

### A1. State-dependent edge weights (rung 2, leverage form)
- **Change:** the per-relation weight is w_r(state) = a_r · p(1-p) · (b_r + c_r · season_left), or
  g_r(p(1-p), season_left, games_behind, meetings_next) as a small MLP with shared weights. Apply
  it to both the own-team shock and the rival messages. Drop games_behind as a main input.
- **Evidence** (`linear_weights_4b.txt`): out-of-sample R², 4a → 4b at 12 h. MLB division 0.29 →
  0.445, NBA playoffs 0.162 → 0.299, NFL division 0.48 → 0.61, NFL playoffs 0.425 → 0.57.
  Interactions with t across folds of 5-7:
  - z × p(1-p): +17.3 (MLB division), +87.8 (NFL division), +102 (NFL playoffs);
  - z × season left: −5.4 (MLB division, t −7.1);
  - rival × season left: +2.0 / +5.4;
  - games_behind: ≈ 0 throughout.

  Marginal effects are monotone and plausible: an open race late in the season gets 26c per unit z
  in NFL division, against about 4c for an early longshot. There is no gain on champ, NBA
  conference, wins or MLB playoffs, where the chosen penalty went to 1,000.
- **Expected effect:** the 4b gains plus per-pair resolution, and recovery of leverage structure.
  That is the "can the graph learn the relations" result.
- **Cost:** medium. It needs the panel builder (nights × nodes × relations) and a relation-typed
  message-passing layer.
- **Test:** rung 2 vs 4a/4b in the same purged walk-forward, with the per-fold penalty and the live
  set. Report marginal-effect surfaces, not raw weights. The product-form variant should match or
  beat 4b with fewer parameters, and must behave better at the tails (see A5).

### A2. Rival edges by competition proximity
- **Change:** split `rival_*` into `rival_division` (applied across kinds: into the division,
  conference and playoffs contracts) and `rival_wide` (same event, outside the division). Weight
  `rival_wide` by price (A3) or drop it. Remove cross-conference pairs from champ messages, or keep
  them only as a placebo relation.
- **Evidence** (`design_studies_A.txt` §2, 12 h, live):
  - division rivals → division: MLB −0.705 (t −9.6), NFL −4.75 (t −9.5);
  - division rivals → other kinds: MLB conference −0.058 (t −2.8), NFL playoffs −2.27 (t −3.8),
    NFL conference −0.30 (t −1.6), NFL champ −0.19 (t −1.7);
  - other same-conference teams: about 0 everywhere except NFL division (−0.46, t −2.1).

  Study 7 (`design_studies_B.txt`) agrees: NFL playoffs rivals with a meeting coming up −5.6
  (t −2.7) vs others −0.6.
- **Expected effect:** fewer noisy edges (champ alone has 870-992), cleaner recovery scoring, and
  probably small out-of-sample gains on conference and playoffs.
- **Cost:** low. These are new relation names in `graph.py`, built from the division and
  conference memberships already in `nodes`.
- **Test:** an ablation with (a) the current edges, (b) proximity edges, (c) proximity + wide.
  Score out-of-sample R² and the sign-recovery rate.

### A3. Reallocation prior for exclusive division events
- **Change:** in division events, give the rival message a fixed-form prior,
  Δp_j ≈ −S · p_j / Σp_nonplayers, as a supplied rung-3 edge or a feature. Don't impose it on
  champ or conference.
- **Evidence** (`design_studies_A.txt` §1): NFL division slope 0.885 (t 19.8), proportional
  R² 0.648 vs equal-split 0.429, and in a head-to-head the proportional term wins (0.913 vs
  −0.042). MLB division slope 0.34 (t 3.1). Champ and conference show no reallocation within 12 h
  (slopes −0.23 to 0.11).
- **Expected effect:** a near-exact relation for NFL division, free of parameters. It also flags
  that champ/conference sums are *not* conserved overnight, which feeds the hierarchy-consistency
  trading test.
- **Cost:** low. **Test:** adding it as a feature to 4b, and as a supplied edge in rung 3.

### A4. Two-dimensional shocks; fix MLB margin
- **Change:** shock = (z, margin surprise ms) for the NBA and NFL. MLB's ms is null for all 4,186
  team-nights: only 36 MLB games in 2025 have spread (run-line) markets, so the ladder fit fails.
  **Fix:** a price-implied margin fallback, μ = σ_league · Φ⁻¹(p_pre), with σ_league fitted on
  training folds only. Or take MLB 2026 run lines once available.
- **Evidence:**
  - `design_studies_A.txt` §4, NBA playoffs: ms 0.036c/pt (t 3.2); z falls 3.27 → 2.42 when ms is
    added. NBA conference 0.009 (t 2.9). NFL conference 0.056 (t 3.4).
  - NFL division and playoffs: ms is not significant once z is in.
  - `nextgame_check.txt`: next-game markets move on margin only (NFL 0.19c/pt, t 6.0; NBA 0.06,
    t 3.1), not on z.
- **Cost:** low. **Test:** a 4b ± ms ablation by league and kind. Validate the fallback against
  the ladder-based ms on the NBA, where both exist.

### A5. Units and saturation
- **Change:** train in log-odds of the price, as the main robustness check, or keep cents with a
  saturating message, e.g. w ∝ p(1-p) as in A1, so effects vanish at 0/100.
- **Evidence:** the linear form gives an NFL playoffs contract at 5c a weight of 14c per unit z
  late in the season (marginal-effect table, this session), too large for a 5c contract. Some
  asymmetry is consistent with saturation (`design_studies_B.txt` §5): NFL division loses 21.1 vs
  wins 15.3. Own × rival interactions are positive in division races (NFL +3.48, t 3.2; MLB −0.38,
  t −2.4). Otherwise wins and losses are symmetric (t-tests of the parts overlap), so no
  asymmetric messages.
- **Cost:** low. **Test:** the same models in log-odds on the live set (already supported in
  `linear_weights.py`).

### A6. Memory, direction and horizons (lead-lag)
- **Change:**
  - directed within-team edges from the fast nodes (champ, division) into the slower kinds;
  - one recurrent / memory step across the night's horizons, or simply the leader's `moved` as an
    input;
  - multi-horizon outputs (1/12/24 h) for MLB/NBA, mainly 1 h for the NFL.
- **Evidence** (`design_studies_A.txt` §3):
  - share of the 48 h response already in place at the decision: NFL 0.80-0.96, NBA 0.41-0.50,
    MLB 0.16-0.33;
  - the leader's `moved` predicts the follower's remaining move: MLB conference ← champ 0.165
    (t 4.1) and ← division 0.045 (t 4.8); NFL conference ← champ 0.227 (t 4.0); NFL division ←
    champ 0.284 (t 2.4);
  - the follower's own `moved` reverts (−0.07 to −0.53, t −3 to −11). Partly real: in
    `checks_4a.txt` it survives the 1 h skip outside MLB division.
- **Cost:** medium (the event clock / memory in `agcrn.py`).
- **Test:** 4b + leader `moved` (linear) first. Use the recurrent version only if it beats that.

### A7. Things not to add (for now)
- **A league-wide common-factor node:** night R² of the residuals is at chance (`design_studies_B.txt`
  §6b: e.g. MLB ALL 0.020 vs chance 0.026). The exception is NFL wins (0.433 vs 0.062), probably
  ladder-fit drift rather than news; treat it as a node-value problem.
- **Statistical co-movement edges:** unexposed rows move about as much as exposed ones in MLB/NBA
  (§6a). Most "news without games" is noise around 1c ticks; co-movement edges would mostly fit it.

### A8. Evaluation and recovery scoring
- **Rungs:** 1 (free, AGCRN `adjacency="signed"`), 2 (A1 + A2 attributes), 3 (supplied A2/A3
  edges). The baselines are 4a, 4b and a no-graph model, all on identical folds.
- **Recovery:** the Jacobian readout of each rung against the supplied signs per relation, plus a
  comparison of the leverage surface (marginal effect vs p(1-p) and season left) with 4b's.
- **Data limits:** the NFL has 57 nights, too few for flexible models. Pool leagues with shared g
  and league-specific scales (plan §10), and report NFL results with that caveat.

## Part B: trading-oriented changes
- **Taker propagation is shelved.** The predictions are well calibrated (`trade_edge.txt`: realised
  ≈ 0.7-0.9 × predicted) but rarely exceed spread + fees. With calibrated edge > cost there are 80
  trades at −1.34c net; the best cell is MLB division at 24 h, +0.65c gross / −0.83c net. Don't
  optimise the graph for taker P&L.
- **Maker risk control:** use the graph's node → game links to pull or widen quotes during the
  own team's live game, the only clearly adverse state (`markouts_futures.txt`: division own game
  live −2.03c at 24 h, t −3.0; NFL −0.39). Quote after results (own result 0-3 h: playoffs +2.12,
  t 4.6; rival result 0-3 h: +0.5 to +0.7, t 6-8). Game markets: in-game resting orders turn
  negative by 2 h (`markouts.txt`: MLB −0.89, NBA −0.37).
- **Next-game nodes:** the margin under-reaction to the close (NFL 0.09c/pt, t 2.7) is the
  candidate. Test it out of sample with the closing-line check before adding next-game nodes to the
  learned graph.

## Fragile results: check before they shape the design
- **Everything in the design studies is in-sample OLS.** A2-A6 need the same terms re-tested out of
  sample (4b-style walk-forward) before being adopted.
- **Small n:** MLB playoffs (131-222 rows), NFL cells (190-300 played rows, 57 nights), study 7's
  NFL cells (6-16% of rows with a meeting), study 1's MLB champ (few non-players per night).
- **Markouts are contract-weighted** (a few huge trades dominate). The "quiet" state is mostly
  pre-season dormancy. **Settlement t-stats are invalid** (one outcome per team for the season,
  clustered by day). Re-run trade-weighted, and cluster by team-season for settlement.
- **The `moved` reversal** may partly be stale hourly quotes that a 1 h skip doesn't remove.
- **The maker fee base rate is unverified** (`reports/sports_data_sources.md`).
- **Wins nodes:** NBA R² ≈ 0 and NFL wins night R² 0.43 point to ladder-fit noise. Consider
  excluding wins from rung 2 until fixed-σ fits exist.

## Decisions for the user
1. Adopt the proximity rival keys (A2) and drop the wide rival edges, or keep them price-weighted?
2. Main units: log-odds for graph training, or cents with the p(1-p) form?
3. Pool leagues with a shared g (more data, especially for the NFL) or fit per league?
4. MLB margin: the price-implied fallback now, or wait for 2026 run lines?
5. Keep wins nodes in the learned graph?

## Proposed order of work (each chunk reviewed before the next)
1. **Out-of-sample re-test of A2-A4 and the A6 leader input inside 4b** (cheap; decides the edge
   set and the shocks).
2. **Graph panel builder** (nights × nodes × relation adjacencies, masks), plus a 1-layer
   relation-typed model that must reproduce 4a.
3. **Rung 2 (A1)** with the chosen edges and units; marginal-effect surfaces; recovery scoring.
4. **Rungs 1 and 3 plus the no-graph baseline;** the comparison table (plan §8).
5. **Memory / lead-lag (A6),** only if chunk 1 shows the leader input helps.
6. **In parallel (trading):** trade-weighted markouts, the hierarchy-consistency test, the
   next-game closing-line test.

## Two tracks (agreed with the user, 2026-10-04)

The work splits into two versions of the graph sharing one architecture (data layer, walk-forward,
panel builder, relation-typed model with a learned weight g, recovery scoring):

| | A. Between-game (games → futures) | B. In-game (plays → same-game markets) |
|---|---|---|
| step | a night of results | a play (or only significant plays) |
| nodes | team × futures kind | one game's winner (merged), spread and total rungs, halves / first 5, team totals; props later |
| shock | result z (+ margin surprise) | change in win probability on the play (league feed) + play type, score, time left |
| edges | competition-proximity rivals, team hierarchy, leader → follower | exact edges from the edge table (complement, ladder, part-of-whole) |
| g's inputs | p(1-p), season left, meetings_next | strike relative to the current state (line minus margin / points so far), time left |
| horizon | hours (first hourly close; labels 1-48 h) | seconds to minutes (1-min candles + trades) |
| data | one season; thin for the NFL (pool leagues) | plentiful, NFL included |
| trading | maker risk control | latency: does a fast feed beat the market |

Bridge B → A later: in-game swings → futures at minute resolution, once the targeted 1-min
futures pull exists (more data for the NFL futures relations).

Order, chunk by chunk:
1. A1: the out-of-sample re-test above.
2. B1: in-game sample table and a descriptive check (NFL first): absorption speed per play,
   winner → spread / total lead-lag, the strike-relative-to-state effect.
3. A shared panel builder for both, and a 1-layer model that reproduces the linear baseline in each.
4. Rung 2 in both, then rungs 1 and 3 plus the no-graph baselines.
