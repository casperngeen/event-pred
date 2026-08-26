## Progress Report (Week 3) - 26 August 2026 

## By: Justyn Lae (FYP, Semester 2)

## Feasibility Assessment: Structural Arbitrage in Kalshi's High-Dimensional Event Clusters (Direction B)

## 1. Background

Following the semester 1 work on the spatio-temporal graph builder, two candidate directions were considered for the deep-learning phase:

- **Direction A**: let the model search for lead-lag relationships across the full, unrestricted Kalshi universe, with no event categorisation.
- **Direction B**: narrow scope to Kalshi's high-dimensional, multi-outcome ("N-way") event clusters (strike ladders, score grids, categorical standings markets) and use graph attention to price illiquid outcomes from their structural neighbours, exploiting deviations from the hard mathematical constraint that a complete set of mutually exclusive outcomes must sum to their true value.

This document summarises a baseline finding that motivated moving away from Direction A, and a sequence of empirical feasibility checks run against real October 2025 Kalshi data to test whether Direction B's premise actually holds before committing the semester to it.

## 2. Baseline Finding: Evidence Against Unrestricted Discovery

An initial statistics-only pairs-trading baseline (correlation and half-life-based pair selection, 1% per-turn transaction cost, October 2025) produced a Sharpe ratio of 5.52 and ~46% return. Inspecting the actual pairs selected, however, most were not economically related: for example, a market on a musician's chart performance was paired with an unrelated album-ranking market, and a college sports coaching-change market was paired with a public-health case-count market.

Given a 20-day selection lookback, a loose correlation threshold (0.20), and a large pool of candidate pairs (50 selected from a much larger combinatorial space), this result is best explained by spurious correlation from multiple testing rather than a genuine market relationship. At n=20 observations, a correlation of roughly 0.44 is already nominally significant at p<0.05 under a naive test; testing tens of thousands of candidate pairs without correction all but guarantees many will clear that bar by chance alone.

This finding directly motivated Direction B: a higher-capacity model (graph attention) applied to the same unrestricted universe would have strictly more capacity to fit this same noise, not less, without a strong structural prior to anchor it.

## 3. Market Structure Analysis

The October 2025 Kalshi universe (106,632 unique events) was categorised by ticker count per event:

| Tier | Count | Share |
|---|---|---|
| Implicit binary (1 ticker, single YES/NO market) | 64,443 | 60.4% |
| Explicit binary (2 tickers, independently calibrated) | 16,082 | 15.1% |
| N-way clusters (3+ tickers) | 26,107 | 24.5% |

## 4. Feasibility Testing for Direction B

### 4.1 Full-basket constraint test (markets quote data): inconclusive due to a data limitation

The first test attempted to check directly whether the sum-to-one (or sum-to-K, see §4.2) constraint is ever violated by enough to be profitable after transaction costs, using the markets file's `yes_bid`/`yes_ask` quotes. After several iterations tightening how simultaneity was measured across legs (same day -> same hour bucket -> an as-of/staleness-aware lookup), the test consistently returned almost no usable snapshots.

The root cause was structural, not a script bug: the `markets_kalshi_even` data file contains **exactly one row per ticker for the entire month**, a single static snapshot, not a quote time series. Across all evaluated N-way clusters (≥20 legs), only 268 of 531,127 legs (0.05%) ever showed a real, non-boundary-default quote at all; the rest showed the exchange's empty-order-book defaults (`bid=0, ask=100`), meaning no market maker had ever quoted them in this file.

**Conclusion**: this specific data source cannot support a live-quote-based feasibility test on the N-way tier. This is itself informative: it is direct, quantified evidence that these legs are illiquid enough that a model inferring price from neighbours is necessary, not optional, since there is often no live market to read a price from at all.

### 4.2 Resolved-outcome validation: structural confirmation of three distinct constraint families

To validate the constraint structure without depending on live quotes, already-*resolved* markets were used: for each fully-settled N-way cluster (≥20 legs), the number of legs that actually resolved "yes" was counted directly from the `result` field. This requires no assumptions about quote timing or liquidity.

Across 6,755 fully-resolved clusters, the distribution of yes-resolutions was:

| Yes-legs resolved | Count |
|---|---|
| 0 (void/no-resolution) | 681 |
| 1 | 2,234 |
| 2 | 134 |
| 3 | 115 |
| 4 | 76 |
| … (small counts, e.g. 6, 8) | matches "Top N" sports-standings pattern |
| Large / scaling with cluster size (e.g. 46, 195, 61) | a distinct pattern, investigated below |

This revealed **three structurally distinct families** within the N-way tier, not one uniform type:

1. **Single-winner MECE partitions** (~33% of resolved clusters, `n_yes == 1`): a genuine "exactly one of N wins" structure, where the correct no-arbitrage target is sum-to-$1.
2. **K-of-N categorical events** (small fixed counts of 2-10): e.g. "Top 8," "Top 6," "Top 4" sports-standings markets, where exactly K outcomes resolve yes (K teams advance, K teams relegated). The correct no-arbitrage target here is sum-to-$K, not sum-to-$1, an important correction, since naively assuming K=1 for these produces false "arbitrage" signals in the opposite direction of what's actually happening.
3. **Nested threshold ladders** (large, variable counts, e.g. "will the S&P 500 / Bitcoin / USD-JPY be above $X" repeated across many strikes): these are not a fixed-sum partition at all. The number of legs resolving "yes" simply reflects how many strikes fell below the actual settlement price. The correct no-arbitrage condition for this family is **monotonicity**, not a fixed sum: for adjacent strikes T_low < T_high, no-arbitrage requires `yes_ask(T_low) ≥ yes_bid(T_high)`, since a portfolio of long YES(T_low) / short YES(T_high) pays exactly $1 if the price lands between the two strikes and $0 otherwise, a payoff that can never be negative, so it can never be free to enter.

### 4.3 Pairwise monotonicity test (trades-based): validated evidence of real mispricing

Since the ladder family's no-arbitrage condition is pairwise (only two adjacent legs need to be live at once, not the whole basket), and the markets file could not support a live-quote test, this was tested using **trade data** instead, which is the only genuinely time-varying price signal available in this dataset (last-traded price per leg per day, used as a fair-value proxy).

An initial keyword-based method for identifying ladder events (titles containing "above"/"below"/"over"/"under" etc.) was too permissive and mistakenly included NFL player-prop combo markets, whose "legs" are different players' unrelated stat lines rather than different thresholds of one shared underlying. This was corrected by requiring every leg of an event to share an identical title template (masking out only the numeric threshold): a structural check that a genuine ladder passes and a combo-prop market does not. This left **669 verified single-underlying ladder events** (144,682 legs; crypto, FX, and equity-index price-level markets).

For each pair of adjacent strikes, monotonicity requires that the lower strike's price be at least as high as the higher strike's price (see §4.2), since it is always at least as easy to clear the lower bar. A **violation** is a pair where this is reversed: the higher strike traded at a higher price than the lower strike, which should not happen if the market is priced consistently. The **violation rate** reported below is simply the share of all adjacent-strike pairs checked that showed this reversed pricing at a given time gap.

An initial same-calendar-day comparison found a 33.8% violation rate, but wide-strike-gap pairs violated almost as often as narrow ones (31.1% vs. 34.8%), suggesting the real confound was time desynchronisation (the underlying moving between two same-day but non-simultaneous trades) rather than strike proximity. Splitting by actual clock-time gap between the two trades gave the decisive result:

| Time gap between trades | Pairs | Violation rate | Median violation size |
|---|---|---|---|
| 0-15 min | 2,092 | 29.3% | $0.08 |
| 15-60 min | 526 | 46.2% | $0.07 |
| 1-4 hours | 130 | 44.6% | $0.035 |
| 4-12 hours | 42 | 69.0% | $0.08 |
| 12-24 hours | 5 | 40.0% | $0.07 |

If the earlier result were purely a timing artifact, the violation rate should approach zero as the time gap shrinks toward true simultaneity. It does not: even within a 15-minute window, which holds the large majority of all tested pairs: 29.3% of adjacent-strike pairs violate monotonicity, with a median magnitude of 8 cents on a dollar. This rules out ordinary intraday price movement as a complete explanation.

**Out-of-sample replication (November 2025).** The table above was October-only. The same test was re-run including November 2025 to check whether the finding was an October quirk:

| Time gap | Oct 2025 only | Nov 2025 only | Combined |
|---|---|---|---|
| 0-15 min | 612/2,092 (29.3%), median $0.08 | 669/2,001 (33.4%), median $0.07 | 1,281/4,093 (31.3%), median $0.07 |
| 15-60 min | 243/526 (46.2%), median $0.07 | 261/500 (52.2%), median $0.09 | 504/1,026 (49.1%), median $0.08 |
| 1-4 hours | 58/130 (44.6%), median $0.035 | 43/96 (44.8%), median $0.03 | 101/226 (44.7%), median $0.03 |
| 4-12 hours | 29/42 (69.0%), median $0.08 | 11/29 (37.9%), median $0.04 | 40/71 (56.3%), median $0.06 |
| 12-24 hours | 2/5 (40.0%), median $0.07 | 4/9 (44.4%), median $0.09 | 6/14 (42.9%), median $0.07 |

November reproduces October's pattern independently, at every time-gap bin, with violation rates and median magnitudes in the same range in both months separately (not just when pooled). This is meaningful confirmation that the finding is a genuine, persistent property of this market tier rather than a one-month artifact, the strongest evidence in this report for any of the three constraint families.

### 4.4 K-of-N full-basket test, part two: a direct STG coverage demonstration

The full-basket test for the K-of-N family (§4.1's trades-based analogue) was extended in two ways: Firstly, a dedicated script (`k_to_n_check.py`) tested all 5 candidate K-of-N events (≥20 legs, non-ladder) directly against trades data, scoped to the month(s) available. Secondly, the same 5 events were run through the **actual semester-1 STG pipeline** (`GraphBuilder` + `KalshiTickerNodes` + `FixedWindowTemporal` + `KalshiEventEdges`), so the coverage gap could be read directly off the graph object the model would actually train on, rather than off a bespoke script's counters.

Two runs were made, both selecting the *best-covered* of the 5 candidates by total trade volume (i.e. the most generous case available, not a cherry-picked worst case):

| Run | Event selected (best of 5) | Total legs | Legs ever traded | Mean per-snapshot node coverage | Snapshots with ≥2 nodes present |
|---|---|---|---|---|---|
| October 2025 only | `KXEPLTOP6-26` | 20 | 3 (15.0%) | 5.8% | 1 of 6 |
| October-November 2025 | `KXEPLRELEGATION-26` | 20 | 2 (10.0%) | 5.4% | 1 of 12 |

In both runs, the other 4 of the 5 candidate events had **zero trades at all** in the scanned window, making the event shown here the best case, not a representative or worst case.

Two findings follow directly from the graph object itself, independent of any script's counting logic:

1. **The coverage ceiling is structural, not a windowing artifact.** `SpatioTemporalGraph.all_node_ids()`, the union of every node that appears in *any* snapshot, under any window choice, was 2-3 of 20 legs in both runs. A leg with zero trade rows anywhere in the data cannot become a node under any temporal strategy, forward-fill included, because `KalshiTickerNodes.identify_nodes()` has no rows to construct a `NodeState` from. This is the STG's own data model confirming the same limit the trades-based diagnostic found.
2. **Almost no snapshot ever has a connected basket.** Using `KalshiEventEdges` (which connects every node present in a snapshot to every other node of the same event: i.e. exactly the full-basket structure the sum-to-K constraint needs), 11 of the 12 November-window snapshots had 0 or 1 node present, and therefore 0 edges: there usually is not even a second leg simultaneously present to compare against, let alone the full 20-leg basket a genuine sum-to-K check requires.

This is a stronger and more direct version of §4.1's conclusion: it is not just that the *markets* (quote) file lacks live prices for this tier. The *trades* (time-varying) file, run through the actual graph-construction pipeline the GAT would consume, produces a graph that is disconnected or empty at the basket level in the overwhelming majority of snapshots. 

A caveat worth carrying into discussion: the underlying markets do not close until 2026-05-25, while the available dataset ends in November 2025, so this reflects only the first ~2 months of each market's life, not its full run.

This should not be read as a permanent verdict on the K-of-N family. There is a plausible reason coverage could improve later in the season: standings and relegation markets are known to see trading interest concentrate near the end of a season, once only a handful of teams are still mathematically live for the outcome and the stakes become concrete rather than abstract. October and November, when almost nothing is decided yet, are close to the worst possible window to test this family in.

That said, a competing limitation would remain even with a full season of data: typically only the *contending* legs (teams whose fate is genuinely uncertain) attract real trading, even at peak stakes, while legs that are already clearly safe or clearly doomed may never trade meaningfully. Full 20-leg coverage might therefore never be achieved even at season's end, though a smaller "live contenders" subset could become genuinely testable. Our own data offers a small, mixed signal on this already: trade volume for `KXEPLTOP6-26` grew substantially between the October-only and October-November windows (8 trades to 37), but leg coverage did not broaden alongside it, the best-covered candidate in the two-month window was still only 2 of 20 legs. That is consistent with the same narrow set of already-liquid legs trading more often rather than new legs joining in, though two data points is not enough to call this a settled trend.

**This is a testable question, not a closed one**: once more months of this season's data become available, re-running `k_to_n_check.py` / `kof_n_stg_sparsity_demo.py` on the later months would show directly whether leg coverage broadens toward the full basket, plateaus at a small contending subset, or stays flat, before deciding whether to invest further model-building effort here.

### 4.5 Single-winner MECE full-basket test: a second confirmed positive result

§4.1 left the single-winner MECE family (exactly one leg resolves "yes": e.g. "which of these six temperature ranges will be the day's high?") untested for live mispricing, since the markets file couldn't support it and the sum-to-$1 constraint needs every leg trading at once, not just a pair. This was tested directly, following the same methodology discipline the ladder test required:

1. **Resolved-outcome validation** (as in §4.2) identified fully-resolved, single-winner clusters (3-60 legs).
2. **Family-level consistency check**: a naive per-instance check let through several threshold-ladder structures that only *coincidentally* resolved with one "yes" leg for a specific game (e.g. "total runs over 3/4/5/.../14" resolving with exactly one leg true because that game's score happened to clear only the lowest bucket), these evaded title-keyword ladder detection because the threshold lived in the ticker suffix, used "15+" notation instead of a keyword, or bundled two interleaved per-team ladders into one event. The fix: check whether each event *family* (e.g. `KXHIGHLAX`, the ticker prefix stripped of the per-instance date/game suffix) resolves with exactly one winner on *every* historical instance, not just this one. Only families consistent across ≥3 independent instances were kept: 27 families survived, dominated by daily weather markets (`KXHIGHLAX`/`KXHIGHNY`/`KXHIGHDEN`/etc.) and polling-bracket markets (`KXAPRPOTUS`).
3. The same combo-prop exclusion from §4.3/§4.4 (`SINGLEGAME`/`MULTIGAME` ticker families) was applied here too.

Unlike the K-of-N season-long standings markets, these MECE events are short-duration and recur daily, which shows up directly in coverage: **618 of 1,326 (event, day) rows with any trade had every leg trading (47% full coverage)**, the full-basket requirement the sum-to-$1 test needs, versus K-of-N's near-total absence of coverage.

Binning by the time-gap span between the day's legs' last trades (same discipline as §4.3):

| Time-gap span | Snapshots | Mean \|deviation\| | Median \|deviation\| | % exceeding $0.05 |
|---|---|---|---|---|
| 0-15 min | 59 | $0.053 | $0.030 | 44.1% |
| 15-60 min | 62 | $0.073 | $0.045 | 50.0% |
| 1-4 hours | 199 | $0.061 | $0.040 | 37.7% |
| 4-12 hours | 257 | $0.060 | $0.040 | 42.8% |
| 12-24 hours | 41 | $0.062 | $0.050 | 61.0% |

As with the ladder test, the violation rate does not shrink monotonically toward zero as the gap tightens (44.1% at <15min vs. 61.0% at 12-24h), arguing against pure staleness as a complete explanation. The largest individual examples (verified by inspecting each leg's title directly) reached $0.23-$0.47 deviation from $1.00, larger in magnitude than the ladder family's typical 7-8 cent violations, e.g. a Chicago daily-high-temperature market summing to $1.47 across its six mutually exclusive buckets.

**This makes the single-winner MECE family a second empirically-confirmed candidate, not just a structurally-confirmed one.** It tests a different (and arguably more demanding) constraint than the ladder family (the full basket, not just a pair) which is a closer match to the GAT's eventual all-legs attention mechanism.

## 5. Limitations

- This is trade-print evidence of pricing inconsistency, not proof of a live, executable arbitrage, that would require genuine simultaneous bid/ask data, which this specific markets file does not provide.
- Bid-ask bounce (the last trade on one leg being a buy at the offer, the last trade on the other being a sell at the bid) is a known alternative explanation for apparent violations between near-simultaneous trades, and cannot be fully ruled out without real quote data. The 8-cent median magnitude is larger than a typical bid-ask spread alone would suggest, but this remains an open question rather than a fully closed one.
- The single-winner MECE family is now empirically confirmed via trades data (§4.5), not just structurally confirmed via resolved outcomes: the same bid-ask-bounce caveat as the ladder family applies (last-traded price is a proxy, not a literal executable quote), and the full-basket sample size (618 snapshots) is smaller than the ladder test's pairwise sample (thousands of pairs), so it warrants continued monitoring as more months of data accumulate rather than being treated as fully closed.
- The K-of-N categorical family, by contrast, now has a direct negative result rather than an open item (§4.4): the best-covered of 5 candidate events had only 2-3 of 20 legs ever trade at all, and the resulting STG graph is disconnected at the basket level in nearly every snapshot. This was tested through the actual STG pipeline, not just a counting script, so it reflects the data the GAT would actually have to learn from.

## 6. Conclusion and Proposed Direction

Direction B seems to be valid and feasible. The strongest, most defensible evidence specifically supports the **threshold-ladder sub-family**: a controlled, near-simultaneous empirical test shows a persistent monotonicity violation rate of roughly 30% with a non-trivial median magnitude, in a family with the cleanest mathematical constraint (pairwise, not full-basket) and the most natural graph structure (strikes are inherently ordered, so "neighbour" is unambiguous: unlike, say, sports-standings categorical events, where there is no obvious adjacency between candidates).

**Proposed direction**: given the timeline, scope the GAT implementation around **two** empirically-confirmed structures rather than one: the threshold-ladder monotonicity family (§4.3: strongest sample size, out-of-sample replicated across two independent months) as the flagship case, and the single-winner MECE family (§4.5: smaller sample so far, but tests the full-basket constraint directly, and its deviations run larger in magnitude) as a strong second case rather than deferred future work. Both share the same underlying evidence quality: contamination identified and excluded, time-gap-controlled, and violation rates that don't shrink to zero as trades get closer in time. 

The K-of-N categorical family is not proposed as part of the initial GAT scope: §4.4's direct STG test shows the best-case candidate event still has a disconnected, near-empty basket graph in almost every snapshot across the two months of data available. As discussed in §4.4, this reflects a data-coverage limit in the current dataset rather than a settled structural dismissal, whether it changes later in the season is a genuinely open question rather than one this report can resolve, so it is left for discussion below rather than ruled out entirely.

**Open items for discussion**:
- Whether to pursue acquiring genuine top-of-book quote history (rather than the single-snapshot markets file used here) to fully settle the live-arbitrage question and rule out bid-ask bounce, for both the ladder and MECE cases.
- Whether to scope the GAT to handle both structures (ladder monotonicity + MECE sum-to-$1) in one architecture, or pick one as the primary deliverable given the timeline and treat the other as an extension.
- Whether the K-of-N categorical family should be revisited later if a live/streaming data feed (rather than this historical snapshot) becomes available, since §4.4's result reflects data coverage in the dataset collected so far, not a permanent property of the market structure itself.
- Confirming the GAT's training objective (e.g., self-supervised reconstruction: masking a liquid leg's price and predicting it from neighbours, versus another formulation), and whether that objective differs between the pairwise ladder case and the full-basket MECE case.

## Appendix A: Checks Performed ([casperngeen/event-pred](https://github.com/casperngeen/event-pred/tree/justyn), branch `justyn`)

The repository accompanying this report contains every script referenced above, each runnable directly (`python -m stg_infra.examples.<name>`). Listed in the order they were run, with the report section each one supports:

- **`stg/pairs/pipeline.py`, `examples/event_pairs_arb_baseline.py`**: the baseline pairs-trading strategy (§2) that produced the Sharpe 5.52 / spurious-correlation finding motivating Direction B. Re-resolves representative tickers per backtest window rather than reusing training-window tickers, since Kalshi event tickers are per-instance and don't persist month to month.
- **`examples/overlap_test.py`**: confirms Kalshi event tickers do not recur across months (0 of 78 October events also present in November), the structural fact behind the pipeline fix above.
- **`examples/testrun.py`**: market structure categorisation (§3): implicit binary / explicit binary / N-way tiers across the full October universe (106,632 events).
- **`examples/sum_to_one_check.py`**: full-basket sum-to-1 / sum-to-K constraint test against the markets (quote) file (§4.1). Three iterations (same-day -> hourly-bucket -> as-of/staleness-aware) before identifying the markets file's single-snapshot-per-ticker limitation as the root cause of the inconclusive result.
- **`examples/sum_to_k_check.py`**: resolved-outcome validation (§4.2): counts actual "yes" resolutions per fully-settled N-way cluster to structurally distinguish the three constraint families (single-winner MECE, K-of-N categorical, threshold ladder) without depending on quote timing.
- **`examples/pairwise_monotonicity_check.py`**: the threshold-ladder pairwise monotonicity test (§4.3), trades-based. Includes the title-template-homogeneity fix (excludes combo-prop markets falsely matched by keyword search) and the time-gap-binning methodology that produced the headline 29-33% violation-rate finding, replicated independently across October and November.
- **`examples/k_to_n_check.py`**: full-basket sum-to-K test on trades data for the K-of-N categorical family (§4.4), scoped to the 5 identified candidate events (20-60 legs). Found near-zero simultaneous coverage.
- **`examples/kof_n_stg_sparsity_demo.py`**: the same K-of-N coverage question (§4.4), tested a second way: builds the actual `GraphBuilder` + `KalshiTickerNodes` + `FixedWindowTemporal` + `KalshiEventEdges` pipeline on the best-covered candidate event's legs and reads node/edge sparsity directly off the resulting graph, confirming the negative result independent of any bespoke counting script.
- **`examples/mece_sum_to_one_check.py`**: full-basket sum-to-$1 test on trades data for the single-winner MECE family (§4.5). Includes the combo-prop exclusion and a family-level consistency check (an event family must resolve with exactly one winner across ≥3 independent historical instances to be trusted as a genuine partition, not a ladder that coincidentally resolved with one "yes" leg for a specific game).