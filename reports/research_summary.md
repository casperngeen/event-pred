# Research Summary: Cross-Market Influence Propagation in Event Prediction Markets

*Based on `FYP_CA_Report.pdf` (CA report, 2026-08-12), the `event-pred/` codebase, direct inspection of the collected data, and a literature scan. Data findings and literature current as of 2026-08-15.*

**Submission target: November 2026.** Roughly 3 months, with analysis realistically needing to freeze early October. This constraint drives the triage in §8.

---

## 1. Current Proposed Methodology (per CA report)

**Core question:** Does the resolution of one Kalshi macroeconomic event contract (CPI, FOMC, NFP, GDP, PCE, PPI, unemployment) act as an informational shock that produces systematic, predictable repricing in other *still-active* related contracts?

**Pipeline (4 components):**

1. **Data pipeline & feature engineering** (§4.1) — raw Kalshi trade-level data → daily OHLCV + derived features (order flow imbalance, trade arrival intensity) per market, then volume-weighted to a single event-level implied-mean signal. ~400 macro events, 2022–2025.
2. **Spatial-temporal graph construction** (§4.2) — strategy-pattern framework (`NodeStrategy`/`EdgeStrategy`/`TemporalStrategy`/`LabelStrategy`) producing resolution-triggered graph snapshots (Discrete-Time Dynamic Graph formulation). ~33 active event-nodes per snapshot; edges from co-activity/thematic proximity.
3. **Inductive representation + adaptive graph structure learning** (§4.3) — inductive adaptation of AGCRN (Bai et al. 2020): shared-MLP node embeddings (not a fixed table), adaptive adjacency `Ã = softmax(ReLU(EEᵀ))`, node-adaptive convolution weights `Wᵢ = MLP(xᵢ)Ŵ` inside GRU gating.
4. **Integration into trading strategy** (§4.4) — via `PortfolioBench`: Alpha layer (implied-mean-change predictions) → Strategy layer (sizing/entry-exit vs. momentum/mean-reversion) → Portfolio layer (Universal Portfolios, Online Newton Step; Sharpe, max drawdown, cumulative return).

**Empirical motivation (§4.3.1).** Event study picks the ATM ticker (closing price nearest 50¢) as event representative, tracks implied-probability change over a [−3, +14] day window normalised to zero at day 0, and computes mean reaction + 95% CI across trigger-target pairs. Applied exhaustively across all type-pairs with n ≥ 5, it identifies **22 relationships where the CI excludes zero for ≥3 consecutive days** — clustering into recognisable channels (WTI→CPI, WTI→FED, Payrolls→GDP, Unemployment→GDP, CPI→FedDecision). Drift is **gradual rather than immediate**, which the report uses to motivate a temporal model.

---

## 2. Repo Reality Check

Surveyed 2026-08-14. What exists versus what the report implies:

**Functional (~2,500 LOC):**
- Strategy-pattern graph framework (`stg/{nodes,edges,temporal,labels,strategies,builders}`) — genuinely built, no stubs. `KalshiTickerNodes`/`KalshiEventNodes`; five edge strategies incl. `KalshiSemanticTopicEdges` (sentence-transformers); four temporal strategies; post-processing (self-loops, pruning, top-K, normalisation). Produces real tensors via `feature_tensor()`/`adjacency_tensor()`.
- `stg/events/` (~1,770 LOC) — per-series analysis for CPI/fed/GDP/payrolls/recession/unemployment, **plus a complete PDF/CDF recovery module** (`implied.py`).
- `stg/pairs/` (~460 LOC) — correlation, OLS hedge ratio, AR(1) half-life, z-score pairs backtest.
- `stg/io/` (~410 LOC) — data loading.

**Missing or disconnected — the four that matter:**

1. **Zero lines of model code.** No AGCRN, no GNN, no regression. The graph framework produces tensors that nothing consumes. A NeuralCDE temporal-interpolation module (286 lines) existed and was deleted in commit `0aa92e9`. `requirements.txt` still carries dead `torch`/`torchcde` deps, and diverges from `pyproject.toml`.
2. **`implied.py` is unwired.** It implements `recover_pdf()` (successive differences over the threshold ladder → discrete PDF), `pdf_implied_stats()` (mean, std, skew, kurtosis, entropy, median), `build_daily_implied_means()`, and `resolved_value()`. It is imported *only* by `events/*.py` analysis scripts — never by `builders`, `nodes`, `labels`, or `pairs`. The actual pipeline instead uses the cruder VWAP collapse in `io/kalshi.py:aggregate_to_event_level` (lines 104–115), which volume-weights raw yes-prices **across different strike thresholds without reference to what each threshold means**. Two competing aggregation schemes coexist; the weaker one feeds the results.
3. **Strike values are discarded at ingest.** `scripts/fetch_kalshi_data.py:normalize_markets` (lines 216–280) drops Kalshi's `cap_strike`/`floor_strike`/`strike_type`. `implied.py` compensates by regex-parsing thresholds out of ticker strings (`-T3.2` suffix) — works, but fragile against any naming change.
4. **No tests at all.** No `pytest`/`unittest` files. `test.ipynb` is a scratch notebook. README references a nonexistent `stg.adapters` module. `events/config.py` defaults to a data path that doesn't match the repo layout.

**Note on §4.3.1's code location.** The report *does* define significance (95% CI excluding zero for ≥3 consecutive days). The `pairs/` module implements a *different* analysis (correlation + half-life screening, no p-values), so the event-study code presumably lives in `events/` or a notebook. The methodological gap is therefore **not** an absent test — it is the absence of **multiple-testing correction** across an exhaustive type-pair search (see §5).

---

## 3. What the Data Actually Contains

Direct inspection, 2026-08-14.

### `data/trades/` — 1.8 GB, 3,890 parquet files
- **39.4M trades, 418,791 tickers, 2021-06-30 → 2025-12-30**
- Schema: `trade_id, ticker, count, yes_price, no_price, taker_side, created_time`
- **`taker_side` is a genuine data advantage.** Trade direction is labelled, not inferred — most equity microstructure work has to estimate this (Lee-Ready etc.). Signed order flow is available for free.

### `data/markets/` — 136 MB, 427 files
- Metadata (titles, `event_ticker` grouping, open/close times, `result`) is historical and useful.
- **`yes_bid`/`yes_ask`/`no_bid`/`no_ask`/`volume`/`open_interest` are a single point-in-time crawl** — all rows fetched within ~50 seconds on 2025-11-23. Not a time series. There is **no historical quote data** for 2021–2025.

### Composition — three facts that change the plan

**(a) The archive is overwhelmingly non-macro.** Top series: `NCAAFGAME` (7.8M trades), `NBAGAME` (5.6M), `MLBGAME` (4.2M), `BTCD` (1.8M). The macro universe is a small slice of a large dataset.

**(b) The archive is overwhelmingly 2025.**

| Year | Trades | Tickers |
|---|---|---|
| 2021 | 86K | 751 |
| 2022 | 709K | 6,929 |
| 2023 | 882K | 10,596 |
| 2024 | 2.1M | 21,984 |
| 2025 | **35.6M** | **380,401** |

~90% of trades are 2025. **This kills the 2022→2025 maturity-trend test** as a headline robustness check — early-period estimates would carry error bars too wide to interpret. Demoted in §8.

**(c) Macro coverage, ranked:**

| Series | Trades | Tickers | Span |
|---|---|---|---|
| FEDDECISION | 445K | 125 | 2023-04 → 2025-12 |
| FED | 126K | 376 | 2021-07 → 2025-12 |
| CPIYOY | 112K | 354 | 2022-12 → 2025-12 |
| CPI | 77K | 383 | 2021-07 → 2025-12 |
| GDP | 59K | 152 | 2021-07 → 2025-11 |
| PAYROLLS | 54K | 162 | **2024-02** → 2025-12 |
| WTI | 34K | 1,530 | 2022-09 → 2025-11 |
| CPICORE | 16K | 299 | 2022-07 → 2025-12 |
| CPICOREYOY | 8.3K | 260 | 2023-02 → 2025-12 |
| PCECORE | 3.5K | 58 | 2022-12 → 2025-12 |

Plus CPI subcomponents (`CPIGAS`, `CPISHELTER`, `CPIUSEDCAR`, `CPIFOOD`, `CPIAPPAREL`) and international variants (`CPIEU`, `GDPCN`, `CPICN`), a few hundred to ~2K trades each.

**PAYROLLS only starts 2024-02** — ~22 monthly releases, thin for a core trigger type.

### Macro contracts behave nothing like sports contracts

Relevant both to positioning against Angelini & De Angelis (§7.1) and to the friction argument. Median trade density from the archive:

| Series | Markets | Median trades/market | Median active days | **Median trades/day** |
|---|---|---|---|---|
| NBAGAME | 778 | 5,303 | 3 | **1,520** |
| MLBGAME | 4,406 | 546 | 2 | **341** |
| NCAAFGAME | 1,644 | 1,814 | 9 | **191** |
| FEDDECISION | 125 | 190 | 31 | **6.3** |
| PAYROLLS | 162 | 90 | 20 | **5.1** |
| GDP | 152 | 108 | 35 | **3.1** |
| CPIYOY | 354 | 25 | 9 | **2.8** |

An NBA contract sees **240–540× the trade density** of a macro contract and lives 3 days rather than a month. The median CPIYOY market trades **25 times in its entire life**.

Structural differences beyond density:

- **Information arrival.** Sports contracts sit in a continuous dense public signal stream; macro contracts receive *one* scheduled discrete signal, then nothing for weeks. Macro price paths are long flat stretches punctuated by jumps.
- **Observability of the correct price.** Game state (score, time) is fully observable, so an out-of-sample benchmark win probability is computable — this is precisely how Angelini & De Angelis measure a 0.64-for-one underreaction. **There is no observable state variable that pins down a CPI contract's correct price.** This is the single deepest difference and it dictates methodology (see §5).
- **Inference depth.** Their signal is *directly* relevant (a scoring run obviously bears on the winner). The signal here is *indirectly* relevant — a CPI print bears on a Fed contract only via an inferential step someone must actually perform.
- **Convergence.** Sports prices converge monotonically to 0/1 as the clock runs; macro prices can sit flat then gap.
- **Schedule.** Macro releases are known to the minute months ahead; game events are stochastic.

**Two consequences.** *(a)* The scheduled, inference-dependent nature of macro releases arguably makes underreaction here **more** interesting: attention is maximally focused at a known instant, so "nobody was watching" is ruled out and what remains is the sharper claim that participants saw the release and failed to work out its second-order implications. *(b)* Measurement is correspondingly harder — at ~3 trades/day with no historical quotes, minute-level midpoint regressions of the kind Angelini & De Angelis run are simply unavailable, and the forward-fill problem (§4.1) is severe rather than marginal. Do not attempt to mirror their design.

### Three under-exploited assets already in hand

1. **Jobless claims exist and are unused.** `JOBLESSCLAIMS` (132 tickers) and `JOBLESS` (74 tickers) appear in the markets metadata, plus `ISMPMI` (49) and `RETAIL` (5). Claims are **weekly** — the single best available fix for the small-N problem, requiring no new data.
2. **CPI subcomponents give ground-truth validation.** `CPISHELTER`/`CPIGAS`/`CPIUSEDCAR` → headline `CPI` have *known mechanical BLS basket weights* (shelter ≈ ⅓). If Stage-1 discovery recovers effect magnitudes consistent with published weights, that validates the method against known truth — stronger than a placebo test, and it partly addresses the identification problem in §5.
3. **Sports/crypto markets as a liquidity-contrast control.** Same venue, same microstructure, same fees, orders of magnitude more liquid. Any measurement artifact should behave differently at 7.8M trades than at 3K. Zero new data, zero scope expansion — used as control, not object of study.

---

## 4. Measurement Threats (new — none of this is in the CA report)

### 4.1 Stale prices and forward-fill may be manufacturing the headline result

**This is the most serious threat, and it is currently baked into the pipeline rather than hypothetical.**

`io/kalshi.py:build_daily` sets `close = yes_price.last()` — the last trade of the day — then **forward-fills across days with no trades** (lines 51–75, `volume` set to 0). A market that doesn't trade for five days gets five identical prices followed by a jump. Averaged across many events with different gap patterns, this produces a **smooth-looking gradual drift curve out of what are really discrete jumps at irregular times** — precisely the §4.3.1 signature.

Nonsynchronous trading is the classic source of spurious lead-lag in equities (Lo–MacKinlay). Worse, it is *confounded with the hypothesis*: the friction story predicts thinner markets show more drift, and stale pricing predicts exactly the same thing. **The key confirmatory test and the main artifact make the same prediction**, so it cannot discriminate.

*Diagnostics:* rerun the event study (a) in **trade time** — horizons defined as next-N-trades rather than next-N-days; (b) **excluding forward-filled rows** (`volume`/`trade_count` already flag them); (c) restricted to target markets that actually traded in the trigger window.

### 4.2 Bid-ask bounce

Each trade is a taker crossing the spread — `taker_side='yes'` lifts the ask, `taker_side='no'` hits the bid. So the daily close sits at the ask or the bid depending on the *direction of the last trade*, injecting spurious negative serial correlation (Roll 1984). If taker-side composition shifts systematically after a resolution — plausible, since directional flow arrives — part of the measured drift is just the close migrating across the spread.

The bounce is directly visible in the raw data. A representative sequence on one liquid market: prints of 19, 19, 21, 20, 19 (all `taker_side='yes'`), then a single `no`-taker at **15**, then straight back to 19. Nothing about beliefs changed; that 4–6¢ drop is one trade hitting the bid instead of lifting the ask. If such a print is the last of the day it becomes the daily `close` and enters the event study as a real move.

**What the trade data does and does not contain.** Trade records satisfy `yes_price + no_price = 100` *exactly, always* — `no_price` is a pure accounting identity for the same execution and carries zero extra information. There is **one price per trade, not a bid and an ask.** Note the practical corollary: in a `taker_side='yes'` print at 19¢, buying *no* at that instant would **not** have cost 81¢ — it would cost `100 − yes_bid`, i.e. ~88¢ at a 7¢ spread. Aggressively buying both legs costs `yes_ask + no_ask = 100 + spread`.

Quote parity holds exactly in the snapshot (all 101 two-sided quotes, zero deviation): `yes_bid = 100 − no_ask`, `yes_ask = 100 − no_bid`, `yes_ask − yes_bid = no_ask − no_bid`, and **`yes_ask + no_ask − 100 = spread`** (the overround).

### 4.2.1 Recovering historical spreads from labelled taker direction — a reusable result

Kalshi retains no historical quotes (§4.3), so the spread must be estimated from trades. `taker_side` makes this unusually clean: **each trade reveals one side of the book** — a yes-taker print *is* the ask, a no-taker print *is* the bid. Pairing temporally adjacent opposite-direction trades therefore yields a direct effective-spread estimate. (Same-timestamp fills are collapsed to the first, since sweeps walk the book and only the first fill sits at the best quote.)

Median effective spread, cents:

| Series | ≤60s gap | ≤600s | ≤3600s | n (60s) |
|---|---|---|---|---|
| FEDDECISION | **1.0** | 1.0 | 1.0 | 45,042 |
| CPI | **2.0** | 2.0 | 2.0 | 1,147 |
| CPIYOY | **2.0** | 2.0 | 2.0 | 4,371 |
| GDP | **2.0** | 2.0 | 2.0 | 733 |
| PAYROLLS | **3.0** | 2.0 | 2.0 | 1,366 |

**The window-invariance is the validity check.** If mid-price drift were contaminating the estimate, widening the pairing window 60× would inflate it. It does not move at all. FEDDECISION at 1¢ sits **at the tick floor** — Kalshi prices in whole cents, so its most liquid macro market is as tight as the exchange permits.

**This supersedes the 7¢ figure from the quote snapshot.** Both are real but measure different things: the snapshot is **time-weighted** (spread at a random instant, including dead time, and drawn from a sample dominated by inactive/settled markets), whereas this is **trade-weighted** (spread conditional on execution). For a strategy that can choose its timing, the trade-weighted figure is the relevant one. Caveats to state: it observes only moments when both sides traded, so quiet periods are invisible and it is a lower bound on the time-weighted spread; and it is an *effective* spread, which diverges from the quoted spread under hidden liquidity or price improvement.

**Worth writing up as a small methods contribution.** Anyone studying pre-2026 Kalshi microstructure hits the same missing-quote wall. A taker-direction-based effective-spread estimator, validated by window-invariance, is a reusable answer — and it is the input that decides Phase 4.

*Other corrections available:* Roll's estimator (effective spread from return autocovariance) as an independent cross-check; VWAP instead of last-trade to partially diversify the bounce.

### 4.3 Historical quotes are unobtainable; a ~3-month window is recoverable

Kalshi's candlestick endpoint **does** return historical quotes:

```
GET /trade-api/v2/markets/candlesticks?market_tickers=...&start_ts=...&end_ts=...&period_interval=1440
```

Each candle carries full OHLC for **both `yes_bid` and `yes_ask`**, plus `price` (with `mean_dollars` = VWAP, `previous_dollars` = last trade), `volume_fp`, and **`open_interest_fp`**. `period_interval` accepts 1 (minute), 60 (hour), 1440 (day); up to 100 tickers and 10,000 candles per request.

**Critically, quotes update on no-trade periods.** On one market checked, 53 of 78 daily candles had *no trade price at all* — but all 78 had live bid/ask that moved (e.g. bid 0.11→0.10, ask 0.14→0.16 on a zero-volume day). This is the direct fix for §4.1: the mid stays live where last-trade goes stale.

**But retention is a rolling ~3 months.** Enumerated 2026-08-14:

| Series | Retrievable event periods |
|---|---|
| KXCPIYOY | 26MAY … 26NOV |
| KXPAYROLLS | 26JUN … 26NOV |
| KXFEDDECISION | 26JUN … 28JAN |

Older markets are delisted entirely — `KXFEDDECISION-25MAY-C25` and `-24DEC-C25` return empty; pre-KX tickers like `FED-23JUN-T5.25` return `not_found` on the market endpoint itself.

**Implication:** 2021–2025 quotes are gone permanently; the trade archive is irreplaceable. But **one harvest now captures ~3 CPI prints, 2 FOMC decisions, 3 payrolls, ~12 jobless claims with true mids** — enough for an honest robustness subsample. This is the only time-sensitive item in the entire plan. Forward collection adds little before submission; a weekly opportunistic run suffices (candlesticks are historical, so no always-on process is needed — unlike order-book depth, which is why depth is dropped).

Also worth capturing going forward, at near-zero cost: **market metadata snapshots preserving `cap_strike`/`floor_strike`/`strike_type`** (fixes §2.3), since delisting destroys metadata for old markets too.

---

## 5. Identification: "Diffusion" vs. Shared Information Arrival

**The threat.** If a CPI print resolves a CPI contract and also moves a Fed-hike contract, that is not necessarily information propagating *through the prediction market* — both contracts are functions of the same public release, which reached both simultaneously. The planned placebo test (shuffled resolution dates) rules out common *latent* factors but **not mechanically shared inputs**.

This is the difference between "markets are informationally connected" (trivially true, not a contribution) and "participants learn from prices in one market and update another" (interesting, and what the framing claims).

**Mitigations:**
- **Use official release timestamps adversarially.** Phase 0 already plans to pull exact BLS/BEA release times. The interval between the *official release* and the *Kalshi resolution timestamp* is where "reacting to the data" can be separated from "reacting to the market."
- **Decompose into a mechanically-implied update plus a residual.** The residual is where the actual claim lives. CPI subcomponents (§3, asset 2) make this concrete: the mechanical part is a published basket weight.

**Related gap — no normative benchmark.** "Underreaction" is meaningless without a correct-magnitude baseline. Currently drift is documented but never compared against what the price *should* have done, so underreaction cannot be distinguished from overreaction-then-reversal, or from drift toward a *wrong* price. A coherence/Bayesian benchmark (given the trigger's realised outcome, what should the target's implied distribution be?) converts the hypothesis from "prices move predictably" (momentum — well-trodden) into "prices converge to a normatively correct value slowly" (information diffusion — the actual claim).

**Cheap standalone contribution in the same family: internal coherence violations.** Within an event's strike ladder, prices must be monotone in threshold and the implied PDF non-negative. Violations are unambiguous mispricings requiring no model, no benchmark, no return prediction. Measuring their frequency, size, and **persistence** gives a direct, independent thermometer for limits-to-arbitrage on Kalshi — currently the mechanism the whole thesis rests on but only *asserts*. Falls out naturally from the `implied.py` wiring. (Distinct from Gebele & Matthes 2026, which studies *cross-platform* LOP violations — see §7.)

---

## 6. Reframed Hypothesis and Feature Set

### 6.1 Two framing corrections

**"Superior to other markets" is out of scope.** Nothing in the design compares Kalshi's diffusion against Fed funds futures or TIPS breakevens around the same releases. The actual claim is narrower: *is diffusion structured within Kalshi at all*. (If revisited later, FRED has daily Treasury yields and the 10-year breakeven `T10YIE` free and permanently available — no urgency.)

**"Systematic ⇒ tradeable" is not automatic.** The standard objection is "wouldn't it already be priced in?" Semi-strong efficiency requires the *reaction* to be quick and complete, not that price never predictably moves. The report's own evidence — gradual multi-day drift rather than an instant day-0 jump — is itself evidence against full instantaneous pricing-in, analogous to post-earnings-announcement drift, which persists via limits to arbitrage.

**Why full pricing-in may specifically fail on Kalshi:**
- No structural arbitrageur enforcing cross-market consistency (unlike market makers enforcing put-call parity in listed options).
- Retail-heavy, limited-attention participants — inferring second-order implications takes effort nobody is paid to expend instantly.
- Thin books: price discovery throttled by trade arrival rate, not information arrival rate.
- Capital/participation constraints (position limits, KYC) shrink the corrective pool.

**Tightened hypothesis:**
> Cross-market belief updates following event resolutions on Kalshi are systematic rather than random — the relationships governing which markets influence which, and with what lag and magnitude, can be modelled — and the market underreacts to this information rather than pricing it in instantly, leaving a residual drift predictable at the individual-event level.

*(Note the deliberate removal of "with enough signal to be traded profitably net of costs" from the hypothesis statement — see §6.4.)*

### 6.2 Surprise factor

- Gap between realised outcome and the market's own pre-resolution implied probability (surprise ≈ outcome − pre-resolution price), or information-theoretic surprisal (−log p) if grounding in Shannon information.
- **Falsifiable prediction this enables:** drift magnitude should scale with surprise magnitude, and near-zero-surprise resolutions should produce near-zero drift. Sharper than "is average drift across all X→Y pairs nonzero," and it operationalises the report's "informational shock" language.
- Split by **sign** — behavioural finance predicts asymmetric underreaction to good vs. bad news.
- Cross-check against FRED actuals (realised-value-relative-to-trend) as a measure independent of Kalshi's own price.

### 6.3 Friction / attention proxies (target-market side)

- **Liquidity at trigger time**: volume, spread, **open interest** (available per-period from candlesticks; absent from the historical archive), trade count. Thinner markets should show larger/slower drift.
- **Drift decay / half-life as the label**, replacing the binary "CI nonzero for ≥3 days" test. This is what determines viable holding period, and it turns the target into something a regression can forecast and a trading rule can consume.
- **Pre-trigger leakage control** — verify the target wasn't already drifting *before* the trigger resolved (informed positioning ahead of release), to avoid misattributing pre-existing drift.
- ~~Calendar-time maturity trend~~ — demoted; see §3(b).

### 6.4 On profitability as the goal — recommended reframe

The stated ultimate goal is a profitable deployable strategy. Whether that is arithmetically reachable turns entirely on the spread — and per §4.2.1 the execution-relevant figure is **1–2¢, not 7¢**. If §4.3.1 drift magnitudes are in the 3–5¢ range, that is a genuinely tradeable margin rather than an impossibility. **Treat tradeability as an open empirical question, not a foregone conclusion in either direction.**

A structural tension nonetheless remains: the hypothesis says drift persists *because* books are thin and no arbitrageur enforces consistency — and thin books are what widen spreads. **The friction that creates the signal is related to the friction that eats it.** That is the real economics, and it is why the anomaly can survive at all.

So making profitability the *criterion* is still the wrong call. A plausible outcome — signal real, statistically robust, margin too thin after costs — would read as failure when it is not. **Reframe the trading evaluation as a test of economic significance:** how large is the signal relative to the frictions that sustain it? Both outcomes become results, and a negative one *quantifies the limits to arbitrage that explain the drift's persistence*, closing the §6.1 loop more elegantly than a positive backtest would. Angelini & De Angelis (2026) reach exactly this conclusion on Kalshi (§7.1).

**Execution design follows from the spread analysis: be a maker, not a taker.** The signal predicts drift over *days*, so immediacy is not required — positions can be established with resting orders rather than by crossing. That flips the spread from a ~2¢ cost to a potential ~1¢ credit, at the cost of fill risk. Given how decisive the spread is here, this is the single most important execution assumption in Phase 4, and it should be modelled explicitly rather than assumed either way. See "Optimal Market Making in Prediction Markets" (§7.5) if this is pursued.

Practically: with a November submission, "deployed" is not realistic. Spread-aware backtesting with realistic fills is the honest scope, and §5.4 of the report currently does not mention execution assumptions at all.

### 6.5 The horizon / term-structure dimension — currently unmodelled

Many macro series list **multiple concurrent expiries** (successive FOMC meetings, "rate cut by 2027 / 2028"), forming a term structure of expectations analogous to the fed funds futures strip.

**Where the pipeline stands.** The expiry dimension survives ingest: `event_ticker` is `KXFEDDECISION-26SEP`, so **each meeting is its own event node**, with the strike (`H0`, `H25`, `C26`) as the submarket beneath it. `days_to_close` is carried as node feature 5. But §4.3.1 operates at the **event-type** level — so estimating "the CPI→FedDecision effect" **pools every FOMC expiry regardless of horizon**, averaging the three-weeks-out meeting together with the eighteen-months-out one. Economically these should respond very differently to the same surprise. The term dimension is present in the data representation and absent from the analysis.

**The historical data barely supports studying it.** Measured across FEDDECISION expiries:

- Over **951 days** with FEDDECISION trading, the **median number of concurrently-trading expiries is 1**
- Only **125 days (13%)** had ≥2 expiries active; **38 days** had ≥3; maximum 5
- For a fixed strike (`H0`), only **3 expiry pairs** have ≥15 overlapping days

Through 2021–2025, flow concentrated almost entirely on the front meeting — there is essentially no curve to study. (The `corr_max = 0.95` filter in `pairs/config.py` was *not* the binding constraint; observed same-series correlations are 0.81 / 0.38 / −0.38, well inside the band. Raw overlap is the constraint.)

**This has changed since the archive was collected.** The live API currently lists KXFEDDECISION expiries from 26JUN through **28JAN — fourteen concurrent meeting periods.** Kalshi has built out a genuine rate-expectations curve in 2026, and the Phase 0 candlestick harvest would capture it.

**Two actions:**
1. **Now, cheap:** condition the drift estimate on the target's **horizon** (`days_to_close`, already computed) rather than pooling. This yields a *term structure of the diffusion effect* — how far along the curve a surprise propagates — and needs no concurrency and no new data, only trigger-target pairs at varying horizons, which already exist. Folded into Phase 1.
2. **Future work, stated as a limitation:** the term structure is the strongest natural extension. The rate-expectations curve is *the* canonical macro object; cross-meeting coherence supplies hard consistency constraints that cross-indicator relationships lack (cumulative hike probabilities must be mutually consistent — the same logic as the strike-ladder check in §5); and it is the cleanest available multi-hop setting, i.e. exactly the capability that motivated graph learning. "Kalshi only built out the term structure in 2026" is a far better-stated limitation than silence, and a descriptive figure of the current 14-expiry curve from the harvest would motivate it well.

---

## 7. Literature Landscape (scanned 2026-08-14)

The Kalshi literature has grown very fast in 2026. Four papers materially affect positioning.

### 7.1 The one that most demands differentiation

**Angelini & De Angelis (2026), "When Do Markets Fully Process Public Information? Evidence from Real-Time Prediction Markets."** arXiv:2606.07811, **posted 2026-08-11 — three days before this scan.**

Kalshi NBA winner contracts: 1,438 games, 2,876 contracts, 409,512 contract-minute observations (Apr 2025–May 2026). Builds an out-of-sample logit benchmark for win probability from game state, then measures price response to real-time signals. Finds **a one-minute change in benchmark probability produces only ~0.64-for-one contemporaneous price change** — i.e. underreaction — and that the resulting gaps **predict further drift**. Uses **bid-ask midpoints**. Concludes the predictable drift is **"largely absorbed by bid–ask spread"**, with executable returns negative even after large underreaction gaps, best interpreted as *gradual price discovery under trading frictions*. **No graph or network model.**

This is the same skeleton as the thesis — underreaction on Kalshi, drift predictable, tradeability constrained by spreads. **The contribution still survives, on three clear axes:**
- **Sports, not macro.** No economic transmission channel; no theory about *which* contracts should influence which. The two contract classes are also structurally different in ways that matter (see §3, "Macro contracts behave nothing like sports contracts"): 240–540× the trade density, continuous vs. discrete-scheduled information arrival, and — decisively — an observable state variable that makes a benchmark probability computable for basketball and impossible for CPI.
- **Within-contract reaction to an external signal**, not **cross-market propagation from one contract's resolution to a different contract.** The §3.2 gap claim ("whether cross-market belief updates are structured") is untouched. Their signal is directly relevant to the contract; the signal here requires an inferential step, which is a stronger prior for underreaction and a cleaner behavioural story.
- **No graph structure.** The multi-hop / joint-diffusion question is not addressed.

**Action:** cite prominently, adopt as the closest antecedent, and state the differentiation explicitly and early. Their use of midpoints is a direct precedent for the §4.2/§4.3 measurement concerns — but note their method is *unavailable* here: minute-level midpoint regressions require quote data and trade density this project does not have.

**Note on their tradeability result.** They find drift "largely absorbed by bid–ask spread" on NBA contracts. Given §4.2.1 measures macro spreads at 1–2¢ (FEDDECISION at the 1¢ tick floor), their conclusion should **not** be assumed to transfer — the spread environment differs, and this is an empirical question for Phase 2, not a settled one.

### 7.2 Already cited — verify the gap claim survives

**Diercks, Katz & Wright (2026), "Kalshi and the Rise of Macro Markets."** NBER WP 34702 / Fed FEDS 2026-016, Feb 2026. Evaluates Kalshi macro forecast accuracy (CPI, payrolls, unemployment, GDP, Fed decisions, 2022→) against Bloomberg/Blue Chip surveys and Fed funds futures. Kalshi matches or beats professional forecasters; continuously-updating full distributions rather than six-weekly modal snapshots.

Establishes that Kalshi macro prices are *accurate* — not that cross-market updates are *structured*. The gap holds. **Note: they publish a public data/code repo (`github.com/jdkatz21/Prediction_Markets_Public`)** — worth checking for release-aligned data that could save Phase 0 effort.

### 7.3 Methodological precedent for the PDF recovery

**Angelini (2026), "The Shape of Macroeconomic Beliefs."** arXiv:2606.30040, Jun 2026 (verified from source PDF).

Constructs a panel of **Kalshi-implied distributions for CPI and core CPI by converting adjacent threshold contracts into probability mass** — exactly the §5.1 / `implied.py` approach — recovering implied means, uncertainty, and upper-tail probabilities from 30 days to 1 hour before each release. Finds the *distributional* signal dominates the mean: large lagged Reuters Poll surprises raise implied uncertainty, and positive lagged surprises raise probability on high-inflation outcomes (a 0.1pp positive lagged surprise raises P(monthly inflation > 0.3%) by ~4.7pp). Kalshi upper-tail probabilities predict realised high-inflation states.

**This helps rather than threatens.** It is published precedent that threshold-ladder PDF recovery on Kalshi is a legitimate, publishable method — direct support for pivoting off the ATM-ticker proxy — and its surprise→distribution-shape result independently motivates the §6.2 surprise factor. It is *within-indicator* (does inflation news predict inflation beliefs), **not cross-indicator propagation**. Cite as methodological grounding.

### 7.4 Microstructure and limits to arbitrage

**Whelan et al. (2026), "Makers or Takers: The Economics of the Kalshi Prediction Market."** GWU WP 2026-001 / karlwhelan.com. Transaction-level data on 300,000+ contracts — first systematic evidence on Kalshi pricing. Prices informative and improving toward close, but a clear **favourite–longshot bias**: low-price contracts win far less often than needed to break even after fees; high-price contracts yield small positive returns. Directly relevant to the §6.4 tradeability argument and to any position-sizing rule.

**Gebele & Matthes (2026), "Semantic Non-Fungibility and Violations of the Law of One Price in Prediction Markets."** arXiv:2601.01706, Jan 2026. Ten venues, 2018–2025; ~6% of events list concurrently across platforms, with persistent execution-aware deviations of 2–4%. Fragmentation is structural (heterogeneous resolution semantics, institutional segmentation, limits to cross-platform arbitrage).

**Cross-platform**, so the §5 within-venue coherence-violation idea remains distinct — and this paper is a strong citation for why such violations persist.

### 7.5 Peripheral but worth knowing

- **Le (2026)**, arXiv:2602.19520 — calibration across 292M trades, 327K binary contracts.
- **"Unlocking the Forecasting Economy"**, arXiv:2604.20421 — prediction-market dataset suite; possible auxiliary data.
- **Dalen (2025)**, arXiv:2510.15205 — options-style pricing for prediction markets; "event vega."
- **"Optimal Market Making in Prediction Markets"**, arXiv:2607.17991 — relevant if a maker-side execution strategy is considered (§6.4 implies posting rather than crossing).
- **"Do Prediction Markets Forecast Cryptocurrency Volatility? Evidence from Kalshi Macro Contracts"**, arXiv:2604.01431.
- On the GNN side: no published work applies spatial-temporal GNNs to prediction-market contracts. The STG-in-finance literature is equities/volatility (MST-GNN, STGAT, Temporal GAT). **The novelty of the graph framing holds** — though §8 argues it should be scoped empirically rather than assumed.

---

## 8. The Pivot: Structure/Learning Decomposition, and Its Justification

### 8.1 The pivot

Replace AGCRN-as-primary-model with a **two-stage decomposition**, keeping graph *structure* while making graph *learning* an empirical question:

- **Stage 1 — Structure (the discovery deliverable).** Estimate the diffusion graph via transparent statistics: per type-pair surprise-conditioned effect sizes and decay/half-life curves, with multiple-testing correction. Every edge is a hypothesis test with a CI — interpretable and independently validatable, unlike an adjacency fit purely to minimise downstream loss.
- **Stage 2 — Consumption (the prediction model).** Given that structure, empirically determine how much capacity is needed to *use* it, via an ablation ladder from regression up to constrained graph models. This stage no longer has to *discover* structure from limited data — only learn how to weight structure already estimated. A far better-conditioned problem.

### 8.2 Justification for the supervisor

Frame as **resolving a contradiction already present in the report**, not as retreat.

**(a) The report's own evidence undercuts its stated premise.** §2 and §3.3 justify AGCRN on the grounds that inter-market relationships are "latent" and that end-to-end adjacency learning "removes the need for any pre-defined structural assumptions." But §4.3.1 then recovers those relationships with elementary event-study statistics and finds them **economically interpretable** — clustering into recognisable transmission channels. The structure is neither latent nor unrecoverable. Where structure is recoverable and interpretable, learning it from prediction loss is strictly worse: less interpretable, more prone to fitting shared latent factors rather than genuine transmission (§5), and **unvalidatable** — a loss-optimised adjacency has no CI and no independent test.

**(b) Capacity/data mismatch.** AGCRN carries a dense N×N adjacency plus a weight pool Ŵ ∈ ℝ^(d×c×c′). Against ~33 nodes/snapshot and a few hundred snapshots, parameters exceed observations substantially. Separately, **node-adaptive parameter learning exists to let *persistent* nodes specialise** — but in the inductive DTDG setting node identities do not persist across snapshots, weakening that mechanism's motivation independently of sample size.

**(c) It elevates a promised experiment rather than dropping a deliverable.** §5.3 already commits to "a controlled ablation study... isolating the contribution of graph structure and adaptive adjacency learning over simpler temporal baselines." The pivot **makes that ablation the central methodological result**: *does adaptive graph learning earn its complexity in a low-data, non-stationary market setting?* A rigorous negative answer is a genuine finding.

**(d) The claimed contribution is untouched.** §3.2 states the gap as "no existing work has characterised whether these cross-market belief updates are **structured**." Characterising structure *is* Stage 1. AGCRN was the consumption layer, never the source of the novelty claim.

**(e) External support.** Angelini & De Angelis (2026) establish underreaction and spread-absorbed drift on Kalshi with **no graph model at all** (§7.1) — evidence that the phenomenon is tractable without heavy machinery, and that the marginal contribution lies in the cross-market structure, not model complexity.

### 8.3 Economic intuition as inductive bias — making it rigorous

The instinct to ground the ML in economics is right; the principled framing is that **economic priors are structural regularisation**, substituting domain constraints for parameters the data cannot support. Four concrete mechanisms:

1. **Constrain the graph to economically-motivated channels** rather than a dense learned adjacency — a hard sparsity prior, the single largest available reduction in overfitting risk.
2. **Sign restrictions.** Theory predicts direction for many channels (higher WTI → higher CPI). Testing whether estimated coefficients respect predicted signs is falsification that validation loss cannot provide — a model can achieve good MAE with economically nonsensical coefficients and never reveal it.
3. **Theory-derived features, not learned representations.** The surprise factor comes from Kuttner (2001) / Gürkaynak et al. (2005), already cited in §3.2.
4. **CPI subcomponents as ground-truth calibration.** Known BLS basket weights (§3, asset 2) let Stage-1 estimates be checked against published truth.

This also directly serves identification (§5): economic structure is what separates the mechanically-implied update from the residual where the actual claim lives.

---

## 9. Revised Research Plan (triaged for November 2026)

Replaces the CA report's §5. Ordered by priority, with explicit cuts.

### Phase 0 — Time-sensitive data capture (this week, ~2 hours)
Only one item in this entire plan is time-sensitive:
- **Harvest the ~3-month candlestick window now** (daily + hourly; minute-level only for liquid macro tickers in a ±3–5 day window around release dates). Rolling retention means this is actively being lost. Yields true mids, spreads, and open interest for ~3 CPI, 2 FOMC, 3 payrolls, ~12 claims.
- **Preserve `cap_strike`/`floor_strike`/`strike_type` in `normalize_markets`** going forward (fixes §2.3).
- Weekly opportunistic re-run. No cron/CI needed — candlesticks are historical, so gaps self-heal.
- **Not time-sensitive, defer:** FRED actuals, BLS/BEA release calendars, FOMC dates. Permanently available, fetchable in an afternoon. **Dropped:** Google Trends, order-book depth (needs an always-on process for a few months of data that could not support any conclusion).

### Phase 1 — Must do (fixes defects in existing claims)
1. **Multiple-testing correction (FDR/Bonferroni)** on the §4.3.1 pair search, and report *how many* type-pairs were tested to find 22. An exhaustive 95%-CI search manufactures false positives mechanically; this is the most likely examiner question.
2. **Trade-time robustness check** (§4.1) — rerun the event study with trade-based horizons and excluding forward-filled rows. **Do this first among the analyses:** if drift does not survive, the thesis needs to change in August, not October.
3. **Wire `implied.py` into the pipeline** — replace the VWAP collapse in `io/kalshi.py:aggregate_to_event_level` with `build_daily_implied_means`; update `pairs/kalshi_inputs.py` and `nodes/kalshi.py` to consume `implied_mean/std/skew/kurtosis/entropy`. Unlocks the surprise measure; `implied_std`/`implied_entropy` double as uncertainty features. Methodological precedent: Angelini (2026), §7.3.
4. **Surprise factor** as a first-class feature, split by sign (§6.2).
5. **Horizon-condition the drift estimates** (§6.5) — replace pooled type-pair effects with effects estimated as a function of target `days_to_close`. No new data; turns one coefficient into a term structure of diffusion.

### Phase 2 — Should do
5. **Ablation ladder rungs 1–2** — (1) single-pair regression on Stage-1 structure (surprise × edge weight → target drift); (2) multi-trigger regression aggregating surprise-weighted signals over active Stage-1 neighbours. This produces the first actual predictive result in the project. Evaluate on chronological held-out splits with MAE/RMSE **and** directional hit-rate.
6. **Placebo test** (shuffled resolution dates) — separates correlation from transmission.
7. **Spread-aware backtest** using the §4.2.1 per-series effective spreads — model **maker vs. taker execution explicitly** (§6.4), since that choice flips the spread from cost to credit and likely decides the result. Plus a "no cross-market signal" baseline (per-node univariate) to isolate whether cross-market information adds value over own-autocorrelation.
8. **Add jobless claims** to the event universe (§3, asset 1) — weekly cadence, data already local, best available fix for small-N.
9. **Write up the effective-spread estimator** (§4.2.1) as a short methods subsection — self-contained, reusable, and the input on which Phase 4 turns.

### Phase 3 — If time permits
10. **Internal coherence violations** (§5) — frequency, size, persistence. Cheap once `implied.py` is wired; strong independent evidence for the limits-to-arbitrage mechanism.
11. **CPI-subcomponent ground-truth validation** against BLS basket weights (§3, asset 2).
12. **Quote-based robustness subsample** using the Phase 0 harvest — "drift measured on true mids is X vs. Y on last-trade prices."
13. **Roll's estimator** as an independent cross-check on §4.2.1.

### Cut
- **Ablation rungs 3–4 (GCN, AGCRN).** Not trainable to a competitive standard on ~400 events in three months; a half-trained model is worse than none. Present the ladder's stopping point as the empirical answer to "does graph learning earn its complexity here."
- **Rolling-window maturity trend.** Doubly weak: 90% of trades are 2025 (§3b).
- Google Trends, order-book depth, the comparison-market arm.

### Phase 4 — Write-up
- **Stage 1 (diffusion structure + decay curves) as the primary standalone contribution**, independent of trading results.
- The ablation ladder as **direct evidence on model-capacity requirements**, framed as a finding rather than an unfinished deliverable.
- Trading evaluation framed as **economic significance** (§6.4), where a negative result quantifies the limits to arbitrage sustaining the drift.
- Position explicitly against Angelini & De Angelis (2026) (§7.1) early and clearly, using the sports-vs-macro structural contrast (§3) as the differentiation.
- **State the term-structure limitation explicitly** (§6.5) and name it as the primary extension, ideally with a descriptive figure of the current 14-expiry FEDDECISION curve from the Phase 0 harvest.
- Include the effective-spread estimator (§4.2.1) as a methods subsection.

---

## 10. Open Items Requiring the User's Decision

1. **Supervisor sign-off on the AGCRN pivot.** The CA report promises an inductive AGCRN adaptation as the methodological contribution. §8.2 gives the justification, but if the supervisor expects that specific deliverable, renegotiate now rather than in November.
2. **Whether to reframe profitability from goal to test** (§6.4). This changes what counts as success and should be agreed before results arrive, not after.
3. **Read Diercks et al. (2026) and Angelini & De Angelis (2026) in full** before finalising the gap statement. Both are close enough that the differentiation must be precise.
