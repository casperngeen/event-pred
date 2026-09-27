# STG models in finance: what the literature does, and where this thesis differs

*2026-09-26. Reading notes on five papers, written for the literature review
and the "why doesn't your STG match published gains" question. PDFs in
`papers/`.*

## The pattern across the literature

Published STG-in-finance gains are large, but they are measured in ways this
project's evaluation rules out:
- levels instead of changes;
- labels that are partly known at prediction time (paper 3's label gives a
  zero-parameter rule 77–81% on the real series);
- random-day splits;
- graphs and scalers fitted on the full sample;
- no costs and no null.

Where a paper does report an honest out-of-time directional number, it is at
chance, which matches this project's null. The papers also routinely read the
learned adjacency as economic structure. `recovery_test.md` shows that reading
fails under a known truth.

## 1. Luo, Ma & Cucuringu (2024), *Spatial-temporal stock movement prediction … Semantic Company Relationship Graph* — `papers/STG_stocks.pdf`

- **What it does.** Builds a company graph from news co-occurrence
  (1.2M Factiva articles, GloVe cosine similarity, threshold 0.45). The model
  is a GCN feeding an LSTM with per-stock weights (NIST-GNN). The target is
  next-day up/down for 411 S&P 500 stocks. Portfolios trade only stocks whose
  past accuracy clears 0.51–0.54.
- **Result.** 5.79 bps/day vs 5.40 for the S&P, Sharpe 0.91, p ≈ 0.05.
  No transaction costs. The universe is the 2021 constituents (survivorship).
- **Relevance: high, as a parallel.**
  - Its key comparison is a prior graph against a correlation graph with the
    same model (5.79 vs 4.17 bps/day), which is the same shape as this
    project's imposed-vs-learned result.
  - The accuracy-threshold rule is the honest form of walk-forward channel
    selection.
  - Its economic claim (news reaches linked firms with a lag, per
    Cohen–Frazzini) is the equity analogue of the diffusion question here.
- **Does not carry over.**
  - Per-node LSTM weights: its 1.66M rows against about 900 labelled cells here.
  - The symmetric GCN normalisation: it cannot take the signed, directed edges
    used here.

## 2. Feng, Jiang, Liang & Xia (2025), *STGAT* (Appl. Sci.) — `papers/STG_stock2.pdf`

- **What it does.** The graph is a Pearson price correlation, passed through a
  sigmoid, which discards the sign. The model is graph attention plus a
  temporal convolutional network on STL trend/seasonal parts; the residual goes
  to an MLP as "noise". The target is the next price *level* for 304 CSI 500
  and 462 S&P 500 stocks.
- **Result.**
  - R² of 0.95–0.99 on price levels, with no persistence baseline.
  - On the out-of-time split, directional accuracy is 49–54%. On US stocks
    STGAT gets 49.33%, below GCN and MLP.
  - A random-day split raises that to about 62%, but its 20-day windows with a
    5-day step share up to 15 of 20 days between train and test.
  - Sharpe 4.15 vs 3.93 for the S&P over 8 months, with no costs.
- **Relevance: medium, as a contrast.**
  - Its honest-split number agrees with this project's null.
  - It treats "sudden market shocks, unexpected news" as the unpredictable
    residual. That residual is the object of study here, so it makes a useful
    framing line.

## 3. Foroutan & Lahmiri (2024), *Deep learning-based STGNNs for price movement classification in crude oil and precious metal markets* (MLWA 16) — `papers/STG_commods.pdf`

- **What it does.**
  - 25 daily nodes: WTI, Brent, gas, gold, silver, 9 equity indices,
    5 FX rates, Fed funds, US unemployment, US CPI, oil production and
    consumption, and a mining index. 28 technical indicators are added as
    extra nodes in a second run.
  - Three models: MTGNN with a learned directed adjacency, SGA-TCN and ASTGCN,
    the last two on a KNN graph with k = 18 of 25 (nearly complete).
  - Train/validation/test is 2001–16 / 2016–19 / 2019–22.
- **Result.** 74–85% accuracy against 73–79% for TCN.
- **The label gives most of that accuracy away.** The label is
  Y_t = 1[X_{t+1} ≥ EMA_t] with a 9-day EMA, and whether today's price sits
  above its own EMA is known at time t.
  - On a **pure random walk**, the zero-parameter rule "predict up if
    X_t ≥ EMA_t" scores **79.5%** on this label, at any daily volatility.
    Analytically it is ½ + arcsin(0.8)/π: today's gap to a 9-day EMA has
    variance 1.78σ², against σ² for the next day's move. So accuracy on this
    label does not measure forecasting skill.
  - On the **real series over their test window** (2019-01-18 to 2022-12-28,
    EMA started 2001-07-12), the same rule scores:

    | Market | Rule | Paper best | Paper TCN |
    |---|---|---|---|
    | WTI | 0.808 | 0.849 / 0.852 | 0.786 |
    | Brent | 0.796 | 0.818 / 0.817 | 0.772 |
    | Gold | 0.771 | 0.792 / 0.792 | 0.732 |
    | Silver | 0.769 | 0.792 / 0.784 | 0.745 |

    Sources: WTI and Brent are FRED spot. Gold and silver are COMEX front-month
    futures, not the paper's Kitco spot. The rule scores within 0.005 of these
    figures on the dates all four series share.
  - **So their best graph models sit about 2–4 points above a zero-parameter
    rule, not the 5–7 points above TCN that the paper reports.** Their TCN
    baseline is 2–4 points *below* the rule in every market, so their
    "graph beats non-graph" margin is mostly a weak baseline.
    - With about 975 test days, the standard error of one accuracy is about
      1.3 points, so a 2-point edge is roughly 1.5 SE, before accounting for
      hyperparameters tuned on the validation set.
    - Their exact day grid is unknown. They drop any date where any of 25
      variables is missing, including Tadawul, which trades Sunday–Thursday.
  - The paper reports no such baseline. Adding technical indicators (EMA5, TEMA,
    ROC) "helps" because they encode X_t − EMA_t directly.
- **Other issues.**
  - Monthly CPI and unemployment are **linearly interpolated to daily**. That
    puts next month's print into this month's days, before it is released: a
    look-ahead in exactly the macro series this thesis times to the second.
  - Min-max scaling over an unstated range.
  - The hyperparameters, including k and the window, were tuned by TPE on the
    validation set.
- **Relevance: high, as the closest domain and the clearest contrast.**
  1. It puts macro indicators in an STG as *continuous interpolated series*.
     This thesis treats them as *timestamped release events with a measured
     surprise*. Its interpolation look-ahead is the concrete reason the
     event-time design matters.
  2. It reads the learned adjacency as economics (its Table 7, e.g.
     "NASDAQ → WTI as a safe haven", Tadawul and USDQAR → silver) with no
     validation. `recovery_test.md` shows AGCRN's Ã ranks true edges *below
     chance* under a known graph, including when there is no signal.
  3. MTGNN's graph learner, ReLU(tanh(α(M₁M₂ᵀ − M₂M₁ᵀ))) with top-k, is
     **directed and one-way by construction**: if A_ij > 0 then A_ji = 0. That
     suits one-way economic channels (labour→policy) better than AGCRN's
     symmetric EEᵀ. It is a candidate learner for the recovery test, though the
     dilution result predicts it would fail at this sample size for the same
     reason the signed AGCRN did.
  4. The label failure is the same mechanism as the retracted direction-study
     rung, which was reading the target's bounded price level. It is worth
     citing together with that lesson.

## 4. Moges & Moodley (2025), *Lite-STGNN* — arXiv 2512.17453

- **What it does.** A DLinear forecast plus a gated graph correction.
  The correction is (A − I)·Ŷ_base, with A = TopK(ReLU(E_src E_dstᵀ)) at
  rank 16, and the gate starts at σ(−4) ≈ 2%. Tested on long-horizon
  benchmarks (Electricity, Traffic, Weather, Exchange).
- **Relevance: design reference.**
  - The zero-start gate equals `zero_head`, and the directed low-rank adjacency
    is the right shape.
  - Its graph term mixes neighbours' *forecasts*, not their *surprises*. Adapted
    to mix surprises, it becomes the rank-4 low-rank graph in
    `recovery_test.md`, the best STG-shaped learner there.

## 5. Huai, Zhang, Yang & Tao (2023), *Spatial-temporal Knowledge Graph Network for Event Prediction* (STKGN) — `papers/STKGN.pdf`

*SSRN preprint submitted to Neurocomputing, not peer reviewed.*

- **What it does.** Predicts which political event types occur in a country
  the next day (multi-label), from ICEWS and GDELT for 2000–2010.
  - Three countries per dataset (Iraq, Afghanistan or Turkey, Iran), with
    80–86k events and about 200–245 event types.
  - The graph has a fixed, symmetric "trans-regional influence" edge between
    every pair of countries. It is imposed, not learned.
  - The model is TGN-style continuous time: each entity has a memory that is
    updated *only when an event touches it*. Event text is embedded with a
    text CNN. A 2-layer relational GCN then broadcasts the update to
    neighbours ("the 9/11 events change USA, which indirectly affects
    Afghanistan").
- **The results don't support the claims.**
  - The STKGN row is **identical** in the ICEWS and GDELT tables
    (83.06 / 36.13 / 84.74 / 38.12 / 82.04 / 37.32), across different
    datasets and different countries.
  - Against that row, STKGN **loses to the best baseline in 5 of 6 ICEWS
    columns** (e.g. Iraq recall 83.06 vs EvoKG 85.13). The text says it
    "consistently achieves the best performance", and the ▲% row reports
    positive gains.
  - Their own layer study gives ICEWS Iraq at the same configuration
    (L = 2) as 88.35 / 40.02, not 83.06 / 36.13. Their ablation's
    "STKGN-mean" variant (85.83) beats the full model.
  - Metrics are recall and Hit@3, with no precision, so predicting more types
    raises recall. There is no persistence or frequency baseline, which is
    known to be strong in temporal-KG forecasting (Gastinger et al., IJCAI
    2024, "History repeats itself").
  - Interpretability rests on one case: removing the Iraq–Iran edge drops
    P(Iran "provide aid") from 0.61 to 0.07.
- **Relevance: high, as a design reference.** Its framing is the one this
  thesis needs, even though its evidence is weak.
  1. **"The cause of an event is somewhere else"** is its phrase for
     cross-regional influence. It is this project's cross-market influence:
     B reprices because A released.
  2. **Continuous-time dynamic graphs (CTDG: TGN, Jodie, TGAT) versus
     discrete snapshots (DTDG: AGCRN, T-GCN, Glean).** This is the ML
     vocabulary for the move this project made, from the daily node panel to
     the release clock (`agcrn_checklist.md` §3c): node state changes only
     when a release touches it. Cite it that way.
  3. **Its architecture is a memory update on the event, then a broadcast
     to neighbours.** Stripped down to what this sample size supports, that is
     the recovery test's low-rank graph: the message is the surprise (not a
     text embedding) and the broadcast goes through a directed, signed
     adjacency. A full TGN memory adds a GRU-scale parameter block per node,
     which the recovery test predicts would fail here for the same reason
     AGCRN does. It works for STKGN because it has about 300× the events.
  4. **The graph is imposed and its structure goes untested,** checked only by
     one edge-removal anecdote. The channel-level sign tests, permutation nulls
     and recovery test here are the rigorous version of the same claim.

## How to use these in the thesis

- **Motivation.** Equity and commodity STGs model smooth co-movement and route
  news to a residual they call noise (2, 3), or smear releases into
  interpolated series (3). The event-forecasting literature (5) has the right
  event-driven, continuous-time frame, but works with text events at about 300×
  this sample and does not test its graph. Prediction markets give each shock a timestamp, a
  surprise against the market's own implied distribution, and a target priced
  as a probability.
- **Evaluation.** Every inflating pattern above has a counter in this project's
  pipeline:

  | Pattern in the papers | Counter in this project |
  |---|---|
  | Levels, persistence-inflated R² | R² vs zero on changes |
  | Labels known at prediction time | Label and purge rules |
  | Random-day splits | Walk-forward folds purged on label end |
  | No null | Permutation nulls |
  | No costs | Taker/maker costs |
  | Reading the learned graph as economics | The recovery test |

- **The comparable result.** Paper 1's prior graph beats its correlation
  graph, and paper 2's honest split is at chance. Both agree with the
  imposed > learned finding here and with the R² ≈ 0 null.
