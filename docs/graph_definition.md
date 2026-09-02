# The spatio-temporal graph: node, edge, snapshot, label

*Companion to `research_summary.md` / `update_2026_08.md`. This is the single
authoritative definition; the code in `stg/panel/` and `stg/nodes|edges|labels/`
implements exactly this and nothing else.*

## Why this document exists

The repo previously carried **three incompatible node definitions**
(contract-level `KalshiTickerNodes`, event-super-node `KalshiEventNodes`, and the
series-level construction used in the August analysis scripts). They pulled in
different directions and made "which graph are we even estimating" unanswerable.
The August 2026 pivot (`update_2026_08.md` §2) settles it: the object of study is
the **series-level influence structure**, so that is the only definition the code
now supports. The other two were deleted (recoverable from git history / branch
`main`).

---

## Node

A node is **a macro series' current market-implied belief about its nearest
unresolved event**.

- ~20 nodes at `min_events = 5` (`CPI`, `CPICORE`, `CPIYOY`, `PAYROLLS`, `U3`,
  `GDP`, `WTI`, `FED`, `FEDDECISION`, `JOBLESSCLAIMS`, …). **N is a consequence
  of `min_events` and the alias table**, not a fixed number — see
  `stg/panel/registry.py`.
- The node **persists across time**. The event it points at rolls forward as
  each print resolves; between resolutions the belief is continuously updated.
- A node is present in snapshot *t* only if it has an event with
  `close_time > t` (and, for labelled training, `> t + clearance` so a node's
  own resolution never lands inside its own label window).

**Canonicalisation happens before the `min_events` filter.** `PROLLS` and
`PAYROLLS` are one BLS release under two ticker conventions; `JOBLESS` (old
single-threshold binaries) and `KXJOBLESSCLAIMS` (new ladders) are one weekly
release. Merging first raises events-per-node, which matters for the AGCRN
capacity accounting. `rules_primary` is not in the archive, so the alias table is
hand-curated from titles.

### Node features (11-D, `stg.nodes.kalshi.FEATURE_ORDER`)

| # | feature | meaning |
|---|---|---|
| 0 | `implied_mean` | centre of the implied PDF of the nearest event |
| 1 | `implied_std` | dispersion in the units of the underlying |
| 2 | `implied_entropy` | dispersion across ladder buckets, unit-free (attention proxy) |
| 3 | `implied_skew` | 3rd standardised moment |
| 4 | `implied_kurtosis` | excess kurtosis |
| 5 | `max_stale_days` | worst per-leg staleness in the ladder that day (0 = all fresh) |
| 6 | `d_implied_mean` | change in `implied_mean` over the last 5 days (belief momentum) |
| 7 | `days_to_close` | calendar days to the nearest event's resolution |
| 8 | `recent_volume` | contracts traded over the last 7 days |
| 9 | `net_flow` | signed taker imbalance over the last 7 days |
| 10 | `is_bucket` | 1 if the series prices bucket ladders (WTI), else 0 |

Standardisation is deferred to a downstream `FeatureStrategy` so it can be fit on
the train fold only.

### Rejected node definitions

- **Contract nodes** (one strike). ~13k nodes for ~400 events; within-ladder
  edges are mechanical arithmetic (`P(X>3.1) ≥ P(X>3.2)`), not influence.
- **Event super-nodes** (one per `CPI-24JUN`). Do not persist, so no per-node
  trajectory and no basis for node-adaptive parameters.

---

## Edge

Directed, `A → B` = "a resolution surprise in series A predicts a subsequent
revision in series B's belief". Two edge types:

### `SurpriseInfluenceEdges` — the estimated structure

Loaded from `artifacts/adjacency_is.parquet` (Stage-1 estimator). Weight =
signed `ρ̂`; metadata carries `n`, the BH-FDR verdict and the same-release flag.
FEDDECISION hike/cut collapse to one edge (larger `|ρ̂|`).

### `SameReleaseEdges` — the common-signal channel

Deterministic, bidirectional. Connects series that resolve from the same official
print (BLS CPI → every CPI ladder; BLS Employment Situation → `PAYROLLS` + `U3`;
FOMC → `FED` + `FEDDECISION`). Kept **separate** so identification work
(`update_2026_08.md` §7) can condition on "two contracts reading one number"
rather than propagation.

---

## Snapshot / temporal axis

`MacroResolutionTemporal`, config flag `cadence`:

- **`event`** (default) — one snapshot per date a universe event resolves. The
  mechanism-matched grid: a surprise appears when a print resolves, and that is
  when other beliefs can move. Irregular Δt is handed to the temporal model /
  carried as `days_to_close`.
- **`weekly`** — every Monday, uniform Δt, for a robustness check.

---

## Label

`ImpliedMeanChangeLabels`: for node B at snapshot *t*, the change in B's
`implied_mean` over the next *k* snapshots (default `k = 3`, the "dormant"
horizon) or *k* calendar days. `direction = sign(Δ)` is exposed for
directional-accuracy scoring.

---

## How to read a fitted model

| object | interpretation |
|---|---|
| adjacency **row** for A | which beliefs move when A's print surprises |
| edge **weight** | strength of transmission |
| edge **sign** | direction (hawkish CPI → higher P(hike); dovish for U3). In the direct estimator this is `sign(ρ̂)`; an AGCRN's `softmax(ReLU(EEᵀ))` **cannot represent it** and must carry it in the convolution weights — a known architectural mismatch |
| **term structure** (`stg.structure.estimate_by_horizon`) | how far along the target's curve a surprise propagates |
| **mediation** (`test_mediation`) | large drop in `partial(A,C\|B)` ⇒ genuine multi-hop graph; unchanged ⇒ a list of bilateral edges |
| Stage-1 adjacency **vs** learned Ã | recovery ⇒ adaptive graph learning works in low data; divergence ⇒ finding about an unvalidated method |
