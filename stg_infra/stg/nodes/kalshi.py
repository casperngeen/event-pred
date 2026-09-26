"""Kalshi-specific node strategies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import polars as pl

from stg.core import NodeState
from stg.util import seconds_to_close

# GraphSnapshot.feature_matrix() (and by extension feature_tensor_padded())
# stacks every node's feature vector into one matrix, which requires every
# node in a graph to share the SAME feature width -- but this project
# deliberately mixes two structurally different node types in one combined
# graph: individual market legs (KalshiTickerNodes) and synthetic MECE
# basket hubs (KalshiMeceBasketNodes), which naturally have different,
# semantically unrelated feature sets. Rather than a full heterogeneous-
# graph redesign, both node types pad to this shared width and append an
# explicit type indicator as the LAST slot -- node.metadata["node_type"]
# already distinguishes them for our own bookkeeping, but metadata never
# reaches the actual tensors a model trains on, only .features does, so
# the indicator has to live there too. Slots beyond a node type's own raw
# feature count are zero-padding; the model is expected to learn
# different interpretations of the shared slots per type (a standard,
# if simple, pattern for small heterogeneous graphs -- typically paired
# with a type-specific input projection layer in the model itself).
def _as_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalizes a possibly-tz-aware datetime to naive-UTC, so it can be
    safely compared against this pipeline's other datetimes (window_end,
    created_time) -- confirmed directly on real data that markets_df's
    close_time comes back tz-AWARE while window_end (built from
    FixedWindowTemporal, itself derived from created_time, whose dtype is
    Datetime(time_zone=None)) is naive. Comparing a naive and an aware
    datetime with a bare `>=`/`-` raises TypeError in Python -- this
    normalizes rather than stripping tzinfo blindly, since a bare strip
    would silently shift an aware time that ISN'T already UTC by its
    offset. A naive input is returned unchanged (assumed already UTC,
    matching this pipeline's convention throughout)."""
    if dt is not None and dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


_TICKER_RAW_FEATURES = 9
_BASKET_RAW_FEATURES = 6
COMBINED_N_FEATURES = max(_TICKER_RAW_FEATURES, _BASKET_RAW_FEATURES) + 1  # +1 for the type indicator


class KalshiTickerNodes:
    """Node features reconstructed from the Kalshi trade data.

    ``yes_price`` is used as the canonical market probability throughout.
    YES-side trades (``taker_side == "yes"``) represent buying pressure;
    NO-side trades represent selling pressure on YES.

    Feature vector (COMBINED_N_FEATURES-D -- see module docstring for why):
        --- Price (yes_price as canonical probability) ---
        0.  last_yes_price      yes_price of the last trade in the window
        1.  yes_vwap            volume-weighted average yes_price across all trades
        2.  price_return        last_yes_price – first_yes_price
        3.  price_std           std dev of yes_price  (realised volatility)

        --- Volume ---
        4.  window_volume       total contracts traded in this window

        --- Order flow ---
        5.  net_flow            (buy_vol – sell_vol) / total_vol  (normalised, signed)
        6.  buy_ratio           YES-side volume / total_vol  (volume-weighted)

        --- Activity ---
        7.  trade_intensity     trades per second in window
        8.  time_to_close       seconds from window_end to market close_time  (0 if past)

        --- Type indicator (shared slot with KalshiMeceBasketNodes) ---
        9.  is_mece_basket      always 0.0 for a ticker node

    CARRY-FORWARD FOR TRACKED TICKERS: by default (``tracked_tickers=None``),
    a ticker only exists as a node in windows where it actually traded --
    the original behaviour. If ``tracked_tickers`` is given (the union of
    every MECE-leg and ladder-leg ticker your validated pipelines care
    about), those SPECIFIC tickers ALSO stay present in a window via their
    last-known feature vector even if they didn't trade fresh that window,
    mirroring the same carry-forward pattern already used by
    KalshiMeceBasketNodes for its aggregate sum_cents/deviation.

    This was added after discovering that MECE hyperedges only ever
    connected FRESHLY-trading legs to their hub (since a leg with no fresh
    trade was never even a node to connect), even though the hub's own
    features already reflected carry-forward knowledge of quieter legs.
    In practice this meant many basket-hub softmax groups at low-freshness
    snapshots degenerated to a single member -- a softmax over one value
    is trivially always 1.0 regardless of what the model learned, wasting
    that training instance entirely. Restricted to `tracked_tickers`
    specifically (not every ticker that ever trades) so this doesn't
    explode node counts for the vast majority of tickers no mechanism
    here cares about.

    NOT extended with a dedicated per-leg "is this reading stale" feature
    slot in this pass -- that would change COMBINED_N_FEATURES and cascade
    into every downstream file (output_heads.py, training_objective.py's
    raw_feature_dim, etc.). Documented here as a known follow-up rather
    than rushed: for now the model can infer staleness only indirectly,
    via the basket hub's own freshness_ratio feature, not per individual
    leg.

    CARRY-FORWARD EXPIRY -- REQUIRED AT MULTI-MONTH SCALE, NOT OPTIONAL:
    the FIRST version of this carry-forward (single-day verification,
    T=10 snapshots) never expired a tracked ticker once it started being
    carried forward -- ``self._last_known_features.keys()`` only grew.
    That's harmless over one day, but confirmed directly (not assumed) to
    be the actual cause of a real 5-month run OOM-hanging a machine: over
    ~1800 two-hour snapshots, every tracked ticker that EVER traded once
    stayed a live node in EVERY subsequent snapshot for the rest of the
    whole build, forever, including long after its own market resolved
    and closed and could never trade again. Node count per snapshot grew
    roughly monotonically across the run instead of tracking real
    activity, and KalshiLadderChainEdges.build_edges (which scans
    snapshot.node_ids every snapshot) paid that same growth right back as
    edge-reconstruction cost. Fixed here with two independent cutoffs,
    checked in identify_nodes before a stale ticker is even considered
    for carry-forward:
      1. ``close_time`` (already looked up via ``_ticker_meta``) is a hard
         FACT, not a guess -- once a window's time passes it, the market
         cannot trade again, ever, so it must stop existing as a node.
      2. ``max_staleness`` is a fallback cap for tickers with missing or
         wrong close_time data, so a data gap can't silently reopen this
         same unbounded-growth hole.
    A ticker that expires from carry-forward isn't gone forever -- if it
    genuinely trades again, it re-enters via ``fresh_ids`` the same as any
    other ticker, unaffected by this cutoff (which only ever governs
    whether a STALE ticker gets carried forward, never a fresh one).

    Requires
    --------
    ``auxiliary["markets"]`` — the markets table, used to look up
    ``close_time`` and metadata columns (``event_ticker``, ``title``,
    ``status``, ``market_type``).  Only the last-seen row per ticker is
    used so the stale order-book values are intentionally ignored.
    """

    N_FEATURES = COMBINED_N_FEATURES
    _META_COLS = ("event_ticker", "title", "status", "market_type", "close_time")

    def __init__(
        self,
        node_col: str = "ticker",
        markets_df: Optional[pl.DataFrame] = None,
        tracked_tickers: Optional[set] = None,
        max_staleness: Optional[timedelta] = timedelta(days=14),
    ) -> None:
        self.node_col = node_col
        self.tracked_tickers = {str(t) for t in tracked_tickers} if tracked_tickers else set()
        # Fallback cap on how long a tracked ticker can be carried forward
        # with NO close_time on record before it's dropped anyway -- see
        # the "CARRY-FORWARD EXPIRY" docstring section above. None disables
        # this fallback entirely (close_time remains the primary cutoff).
        self.max_staleness = max_staleness
        # Pre-build O(1) ticker → metadata dict to avoid per-node DataFrame filters at build time.
        self._ticker_meta: Dict[str, Dict[str, Any]] = {}
        if markets_df is not None:
            cols = [c for c in (node_col, *self._META_COLS) if c in markets_df.columns]
            for row in (
                markets_df.select(cols)
                .unique(subset=[node_col], keep="last")
                .iter_rows(named=True)
            ):
                self._ticker_meta[str(row[node_col])] = row
        # Precomputed once at construction (not per snapshot): ticker -> its
        # own close_time, the HARD carry-forward cutoff (a closed market can
        # never trade again, period -- see docstring). None if unknown.
        # Normalized to naive-UTC here, ONCE, rather than at every
        # comparison site -- confirmed directly on real data that this
        # column comes back tz-AWARE while window_end (derived from
        # created_time, which is tz-naive) does not, and Python raises
        # TypeError comparing the two as-is (see _as_naive_utc's docstring).
        self._ticker_close_time: Dict[str, Optional[datetime]] = {
            tic: _as_naive_utc(meta.get("close_time")) for tic, meta in self._ticker_meta.items()
        }
        # Cache of per-ticker slices for the current window, populated in identify_nodes
        # and consumed in build_node_state to avoid re-scanning wd per node.
        self._window_slices: Dict[str, pl.DataFrame] = {}
        # NOTE: this must be a real reference to the last-seen window DataFrame, not
        # id(data). GraphBuilder keeps every window alive for the duration of one
        # .build() call, so id() reuse can't happen there — but IncrementalGraphBuilder
        # reuses this same strategy instance across *separate* .build() calls on
        # different chunks. Once one chunk's DataFrame is garbage-collected between
        # ingest() calls, Python is free to hand the next chunk's DataFrame the exact
        # same id(), which would silently reuse a stale, unrelated window's cache.
        # Holding the actual object (and comparing with `is`) makes that impossible:
        # as long as we hold the reference, its id() can't be reassigned to anything
        # else.
        self._window_data_ref: Optional[pl.DataFrame] = None
        # Carry-forward cache for tracked tickers ONLY -- ticker -> its last
        # computed full feature vector (all raw slots, pre-nan_to_num'd).
        # See class docstring.
        self._last_known_features: Dict[str, np.ndarray] = {}
        # ticker -> window_end of the last window it had an ACTUAL fresh
        # trade (not merely a window where it was displayed via carry-
        # forward) -- the clock max_staleness measures against.
        self._last_known_time: Dict[str, datetime] = {}

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        fresh_ids = {str(x) for x in data[self.node_col].unique().to_list()}
        # Partition wd once here so build_node_state can do O(1) dict lookup
        # instead of re-scanning the window DataFrame for every node.
        if data is not self._window_data_ref:
            self._window_slices = {
                str(k): v
                for k, v in data.partition_by(self.node_col, as_dict=True).items()
            }
            self._window_data_ref = data

        carried_ids: set = set()
        if self.tracked_tickers:
            window_end: Optional[datetime] = _as_naive_utc(kwargs.get("window_end"))
            candidates = (self.tracked_tickers & self._last_known_features.keys()) - fresh_ids
            for t in candidates:
                if window_end is not None:
                    close_time = self._ticker_close_time.get(t)
                    if close_time is not None and window_end >= close_time:
                        # Hard cutoff: this market has resolved and can
                        # never trade again -- stop carrying it forward,
                        # and evict its cache entries since they can never
                        # legitimately be needed again either (see
                        # "CARRY-FORWARD EXPIRY" docstring section).
                        self._last_known_features.pop(t, None)
                        self._last_known_time.pop(t, None)
                        continue
                    last_seen = self._last_known_time.get(t)
                    if (self.max_staleness is not None and last_seen is not None
                            and (window_end - last_seen) > self.max_staleness):
                        # Fallback cutoff for missing/bad close_time data --
                        # NOT evicted (unlike the close_time case above):
                        # unlike a resolved market, this ticker could
                        # legitimately trade again later, at which point
                        # it re-enters via fresh_ids regardless of this
                        # cutoff, so its cache stays available for that.
                        continue
                carried_ids.add(t)

        return sorted(fresh_ids | carried_ids)

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        # Normalized the same way identify_nodes' window_end is, so the
        # value stored into _last_known_time here is directly comparable
        # (same naive-UTC convention) to what identify_nodes reads back
        # out of it later -- see _as_naive_utc's docstring.
        window_end: Optional[datetime] = _as_naive_utc(kwargs.get("window_end"))
        auxiliary: Dict[str, pl.DataFrame] = kwargs.get("auxiliary") or {}
        markets: Optional[pl.DataFrame] = auxiliary.get("markets")
        node_id_str = str(node_id)

        t = self._window_slices.get(node_id_str) if self._window_slices else None
        if t is None:
            t = data.filter(pl.col(self.node_col) == node_id)
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)
        meta: Dict[str, Any] = {"node_type": "ticker"}

        close_time: Optional[datetime] = None
        cached_meta = self._ticker_meta.get(node_id_str)
        if cached_meta is not None:
            for col in ("event_ticker", "title", "status", "market_type"):
                if col in cached_meta:
                    meta[col] = cached_meta[col]
            ct = cached_meta.get("close_time")
            if ct is not None:
                close_time = _as_naive_utc(ct)
        elif markets is not None and not markets.is_empty():
            mkt = markets.filter(pl.col("ticker") == node_id)
            if not mkt.is_empty():
                row = mkt.tail(1)
                for col in ("event_ticker", "title", "status", "market_type"):
                    if col in row.columns:
                        meta[col] = row[col][0]
                if "close_time" in row.columns:
                    ct = row["close_time"][0]
                    if ct is not None:
                        close_time = _as_naive_utc(ct)

        if t.is_empty():
            # No fresh trade this window. A tracked ticker with a carried-
            # forward state replays it (see class docstring) instead of
            # returning near-all-zero features -- time_to_close is still
            # recomputed fresh, since it depends only on window_end, not
            # on anything that goes stale.
            cached_feats = self._last_known_features.get(node_id_str)
            if node_id_str in self.tracked_tickers and cached_feats is not None:
                feats = cached_feats.copy()
                if window_end is not None and close_time is not None:
                    feats[8] = seconds_to_close(window_end, close_time)
                return NodeState(node_id, feats, meta)

            if window_end is not None and close_time is not None:
                feats[8] = seconds_to_close(window_end, close_time)
            return NodeState(node_id, feats, meta)

        if "created_time" in t.columns:
            t = t.sort("created_time")

        yes_prices = t["yes_price"].cast(pl.Float64)
        counts = t["count"].cast(pl.Float64)
        n = t.height
        total_vol = float(counts.sum())

        is_buy = t["taker_side"] == "yes"
        t_yes = t.filter(is_buy)
        t_no = t.filter(~is_buy)
        buy_vol = float(t_yes["count"].sum()) if not t_yes.is_empty() else 0.0
        sell_vol = float(t_no["count"].sum()) if not t_no.is_empty() else 0.0

        feats[0] = float(yes_prices[-1])
        feats[1] = float((yes_prices * counts).sum()) / max(total_vol, 1.0)
        feats[2] = float(yes_prices[-1]) - float(yes_prices[0]) if n >= 2 else 0.0
        feats[3] = float(yes_prices.std()) if n >= 2 else 0.0  # type: ignore[arg-type]
        feats[4] = total_vol
        feats[5] = (buy_vol - sell_vol) / max(total_vol, 1.0)
        feats[6] = buy_vol / max(total_vol, 1.0)
        if "created_time" in t.columns and n >= 2:
            span = (t["created_time"][-1] - t["created_time"][0]).total_seconds()
            feats[7] = n / max(span, 1.0)
        if window_end is not None and close_time is not None:
            feats[8] = seconds_to_close(window_end, close_time)

        result = np.nan_to_num(feats, nan=0.0)
        if node_id_str in self.tracked_tickers:
            self._last_known_features[node_id_str] = result.copy()
            # Record when this ticker was last seen with an ACTUAL fresh
            # trade -- this is the point we reached by falling through
            # past the "t.is_empty()" branch above, so this call really is
            # a fresh observation, not a carried-forward replay (that
            # branch returns early and never reaches here). window_end may
            # be None (temporal strategy didn't supply one); in that case
            # the max_staleness fallback in identify_nodes simply can't
            # fire for this ticker, leaving close_time as the only cutoff.
            if window_end is not None:
                self._last_known_time[node_id_str] = window_end
        return NodeState(node_id, result, meta)


class KalshiEventNodes:
    """Event-level super-nodes constructed directly from trade data.

    One node per unique ``event_ticker`` active in the current time window.
    Features are aggregated across every market belonging to the event —
    completely independent of ``KalshiTickerNodes``.

    Feature vector (7-D):
        0.  total_volume        total contracts traded across all event markets
        1.  event_vwap          volume-weighted yes_price across all trades
        2.  buy_ratio           YES-side volume / total_vol  (volume-weighted)
        3.  net_flow            (buy_vol - sell_vol) / total_vol  (normalised)
        4.  num_markets         number of active markets for this event
        5.  min_time_to_close   seconds to the earliest-resolving market  (0 if past)
        6.  max_time_to_close   seconds to the latest-resolving market    (0 if past)

    Requires ``auxiliary["markets"]`` to map ``ticker → event_ticker`` and to
    provide ``close_time``.

    Super-node IDs use the prefix ``"EVENT:"`` (e.g. ``"EVENT:CPI-21JUN"``) so
    they never collide with market ticker strings.
    """

    _PREFIX = "EVENT:"
    N_FEATURES = 7

    def __init__(self, node_col: str = "ticker") -> None:
        self.node_col = node_col

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        markets: Optional[pl.DataFrame] = (kwargs.get("auxiliary") or {}).get("markets")
        if markets is None or data.is_empty():
            return []
        active = data[self.node_col].unique()
        event_tickers = (
            markets.filter(pl.col(self.node_col).is_in(active))
            .select("event_ticker")
            .unique()
            ["event_ticker"]
            .drop_nulls()
            .to_list()
        )
        return [f"{self._PREFIX}{et}" for et in event_tickers]

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        evt = str(node_id)[len(self._PREFIX):]
        markets: Optional[pl.DataFrame] = (kwargs.get("auxiliary") or {}).get("markets")
        window_end = kwargs.get("window_end")
        meta: Dict[str, Any] = {"node_type": "event", "event_ticker": evt}
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)

        if markets is None:
            return NodeState(node_id, feats, meta)

        event_mkts = markets.filter(pl.col("event_ticker") == evt)
        event_tickers_list = event_mkts[self.node_col].unique().to_list()
        feats[4] = float(len(event_tickers_list))

        t = data.filter(pl.col(self.node_col).is_in(event_tickers_list))
        if not t.is_empty():
            if "created_time" in t.columns:
                t = t.sort("created_time")
            counts = t["count"].cast(pl.Float64)
            total_vol = float(counts.sum())
            feats[0] = total_vol

            yes_prices = t["yes_price"].cast(pl.Float64)
            feats[1] = float((yes_prices * counts).sum()) / max(total_vol, 1.0)

            is_buy = t["taker_side"] == "yes"
            buy_vol = float(t.filter(is_buy)["count"].sum())
            sell_vol = float(t.filter(~is_buy)["count"].sum())
            feats[2] = buy_vol / max(total_vol, 1.0)
            feats[3] = (buy_vol - sell_vol) / max(total_vol, 1.0)

        if window_end is not None and "close_time" in event_mkts.columns:
            close_times = event_mkts["close_time"].drop_nulls()
            if close_times.len() > 0:
                feats[5] = seconds_to_close(window_end, close_times.min())  # type: ignore[arg-type]
                feats[6] = seconds_to_close(window_end, close_times.max())  # type: ignore[arg-type]

        return NodeState(node_id, np.nan_to_num(feats, nan=0.0), meta)


class KalshiMeceBasketNodes:
    """Synthetic hub node per validated MECE (single-winner, sum-to-$1) basket.

    Mirrors ``KalshiEventNodes``' super-node pattern, but scoped ONLY to
    events already validated as genuine MECE structures by
    ``mece_sum_to_one_check.py``'s family-consistency heuristic -- not every
    event sharing a raw ``event_ticker``. Combined with
    ``KalshiMeceHyperedges``, this gives each basket's N legs a
    star-expansion hyperedge: every leg connects to one shared hub instead
    of a fully-connected clique of N(N-1) pairwise edges. That matters
    because the sum-to-$1 constraint is a genuine N-ary relation over the
    whole basket at once, not N(N-1)/2 independent pairwise relations --
    representing it as a clique forces a GAT to re-derive "these all belong
    to one constraint" indirectly through many separate attention weights;
    a hyperedge represents it directly.

    Feature vector (COMBINED_N_FEATURES-D -- see module docstring for why):
        0.  sum_cents        sum of each leg's last KNOWN yes_price (fresh this window, or
                              carried forward from an earlier one) -- cents
        1.  deviation        sum_cents / 100 - 1.0 -- the actual mispricing signal
        2.  legs_known       number of legs with any observed price so far (fresh or carried forward)
        3.  legs_total       total legs in the validated basket
        4.  coverage_ratio   legs_known / legs_total (1.0 once every leg has traded at
                              least once, ever -- not just in this window)
        5.  freshness_ratio  fraction of legs_known that got an ACTUAL trade in this
                              specific window, vs. relying on a carried-forward stale
                              price (1.0 = everything just repriced; low = mostly stale)
        6-8.  unused padding (always 0.0) -- shared-width slots only meaningful for
                              KalshiTickerNodes' feature 6-8
        9.  is_mece_basket   always 1.0 for a basket hub (0.0 for a ticker node)

    CARRY-FORWARD, AND WHY: a market that trades once every few hours would
    otherwise "disappear" from every window it doesn't trade in, and the
    naive sum would implicitly treat its price as $0 -- biasing deviation
    toward looking underpriced purely because a leg went quiet, not because
    of any real mispricing. This keeps a running per-ticker last-known-price
    cache, updated only when a leg actually trades, so a quiet leg still
    contributes its last real price instead of vanishing from the sum.
    KNOWN SIMPLIFICATION: no staleness cutoff is applied -- a price from
    days ago is still carried forward indefinitely. freshness_ratio is
    provided specifically so the model can learn to discount stale readings
    itself, rather than this class silently dropping them past some
    arbitrary age.

    STATEFUL -- IMPORTANT FOR TRAIN/VAL/TEST SPLITS: this cache accumulates
    causally in temporal order (a snapshot only ever sees prices from itself
    or earlier, never the future), so it's safe and CORRECT to build one
    continuous graph across your full master calendar in a single
    .build() call, then slice into train/val/test via
    SpatioTemporalGraph.window() afterward. Do NOT construct separate
    instances for separate train/val/test builds -- that would reset the
    cache at each split boundary and lose legitimate carried-forward
    knowledge that genuinely existed before the split (e.g. val's first
    snapshot not "knowing" a price train's last snapshot already knew).

    Hub node IDs use the prefix ``"MECE:"`` (distinct from
    ``KalshiEventNodes``' ``"EVENT:"`` prefix) so the two never collide if
    both are registered on the same graph.

    Parameters
    ----------
    leg_prices_df : pl.DataFrame
        ``mece_sum_to_one_leg_prices.parquet`` (or equivalent) -- must have
        columns ``"event_ticker"``, ``"ticker"``. Defines basket membership
        (which legs belong to which validated basket) as ground truth from
        the already-validated offline pipeline; this class does NOT
        re-derive membership from raw data, only reads live per-window
        prices for legs whose membership is already established.
    """

    _PREFIX = "MECE:"
    N_FEATURES = COMBINED_N_FEATURES
    _RAW_FEATURES = 6

    def __init__(self, leg_prices_df: pl.DataFrame, node_col: str = "ticker") -> None:
        self.node_col = node_col
        self._basket_legs: Dict[str, List[str]] = {}
        self._leg_basket: Dict[str, str] = {}
        for row in leg_prices_df.select(["event_ticker", "ticker"]).unique().iter_rows(named=True):
            evt, tic = str(row["event_ticker"]), str(row["ticker"])
            self._basket_legs.setdefault(evt, []).append(tic)
            self._leg_basket[tic] = evt
        # Carry-forward cache: ticker -> last known yes_price (cents). Updated
        # only when a leg actually trades; consulted (without being cleared)
        # when a leg is absent from the current window. See class docstring
        # for why this must stay ONE instance across a whole chronological
        # build, not reset per train/val/test split.
        self._last_known_price: Dict[str, float] = {}

    def identify_nodes(self, data: pl.DataFrame, **kwargs: Any) -> List[Hashable]:
        if data.is_empty():
            return []
        active = set(data[self.node_col].unique().to_list())
        baskets_present = {
            self._leg_basket[t] for t in active if t in self._leg_basket
        }
        return [f"{self._PREFIX}{evt}" for evt in baskets_present]

    def build_node_state(self, node_id: Hashable, data: pl.DataFrame, **kwargs: Any) -> NodeState:
        evt = str(node_id)[len(self._PREFIX):]
        legs_total = self._basket_legs.get(evt, [])
        meta: Dict[str, Any] = {
            "node_type": "mece_basket", "event_ticker": evt, "n_legs_total": len(legs_total),
        }
        feats = np.zeros(self.N_FEATURES, dtype=np.float64)
        feats[-1] = 1.0  # is_mece_basket indicator -- see module docstring
        feats[3] = float(len(legs_total))

        if not legs_total:
            return NodeState(node_id, feats, meta)

        t = data.filter(pl.col(self.node_col).is_in(legs_total))
        legs_fresh_this_window = 0

        if not t.is_empty():
            if "created_time" in t.columns:
                t = t.sort("created_time")
            fresh_prices = (
                t.group_by(self.node_col)
                .agg(pl.col("yes_price").cast(pl.Float64).last().alias("last_price"))
            )
            legs_fresh_this_window = fresh_prices.height
            # Update the carry-forward cache with this window's actual trades.
            for row in fresh_prices.iter_rows(named=True):
                self._last_known_price[row[self.node_col]] = row["last_price"]

        # Sum using every leg with a KNOWN price -- fresh this window, or
        # carried forward from an earlier one -- not just legs that traded
        # in this exact window.
        sum_cents = 0.0
        legs_known = 0
        for leg in legs_total:
            price = self._last_known_price.get(leg)
            if price is not None:
                sum_cents += price
                legs_known += 1

        feats[0] = sum_cents
        feats[1] = sum_cents / 100.0 - 1.0
        feats[2] = float(legs_known)
        feats[4] = legs_known / max(len(legs_total), 1)
        feats[5] = legs_fresh_this_window / max(legs_known, 1)

        return NodeState(node_id, np.nan_to_num(feats, nan=0.0), meta)