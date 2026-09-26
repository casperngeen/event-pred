"""
model/data_validation.py

Drops or repairs feature values that are PHYSICALLY IMPOSSIBLE, as
opposed to merely extreme, before they reach the model.

WHY THIS EXISTS -- MEASURED, NOT SUSPECTED. diagnose_features.py over
1,075,725 real observations from five months of Kalshi data found:

    last_yes_price outside [0, 100]     26,848   2.4958%   (max 218.0)
    yes_vwap       outside [0, 100]      7,817   0.7267%   (min -0.99)
    time_to_close  outside [0, 2 years]  3,511   0.3264%   (max 73.9 YEARS)

Three separate harms, which is why this is not cosmetic:

1. UNREACHABLE TARGETS. model/train.py's true_prices() feeds slot 0 / 100
   to the loss as the regression target, while MeceOutputHead (softmax)
   and LadderOutputHead (sigmoid) are bounded to [0, 1] BY CONSTRUCTION.
   A price of 218 becomes a target of 2.18 that no parameter setting can
   reach, contributing irreducible loss forever.

2. BROKEN MECHANISM STRUCTURE, which is worse than the raw 2.5% implies.
   The MECE mechanism is learnable only because a basket's legs sum to
   about 1. At a 2.5% per-leg corruption rate, a 5-leg basket has a
   ~12% chance of containing at least one corrupt leg -- so roughly one
   basket in eight has the very relationship being learned destroyed by
   bad data, and the model is asked to fit that noise.

3. OUTLIERS THAT SURVIVE STANDARDIZATION. Standardizing does not rescue
   a corrupt value, it just produces a well-scaled corrupt value: a
   74-year time_to_close sits roughly 344 standard deviations from a
   mean of ~7.8 days even after (x - mean) / std.

POLICY, AND WHY IT DIFFERS PER FIELD:

  DROP (mask -> False) for the core price fields (slots 0, 1). An
  impossible price means the reading itself cannot be trusted, and the
  honest representation of "we have no valid observation here" is the
  one the pipeline already has: mask=False. Clamping instead would
  fabricate a value -- pinning 218 to 100 asserts "this market is
  certain", a strong and false claim the model would then train on.

  CLAMP for secondary fields (slots 2-8). A bad close_time does not make
  the trade prices wrong, so discarding an otherwise-good observation
  over it would throw away real data. Clamping bounds the damage to the
  one field that is broken.

EDGE CONSISTENCY IS NOT OPTIONAL HERE. Setting mask=False without also
dropping that node's edges breaks an invariant the rest of the pipeline
relies on: model/sparse_utils.py's to_local_edge_index maps global edge
indices into a snapshot's active-node ordering with searchsorted, which
is only correct because every edge endpoint is guaranteed to be active in
that snapshot (see that module's docstring). Invalidate a node without
filtering its edges and searchsorted silently returns an insertion point
instead of a position -- wiring the graph to the wrong nodes while
producing correctly-shaped, non-NaN output. So edges lose any endpoint
that is no longer active.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import torch

_TWO_YEARS_SECONDS = 2 * 365 * 24 * 3600.0
_INF = float("inf")

# RULES ARE PER NODE TYPE, AND MUST BE. The two node types share one
# padded feature width but NOT one feature meaning -- see the class
# docstrings in stg/nodes/kalshi.py. Slot 0 is last_yes_price for a
# ticker leg and sum_cents for a MECE basket hub; slot 1 is yes_vwap for
# a ticker and DEVIATION (sum_cents/100 - 1) for a hub.
#
# An earlier version of this module applied the TICKER ranges to every
# node. On real data that was not a small error: requiring hub slot 0 in
# [0, 100] discards every basket whose legs sum to more than $1 (the
# OVERPRICED case) and requiring hub slot 1 in [0, 100] discards every
# basket with negative deviation (the UNDERPRICED case) -- between them,
# essentially every mispriced basket, which is the exact signal this
# project exists to detect. Because hubs are high-degree (one edge per
# leg), losing 3.2% of observations destroyed 55% of all edges and left
# the validation split with almost no MECE structure at all. Selection
# bias that removes the phenomenon under study is worse than the corrupt
# data it was meant to remove, so the type split below is load-bearing,
# not tidiness.

# --- ticker legs (type indicator == 0.0) ---
TICKER_DROP: Dict[int, Tuple[float, float]] = {
    0: (0.0, 100.0),    # last_yes_price -- a YES price is 0-100 cents
    1: (0.0, 100.0),    # yes_vwap -- a volume-weighted average of prices in 0-100
}
TICKER_CLAMP: Dict[int, Tuple[float, float]] = {
    2: (-100.0, 100.0),            # price_return
    3: (0.0, 100.0),               # price_std
    4: (0.0, _INF),                # window_volume
    5: (-1.0, 1.0),                # net_flow
    6: (0.0, 1.0),                 # buy_ratio
    7: (0.0, _INF),                # trade_intensity
    8: (0.0, _TWO_YEARS_SECONDS),  # time_to_close
}

# --- MECE basket hubs (type indicator == 1.0) ---
# Deliberately permissive: only genuinely impossible values are dropped.
# sum_cents ABOVE 100 and deviation BELOW 0 are both real, meaningful
# market states, not corruption.
BASKET_DROP: Dict[int, Tuple[float, float]] = {
    0: (0.0, 10000.0),  # sum_cents -- non-negative; 10000 allows a 100-leg basket at 100c each
    4: (0.0, 1.0),      # coverage_ratio -- a ratio by construction
    5: (0.0, 1.0),      # freshness_ratio -- a ratio by construction
}
BASKET_CLAMP: Dict[int, Tuple[float, float]] = {
    1: (-1.0, 100.0),   # deviation = sum_cents/100 - 1: >= -1 by construction, negative is NORMAL
    2: (0.0, _INF),     # legs_known -- a count
    3: (0.0, _INF),     # legs_total -- a count
    # slots 6-8 are unused padding for this type (always 0.0); nothing to bound.
}


class ValidationStats:
    """Counts what was changed, so a run reports its own data quality
    rather than silently discarding a chunk of the dataset."""

    def __init__(self):
        self.observations_seen = 0
        self.observations_dropped = 0
        self.values_clamped = 0
        self.edges_dropped = 0
        self.edges_seen = 0
        # Broken out BY NODE TYPE, because the aggregate hid a serious
        # bug: a 3.2% overall drop rate looked tolerable while actually
        # being a near-total wipeout of basket hubs (which then took 55%
        # of all edges with them). A per-type line makes that visible
        # immediately instead of only via the edge count.
        self.ticker_seen = 0
        self.ticker_dropped = 0
        self.basket_seen = 0
        self.basket_dropped = 0

    def merge(self, other: "ValidationStats"):
        self.observations_seen += other.observations_seen
        self.observations_dropped += other.observations_dropped
        self.values_clamped += other.values_clamped
        self.edges_dropped += other.edges_dropped
        self.edges_seen += other.edges_seen
        self.ticker_seen += other.ticker_seen
        self.ticker_dropped += other.ticker_dropped
        self.basket_seen += other.basket_seen
        self.basket_dropped += other.basket_dropped

    def summary(self) -> str:
        if self.observations_seen == 0:
            return "  (no observations seen)"

        def _pct(a, b):
            return 100.0 * a / b if b else 0.0

        lines = [
            f"  observations: {self.observations_seen:,} seen, {self.observations_dropped:,} "
            f"dropped ({_pct(self.observations_dropped, self.observations_seen):.3f}%) as physically impossible",
            f"    ticker legs:  {self.ticker_seen:,} seen, {self.ticker_dropped:,} dropped "
            f"({_pct(self.ticker_dropped, self.ticker_seen):.3f}%)",
            f"    basket hubs:  {self.basket_seen:,} seen, {self.basket_dropped:,} dropped "
            f"({_pct(self.basket_dropped, self.basket_seen):.3f}%)",
            f"  values clamped into range: {self.values_clamped:,}",
            f"  edges: {self.edges_seen:,} seen, {self.edges_dropped:,} dropped "
            f"({_pct(self.edges_dropped, self.edges_seen):.3f}%) for referencing a dropped observation",
        ]
        if _pct(self.basket_dropped, self.basket_seen) > 20.0:
            lines.append(
                "  *** WARNING: a large fraction of MECE basket hubs was dropped. Hubs are\n"
                "      high-degree, so this destroys the MECE graph structure, and the rules\n"
                "      may be rejecting REAL mispricings (sum_cents > 100, negative deviation)\n"
                "      rather than corrupt data. Check BASKET_DROP in model/data_validation.py."
            )
        return "\n".join(lines)


def apply_validity(
    features: torch.Tensor,
    mask: torch.Tensor,
    adjacency_by_type: Dict[str, List],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List], ValidationStats]:
    """Returns (features, mask, adjacency_by_type, stats) with impossible
    observations masked out, secondary fields clamped, and edges filtered
    to match.

    Operates in whatever node index space it is handed -- it only ever
    indexes ``mask[t]`` by edge endpoints, so it is correct for both the
    month-local and chunk-local spaces, as long as features, mask and the
    edge indices all share one space (which they do at every call site).
    """
    stats = ValidationStats()
    T, N, F = features.shape

    stats.observations_seen = int(mask.sum())

    # Node type comes from the LAST slot (COMBINED_N_FEATURES - 1):
    # 1.0 = MECE basket hub, 0.0 = ticker leg. Same convention
    # TypeSpecificProjection routes on.
    is_basket = features[..., -1] > 0.5
    is_ticker = ~is_basket
    stats.ticker_seen = int((mask & is_ticker).sum())
    stats.basket_seen = int((mask & is_basket).sum())

    # 1. Which observations are untrustworthy at their core, per type?
    invalid = torch.zeros_like(mask)
    for rules, type_sel in ((TICKER_DROP, is_ticker), (BASKET_DROP, is_basket)):
        for slot, (lo, hi) in rules.items():
            if slot < F:
                col = features[..., slot]
                invalid |= ((col < lo) | (col > hi)) & type_sel
    invalid &= mask                       # only observed positions can be dropped
    stats.observations_dropped = int(invalid.sum())
    stats.ticker_dropped = int((invalid & is_ticker).sum())
    stats.basket_dropped = int((invalid & is_basket).sum())
    new_mask = mask & ~invalid

    # 2. Bound the secondary fields rather than discarding the row,
    #    again per type since the same slot means different things.
    new_features = features.clone()
    for rules, type_sel in ((TICKER_CLAMP, is_ticker), (BASKET_CLAMP, is_basket)):
        for slot, (lo, hi) in rules.items():
            if slot >= F:
                continue
            col = new_features[..., slot]
            applies = type_sel & mask
            out_of_range = ((col < lo) | (col > hi)) & applies
            stats.values_clamped += int(out_of_range.sum())
            clamped = col.clamp(min=lo, max=hi if hi != _INF else None)
            new_features[..., slot] = torch.where(applies, clamped, col)

    # 3. Keep the edge set consistent with the new mask (see module docstring).
    new_adjacency: Dict[str, List] = {}
    for edge_type, snapshots in adjacency_by_type.items():
        kept = []
        for t, edges_t in enumerate(snapshots):
            ei = edges_t.edge_index
            if ei.numel() == 0:
                kept.append(edges_t)
                continue
            stats.edges_seen += ei.shape[1]
            alive = new_mask[t]
            keep = alive[ei[0]] & alive[ei[1]]
            n_drop = int((~keep).sum())
            if n_drop == 0:
                kept.append(edges_t)
                continue
            stats.edges_dropped += n_drop
            kept.append(replace(
                edges_t,
                edge_index=ei[:, keep],
                weight=edges_t.weight[keep] if edges_t.weight is not None else None,
                features=edges_t.features[keep] if edges_t.features is not None else None,
            ))
        new_adjacency[edge_type] = kept

    return new_features, new_mask, new_adjacency, stats