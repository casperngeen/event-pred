"""
model/output_heads.py

Constraint-respecting output heads: the whole point of building these
(rather than free-form regression + hoping the network learns the
constraint from a loss penalty) is that the sum-to-$1 / monotonicity
constraint holds BY CONSTRUCTION, not as something the loss has to coax
the network toward -- see the STGAT architecture discussion early in this
project.

MECE head: for each basket, produces a fair-value distribution over its
own legs ONLY, via a softmax within each basket's own leg group (not a
global softmax over every node in the graph). Guarantees the predicted
prices sum to exactly 1.0 per basket, since a softmax's outputs always
sum to 1 by definition. Implemented with torch_geometric.utils.softmax,
which performs exactly this "softmax within groups defined by an index"
operation -- the same primitive GAT itself uses internally to normalise
attention per target node -- reused here rather than hand-rolled, same
reasoning as using GATv2Conv over a hand-rolled dense implementation in
spatial_attention.py.

Ladder head: for each validated pair (leg_a, leg_b), predicts a
non-negative gap via softplus and subtracts it from leg_a's predicted
base value. Guarantees fair(leg_a) >= fair(leg_b) BY CONSTRUCTION,
matching the actual constraint (leg_a is the easier-to-satisfy
threshold, so its price must be at least as high). Operates PAIRWISE,
not on fully-reconstructed multi-leg chains -- this matches how the rest
of this project already frames and validates the ladder mechanism
(same_side_yes_violation is a per-PAIR statistic throughout, computed by
pairwise_monotonicity_taker_side_check_v2.py on exactly these validated
pairs, never on a reconstructed chain), so operating pairwise here isn't
a simplification that loses rigor relative to the validated methodology
-- it's consistent with it.

Both heads are grouped/per-edge operations with ragged sizes per
snapshot (a different number of baskets/pairs is active at every
timestep), so -- same as spatial_attention.py's per-snapshot loop, and
for the identical reason -- each snapshot is processed individually
rather than vectorised into one dense batch.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as grouped_softmax


class MeceOutputHead(nn.Module):
    """Predicts a fair-value distribution over each MECE basket's legs,
    guaranteed to sum to exactly 1.0 per basket by construction."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.logit_proj = nn.Linear(embed_dim, 1)

    def forward(
        self,
        h: torch.Tensor,                        # (T, N, D)
        mece_leg_to_basket: list,                # length-T list of SparseSnapshotEdgesT, leg -> hub, GLOBAL indices
    ) -> Dict[int, Optional[dict]]:
        """Returns, per snapshot t (ragged group sizes, so no single fixed
        shape): a dict with 'leg_idx', 'hub_idx' (both LongTensor(E_t,))
        and 'fair_price' (Tensor(E_t,)), or None if no basket is active
        that snapshot. fair_price sums to exactly 1.0 within each distinct
        hub_idx group -- verified in tests, not just claimed.

        No ``mask`` parameter needed (unlike the old dense-adjacency
        version): ``mece_leg_to_basket[t].edge_index`` already holds
        GLOBAL leg/hub indices directly -- an edge only ever exists
        between nodes that coexisted in that snapshot in the first
        place, so there's no active-node restriction left to apply."""
        T, N, D = h.shape
        results: Dict[int, Optional[dict]] = {}
        for t in range(T):
            edges_t = mece_leg_to_basket[t]
            if edges_t.edge_index.numel() == 0:
                results[t] = None
                continue

            leg_global = edges_t.edge_index[0]
            hub_global = edges_t.edge_index[1]

            leg_h = h[t, leg_global]
            logits = self.logit_proj(leg_h).squeeze(-1)
            fair_price = grouped_softmax(logits, hub_global)

            results[t] = {"leg_idx": leg_global, "hub_idx": hub_global, "fair_price": fair_price}
        return results


class LadderOutputHead(nn.Module):
    """Predicts a monotonicity-respecting pair of fair values for each
    validated ladder pair: fair(leg_a) >= fair(leg_b) guaranteed by
    construction (a non-negative fraction of the base subtracted from
    that same base), not learned via a loss penalty.

    BOTH fair_a and fair_b are bounded to [0, 1] by construction
    (sigmoid base, sigmoid-fraction gap) -- NOT the original formulation
    (base_a - softplus(gap), softplus is non-negative but unbounded
    ABOVE). That version still satisfied fair_a >= fair_b, but on a
    random-init model its outputs could land anywhere on the real
    number line, while the true price target and MeceOutputHead's
    softmax output both live in [0, 1] -- found directly, not assumed,
    from a real run against 2025-02-02 data: an untrained model's ladder
    loss came out at ~58,767 against a MECE loss of ~0.14. In
    total = mece_loss + ladder_loss, that scale gap means the shared
    backbone's gradient during training is almost entirely driven by the
    ladder term regardless of how well or badly it's actually doing,
    effectively starving MECE learning until the ladder head's outputs
    happen to shrink on their own. Bounding both heads to the same [0,1]
    scale as their shared target fixes this at the source rather than
    reweighting the loss terms after the fact."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.base_proj = nn.Linear(embed_dim, 1)
        self.gap_proj = nn.Linear(embed_dim * 2, 1)

    def forward(
        self,
        h: torch.Tensor,               # (T, N, D)
        ladder_adj: list,              # length-T list of SparseSnapshotEdgesT, leg_a -> leg_b, GLOBAL indices
    ) -> Dict[int, Optional[dict]]:
        """Returns, per snapshot t: 'leg_a_idx', 'leg_b_idx'
        (LongTensor(E_t,)), 'fair_a', 'fair_b' (Tensor(E_t,)), or None if
        no ladder pair is active that snapshot. fair_a >= fair_b >= 0 and
        both are <= 1 hold for every single pair, unconditionally --
        verified in tests across many random weight initialisations,
        since it must be a mathematical guarantee (sigmoid outputs live
        in (0, 1) everywhere), not something that merely tends to hold.

        No ``mask`` parameter needed -- see MeceOutputHead.forward's
        docstring; the same reasoning applies here."""
        T, N, D = h.shape
        results: Dict[int, Optional[dict]] = {}
        for t in range(T):
            edges_t = ladder_adj[t]
            if edges_t.edge_index.numel() == 0:
                results[t] = None
                continue

            a_global = edges_t.edge_index[0]
            b_global = edges_t.edge_index[1]

            h_a = h[t, a_global]
            h_b = h[t, b_global]

            base_a = torch.sigmoid(self.base_proj(h_a).squeeze(-1))                              # (0, 1)
            gap_fraction = torch.sigmoid(self.gap_proj(torch.cat([h_a, h_b], dim=-1)).squeeze(-1))  # (0, 1)
            fair_a = base_a
            fair_b = base_a * (1.0 - gap_fraction)  # in [0, base_a] subset [0, 1] -- fair_a >= fair_b >= 0 always

            results[t] = {"leg_a_idx": a_global, "leg_b_idx": b_global, "fair_a": fair_a, "fair_b": fair_b}
        return results