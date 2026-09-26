"""
model/sparse_utils.py

Shared helper for every per-snapshot consumer of a TorchGraphBundle's
sparse ``adjacency_by_type`` (see stg/bridge.py's rewrite, and its
docstring for WHY adjacency is sparse now): converting a snapshot's
GLOBAL edge indices into LOCAL indices (0..len(active_idx)-1) matching
that snapshot's active-node ordering, without ever building anything
sized to the GLOBAL node count N.

WHY torch.searchsorted WORKS, UNCONDITIONALLY: a real edge only ever
exists between two nodes that are BOTH present in the same snapshot (it
was built by iterating that snapshot's own nx_graph -- see
SpatioTemporalGraph.sparse_edges_by_type()), so every edge_index value
for snapshot t is GUARANTEED to already be a member of ``active_idx``
(that snapshot's own active/masked-in GLOBAL node indices, from
``mask[t].nonzero()``). Given active_idx is ascending and unique
(torch.nonzero() returns indices in ascending order), searchsorted
directly returns each global index's LOCAL position in
O(E log n_active) time -- no lookup table sized to N is ever built, so
this stays cheap regardless of how large the global node universe gets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def to_local_edge_index(edge_index_global: torch.Tensor, active_idx_sorted: torch.Tensor) -> torch.Tensor:
    """(2, E) GLOBAL -> (2, E) LOCAL, via binary search against the
    snapshot's own sorted active-node index list. See module docstring
    for why every value is guaranteed to be found."""
    if edge_index_global.numel() == 0:
        return edge_index_global
    return torch.searchsorted(active_idx_sorted, edge_index_global)


def symmetrize_local_edges(
    local_edge_index: torch.Tensor,          # (2, E) local, directed leg_a -> leg_b
    weight: torch.Tensor,                     # (E,) -- carried through for API symmetry, not currently consumed
    features: Optional[torch.Tensor],         # (E, Fe) or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetrizes a directed local edge list for MESSAGE-PASSING
    purposes only (both endpoints become mutually aware of each other),
    while preserving the ORIGINAL one-way relationship as an explicit
    +1/-1 direction feature -- see TwoChannelSpatialAttention's docstring
    for why (PyG's directed message passing only updates an edge's
    TARGET, which would otherwise leave the SOURCE node permanently
    blind to its own ladder neighbours).

    Returns (edge_index_sym (2, 2E), edge_attr_sym (2E, Fe+1)): forward
    edges get direction=+1, the synthetic reverse copies get -1, and any
    real edge features are duplicated onto both copies (the one real
    value reflected into both positions -- same semantics as the
    original dense implementation's `sub_ef + sub_ef.transpose(0, 1)`).
    """
    if local_edge_index.numel() == 0:
        fe = (features.shape[-1] if features is not None else 0) + 1
        return local_edge_index, torch.zeros((0, fe), dtype=weight.dtype, device=weight.device)

    reversed_idx = local_edge_index.flip(0)
    edge_index_sym = torch.cat([local_edge_index, reversed_idx], dim=1)

    n = local_edge_index.shape[1]
    direction = torch.cat([
        torch.ones(n, dtype=weight.dtype, device=weight.device),
        -torch.ones(n, dtype=weight.dtype, device=weight.device),
    ]).unsqueeze(-1)

    if features is not None:
        feats_sym = torch.cat([features, features], dim=0)
        edge_attr_sym = torch.cat([feats_sym, direction], dim=-1)
    else:
        edge_attr_sym = direction

    return edge_index_sym, edge_attr_sym