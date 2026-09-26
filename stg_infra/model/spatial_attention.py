"""
model/spatial_attention.py

Spatial attention layer for the STGAT: the "structural" half of a
DySAT-style backbone, operating WITHIN each snapshot independently (no
cross-time dependency here -- that's the temporal layer's job).

REWRITTEN TWICE from an earlier dense implementation. First rewrite:
a dense (T, N, N, heads, head_dim) tensor OOM-killed on the actual
verified data scale (N=788) -- dense attention computes a score for
every one of N^2 possible pairs (620,944 at N=788) even though a real
snapshot has on the order of 100 actual edges. That rewrite moved to
PyTorch Geometric's sparse GATv2Conv over real edges (via edge_index),
but still built those edge_index lists by densifying a (N,N) submatrix
first (dense_to_sparse) -- fine at N=788, but the SOURCE of that
submatrix, stg/bridge.py's old dense (T,N,N)-per-type export, OOM-killed
an entire real machine at multi-month scale (N in the tens of
thousands). Second rewrite (this one): stg/bridge.py now hands over
SPARSE per-snapshot edge lists directly (see model/sparse_utils.py), so
this layer never touches a dense (N,N) matrix at any point, not even
transiently -- restricted to the nodes actually active in a given
snapshot (per mask) via searchsorted, not via any structure sized to N.

Two-channel design, not a single shared GAT: MECE hyperedge spokes
(symmetric, no edge features) and ladder chain edges (directed, carrying
historical violation/gap features) are structurally different relations
and are attended over SEPARATELY, then combined -- forcing them through
one shared attention mechanism would repeat the exact mistake this
project already learned from (treating structurally different market
types as if they were the same because they're superficially similar).

Type-specific input projection: ticker-leg nodes and MECE-basket-hub
nodes share the same padded feature width (COMBINED_N_FEATURES in
stg/nodes/kalshi.py), but the SAME slot means a different, unrelated
quantity for each node type. A single shared linear projection would
force one set of weights to do double duty for two unrelated things;
separate per-type projections (selected via the type-indicator feature
already present in the padded vector) let both types project into one
common embedding space properly before attention compares them.

Snapshots are processed one at a time (a Python loop over T), not
vectorised as one dense batch -- each snapshot has a genuinely different
active-node set and edge count (ragged), which is the normal, expected
shape of a temporal graph batch. The per-snapshot cost is small (active
nodes are a small fraction of the padded universe), so the loop overhead
is negligible next to the memory it avoids.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from model.sparse_utils import symmetrize_local_edges, to_local_edge_index


class TypeSpecificProjection(nn.Module):
    """Projects heterogeneous, shared-width padded node features into one
    common embedding space, using a separate linear layer per node type.

    Expects the LAST feature slot to be the type indicator (1.0 = MECE
    basket hub, 0.0 = ticker leg) -- see COMBINED_N_FEATURES in
    stg/nodes/kalshi.py.
    """

    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.ticker_proj = nn.Linear(in_features, embed_dim)
        self.basket_proj = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., F) -> (..., embed_dim)"""
        is_basket = x[..., -1:]
        ticker_out = self.ticker_proj(x)
        basket_out = self.basket_proj(x)
        return is_basket * basket_out + (1.0 - is_basket) * ticker_out


class TwoChannelSpatialAttention(nn.Module):
    """Combines a MECE-hyperedge channel and a ladder-chain channel into
    one spatial attention layer, with a residual connection.

    MECE channel: attends over BOTH ``mece_leg_to_basket`` and
    ``mece_basket_to_leg`` edges together (symmetric spokes, no edge
    features -- see ``KalshiMeceHyperedges``), via ``GATv2Conv``.

    Ladder channel: attends over ``ladder_monotonic`` edges, conditioned
    on each edge's historical violation/gap features (see
    ``KalshiLadderChainEdges``) via a separate ``GATv2Conv`` with
    ``edge_dim`` set. Direction is preserved (leg_a -> leg_b only), never
    symmetrised, since order carries real information here that the MECE
    relation doesn't have.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, ladder_edge_feature_dim: int = 2):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        head_dim = embed_dim // n_heads
        # concat=True + heads*head_dim == embed_dim keeps output width unchanged
        self.mece_conv = GATv2Conv(embed_dim, head_dim, heads=n_heads, concat=True, add_self_loops=False)
        # +1 for the direction indicator this class adds when symmetrising --
        # see forward()'s docstring note on bidirectional message-passing.
        self.ladder_conv = GATv2Conv(
            embed_dim, head_dim, heads=n_heads, concat=True,
            edge_dim=ladder_edge_feature_dim + 1, add_self_loops=False,
        )

    def forward(
        self,
        h: torch.Tensor,                       # (T, N, D) -- already projected via TypeSpecificProjection
        mask: torch.Tensor,                    # (T, N) bool
        adjacency_by_type: dict,               # edge_type -> list of length T, each a SparseSnapshotEdgesT
    ) -> torch.Tensor:
        T, N, D = h.shape
        out = h.clone()  # residual base -- untouched (isolated) nodes just keep their input

        leg_to_hub_list = adjacency_by_type.get("mece_leg_to_basket")
        hub_to_leg_list = adjacency_by_type.get("mece_basket_to_leg")
        ladder_list = adjacency_by_type.get("ladder_monotonic")

        # DIAGNOSTIC INSTRUMENTATION (temporary -- added to localize a hang
        # that reproduces after graph-building/chunking succeed but before
        # any epoch completes; see temporal_attention.py's docstring for
        # the pattern this project keeps finding: a dense/O(N)-ish cost
        # hiding inside a per-snapshot loop, invisible until real N/T
        # scale). Printed every 20 snapshots (t%20==0) plus the last one,
        # so a run that's genuinely progressing (just slowly) is
        # distinguishable from one that's truly stuck at a specific t --
        # and max_active flags a single pathological snapshot (e.g. one
        # with far more active nodes than the ~3.5% average) as a
        # candidate root cause distinct from the three already-fixed bugs.
        _t_start = time.time()
        _max_active = 0
        _max_active_t = -1

        for t in range(T):
            active_idx = mask[t].nonzero(as_tuple=True)[0]  # sorted ascending, GLOBAL indices
            if active_idx.numel() == 0:
                continue
            if active_idx.numel() > _max_active:
                _max_active = active_idx.numel()
                _max_active_t = t
            h_active = h[t, active_idx]  # (n_active, D)

            mece_out_active = torch.zeros_like(h_active)
            # Both MECE spoke directions (leg->hub, hub->leg) are real,
            # DISTINCT edges at different (row, col) positions -- concatenating
            # their local edge lists is exactly equivalent to the old dense
            # version's `leg_to_hub + hub_to_leg` sum-then-nonzero, since the
            # two never collide at the same position.
            mece_local_parts = []
            for edges_list in (leg_to_hub_list, hub_to_leg_list):
                if edges_list is not None and edges_list[t].edge_index.numel() > 0:
                    mece_local_parts.append(to_local_edge_index(edges_list[t].edge_index, active_idx))
            n_mece_edges = sum(p.shape[1] for p in mece_local_parts)
            if mece_local_parts:
                ei = torch.cat(mece_local_parts, dim=1)
                mece_out_active = self.mece_conv(h_active, ei)

            ladder_out_active = torch.zeros_like(h_active)
            n_ladder_edges = 0
            if ladder_list is not None and ladder_list[t].edge_index.numel() > 0:
                # Symmetrised for MESSAGE-PASSING purposes only: a ladder
                # violation is a joint fact about the pair (comparing leg_a
                # vs leg_b), so both legs should be mutually aware of each
                # other -- same reasoning already applied to the MECE
                # channel's leg<->hub spokes. The original one-way
                # leg_a->leg_b RELATIONSHIP isn't lost, it's preserved
                # explicitly as a direction indicator feature (see
                # symmetrize_local_edges), rather than being encoded (and
                # effectively hidden from the source node) purely via
                # one-way edge_index direction -- confirmed via a minimal
                # isolated test that PyG's directed message passing only
                # updates an edge's TARGET, never its SOURCE, which would
                # otherwise leave leg_a's own embedding permanently blind
                # to its own ladder neighbours.
                edges_t = ladder_list[t]
                n_ladder_edges = edges_t.edge_index.shape[1]
                local_fwd = to_local_edge_index(edges_t.edge_index, active_idx)
                ei, ea = symmetrize_local_edges(local_fwd, edges_t.weight, edges_t.features)
                ladder_out_active = self.ladder_conv(h_active, ei, edge_attr=ea)

            out[t, active_idx] = h_active + mece_out_active + ladder_out_active

            if t % 20 == 0 or t == T - 1:
                print(f"      [spatial] t={t:4d}/{T}  active={active_idx.numel():5d}  "
                      f"mece_edges={n_mece_edges:5d}  ladder_edges={n_ladder_edges:5d}  "
                      f"elapsed={time.time() - _t_start:6.1f}s", flush=True)

        print(f"      [spatial] snapshot loop done: max_active_in_one_snapshot={_max_active} "
              f"(at t={_max_active_t})  total_elapsed={time.time() - _t_start:.1f}s", flush=True)
        return out