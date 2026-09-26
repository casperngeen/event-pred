"""
stg/bridge.py

Converts a SpatioTemporalGraph into PyTorch-ready tensors: a dense
feature tensor (features/mask are ALWAYS small enough to be dense --
(T,N,F) with F~10 is at most a few GB even at N in the tens of
thousands), plus SPARSE per-snapshot edge lists per type.

REWRITTEN from an earlier version that used
adjacency_tensor_padded_by_type()/edge_feature_tensor_padded(), which
build one dense (T,N,N) tensor per edge type -- fine at the single-day
scale this bridge was first verified at (N~788, T~10), but catastrophic
at real multi-month scale, where N (every node ever seen across the
whole requested range) runs into the tens of thousands and T into the
thousands: (T,N,N) at that scale is terabytes, not gigabytes. Confirmed
directly, not assumed -- a real 5-month training run OOM-killed the
machine running it. Every consumer (TwoChannelSpatialAttention,
MeceOutputHead, LadderOutputHead) only ever used a snapshot's small
active-node submatrix anyway, so the dense (N,N) padding was always
wasted memory; this bridge now hands out stg/core.py's
sparse_edges_by_type() export instead, whose memory is O(total real
edges) -- this project's own data shows that's hundreds per snapshot,
regardless of how large N gets.

Deliberately keeps torch as a SOFT dependency, imported only inside
to_torch_bundle() -- the rest of stg_infra (graph construction) does not
require torch just to build graphs, only this conversion step does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from stg.core import SpatioTemporalGraph


@dataclass
class SparseSnapshotEdgesT:
    """Torch counterpart of stg.core.SparseSnapshotEdges -- same fields,
    as tensors. edge_index is (2, E) long, GLOBAL node-index space
    (matching TorchGraphBundle.node_ids' ordering); weight is (E,);
    features is (E, Fe) or None if this edge type never carries any."""

    edge_index: "torch.Tensor"
    weight: "torch.Tensor"
    features: Optional["torch.Tensor"]


@dataclass
class TorchGraphBundle:
    """Everything a model needs, all sharing one consistent node ordering
    (``node_ids``) and one consistent time ordering (``timestamps``).

    ``features``/``mask`` stay dense -- (T,N,F) with F~10 is small even
    at large N, nothing like the (T,N,N) adjacency problem this bridge
    was rewritten to avoid (see module docstring). ``adjacency_by_type``
    is SPARSE per snapshot: edge_type -> list of length T, each entry a
    SparseSnapshotEdgesT for that one snapshot. There is no longer a
    combined all-types ``adjacency`` field or a separate
    ``edge_features``/``edge_has_features`` -- features now travel WITH
    their edge type's per-snapshot entry (``.features``, None when that
    type never carries any), since keeping them as separate top-level
    dense tensors would reintroduce the exact (T,N,N) problem this
    rewrite removes.
    """

    features: "torch.Tensor"                              # (T, N, F) float, NaN replaced with 0
    mask: "torch.Tensor"                                   # (T, N) bool -- True where node was actually observed
    adjacency_by_type: Dict[str, List[SparseSnapshotEdgesT]]  # edge_type -> length-T list of sparse snapshots
    node_ids: List
    timestamps: List


def to_torch_bundle(stg: SpatioTemporalGraph, dtype=None) -> TorchGraphBundle:
    """Build a ``TorchGraphBundle`` from a ``SpatioTemporalGraph``.

    NaN handling: ``feature_tensor_padded()`` uses NaN to mark an absent
    node's feature values. NaN is replaced with 0.0 here BEFORE conversion
    to torch, since an unmasked NaN can silently poison gradients if it
    ever reaches a matmul/attention computation. After this conversion,
    the NaN information no longer exists in ``features`` -- always gate on
    ``mask`` to know which entries are real, never by checking for NaN in
    the tensor itself.
    """
    import torch  # lazy import -- see module docstring

    dtype = dtype or torch.float32

    feat_np, mask_np = stg.feature_tensor_padded()
    feat_np = np.nan_to_num(feat_np, nan=0.0)
    features = torch.from_numpy(feat_np).to(dtype)
    mask = torch.from_numpy(mask_np)  # keep as bool, not cast to dtype

    sparse_by_type_np = stg.sparse_edges_by_type()
    adjacency_by_type: Dict[str, List[SparseSnapshotEdgesT]] = {}
    for et, snaps in sparse_by_type_np.items():
        converted = []
        for s in snaps:
            edge_index = torch.from_numpy(s.edge_index)  # already int64
            weight = torch.from_numpy(s.weight).to(dtype)
            feats = torch.from_numpy(s.features).to(dtype) if s.features is not None else None
            converted.append(SparseSnapshotEdgesT(edge_index=edge_index, weight=weight, features=feats))
        adjacency_by_type[et] = converted

    return TorchGraphBundle(
        features=features,
        mask=mask,
        adjacency_by_type=adjacency_by_type,
        node_ids=stg.all_node_ids(),
        timestamps=stg.timestamps,
    )