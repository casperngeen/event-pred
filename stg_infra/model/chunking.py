"""
model/chunking.py

Turns one long TorchGraphBundle (T spanning the full training range) into
a list of bounded, split-safe TIME CHUNKS that the training loop actually
feeds through the model.

WHY THIS EXISTS: temporal_attention.py computes a dense (N, H, T, T)
attention tensor per call. That's cheap for a bounded window (a few
hundred snapshots) but infeasible for a full multi-month range (~7200
snapshots at 2h resolution -- see that module's docstring). The fix used
throughout this project for a scale problem has always been "restrict to
what's actually active/needed" (GATv2Conv over real edges only, per-
snapshot loops over ragged active-node sets) rather than a denser
representation with a bigger buffer -- chunking the time axis is the same
idea applied to the temporal layer.

TWO SEPARATE CONCERNS, DELIBERATELY KEPT SEPARATE:

1. Split boundaries (data_windows.split_for, on the MASTER calendar) --
   which month a snapshot's timestamp falls in determines whether it is
   train/val/test. This is a real, sharp boundary: a chunk must NEVER mix
   snapshots from two different splits, or training would see data from
   inside the nominal val/test window (leakage), just one layer removed
   from the leakage data_windows.py was already built to prevent at the
   raw-data level.

2. Chunk length (CHUNK_LEN) -- a compute/memory bound with no
   train/val/test meaning at all. Chunks are cut to this length ONLY
   after the split boundaries have already carved the timeline into
   train/val/test runs, so a chunk-length cut can never accidentally
   cross a split boundary; it can only ever cut a run of already-same-
   split snapshots into smaller same-split pieces.

TRADE-OFF THIS MAKES EXPLICIT, NOT SILENTLY: cutting the timeline into
independent chunks bounds each node's temporal receptive field to within
one chunk -- a node's representation at the start of a chunk has no
attention path back to the previous chunk's history. For 1-2 week chunks
on 2h-resolution Kalshi data this is an acceptable, documented limitation
(a market's own multi-day history is available; multi-month continuity is
not), not a bug -- the alternative (one dense pass over the full range) is
simply not computable, as the OOM/IndexError investigation in
temporal_attention.py's docstring found directly rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Sequence, Tuple

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.data_windows import split_for  # noqa: E402

from model.data_validation import apply_validity  # noqa: E402


def _month_str(ts: datetime) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


@dataclass
class TemporalChunk:
    """One bounded, single-split window of snapshot indices into a
    TorchGraphBundle. ``end`` is EXCLUSIVE, Python-slice style."""

    split: str    # "train" | "val" | "test"
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"TemporalChunk(split={self.split!r}, [{self.start}:{self.end}), len={self.length})"


def build_split_ranges(timestamps: Sequence[datetime],
                       break_at_month_boundaries: bool = True) -> List[Tuple[str, int, int]]:
    """Groups snapshot indices into maximal contiguous runs that share the
    same split label (train/val/test), per data_windows.split_for() on
    each snapshot's month.

    Returns a list of (split, start, end) with end EXCLUSIVE, covering
    every index in [0, len(timestamps)) exactly once, in order.

    ``break_at_month_boundaries`` (default True) additionally ends a run
    whenever the MONTH changes, even when the split label doesn't. This
    exists to serve MonthlyBundleStore (model/month_store.py): the store
    holds one month's data per file and keeps only ONE month resident at
    a time, which is what makes 12-14+ months of data fit in memory at
    all. A chunk that straddled a month boundary would force two months
    resident simultaneously and defeat that.

    The cost is small and bounded: with chunk_len=168 (~2 weeks) against
    months of ~310 snapshots, most chunks already sit inside one month,
    so this mainly adds one short remainder chunk per month rather than
    reshaping the schedule. It does mean no node's temporal receptive
    field spans a month boundary -- the same class of trade-off this
    module's docstring already documents for chunking in general, and a
    minor one here given most Kalshi markets live only 1-2 days.

    Set it False to restore the old behaviour (a single dense bundle
    covering the whole range, where cross-month chunks cost nothing
    because everything is resident anyway).
    """
    if not timestamps:
        return []

    months = [_month_str(ts) for ts in timestamps]
    labels = [split_for(m) for m in months]

    def _key(i: int):
        return (labels[i], months[i]) if break_at_month_boundaries else (labels[i],)

    ranges: List[Tuple[str, int, int]] = []
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or _key(i) != _key(run_start):
            ranges.append((labels[run_start], run_start, i))
            run_start = i
    return ranges


def chunk_ranges(
    ranges: List[Tuple[str, int, int]],
    chunk_len: int,
    min_chunk_len: int = 1,
) -> List[TemporalChunk]:
    """Cuts each (split, start, end) run into non-overlapping chunks of at
    most ``chunk_len`` snapshots, never crossing a run boundary (so never
    crossing a split boundary, by construction -- see module docstring).

    A trailing remainder shorter than ``min_chunk_len`` is dropped rather
    than kept as a near-empty chunk (a 2-snapshot "chunk" has almost
    nothing for temporal attention to do and mostly wastes a training
    step) -- UNLESS it is the only chunk in its run, in which case it is
    kept regardless of length so a short val/test run is never silently
    discarded entirely.
    """
    chunks: List[TemporalChunk] = []
    for split, start, end in ranges:
        run_len = end - start
        if run_len <= 0:
            continue
        if run_len <= chunk_len:
            chunks.append(TemporalChunk(split, start, end))
            continue
        pos = start
        while pos < end:
            stop = min(pos + chunk_len, end)
            remainder_after = end - stop
            if 0 < remainder_after < min_chunk_len:
                # fold the too-small trailing remainder into this chunk
                # instead of emitting it as its own tiny chunk
                stop = end
            chunks.append(TemporalChunk(split, pos, stop))
            pos = stop
    return chunks


def _remap_edges_to_chunk_local(edges_t, active_sorted: torch.Tensor):
    """Rewrites one snapshot's sparse edge struct from GLOBAL node-index
    space into CHUNK-LOCAL space (positions within ``active_sorted``).

    Safe unconditionally, for the same reason model/sparse_utils.py's
    to_local_edge_index is: an edge at snapshot t only ever connects two
    nodes that were both active AT snapshot t, and every node active at
    snapshot t is by definition a member of the chunk-wide active union
    ``active_sorted``. So every edge_index value is guaranteed present,
    and searchsorted returns its exact position rather than an insertion
    point. That guarantee is CHECKED below rather than trusted -- a
    silent off-by-one here would mis-wire the graph while still producing
    correctly-shaped, non-NaN output, which is precisely the class of bug
    this project has been bitten by repeatedly.
    """
    ei = edges_t.edge_index
    if ei.numel() == 0:
        return edges_t
    local = torch.searchsorted(active_sorted, ei)
    if not torch.equal(active_sorted[local], ei):
        raise AssertionError(
            "slice_bundle_chunk: an edge references a node that is not in this "
            "chunk's active-node union. This should be impossible (an edge only "
            "exists between nodes active in the same snapshot) -- it means either "
            "mask and adjacency disagree about which nodes are active, or the "
            "bundle's edge indices are not in the same global node space as "
            "bundle.mask/node_ids."
        )
    return replace(edges_t, edge_index=local)


def slice_bundle_chunk(bundle, chunk: TemporalChunk, validate: bool = True) -> dict:
    """Slices every T-indexed field of a TorchGraphBundle down to one
    chunk's [start:end) snapshots, AND compacts the node axis down to
    only the nodes actually active somewhere in that chunk.

    NODE-AXIS COMPACTION -- WHY THIS CHANGED. This function originally
    left the node axis as the full global padded universe, on the
    reasoning that "spatial_attention.py and the output heads already
    restrict to each snapshot's actually-active nodes internally, so
    keeping N global here costs nothing extra". Measured on real data,
    that turned out to be false in two ways:

    1. It is not free even for one month. The instrumented run showed
       TypeSpecificProjection alone adding ~671MB of RSS for a single
       chunk, because it runs two dense Linear layers across the FULL
       (T, N, F) grid -- N=14,983 nodes of which only ~7,791 appear
       anywhere in the chunk and only ~200-840 in any given snapshot.
       Every (T, N, D) tensor downstream (h0, h1, h2, and the residual
       clones inside both attention layers) pays the same padding tax.

    2. It gets structurally worse with range, which is the real problem.
       N is the union of every node EVER active across the whole build
       range, so it grows with every month added (~15k/month of
       short-lived Kalshi tickers). The number active within any one
       ~2-week chunk does NOT grow that way -- it is bounded by how many
       markets exist at once. So the padding fraction rises with total
       range: at 1 month N/n_active is ~2x, at 5 months it would be
       ~10x, spent entirely on nodes that are provably all-zero for the
       entire chunk.

    Compacting here fixes both at the one place that knows a chunk's
    node set, and needs no change in any layer downstream: every
    consumer already derives its own active sets from ``mask`` and from
    edge indices, both of which are remapped consistently below.

    INDEX SPACE -- READ THIS BEFORE USING THE RESULT. Node indices in
    the returned tensors (and therefore in anything derived from them,
    including model/inference.py's ``LegSignal.leg_idx``) are CHUNK-LOCAL,
    not global. Two keys are returned to make the mapping back explicit
    rather than something a caller has to reconstruct:
      - ``active_node_idx``: LongTensor (n_active,), chunk-local position
        -> GLOBAL bundle index. So ``bundle.node_ids[active_node_idx[i]]``
        is the real ticker for chunk-local node ``i``.
      - ``node_ids``: the already-resolved list, same ordering, so
        ``node_ids[i]`` is that ticker directly.

    Returns a plain dict (not a new dataclass type) with the same field
    names TwoChannelSpatialAttention / TemporalAttention / the output
    heads already expect, so callers can pass it straight through.
    """
    s, e = chunk.start, chunk.end

    features_chunk = bundle.features[s:e]
    mask_chunk = bundle.mask[s:e]
    adjacency_chunk = {et: adj[s:e] for et, adj in bundle.adjacency_by_type.items()}

    # Drop physically impossible observations BEFORE deciding which nodes
    # are active, so a node whose only readings in this chunk are corrupt
    # is excluded from the chunk entirely rather than carried as a node
    # with no valid data. See model/data_validation.py for the measured
    # rates (2.50% of prices out of range) and the per-field policy.
    stats = None
    if validate:
        features_chunk, mask_chunk, adjacency_chunk, stats = apply_validity(
            features_chunk, mask_chunk, adjacency_chunk,
        )

    active = mask_chunk.any(dim=0).nonzero(as_tuple=True)[0]        # (n_active,) global, sorted ascending

    out = {
        "features": features_chunk[:, active, :],                   # (T_chunk, n_active, F)
        "mask": mask_chunk[:, active],                              # (T_chunk, n_active)
        "adjacency_by_type": {
            et: [_remap_edges_to_chunk_local(snap, active) for snap in adj]
            for et, adj in adjacency_chunk.items()
        },
        "timestamps": bundle.timestamps[s:e],
        "active_node_idx": active,
        "node_ids": [bundle.node_ids[i] for i in active.tolist()] if getattr(bundle, "node_ids", None) else None,
        "validation_stats": stats,
    }
    return out


def describe_chunks(chunks: List[TemporalChunk]) -> str:
    """Human-readable one-line-per-split summary, for logging at the start
    of a training run (how many chunks, total snapshots, per split)."""
    lines = []
    for split in ("train", "val", "test"):
        this = [c for c in chunks if c.split == split]
        total = sum(c.length for c in this)
        lines.append(f"  {split}: {len(this)} chunk(s), {total} snapshot(s) total")
    return "\n".join(lines)