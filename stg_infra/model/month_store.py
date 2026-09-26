"""
model/month_store.py

Holds a multi-month training range as ONE CACHED BUNDLE PER MONTH on
disk, loading at most one month into RAM at a time, instead of as a
single global TorchGraphBundle covering the whole range.

WHY THIS EXISTS -- THE ARITHMETIC, NOT A PREFERENCE. A TorchGraphBundle
stores features densely as (T, N, F), where N is the union of every node
active ANYWHERE in the range. Kalshi markets are short-lived, so N grows
roughly linearly with calendar range (~15k new tickers per month), while
the number active in any one snapshot stays flat at ~200-850. Both axes
of the dense tensor therefore grow with range, so its size grows with
the SQUARE of the range while the amount of real data in it grows only
linearly. Measured against this project's own real one-month figures
(T=310, N=14,983, F=10, ~700 active per snapshot):

    range      dense (T,N,F)      actual observations     padding
    1 month          0.2 GB                   8.7 MB          20x
    5 months         4.4 GB                  43.4 MB         102x
    14 months       34.6 GB                 121.5 MB        285x

A 14-month run -- the stated target for this project -- would need a
34.6GB tensor to carry 121MB of observations. That is not a tuning
problem with a threshold to raise; it is the wrong data structure at
that range, and it is the fourth time this project has hit the same
underlying mistake of sizing something to the GLOBAL node universe
rather than to what is actually active (see model/temporal_attention.py's
docstring for the previous three).

WHAT THIS CHANGES. A month is a natural shard here because the
train/val/test split is already defined per-month on the master calendar
(data_windows.py), so a month never straddles a split. Each month is
built once, cached to its own file, and loaded only when a chunk from
that month is actually being trained on. Peak memory becomes a function
of ONE month's size (~186MB dense at F=10) plus one chunk's compacted
tensors (~54MB), independent of how many months the run covers -- so 14
months costs the same resident memory as 1, and adding months costs disk
rather than RAM.

Chunks are kept inside a single month by build_split_ranges(...,
break_at_month_boundaries=True), so materializing a chunk never needs
two months at once. See that function's docstring for the (small,
bounded) modelling cost of that constraint.

WHAT THIS DOES NOT CHANGE. materialize_chunk() returns exactly the dict
that model/chunking.py's slice_bundle_chunk() returns -- same keys, same
shapes, same chunk-local node index space, same active_node_idx/node_ids
mapping back to real tickers. So model/train.py's forward/loss path is
identical whether it is fed a store or a single in-memory bundle, and
the two are verified to produce identical chunks rather than assumed to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from model.chunking import TemporalChunk, slice_bundle_chunk


class MonthlyBundleStore:
    """Lazily-loaded, one-month-resident view over a multi-month range.

    Exposes the same three attributes model/train.py reads off a
    TorchGraphBundle for planning purposes -- ``timestamps`` (the full
    concatenated timeline, needed to compute splits and chunks) and
    ``features``' trailing dimension via ``feature_width`` -- without
    ever concatenating the months' feature tensors themselves.
    """

    def __init__(self, month_paths: "Dict[str, Path]", verbose: bool = True):
        """``month_paths`` maps 'YYYY-MM' -> path of that month's cached
        TorchGraphBundle (as written by torch.save). Months are processed
        in sorted calendar order, which is also the order their snapshots
        occupy on the concatenated timeline."""
        if not month_paths:
            raise ValueError("MonthlyBundleStore needs at least one month")

        self.month_paths = {m: Path(p) for m, p in sorted(month_paths.items())}
        self.verbose = verbose

        # Cheap pass to learn each month's length and timestamps, so the
        # global timeline (and therefore splits/chunks) can be planned
        # without holding more than one month at a time.
        self.timestamps: List = []
        self._month_spans: List[tuple] = []  # (month, global_start, global_end)
        self.feature_width: Optional[int] = None
        self._node_id_counts: Dict[str, int] = {}
        # Which edge types each month actually carries. Collected during
        # the setup pass that already loads every month, so it costs
        # nothing, and checked by model/run_guards.check_month_schema.
        #
        # WHY: a feature-width mismatch is rejected below, but an EDGE
        # TYPE mismatch was not checked at all -- and it is the more
        # dangerous of the two. A month built before an edge builder
        # existed simply has no entries under that key, so the spatial
        # channel is absent for part of the range while the run reports
        # one number averaged over a with-graph and a without-graph
        # regime. Nothing errors; the result is just quietly a mixture.
        # That risk arrives the moment months are added to the range.
        self.adjacency_keys: Dict[str, List[str]] = {}

        for month, path in self.month_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"cached bundle for {month} not found at {path}")
            bundle = torch.load(path, weights_only=False)
            start = len(self.timestamps)
            self.timestamps.extend(bundle.timestamps)
            end = len(self.timestamps)
            self._month_spans.append((month, start, end))
            if self.feature_width is None:
                self.feature_width = int(bundle.features.shape[-1])
            elif int(bundle.features.shape[-1]) != self.feature_width:
                raise ValueError(
                    f"month {month} has feature width {bundle.features.shape[-1]}, but an earlier "
                    f"month has {self.feature_width} -- months must share one feature layout "
                    f"(COMBINED_N_FEATURES in stg/nodes/kalshi.py). Rebuild the caches after any "
                    f"change to the node feature definitions."
                )
            self._node_id_counts[month] = len(bundle.node_ids)
            self.adjacency_keys[month] = sorted(bundle.adjacency_by_type.keys())
            del bundle  # one month resident at a time, even during setup

        # Single-slot cache: consecutive chunks are usually from the same
        # month (chunks never straddle months), so this avoids reloading
        # the same file once per chunk while still never holding two.
        self._loaded_month: Optional[str] = None
        self._loaded_bundle = None

    # -- introspection -----------------------------------------------

    @property
    def months(self) -> List[str]:
        return [m for m, _, _ in self._month_spans]

    def summary(self) -> str:
        lines = [f"MonthlyBundleStore: {len(self._month_spans)} month(s), "
                 f"{len(self.timestamps)} snapshot(s) total, F={self.feature_width}"]
        for month, start, end in self._month_spans:
            lines.append(f"  {month}: snapshots [{start}:{end}) "
                         f"({end - start})  N_month={self._node_id_counts[month]}"
                         f"  edges={self.adjacency_keys.get(month, [])}")
        return "\n".join(lines)

    # -- chunk materialization ---------------------------------------

    def _month_for(self, chunk: TemporalChunk) -> tuple:
        """Finds the month containing this chunk, and asserts the chunk
        lies entirely inside it (guaranteed by build_split_ranges with
        break_at_month_boundaries=True -- checked, not trusted, since a
        chunk silently spanning months would read the wrong snapshots)."""
        for month, start, end in self._month_spans:
            if chunk.start >= start and chunk.end <= end:
                return month, start, end
        raise ValueError(
            f"{chunk} does not lie within a single month. MonthlyBundleStore requires "
            f"month-bounded chunks -- call build_split_ranges(..., "
            f"break_at_month_boundaries=True) (the default) before chunk_ranges()."
        )

    def month_of(self, chunk: TemporalChunk) -> str:
        """Which month this chunk belongs to. model/train.py uses this to
        group an epoch's shuffled chunk order BY MONTH: chunks are
        shuffled for optimisation reasons, but a naively shuffled order
        would bounce between months and reload a ~186MB file on every
        single chunk. Grouping keeps each month loaded once per epoch
        while still randomising both the month order and the order within
        each month."""
        return self._month_for(chunk)[0]

    def _ensure_loaded(self, month: str):
        if self._loaded_month == month:
            return self._loaded_bundle
        # Drop the old one BEFORE loading the new one, so peak stays at
        # one month rather than briefly two.
        self._loaded_bundle = None
        self._loaded_month = None
        if self.verbose:
            print(f"    [store] loading month {month} ...", flush=True)
        self._loaded_bundle = torch.load(self.month_paths[month], weights_only=False)
        self._loaded_month = month
        return self._loaded_bundle

    def materialize_chunk(self, chunk: TemporalChunk) -> dict:
        """Returns the same dict slice_bundle_chunk() returns, for a
        chunk addressed in GLOBAL timeline coordinates."""
        month, month_start, _ = self._month_for(chunk)
        bundle = self._ensure_loaded(month)
        local = TemporalChunk(chunk.split, chunk.start - month_start, chunk.end - month_start)
        return slice_bundle_chunk(bundle, local)

    def release(self):
        """Drops the resident month. Worth calling between epochs' train
        and val passes if they sit in different months and memory is
        tight; otherwise the single-slot cache handles itself."""
        self._loaded_bundle = None
        self._loaded_month = None


def build_month_paths(months: Sequence[str], cache_dir: Path) -> Dict[str, Path]:
    """Canonical per-month cache filenames, so the runner and the store
    agree on where a month lives without passing paths around."""
    return {m: Path(cache_dir) / f"bundle_month_{m}.pt" for m in months}