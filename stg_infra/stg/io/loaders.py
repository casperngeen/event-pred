"""I/O utilities — pure Polars with lazy scan and native Parquet reader."""

from __future__ import annotations

import glob
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import polars as pl

from stg.core import SpatioTemporalGraph

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load large multi-file datasets using Polars lazy evaluation.

    Supports Parquet, CSV, TSV, and NDJSON.
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        file_format: Optional[str] = None,
        sort_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> None:
        self.patterns = [paths] if isinstance(paths, str) else paths
        self.file_format = file_format
        self.sort_by = sort_by
        self.filters = filters or {}
        self.columns = columns

    def _resolve_files(self) -> List[str]:
        files: List[str] = []
        for pattern in self.patterns:
            matched = sorted(glob.glob(pattern, recursive=True))
            if not matched:
                logger.warning("Pattern matched no files: %s", pattern)
            files.extend(matched)
        files = sorted(set(files))
        logger.info("Resolved %d files from %d patterns", len(files), len(self.patterns))
        return files

    def _detect_format(self, path: str) -> str:
        ext = Path(path).suffix.lower().lstrip(".")
        return {"parquet": "parquet", "pq": "parquet", "tsv": "tsv",
                "ndjson": "ndjson", "jsonl": "ndjson"}.get(ext, "csv")

    def _build_filter_expr(self) -> Optional[pl.Expr]:
        if not self.filters:
            return None
        exprs = []
        for col, val in self.filters.items():
            if isinstance(val, (list, tuple, set)):
                exprs.append(pl.col(col).is_in(list(val)))
            else:
                exprs.append(pl.col(col) == val)
        result = exprs[0]
        for e in exprs[1:]:
            result = result & e
        return result

    def _scan_single(self, path: str, fmt: str) -> pl.LazyFrame:
        if fmt == "parquet":
            lf = pl.scan_parquet(path)
        elif fmt == "tsv":
            lf = pl.scan_csv(path, separator="\t")
        elif fmt == "ndjson":
            lf = pl.scan_ndjson(path)
        else:
            lf = pl.scan_csv(path)
        if self.columns:
            avail = lf.collect_schema().names()
            use = [c for c in self.columns if c in avail]
            if use:
                lf = lf.select(use)
        filt = self._build_filter_expr()
        if filt is not None:
            lf = lf.filter(filt)
        return lf

    def scan(self) -> pl.LazyFrame:
        """Lazy frame over all files — predicates/projections pushed to I/O."""
        files = self._resolve_files()
        if not files:
            return pl.LazyFrame()
        fmt = self.file_format
        lfs = [self._scan_single(f, fmt or self._detect_format(f)) for f in files]
        combined = pl.concat(lfs)
        if self.sort_by:
            combined = combined.sort(self.sort_by)
        return combined

    def load(self) -> pl.DataFrame:
        """Eagerly collect — all lazy optimisations still apply."""
        df = self.scan().collect()
        logger.info("Loaded %d rows × %d cols", df.height, df.width)
        return df

    def load_iter(self, files_per_batch: int = 10) -> Iterator[pl.DataFrame]:
        """Yield DataFrames in batches for out-of-core processing."""
        files = self._resolve_files()
        if not files:
            return
        fmt = self.file_format
        for start in range(0, len(files), files_per_batch):
            batch = files[start:start + files_per_batch]
            lfs = [self._scan_single(f, fmt or self._detect_format(f)) for f in batch]
            chunk = pl.concat(lfs).collect()
            if self.sort_by and self.sort_by in chunk.columns:
                chunk = chunk.sort(self.sort_by)
            yield chunk

    def metadata(self) -> Dict[str, Any]:
        files = self._resolve_files()
        if not files:
            return {"num_files": 0}
        fmt = self.file_format or self._detect_format(files[0])
        sample_lf = self._scan_single(files[0], fmt)
        schema = sample_lf.collect_schema()
        sample = sample_lf.head(5).collect()
        total_size = sum(os.path.getsize(f) for f in files)
        return {
            "num_files": len(files),
            "total_size_mb": round(total_size / 1e6, 2),
            "columns": schema.names(),
            "dtypes": {name: str(dtype) for name, dtype in schema.items()},
            "sample_rows": sample.height,
        }


class IncrementalGraphBuilder:
    """Build a SpatioTemporalGraph incrementally from chunked data."""

    def __init__(self, builder: Any, merge_strategy: str = "append") -> None:
        self._builder = builder
        self._merge = merge_strategy
        self._stg = SpatioTemporalGraph()

    def ingest(self, chunk: pl.DataFrame, auxiliary: Optional[Dict[str, pl.DataFrame]] = None) -> None:
        partial = self._builder.build(chunk, auxiliary=auxiliary)
        if self._merge == "merge":
            existing_ts = {s.timestamp: i for i, s in enumerate(self._stg.snapshots)}
            for snap in partial:
                if snap.timestamp in existing_ts:
                    base = self._stg.snapshots[existing_ts[snap.timestamp]]
                    for nid in snap.node_ids:
                        ns = snap.get_node(nid)
                        if ns and nid not in base.node_ids:
                            base.add_node(ns)
                    for u, v in snap.edges:
                        es = snap.get_edge(u, v)
                        if es and not base._graph.has_edge(u, v):
                            base.add_edge(es)
                else:
                    self._stg.add_snapshot(snap)
                    existing_ts[snap.timestamp] = len(self._stg) - 1
        else:
            for snap in partial:
                self._stg.add_snapshot(snap)

    def finalise(self) -> SpatioTemporalGraph:
        self._stg.sort()
        return self._stg


class GraphSerialiser:
    """Save / load SpatioTemporalGraph."""

    @staticmethod
    def to_pickle(stg: SpatioTemporalGraph, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(stg, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved STG to %s", path)

    @staticmethod
    def from_pickle(path: str) -> SpatioTemporalGraph:
        with open(path, "rb") as f:
            stg = pickle.load(f)
        logger.info("Loaded STG from %s (%d snapshots)", path, len(stg))
        return stg

    @staticmethod
    def to_numpy(stg: SpatioTemporalGraph, directory: str) -> Dict[str, str]:
        os.makedirs(directory, exist_ok=True)
        paths: Dict[str, str] = {}
        try:
            p = os.path.join(directory, "features.npy")
            np.save(p, stg.feature_tensor()); paths["features"] = p
        except Exception as e:
            logger.warning("Feature tensor: %s", e)
        try:
            p = os.path.join(directory, "adjacency.npy")
            np.save(p, stg.adjacency_tensor()); paths["adjacency"] = p
        except Exception as e:
            logger.warning("Adjacency tensor: %s", e)
        p = os.path.join(directory, "timestamps.json")
        with open(p, "w") as f:
            json.dump([str(t) for t in stg.timestamps], f)
        paths["timestamps"] = p
        p = os.path.join(directory, "node_ids.json")
        with open(p, "w") as f:
            json.dump([[str(n) for n in s.node_ids] for s in stg], f)
        paths["node_ids"] = p
        logger.info("Exported %d arrays to %s", len(paths), directory)
        return paths

    @staticmethod
    def to_edge_list(stg: SpatioTemporalGraph, path: str) -> None:
        rows = {"timestamp": [], "source": [], "target": [], "weight": []}  # type: ignore[var-annotated]
        for snap in stg:
            for u, v in snap.edges:
                e = snap.get_edge(u, v)
                rows["timestamp"].append(str(snap.timestamp))
                rows["source"].append(str(u))
                rows["target"].append(str(v))
                rows["weight"].append(e.weight if e else 1.0)
        pl.DataFrame(rows).write_csv(path)
        logger.info("Exported %d edges to %s", len(rows["source"]), path)

    @staticmethod
    def to_edge_list_parquet(stg: SpatioTemporalGraph, path: str) -> None:
        rows = {"timestamp": [], "source": [], "target": [], "weight": []}  # type: ignore[var-annotated]
        for snap in stg:
            for u, v in snap.edges:
                e = snap.get_edge(u, v)
                rows["timestamp"].append(str(snap.timestamp))
                rows["source"].append(str(u))
                rows["target"].append(str(v))
                rows["weight"].append(e.weight if e else 1.0)
        pl.DataFrame(rows).write_parquet(path)
        logger.info("Exported %d edges to %s", len(rows["source"]), path)
