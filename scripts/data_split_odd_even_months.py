"""Odd/Even Month Splitter"""

import glob
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

import polars as pl

logger = logging.getLogger(__name__)


class SplitByOddEvenMonths:
    """Split large multi-file datasets into train/test sets by odd/even months.

    - Train = odd months
    - Test  = even months
    - Supports Parquet, CSV, TSV, NDJSON
    """

    def __init__(
        self,
        paths: Union[str, List[str]],
        file_format: Optional[str] = None,
        time_col: str = "created_time",
        batch_size: int = 10,
    ) -> None:
        self.patterns = [paths] if isinstance(paths, str) else paths
        self.file_format = file_format
        self.time_col = time_col
        self.batch_size = batch_size

    def _resolve_files(self) -> List[str]:
        files: List[str] = []
        for pattern in self.patterns:
            matched = sorted(glob.glob(pattern, recursive=True))
            if not matched:
                logger.warning("Pattern matched no files: %s", pattern)
            files.extend(matched)
        return sorted(set(files))

    def _detect_format(self, path: str) -> str:
        ext = Path(path).suffix.lower().lstrip(".")
        return {
            "parquet": "parquet",
            "pq": "parquet",
            "tsv": "tsv",
            "ndjson": "ndjson",
            "jsonl": "ndjson",
        }.get(ext, "csv")

    def _scan_single(self, path: str, fmt: str) -> pl.LazyFrame:
        if fmt == "parquet":
            return pl.scan_parquet(path)
        elif fmt == "tsv":
            return pl.scan_csv(path, separator="\t")
        elif fmt == "ndjson":
            return pl.scan_ndjson(path)
        else:
            return pl.scan_csv(path)

    def stream_split(self) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """Yield (train_chunk, test_chunk) pairs batch by batch — memory safe."""
        files = self._resolve_files()
        if not files:
            return
        fmt = self.file_format
        for start in range(0, len(files), self.batch_size):
            batch = files[start:start + self.batch_size]
            lfs = [self._scan_single(f, fmt or self._detect_format(f)) for f in batch]
            chunk = pl.concat(lfs).collect()

            # Normalize datetime
            chunk = chunk.with_columns(pl.col(self.time_col).dt.replace_time_zone(None))
            chunk = chunk.drop_nulls(subset=[self.time_col])
            chunk = chunk.with_columns(pl.col(self.time_col).dt.month().alias("month"))

            train_chunk = chunk.filter(pl.col("month") % 2 == 1)
            test_chunk = chunk.filter(pl.col("month") % 2 == 0)

            yield train_chunk, test_chunk

    def collect_split(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Collect all batches into full train/test DataFrames (use only if dataset fits in memory)."""
        train_parts, test_parts = [], []
        for train_chunk, test_chunk in self.stream_split():
            train_parts.append(train_chunk)
            test_parts.append(test_chunk)
        train_df = pl.concat(train_parts) if train_parts else pl.DataFrame()
        test_df = pl.concat(test_parts) if test_parts else pl.DataFrame()
        return train_df, test_df
