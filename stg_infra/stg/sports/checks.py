"""Leakage checks for the sports data layer (reports/sports_research_plan.md §4).

    assert_asof(frame)          no ``*_known_at`` column is after its row's snapshot
    truncation_test(build, C)   build to C and from the full data: samples whose
                                label ended by C must be identical
    final_column_reads(path)    names of the listing's final fields a source
                                file mentions (feature code must mention none)
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Callable

import polars as pl

from stg.sports.asof import SportsData
from stg.sports.loaders import FINAL_COLS


def assert_asof(frame: pl.DataFrame, t_col: str = "t") -> None:
    cols = [c for c in frame.columns if c.endswith("known_at") and c != t_col]
    bad = {c: frame.filter(pl.col(c) > pl.col(t_col)).height for c in cols}
    bad = {c: n for c, n in bad.items() if n}
    if bad:
        raise AssertionError(f"features known after their snapshot ({t_col}): {bad}")


def truncation_test(build: Callable[[SportsData], pl.DataFrame], cutoff: dt.datetime, key: list[str],
                    label_end: str = "label_end", full: SportsData | None = None) -> dict:
    """Build with ``SportsData(cutoff)`` and with the full data; compare the samples
    whose label ended at or before the cutoff. Raises on any difference."""
    c = pl.lit(cutoff, dtype=pl.Datetime("us", "UTC"))
    a = build(SportsData(cutoff=cutoff)).filter(pl.col(label_end) <= c).sort(key)
    b = build(full or SportsData()).filter(pl.col(label_end) <= c).sort(key)
    if a.height != b.height:
        ka, kb = a.select(key), b.select(key)
        raise AssertionError(f"truncation test: {a.height} samples built to {cutoff} vs {b.height} from the "
                             f"full data; only-full e.g. {kb.join(ka, on=key, how='anti').head(3).rows()}, "
                             f"only-truncated e.g. {ka.join(kb, on=key, how='anti').head(3).rows()}")
    diff = [col for col in a.columns if not a[col].equals(b[col])]
    if diff:
        raise AssertionError(f"truncation test: columns differ for samples ended by {cutoff}: {diff}")
    return {"samples": a.height, "cutoff": cutoff}


def final_column_reads(path: str | Path) -> list[str]:
    src = Path(path).read_text()
    code = re.sub(r'"""[\s\S]*?"""', "", src)                  # docstrings may name them
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    return sorted({c for c in FINAL_COLS if re.search(rf"[\"']{c}[\"']", code)})
