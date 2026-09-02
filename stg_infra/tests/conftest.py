"""Shared fixtures. Data-backed tests are skipped when the archive is absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ARCHIVE = Path("data/markets")


def _has_archive() -> bool:
    return _ARCHIVE.exists() and any(_ARCHIVE.glob("*.parquet"))


requires_archive = pytest.mark.skipif(
    not _has_archive(),
    reason="Kalshi archive not found under ./data — run from event-pred/",
)


@pytest.fixture(scope="session")
def markets():
    from stg.panel._io import load_markets
    return load_markets(is_only=True)


@pytest.fixture(scope="session")
def surprise_panel():
    import polars as pl
    p = Path("artifacts/panels/surprise_panel.parquet")
    if not p.exists():
        pytest.skip("surprise panel not built — run scripts/build_panels.py")
    return pl.read_parquet(p)
