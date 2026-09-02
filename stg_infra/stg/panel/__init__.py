"""Durable panel layer: reconstructs every analysis panel from ``data/`` alone.

    from stg.panel import (
        universe, build_surprise_panel, build_node_panel, snapshot_dates,
    )
"""

from stg.panel.registry import (
    SPECS, ALIASES, canonical, universe, event_counts,
    same_release_groups, same_release_pairs, is_same_release,
    ticker_prefixes, series_filter_expr,
)
from stg.panel.surprise import build_surprise_panel, usable_triggers
from stg.panel.nodes import build_node_panel
from stg.panel.targets import representative_tickers, response_panel
from stg.panel.snapshots import snapshot_dates, macro_resolution_dates, weekly_dates

__all__ = [
    "SPECS", "ALIASES", "canonical", "universe", "event_counts",
    "same_release_groups", "same_release_pairs", "is_same_release",
    "ticker_prefixes", "series_filter_expr",
    "build_surprise_panel", "usable_triggers",
    "build_node_panel",
    "representative_tickers", "response_panel",
    "snapshot_dates", "macro_resolution_dates", "weekly_dates",
]
