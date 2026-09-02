#!/usr/bin/env python
"""Rebuild every durable panel artifact from ``data/`` alone.

    venv/bin/python scripts/build_panels.py [--min-events 5] [--cadence event]

Writes to ``artifacts/panels/``:

    surprise_panel.parquet   one row per usable macro event
    node_panel_<cadence>.parquet   (series, date) belief-state features
    universe.parquet         canonical series, event counts, in/out of universe
    MANIFEST.md              row counts, universe N, build timestamp

In-sample only — every loader routes the wall through ``stg.splits``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel import build_node_panel, build_surprise_panel, event_counts, universe
from stg.panel._io import load_markets, scan_trades
from stg.panel.surprise import usable_triggers
from stg.splits import OOS_START, assert_no_oos

OUT = Path("artifacts/panels")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-events", type=int, default=5)
    ap.add_argument("--cadence", default="event", choices=["event", "daily", "weekly"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)
    names = universe(args.min_events, mk)

    counts = event_counts(mk)
    counts.write_parquet(OUT / "universe.parquet")

    print(f"universe (min_events={args.min_events}): N={len(names)}")
    surprise = build_surprise_panel(names, markets=mk, trades=tr)
    assert_no_oos(surprise, time_col="close_time")
    surprise.write_parquet(OUT / "surprise_panel.parquet")
    triggers = usable_triggers(surprise, 10)

    node = build_node_panel(names, cadence=args.cadence, min_events=args.min_events,
                            markets=mk)
    assert_no_oos(node, time_col="date")
    node.write_parquet(OUT / f"node_panel_{args.cadence}.parquet")

    lines = [
        "# Panel artifacts",
        "",
        f"built: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"OOS wall: {OOS_START.date()} (in-sample only)",
        f"min_events: {args.min_events}  ->  universe N = {len(names)}",
        f"cadence: {args.cadence}",
        "",
        "| panel | rows | notes |",
        "|---|---|---|",
        f"| surprise_panel | {surprise.height} | "
        f"{surprise['series'].n_unique()} series; usable triggers (n>=10): "
        f"{', '.join(triggers)} |",
        f"| node_panel_{args.cadence} | {node.height} | "
        f"{node['series'].n_unique()} series, {node['date'].n_unique()} dates |",
        "",
        "## events per series",
        "",
        "| series | events | in universe |",
        "|---|---|---|",
    ]
    for r in counts.filter(pl.col("in_universe")).iter_rows(named=True):
        lines.append(f"| {r['canon']} | {r['n_events']} | yes |")
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/  (surprise {surprise.height} rows, node {node.height} rows)")


if __name__ == "__main__":
    main()
