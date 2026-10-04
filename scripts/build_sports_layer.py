"""Build the sports as-of data layer (stg.sports) from what is on disk.

    venv/bin/python scripts/build_sports_layer.py            # everything (~2 min on 60M trades)
    venv/bin/python scripts/build_sports_layer.py --only KXMLBGAME KXMLBSPREAD
    venv/bin/python scripts/build_sports_layer.py --tables   # contracts / outcomes / edges only

Writes data/sports/derived/ (git-ignored):
    quotes/<series>.parquet    per-market bid/ask series by known_at (stg.sports.prices)
    contracts.parquet          what each market is (known at listing)
    outcomes.parquet           outcome table with labels (labels: settlement-time only)
    edges.parquet              the edge table (stg.sports.edges)
    manifest.json              build time, rows per series, whether candles were used

Rerun after each trades / candles pull: a series is rebuilt when any of its
trade or candle files is newer than its quote file. Never reads the holdout.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stg_infra"))

from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.edges import edge_table                    # noqa: E402
from stg.sports.loaders import derived_dir, safe_path, series_with   # noqa: E402
from stg.sports.outcomes import outcome_table              # noqa: E402
from stg.sports.prices import quotes_path, write_quotes    # noqa: E402


def stale(series: str) -> bool:
    out = quotes_path(series)
    if not out.exists():
        return True
    src = [*safe_path("trades", series).glob("*.parquet"), *safe_path("candles", series).glob("*.parquet")]
    return any(f.stat().st_mtime > out.stat().st_mtime for f in src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--force", action="store_true", help="rebuild every quote file")
    ap.add_argument("--tables", action="store_true", help="contracts / outcomes / edges only")
    a = ap.parse_args()
    d = derived_dir()
    d.mkdir(parents=True, exist_ok=True)
    man_p = d / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"series": {}}

    if not a.tables:
        series = a.only or sorted(set(series_with("trades")) | set(series_with("candles")))
        todo = [s for s in series if a.force or stale(s)]
        print(f"quotes: {len(todo)} of {len(series)} series to (re)build", flush=True)
        t0 = time.time()
        for i, s in enumerate(todo, 1):
            n = write_quotes(s)
            man["series"][s] = {"rows": n, "candles": bool(series_with("candles") and s in series_with("candles")),
                                "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            if i % 25 == 0 or i == len(todo) or n > 2_000_000:
                print(f"  [{i}/{len(todo)}] {s}: {n:,} rows  ({time.time() - t0:.0f} s)", flush=True)

    data = SportsData()
    data.contracts.write_parquet(d / "contracts.parquet")
    oc = outcome_table(data.contracts)
    oc.write_parquet(d / "outcomes.parquet")
    e = edge_table(data)
    e.write_parquet(d / "edges.parquet")
    man["tables"] = {"contracts": data.contracts.height, "outcomes": oc["outcome_id"].n_unique(),
                     "edges": e.height, "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    man_p.write_text(json.dumps(man, indent=1, sort_keys=True))
    print(f"contracts {data.contracts.height:,}, outcomes {oc['outcome_id'].n_unique():,}, edges {e.height:,}")


if __name__ == "__main__":
    main()
