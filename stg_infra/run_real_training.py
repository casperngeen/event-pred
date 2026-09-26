"""
run_real_training.py

The actual entry point for "get real Kalshi data + the training loop
running together" -- wires build_combined_stgat_graph.py's real-data
loading pattern (same even/odd parquet convention already used by
mece_sum_to_one_check.py, pairwise_monotonicity_taker_side_check_v2.py,
etc.) into stg.bridge.to_torch_bundle() and model.train.run_training().

Run from the repo root (the directory that CONTAINS stg_infra/ and
data/), or adjust _REPO_ROOT below:

    python stg_infra/run_real_training.py

WHY THIS LOADS THE FULL MONTH RANGE IN ONE PASS, NOT MONTH-BY-MONTH:
KalshiTickerNodes and KalshiMeceBasketNodes are both STATEFUL --
each carries a `_last_known_*` cache across snapshots so a leg/basket
that goes quiet still carries forward correctly (see their docstrings in
stg/nodes/kalshi.py). That cache lives on the node-builder OBJECT, which
build_combined_graph() constructs exactly ONCE and hands to GraphBuilder.
Calling build_combined_graph() separately per month would silently
RESET that cache at every month boundary -- a real correctness bug, not
just slower. So this script collects trades/markets for the WHOLE
requested month range into one DataFrame and makes ONE
build_combined_graph() call, exactly mirroring how every other real-data
script in this project already treats this window (see
pairwise_monotonicity_taker_side_check_v2.py's load_trades_lazy() for the
same one-pass-over-the-whole-window reasoning, there for a different
statefulness reason -- pair statistics needing every month present).

MEMORY, READ BEFORE RUNNING AT THE FULL RANGE: the raw trades table is
one row per executed trade ACROSS THE WHOLE EXCHANGE, not just the
tickers this project cares about -- pairwise_monotonicity_taker_side_check_v2.py's
own docstring already flags this table as potentially enormous at just
20 ladder months. Collecting MASTER_MONTHS (~20 months, financials +
crypto + weather combined) in one eager .collect() may not fit in
memory on a laptop. Start with PILOT_MONTHS below (a few months) to
confirm the whole pipeline runs end to end before committing to the full
range -- shrinking the DATE RANGE is safe (it just trains on less data),
unlike trying to chunk the SAME range across multiple builder calls
(which would corrupt the carry-forward caches, per above).
"""

import gc
import logging
import resource
import sys
import time
from pathlib import Path

import polars as pl
import torch

# GraphBuilder.build() (stg/builders/builder.py) reports its own progress --
# a %-complete line plus a nodes/edges/feats/post/labels timing breakdown --
# via logger.info() every ~10s, specifically so a long real-data build isn't
# silent. But NOTHING in this pipeline ever called logging.basicConfig(),
# and Python's root logger defaults to WARNING, so every one of those INFO
# lines was being silently swallowed -- a multi-hour build and a genuine
# hang looked IDENTICAL from the outside (both: no output, ever). Confirmed
# directly: this is the exact gap that made "my system died" impossible to
# tell apart from "it's just grinding through 1800 windows and would have
# finished eventually." This must be configured BEFORE build_combined_graph
# (and therefore builder.build()) is ever called.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))

from build_combined_stgat_graph import build_combined_graph
from stg.bridge import to_torch_bundle
from examples.data_windows import MASTER_MONTHS
from model.month_store import MonthlyBundleStore, build_month_paths
from model.train import TrainingConfig, run_training

MECE_LEG_PRICES_PATH = str(_REPO_ROOT / "mece_sum_to_one_leg_prices.parquet")
LADDER_PAIRS_PATH = str(_REPO_ROOT / "pairwise_monotonicity_taker_side_results_corrected.parquet")

# Start here, not at MASTER_MONTHS -- see the memory note above. Widen
# once a pilot run has confirmed the pipeline actually completes and
# checkpoints on your machine. Chosen to span all three of train/val/test
# per data_windows.py's TRAIN_END=2025-06 / VAL_END=2025-08 boundaries so
# even a pilot run exercises every split, not just train.
PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]


def _parity_path(kind: str, month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return str(_REPO_ROOT / f"data/{kind}/{kind}_kalshi_{parity}/{kind}_{month}.parquet")


def _month_bounds(months) -> tuple:
    """[first day of the earliest requested month, first day of the month
    AFTER the latest one) -- an exclusive upper bound. Used to sanity-check
    every ROW's own created_time, not just which FILE it came from -- see
    _drop_out_of_range_rows' docstring for why that distinction matters."""
    from datetime import datetime
    first, last = min(months), max(months)
    fy, fm = (int(x) for x in first.split("-"))
    ly, lm = (int(x) for x in last.split("-"))
    lm += 1
    if lm == 13:
        lm = 1
        ly += 1
    return datetime(fy, fm, 1), datetime(ly, lm, 1)


def _drop_out_of_range_rows(markets: pl.DataFrame, trades: pl.DataFrame, months) -> tuple:
    """Defensive filter -- confirmed directly (not assumed) that a SINGLE
    corrupted/out-of-range created_time value ANYWHERE in a multi-month
    concatenated table can make polars' group_by_dynamic (the core of
    FixedWindowTemporal.slice(), called on the primary trades table right
    at the start of GraphBuilder.build()) hang effectively forever: its
    time buckets span the WHOLE column's min..max, so one absurd outlier
    timestamp (a bad parse, a sentinel/placeholder date, a unit-conversion
    bug -- anything that lands hundreds of years off) forces it to try to
    represent a vastly larger time range than the real data needs, even
    though almost every bucket in that range would be empty. Reproduced in
    isolation: a single ~1000-year-future outlier row hangs
    group_by_dynamic even on an otherwise-small, clean dataset; the same
    magnitude in the PAST does not (confirmed asymmetric), and moderate
    offsets (checked up to +250 years) don't trigger it either -- so this
    needs an outlier of serious magnitude to fire, not just ordinary noise,
    but a single bad row anywhere in 12.6M is entirely plausible.

    The data/{kind}_kalshi_{even,odd}/{kind}_{month}.parquet FILENAME
    convention says which month a file nominally belongs to -- it does
    NOT guarantee every row's own created_time column value actually
    falls inside that month, which is exactly the gap this closes. Applied
    to BOTH tables (builder.py's KalshiTickerNodes/_slice_auxiliary also
    filters "markets" by created_time) for the same reason, even though
    only the TRADES table's outliers can actually hang group_by_dynamic
    (markets only ever gets used for narrower two-sided .filter() lookups,
    not global min/max bucket generation)."""
    lo, hi = _month_bounds(months)
    out = {}
    for name, df in (("markets", markets), ("trades", trades)):
        if "created_time" not in df.columns:
            out[name] = df
            continue
        bad = df.filter((pl.col("created_time") < lo) | (pl.col("created_time") >= hi))
        if bad.height > 0:
            print(
                f"  WARNING: dropping {bad.height} {name} row(s) with created_time outside "
                f"the requested [{lo}, {hi}) range -- min={bad['created_time'].min()}  "
                f"max={bad['created_time'].max()}. A row like this is exactly what can make "
                f"FixedWindowTemporal.slice()'s group_by_dynamic call hang indefinitely "
                f"(confirmed directly, not assumed) -- see _drop_out_of_range_rows' docstring."
            )
        out[name] = df.filter((pl.col("created_time") >= lo) & (pl.col("created_time") < hi))
    return out["markets"], out["trades"]


def load_months(months) -> tuple[pl.DataFrame, pl.DataFrame]:
    market_paths = [_parity_path("markets", m) for m in months if Path(_parity_path("markets", m)).exists()]
    trade_paths = [_parity_path("trades", m) for m in months if Path(_parity_path("trades", m)).exists()]
    missing_markets = [m for m in months if not Path(_parity_path("markets", m)).exists()]
    missing_trades = [m for m in months if not Path(_parity_path("trades", m)).exists()]
    if missing_markets:
        print(f"WARNING: no markets file for {missing_markets}")
    if missing_trades:
        print(f"WARNING: no trades file for {missing_trades}")
    if not market_paths or not trade_paths:
        raise FileNotFoundError(
            f"No markets/trades files found for {months} under {_REPO_ROOT / 'data'} -- "
            f"check the data/{{markets,trades}}/{{markets,trades}}_kalshi_{{even,odd}}/ convention."
        )
    print(f"Loading markets/trades for {months} ...")
    markets = pl.concat([pl.scan_parquet(p) for p in market_paths]).collect()
    trades = pl.concat([pl.scan_parquet(p) for p in trade_paths]).collect()
    print(f"  {markets.height} market rows, {trades.height} trade rows across {len(months)} month(s)\n")

    markets, trades = _drop_out_of_range_rows(markets, trades, months)
    return markets, trades


def _rss_mb() -> float:
    """Current RSS in MB. `ru_maxrss` is a PEAK high-water mark and never
    goes down, so it can't show a free() working -- this reads the live
    value from /proc instead, falling back to the peak where /proc isn't
    available."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * resource.getpagesize() / (1024.0 * 1024.0)
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _build_one_month(month: str, cache_path: Path):
    """Builds ONE month's bundle and writes it to its own cache file,
    freeing every build-time object before returning.

    One month at a time is the whole point (see model/month_store.py's
    docstring for the arithmetic): the SpatioTemporalGraph holds one
    networkx graph PER SNAPSHOT (~310 per month), so building N months in
    a single pass holds N months of graph objects simultaneously, and the
    resulting global bundle's dense (T, N, F) tensor grows with the
    SQUARE of the range. Building per month keeps both bounded by a
    single month's size no matter how long the total range is.
    """
    print(f"\n=== Building month {month} ===", flush=True)
    markets, trades = load_months([month])
    mece_leg_prices = pl.read_parquet(MECE_LEG_PRICES_PATH)
    ladder_pairs = pl.read_parquet(LADDER_PAIRS_PATH)

    stg = build_combined_graph(trades, markets, mece_leg_prices, ladder_pairs)
    print(f"Built graph for {month}: {stg.summary()}")

    bundle = to_torch_bundle(stg)
    print(f"Bundle {month}: T={len(bundle.timestamps)}  N={len(bundle.node_ids)}  "
          f"F={bundle.features.shape[-1]}  mask_coverage={bundle.mask.float().mean().item():.1%}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    print(f"Cached to {cache_path}")

    before = _rss_mb()
    del stg, bundle, trades, markets, mece_leg_prices, ladder_pairs
    gc.collect()
    print(f"Freed build state for {month}: RSS {before:.0f}MB -> {_rss_mb():.0f}MB", flush=True)


def main(months=None, chunk_len=168, use_cache=True, epochs=20):
    months = months or PILOT_MONTHS

    # PER-MONTH CACHE. Two problems solved at once:
    #
    # 1. Rebuild cost. Graph-building takes ~10 minutes per month and is
    #    deterministic given the same input, yet was redone from scratch
    #    on every run -- a quarter of a 40-minute run spent recomputing a
    #    byte-identical result, which is what made experimentation
    #    painful.
    # 2. Scale. A single global bundle over the whole range is infeasible
    #    past a few months (34.6GB of dense tensor at 14 months, to carry
    #    121MB of real observations -- see model/month_store.py). Caching
    #    per month means the range is never materialized as one object,
    #    and training loads one month at a time.
    #
    # Pass --no-cache to force a rebuild after changing anything upstream
    # of the bundle: the graph builders, node/edge definitions, or the
    # raw parquet data.
    cache_dir = _REPO_ROOT / "cache"
    month_paths = build_month_paths(months, cache_dir)

    for month, path in month_paths.items():
        if use_cache and path.exists():
            print(f"Month {month}: using cached bundle at {path}")
        else:
            _build_one_month(month, path)

    store = MonthlyBundleStore(month_paths)
    print(f"\n{store.summary()}\n")

    config = TrainingConfig(
        embed_dim=32, n_heads=4, chunk_len=chunk_len, min_chunk_len=24,
        mask_ratio=0.15, lr=1e-3, epochs=epochs, checkpoint_dir=str(_REPO_ROOT / "checkpoints"),
    )
    history = run_training(store, config)

    # Report what is ACTUALLY on disk. This line used to print
    # "Best checkpoint at .../best.pt" unconditionally, including for
    # runs that saved nothing at all (no val split -> the old save
    # condition never fired). Checked, not assumed.
    ckpt_dir = Path(config.checkpoint_dir)
    best, last = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    print("\nDone.")
    if best.exists():
        print(f"  Best-by-validation checkpoint: {best}")
    if last.exists():
        print(f"  Last-epoch checkpoint (NO val split -- not model-selected): {last}")
    if not best.exists() and not last.exists():
        print("  WARNING: no checkpoint was written.")
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=None,
                         help=f"e.g. --months 2025-05 2025-06  (default: PILOT_MONTHS={PILOT_MONTHS}; "
                              f"pass --months {' '.join(MASTER_MONTHS[:3])} ... for the full "
                              f"MASTER_MONTHS range once a pilot run has succeeded)")
    parser.add_argument("--no-cache", action="store_true",
                         help="Rebuild the graph from raw data instead of loading cache/bundle_*.pt. "
                              "Use this after changing anything upstream of the bundle (graph "
                              "builders, node/edge definitions, raw parquet data).")
    parser.add_argument("--epochs", type=int, default=20,
                         help="Training epochs. Early stopping (TrainingConfig.patience) ends the "
                              "run sooner if val stops improving, so a larger number costs nothing "
                              "when the model has converged. The 5-month run was still improving at "
                              "epoch 19, i.e. it stopped because it ran out of epochs, not because "
                              "it finished learning.")
    parser.add_argument("--chunk-len", type=int, default=168,
                         help="Snapshots per training chunk (default 168 = ~2 weeks at 2h "
                              "resolution). This is the main memory/compute lever: temporal "
                              "attention cost scales with chunk_len SQUARED, so halving it to 84 "
                              "cuts that ~4x, at the price of halving each node's temporal "
                              "receptive field. Given most Kalshi markets live only ~1-2 days, a "
                              "1-week window still covers a typical market's whole lifetime -- "
                              "drop this first if memory is still tight.")
    args = parser.parse_args()
    main(args.months, chunk_len=args.chunk_len, use_cache=not args.no_cache,
         epochs=args.epochs)