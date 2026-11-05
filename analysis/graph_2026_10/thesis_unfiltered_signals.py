"""One-contract comparison without signal-magnitude selection, development folds only.

Reconstruct b4_export's row identities from retained contract_steps, validating
every saved forecast's kind and anchor. Replay one game at a time with b7's
unchanged simulator. First reproduce the original >=5c fills as a parity check.
Trade files are projected and partitioned one file at a time to bound memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out/thesis_unfiltered_signals"
sys.path.insert(0, str(ROOT / "stg_infra"))
sys.path.insert(0, str(HERE))


def imports():
    # Historical modules interpret argv at import time; only simulate is used.
    saved = sys.argv[:]
    sys.argv = [sys.argv[0], "results/ingame/ens5", "NBA", "60"]
    import b7_maker_trades as bt
    from b6_maker_exit import sweep_sorted
    from b7_maker_robust import with_net
    from stg.splits import SPORTS_HOLDOUT_START, sports_holdout_unsealed
    sys.argv = saved
    assert not sports_holdout_unsealed(), "Development comparison only"
    bt.HOLDS = {"120s": 120}
    bt.VARIANTS = [(1, 0)]
    bt.ONE_AT_A_TIME = True
    return bt, sweep_sorted, with_net, SPORTS_HOLDOUT_START


def setup(league):
    tab = ROOT / "data/sports/derived/ingame" / ("MLB_pitch" if league == "MLB" else league)
    games = pl.read_parquet(tab / "games.parquet").sort("first_play")
    ids = games["game_id"].to_list()
    steps = (pl.read_parquet(tab / "steps.parquet", columns=["game_id", "k", "t", "after_end"])
             .filter(~pl.col("after_end"))
             .with_columns(pl.col("game_id").replace_strict(dict(zip(ids, range(len(ids))))).alias("g"))
             .sort("g", "k").with_row_index("sid"))
    con = pl.read_parquet(tab / "contracts.parquet")
    return tab, games, steps, con


def cache_trades(league, games, con, cutoff):
    cache = OUT / "trades" / league
    marker = cache / "complete.json"
    if marker.exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    wanted = con.filter(pl.col("kind").is_in(["spread", "total"]))
    series = wanted["series_ticker"].unique().to_list()
    mapping = wanted.select("ticker", "game_id").unique()
    sources = []
    for series_name in sorted(series):
        for path in sorted((ROOT / "data/sports/trades" / series_name).glob("*.parquet")):
            f = (pl.read_parquet(path, columns=["ticker", "trade_id", "yes_price", "taker_side", "created_time"])
                 .filter(pl.col("created_time") < cutoff)
                 .join(mapping, on="ticker", how="inner")
                 .select("game_id", "ticker", "trade_id", "yes_price", "taker_side",
                         (pl.col("created_time").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t")))
            for (gid,), part in f.partition_by("game_id", as_dict=True).items():
                dest = cache / gid
                dest.mkdir(exist_ok=True)
                part.drop("game_id").write_parquet(dest / f"{series_name}_{path.name}")
            sources.append(dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        print(f"{league}: cached {series_name}", flush=True)
    marker.write_text(json.dumps(sources, indent=2) + "\n")
    return cache


def game_nodes(tab, gid, steps, con):
    # Identical filtering and ordering to b4_export.py, mode=prints.
    cs = pl.read_parquet(tab / "contract_steps" / f"{gid}.parquet",
                         columns=["k", "ticker", "live", "pr_age_s", "pr_px", "value"])
    cs = (cs.join(con.filter(pl.col("game_id") == gid).select("ticker", "kind", "side", "strike"), on="ticker")
          .join(steps.filter(pl.col("game_id") == gid).select("k", "sid", "t"), on="k")
          .with_columns(pl.when(pl.col("pr_age_s") <= 120).then(pl.col("pr_px"))
                        .otherwise(pl.col("value")).alias("anchor"))
          .filter(pl.col("anchor").is_not_null(), pl.col("live") | pl.col("anchor").is_between(2, 98),
                  pl.col("kind").is_in(["winner", "spread", "total"]))
          .sort("sid", "kind", "side", "strike", "ticker"))
    return cs


KEYS = ["game_id", "ticker", "t", "hold", "k", "through", "side", "signal", "entry", "exit", "how", "t_fill"]


def replay(league, max_games=None):
    bt, sweep_sorted, with_net, cutoff = imports()
    tab, games, steps, con = setup(league)
    assert games["feed_end"].max() < cutoff
    cache = cache_trades(league, games, con, cutoff)
    ids = games["game_id"].to_list()
    plays = pl.read_parquet(tab / "plays.parquet", columns=["game_id", "t_event", "seq"])
    delay = 10 if league == "MLB" else 60
    tag = "_h30-60-120-1800_tp" + ("_pitch" if league == "MLB" else "") + "_ens5"
    rule = "_dec5_one" + ("_skip15-60" if league == "MLB" else "")
    checks, processed = [], 0
    for fold in range(1, 6):
        dest = OUT / f"{league}_fold{fold}.parquet"
        audit_path = dest.with_suffix(".json")
        if max_games is None and dest.exists() and audit_path.exists():
            checks.append(json.loads(audit_path.read_text()))
            print(f"{league} fold {fold}: cached", flush=True)
            continue
        source = ROOT / "results/ingame/ens5" / f"{league}_fold{fold}_gru_d{delay}{tag}_seed0.npz"
        hz = json.loads(source.with_suffix(".json").read_text())["horizons"]
        with np.load(source) as z:
            pr = {k: z[k] for k in ("game", "step", "kind", "value", "y")}
            pr["delta"] = z["delta"][:, hz.index("120s")].copy()
        original = pl.read_parquet(HERE / "out" / f"b7_maker_trades_{league}_d{delay}{tag}{rule}_fold{fold}.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")
        parts, original_parts, valid_ids = [], [], []
        groups = np.unique(pr["game"], return_index=True, return_counts=True)
        assert np.all(pr["game"][1:] >= pr["game"][:-1]), "Forecast games must be contiguous"
        zero_signals = 0
        for gi, start, count in zip(*groups):
            gid = ids[gi]
            assert games.row(int(gi), named=True)["fold"] == fold
            a, b = int(start), int(start + count)
            n = game_nodes(tab, gid, steps, con)
            n = n.filter(pl.col("sid").is_in(np.unique(pr["step"][a:b]).tolist()))
            assert n.height == count, (league, fold, gid, n.height, count)
            assert np.array_equal(n["sid"].to_numpy(), pr["step"][a:b]), (gid, "step mismatch")
            assert np.array_equal(n["kind"].replace_strict({"winner": 0, "spread": 1, "total": 2}).to_numpy(), pr["kind"][a:b]), (gid, "kind mismatch")
            assert np.array_equal(n["anchor"].to_numpy().astype(np.float32), pr["value"][a:b]), (gid, "anchor mismatch")
            # Addition stays float32, as in b5_longshort.rows.
            n = n.with_columns(pl.Series("level_120s", (pr["value"][a:b] + pr["delta"][a:b]).astype(float)),
                               pl.Series("y", pr["y"][a:b].astype(float)))
            r = n.filter(pl.col("y").is_finite(), pl.col("kind").is_in(["spread", "total"]))
            if league == "MLB":
                pt = plays.filter(pl.col("game_id") == gid).sort("t_event", "seq")["t_event"].dt.epoch("us").to_numpy() / 1e6
                t = r["t"].dt.epoch("us").to_numpy() / 1e6
                k = np.searchsorted(pt, t - delay, side="right")
                age = np.where(k > 0, t - pt[np.maximum(k - 1, 0)], np.inf)
                r = r.filter(pl.Series(~((age >= 15) & (age < 60))))
            # Match original attach_tickers' first contract for duplicate identity keys.
            identity = ["kind", "side", "strike"]
            remap = con.filter(pl.col("game_id") == gid).select(*identity, "ticker").unique(identity, keep="first")
            r = r.drop("ticker").join(remap, on=identity, how="left", nulls_equal=True)
            paths = sorted((cache / gid).glob("*.parquet"))
            if not paths or not r.height:
                valid_ids.append(gid)
                processed += 1
                continue
            tr = sweep_sorted(pl.read_parquet(paths).unique("trade_id"))
            r = (r.with_columns((pl.col("t").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"))
                 .sort("t", maintain_order=True)
                 .join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref"))
                            .sort("t", maintain_order=True), on="t", by="ticker", strategy="backward", check_sortedness=False)
                 .filter((pl.col("t") - pl.col("t_ref")) <= 60)
                 .with_columns((pl.col("level_120s") - pl.col("ref")).alias("sig_120s"))
                 .with_row_index("row"))
            zero_signals += r.filter(pl.col("sig_120s") == 0).height
            for minimum in (5.0, 0.0):
                bt.MIN_SIGNAL = minimum
                raw = bt.simulate(r, tr)
                if not raw.height:
                    continue
                raw = raw.join(r.select("row", "kind", "ticker", "t", "ref", "strike", "y"), on="row")
                f = raw.with_columns(pl.lit(gid).alias("game_id"), pl.lit(fold).alias("fold"))
                if minimum == 5:
                    original_parts.append(f.select(KEYS))
                else:
                    # Width uses the same last-per-side prints and 120s freshness as liquidity().
                    for side in ("yes", "no"):
                        p = tr.filter(pl.col("taker_side") == side).select("ticker", "t", pl.col("yes_price").alias(f"p_{side}"))
                        f = f.sort("t", maintain_order=True).join_asof(p.sort("t", maintain_order=True), on="t", by="ticker",
                            strategy="backward", tolerance=120.0, check_sortedness=False)
                    f = with_net(f.with_columns((pl.col("p_yes") - pl.col("p_no")).alias("width")))
                    parts.append(f)
            valid_ids.append(gid)
            processed += 1
            if processed % 25 == 0:
                print(f"{league} fold {fold}: {processed} games replayed", flush=True)
            if max_games is not None and processed >= max_games:
                break
        expected = original.filter(pl.col("game_id").is_in(valid_ids)).select(KEYS).sort(KEYS)
        reproduced = pl.concat(original_parts, how="vertical_relaxed").sort(KEYS) if original_parts else expected.head(0)
        from polars.testing import assert_frame_equal
        assert_frame_equal(reproduced, expected, check_dtypes=False, check_exact=True)
        result = pl.concat(parts, how="vertical_relaxed")
        audit = dict(league=league, fold=fold, games=len(valid_ids), original_fills=expected.height,
                     unfiltered_fills=result.height, parity="exact", zero_signal_decisions=zero_signals,
                     forecast=str(source), forecast_bytes=source.stat().st_size,
                     forecast_mtime_ns=source.stat().st_mtime_ns, min_signal=0,
                     direction="sign of forecast minus reference; exactly zero follows original short tie rule",
                     sizing="one contract per fill comparison; no magnitude or width weighting",
                     delay_s=delay, hold_s=120, one_at_a_time=True)
        print(json.dumps(audit), flush=True)
        if max_games is not None:
            return
        result.write_parquet(dest)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        checks.append(audit)
        del pr, result, parts, original_parts, original
        gc.collect()
    (OUT / f"{league}_audit.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("league", choices=["MLB", "NBA", "NFL"])
    parser.add_argument("--max-games", type=int, help="Small reconstruction/parity check; writes no final fills")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        replay(args.league, args.max_games)
    finally:
        print(json.dumps(dict(maximum_resident_set_size_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
