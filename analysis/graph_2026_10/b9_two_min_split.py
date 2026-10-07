"""Where does each model's 2-minute maker profit come from? NBA spreads, 120 s horizon signal, feed delay
10 s, test folds 3-5; current model >= 8c, mid Huber 5 matched to the current model's trade count.
Mid = (last YES-taker price + last NO-taker price) / 2, both prints at most 120 s old (else missing).
Per fill (side +1 buy YES, -1 sell YES), in cents in the trade's favour:
    discount    mid at fill - entry price
    move +h     mid h after the fill - mid at fill (h = 60 s, 120 s, 300 s, 1800 s): the forecast part
    exit cost   fixed-exit net - discount - move +120 s (exit slippage, fees, stuck trades)
plus the share of fills where the mid moved our way by +120 s (among fills where it moved).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_two_min_split.py
    -> out/b9_two_min_split.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_candidates_by_model import entry_state          # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402

MODELS = {"current": "_h30-60-120-1800_tpf", "mid Huber 5": "_h30-60-120-1800_mid"}
FILTERS = {"all": pl.lit(True), "fresh mid": pl.col("older_s") <= 30, "width >= 9c": pl.col("width") >= 9,
           "fresh & width >= 9c": (pl.col("older_s") <= 30) & (pl.col("width") >= 9)}
HS = [60, 120, 300, 1800]


def mid_at(f: pl.DataFrame, tr: pl.DataFrame, col: str, name: str) -> pl.DataFrame:
    """Trade-based mid at time f[col] (both sides <= 120 s old)."""
    out = []
    for side, p in (("yes", "pa"), ("no", "pb")):
        s = tr.filter(pl.col("taker_side") == side).select("ticker", pl.col("t").alias("_t"), pl.col("t").alias(f"_t{p}"), pl.col("yes_price").alias(f"_{p}")).sort("_t")
        out.append(s)
    g = f.with_columns(pl.col(col).alias("_t")).sort("_t")
    for s in out:
        g = g.join_asof(s, on="_t", by="ticker", strategy="backward")
    ok = ((pl.col("_t") - pl.col("_tpa")) <= 120) & ((pl.col("_t") - pl.col("_tpb")) <= 120)
    return g.with_columns(pl.when(ok).then((pl.col("_pa") + pl.col("_pb")) / 2).alias(name)).drop("_t", "_tpa", "_tpb", "_pa", "_pb")


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(60)
    pick = lambda m: pl.read_parquet(OUT / f"b7_maker_trades_NBA_d10{MODELS[m]}.parquet").filter(   # noqa: E731
        pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s", pl.col("fold").is_in([3, 4, 5]))
    raw = {m: pick(m) for m in MODELS}
    n = raw["current"].filter(pl.col("signal").abs() >= 8).height
    cut = {"current": 8.0, "mid Huber 5": float(raw["mid Huber 5"]["signal"].abs().sort(descending=True)[n - 1])}
    tk = pl.concat([f.select("ticker") for f in raw.values()]).unique()["ticker"]
    tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
    rows = []
    for m, f in raw.items():
        f = entry_state(with_net(f.filter(pl.col("signal").abs() >= cut[m])), tr).drop("t_yes", "t_no", "p_yes", "p_no")
        # mid just before the fill (the fill print itself is excluded by taking t_fill - 0.001)
        f = mid_at(f.with_columns((pl.col("t_fill") - 0.001).alias("tf0")), tr, "tf0", "m0")
        for h in HS:
            f = mid_at(f.with_columns((pl.col("t_fill") + h).alias(f"tf{h}")), tr, f"tf{h}", f"m{h}")
        f = f.with_columns((pl.col("side") * (pl.col("m0") - pl.col("entry"))).alias("discount"),
                           *[(pl.col("side") * (pl.col(f"m{h}") - pl.col("m0"))).alias(f"move_{h}") for h in HS])
        f = f.with_columns((pl.col("net") - pl.col("discount") - pl.col("move_120")).alias("exit_cost"))
        for fn, cond in FILTERS.items():
            g = f.filter(cond)
            full = g.filter(pl.col("m0").is_not_null(), pl.col("move_120").is_not_null())

            def mt(c, x=full):
                v = x.filter(pl.col(c).is_not_null())
                return f"{v[c].mean():+.2f} ({clustered_t(v[c].to_numpy(), v['game_id'].to_numpy()):.1f})"
            mv = full.filter(pl.col("move_120") != 0)
            r = dict(model=m, entry=fn, trades=g.height, with_mids=full.height, net=mt("net", g), net_same_rows=mt("net"),
                     discount=mt("discount"), **{f"move_{h}s": mt(f"move_{h}") for h in HS}, exit_cost=mt("exit_cost"),
                     moved_our_way=round(float((mv["move_120"] > 0).mean()), 2))
            rows.append(r)
    t = pl.DataFrame(rows)
    text = ("NBA spreads, 10 s delay, folds 3-5: fixed-exit net split into entry discount, mid move after the fill, exit cost "
            "(cents per contract in the trade's favour, clustered t in brackets)\n" + str(t))
    (OUT / "b9_two_min_split.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
