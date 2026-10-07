"""Screen every league / contract type / hold for maker candidates, split by market width and
trading activity, from the per-trade logs of b7_maker_trades.py (fixed exit, offset 1c,
strictly-through fills). Liquidity measures as in b7_maker_liquidity.py (prints before t only).

Groups: all trades; width < 4c; 4-9c; >= 9c; >= 9c and active (n_5m at or above the league median).
Per (league, kind, hold, min signal 5 / 8c, group, delay): trades, net, t (clustered by game), share
stuck to settlement, and net with stuck trades charged STUCK_C instead of their settlement result.

A group is listed as a candidate when, at EVERY delay available for its league: trades >= MIN_TRADES,
net > 0 with t >= 2, and the stuck-charged net > 0. Many groups are screened, so a candidate is a
lead to test on untouched data, not a result.

    venv/bin/python -W ignore analysis/graph_2026_10/b7_maker_candidates.py
    -> out/b7_maker_candidates.txt, out/b7_maker_candidates.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity            # noqa: E402
from b7_maker_robust import with_net, summ, OUT     # noqa: E402

TAG = "_h30-60-120-1800_tpf"
LEAGUES = ["NBA", "MLB", "NFL"]
KINDS = ["winner", "spread", "total"]
HOLDS = ["60s", "120s"]
SIGNALS = [5.0, 8.0]
STUCK_C = -3.5
MIN_TRADES = 150


def groups(f: pl.DataFrame, med_n: float) -> dict[str, pl.DataFrame]:
    w = pl.col("width")
    return {"all": f, "width < 4c": f.filter(w < 4), "width 4-9c": f.filter((w >= 4) & (w < 9)),
            "width >= 9c": f.filter(w >= 9), "width >= 9c, active": f.filter((w >= 9) & (pl.col("n_5m") >= med_n))}


def main() -> None:
    pl.Config.set_tbl_rows(1000)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    rows = []
    for L in LEAGUES:
        files = sorted(OUT.glob(f"b7_maker_trades_{L}_d*{TAG}.parquet"))
        parts = [with_net(pl.read_parquet(p).filter(pl.col("k") == 1, pl.col("through") == 0))
                 .with_columns(pl.lit(int(p.name.split("_d")[1].split("_")[0])).alias("delay")) for p in files]
        f = liquidity(pl.concat(parts, how="diagonal_relaxed"))
        med_n = f["n_5m"].median()
        delays = sorted(f["delay"].unique().to_list())
        for kind in KINDS:
            for hold in HOLDS:
                for m in SIGNALS:
                    g0 = f.filter(pl.col("kind") == kind, pl.col("hold") == hold, pl.col("signal").abs() >= m)
                    for gname, g1 in groups(g0, med_n).items():
                        for d in delays:
                            g = g1.filter(pl.col("delay") == d)
                            if g.height == 0:
                                continue
                            x = g.with_columns(pl.when(pl.col("how") == "settled").then(STUCK_C).otherwise(pl.col("net")).alias("x"))
                            rows.append(summ(g, league=L, kind=kind, hold=hold, min_signal=m, group=gname, delay=d)
                                        | dict(stuck=round((g["how"] == "settled").mean(), 2), net_stuck_charged=round(x["x"].mean(), 2),
                                               n_delays=len(delays)))
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "b7_maker_candidates.parquet")
    ok = (pl.col("trades") >= MIN_TRADES) & (pl.col("net") > 0) & (pl.col("t") >= 2) & (pl.col("net_stuck_charged") > 0)
    key = ["league", "kind", "hold", "min_signal", "group"]
    cand = (t.with_columns(ok.alias("ok")).group_by(key)
            .agg(pl.col("ok").all().alias("all_ok"), pl.len().alias("delays_seen"), pl.col("n_delays").first(),
                 pl.col("trades").min().alias("min_trades"), pl.col("net").min().alias("worst_net"), pl.col("t").min().alias("worst_t"),
                 pl.col("net_stuck_charged").min().alias("worst_net_stuck_charged"))
            .filter(pl.col("all_ok") & (pl.col("delays_seen") == pl.col("n_delays"))).drop("all_ok").sort(key))
    lines = []
    out = lambda s: (print(s), lines.append(str(s)))   # noqa: E731
    out(f"Candidates: net > 0, t >= 2, stuck-charged ({STUCK_C:g}c) net > 0 and >= {MIN_TRADES} trades at every delay")
    out(cand)
    for L in LEAGUES:
        for kind in KINDS:
            out(f"\n================ {L} {kind}")
            out(t.filter(pl.col("league") == L, pl.col("kind") == kind).drop("league", "kind", "n_delays", "games", "gross")
                .sort("hold", "min_signal", "group", "delay"))
    (OUT / "b7_maker_candidates.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
