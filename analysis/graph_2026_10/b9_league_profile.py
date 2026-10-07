"""How do the contract types differ? Current model, feed delay 10 s, all test folds, hold 120 s, fixed exit,
offset 1c, strictly-through fills, |signal| >= 8c (all widths, so the groups are comparable).
Per contract type: test games and dates, trades per game, width at entry (b7_maker_liquidity), share with
a missing width, prints in the 5 min before entry, share stuck to settlement, and the profit split of
b9_two_min_split.py (entry discount, mid move 2 min after the fill, exit cost), in cents per contract.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_league_profile.py
    -> out/b9_league_profile.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_liquidity import liquidity               # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_two_min_split import mid_at                     # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402

STRATS = [("NBA", "spread"), ("NBA", "total"), ("MLB", "total"), ("MLB", "spread"), ("NFL", "total"), ("NFL", "spread")]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    rows, split = [], []
    for L in ("NBA", "MLB", "NFL"):
        f0 = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10_h30-60-120-1800_tpf.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")
        all_games = f0.group_by("kind").agg(pl.col("game_id").n_unique().alias("games"), pl.col("first_play").min().dt.date().alias("from"),
                                            pl.col("first_play").max().dt.date().alias("to"))
        f = liquidity(with_net(f0.filter(pl.col("signal").abs() >= 8)))
        tk = f["ticker"].unique()
        tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
              .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect()
              .sort("ticker", "t", "yes_price"))
        f = mid_at(f.with_columns((pl.col("t_fill") - 0.001).alias("tf0")), tr, "tf0", "m0")
        f = mid_at(f.with_columns((pl.col("t_fill") + 120).alias("tf120")), tr, "tf120", "m120")
        f = f.with_columns((pl.col("side") * (pl.col("m0") - pl.col("entry"))).alias("discount"),
                           (pl.col("side") * (pl.col("m120") - pl.col("m0"))).alias("move_2m"))
        f = f.with_columns((pl.col("net") - pl.col("discount") - pl.col("move_2m")).alias("exit_cost"))
        for kind in [k for l_, k in STRATS if l_ == L]:
            g = f.filter(pl.col("kind") == kind)
            ag = all_games.filter(pl.col("kind") == kind).row(0, named=True)
            mt = lambda c, x: f"{x[c].mean():+.1f} ({clustered_t(x[c].to_numpy(), x['game_id'].to_numpy()):.1f})"   # noqa: E731
            rows.append(dict(contract=f"{L} {kind}", test_from=str(ag["from"]), test_to=str(ag["to"]), test_games=ag["games"],
                             trades=g.height, trades_per_game=round(g.height / ag["games"], 1), net=mt("net", g),
                             median_width=g["width"].median(), width_ge9=round((g["width"] >= 9).mean(), 2),
                             width_missing=round(g["width"].is_null().mean(), 2), median_prints_5m=g["n_5m"].median(),
                             median_contracts_5m=g["vol_5m"].median(), stuck=round((g["how"] == "settled").mean(), 2)))
            full = g.filter(pl.col("m0").is_not_null(), pl.col("m120").is_not_null())
            split.append(dict(contract=f"{L} {kind}", with_mids=full.height, net_same_rows=mt("net", full), discount=mt("discount", full),
                              move_2m=mt("move_2m", full), exit_cost=mt("exit_cost", full),
                              moved_our_way=round(float((full.filter(pl.col("move_2m") != 0)["move_2m"] > 0).mean()), 2)))
    text = ("Current model, 10 s delay, all test folds, hold 120 s, |signal| >= 8c, all widths (net cents per contract, clustered t)\n"
            "######## market and trade profile\n" + str(pl.DataFrame(rows)) +
            "\n\n######## profit split (fills with a mid before and 2 min after)\n" + str(pl.DataFrame(split)))
    (OUT / "b9_league_profile.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
