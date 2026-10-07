"""How well does a mid built from trade prints alone track the real order book? At each candle close
(Kalshi's best bid / ask at the end of the minute) of a spread / total contract during a game, compare
the candle mid with the trade-based mid: (last NO-taker price + last YES-taker price) / 2, a NO-taker
selling at the bid and a YES-taker buying at the ask, by how old the older of the two prints is.

    venv/bin/python -W ignore analysis/graph_2026_10/b8_candle_check.py NBA
    -> out/b8_candle_check_<L>.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.sports.loaders import derived_dir, scan_candles, scan_trades   # noqa: E402

L = sys.argv[1] if len(sys.argv) > 1 else "NBA"
OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    D = derived_dir() / "ingame" / L
    games = pl.read_parquet(D / "games.parquet").select("game_id", "first_play", "feed_end")
    con = (pl.read_parquet(D / "contracts.parquet").filter(pl.col("kind").is_in(["spread", "total"]))
           .select("ticker", "series_ticker", "game_id", "kind").join(games, on="game_id"))
    series = con["series_ticker"].unique().to_list()
    tk = con["ticker"].unique().implode()
    cd = (scan_candles(series).filter(pl.col("ticker").is_in(tk))
          .select("ticker", pl.col("known_at").alias("t"), pl.col("yes_bid_close").alias("c_bid"), pl.col("yes_ask_close").alias("c_ask"))
          .collect().join(con.select("ticker", "kind", "first_play", "feed_end"), on="ticker")
          .filter(pl.col("t").is_between(pl.col("first_play"), pl.col("feed_end")), pl.col("c_bid") > 0, pl.col("c_ask") < 100,
                  pl.col("c_ask") >= pl.col("c_bid")))
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(tk))
          .select("ticker", pl.col("known_at").alias("t"), "yes_price", "taker_side").collect())
    yes = tr.filter(pl.col("taker_side") == "yes").select("ticker", "t", pl.col("yes_price").alias("p_ask"), pl.col("t").alias("t_ask")).sort("t")
    no = tr.filter(pl.col("taker_side") == "no").select("ticker", "t", pl.col("yes_price").alias("p_bid"), pl.col("t").alias("t_bid")).sort("t")
    x = (cd.sort("t").join_asof(yes, on="t", by="ticker", strategy="backward").join_asof(no, on="t", by="ticker", strategy="backward")
         .filter(pl.col("p_ask").is_not_null(), pl.col("p_bid").is_not_null())
         .with_columns(((pl.col("t") - pl.min_horizontal("t_ask", "t_bid")).dt.total_seconds()).alias("older_age_s"),
                       ((pl.col("c_bid") + pl.col("c_ask")) / 2).alias("c_mid"), ((pl.col("p_bid") + pl.col("p_ask")) / 2).alias("p_mid"))
         .with_columns((pl.col("p_mid") - pl.col("c_mid")).alias("diff"), (pl.col("c_ask") - pl.col("c_bid")).alias("c_width"),
                       (pl.col("p_ask") - pl.col("p_bid")).alias("p_width"),
                       pl.col("older_age_s").cut([10, 30, 120, 600], labels=["<=10s", "10-30s", "30s-2m", "2-10m", ">10m"]).alias("age")))
    lines = [f"{L}: {x.height:,} candle closes of spread / total contracts in games with both a YES- and a NO-taker print before them"]
    agg = [pl.len().alias("n"), pl.col("diff").abs().median().round(2).alias("median_abs_diff_c"),
           (pl.col("diff").abs() <= 1).mean().round(2).alias("within_1c"), (pl.col("diff").abs() <= 3).mean().round(2).alias("within_3c"),
           pl.col("p_mid").is_between(pl.col("c_bid"), pl.col("c_ask")).mean().round(2).alias("inside_candle_bid_ask"),
           pl.col("c_width").median().alias("candle_width_median"), pl.col("p_width").median().alias("print_width_median")]
    lines += ["\n######## trade-based mid vs candle mid, by age of the older of the two prints", str(x.group_by("age").agg(agg).sort("age"))]
    lines += ["\n######## fresh (both prints <= 30 s old), by candle width",
              str(x.filter(pl.col("older_age_s") <= 30).with_columns(pl.col("c_width").cut([2, 4, 9], labels=["<=2c", "3-4c", "5-9c", ">=10c"]).alias("cw"))
                  .group_by("cw").agg(agg).sort("cw"))]
    lines += ["\n######## fresh, by month", str(x.filter(pl.col("older_age_s") <= 30).with_columns(pl.col("t").dt.strftime("%Y-%m").alias("month"))
                                             .group_by("month").agg(agg).sort("month"))]
    (OUT / f"b8_candle_check_{L}.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
