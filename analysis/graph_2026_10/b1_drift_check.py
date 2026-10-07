"""Is the drift of spread / total contracts after +60 s real, or stale prices? (follow-up to b1_response)

b1_response found spread and total contracts still moving the event's way between +60 s and +600 s
(+0.2-0.5c). Those prices come from the quote layer, where a contract that has not traded or
requoted keeps its old price, so a late catch-up can be staleness. Three versions of the same move
(+60 s -> +600 s, signed the event's way), on the same (event, contract) rows:

    quotes, all        as in b1_response
    quotes, fresh      only rows whose quote at +60 s has both sides updated after the event, and
                       whose quote at +600 s has both sides updated after +300 s
    trades             a price from actual trades only: the mean of the last YES-taker price
                       (paid at the ask) and the last NO-taker price (sold at the bid) in a window
                       around each time (+30..+90 s and +540..+660 s); both sides needed in both
                       windows, so the bid-ask bounce cancels

If the drift survives in "fresh" and "trades", contracts really keep catching up; if it vanishes,
it was stale prices.

    venv/bin/python -W ignore analysis/graph_2026_10/b1_drift_check.py NFL > analysis/graph_2026_10/out/b1_drift_check_NFL.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import b1_response as R                                    # noqa: E402  (reads the league from argv)
from stg.sports.asof import SportsData                     # noqa: E402
from stg.sports.loaders import scan_trades                 # noqa: E402
from stg.sports.prices import quotes_at, scan_quotes       # noqa: E402

WIN_A = (30, 90)
WIN_B = (540, 660)


def quote_moves(x: pl.DataFrame, con: pl.DataFrame) -> pl.DataFrame:
    at = pl.concat([x.select("eid", "ticker", "t", (pl.col("t") + pl.duration(seconds=o)).alias("tq"), pl.lit(o).alias("o"))
                    for o in (60, 600)])
    q = quotes_at(scan_quotes(con["series_ticker"].unique().to_list()),
                  at.select("ticker", pl.col("tq").alias("t")).unique(), max_spread=100.0)
    q = q.with_columns(R.value(q).alias("v")).select("ticker", pl.col("t").alias("tq"), "v", "bid_at", "ask_at").unique(["ticker", "tq"])
    a = at.join(q, on=["ticker", "tq"], how="left").with_columns(
        (pl.min_horizontal("bid_at", "ask_at") > pl.col("t") + pl.duration(seconds=pl.when(pl.col("o") == 600).then(300).otherwise(0)))
        .alias("fresh"))
    w = a.pivot(on="o", index=["eid", "ticker"], values=["v", "fresh"], aggregate_function="first")
    return w.rename({"v_60": "q60", "v_600": "q600", "fresh_60": "fresh60", "fresh_600": "fresh600"})


def trade_moves(x: pl.DataFrame, con: pl.DataFrame) -> pl.DataFrame:
    tr = (scan_trades(con["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(x["ticker"].unique().implode()))
          .select("ticker", pl.col("created_time").cast(pl.Datetime("us", "UTC")).alias("tt"),
                  pl.col("yes_price").cast(pl.Float64).alias("p"), "taker_side").collect().sort("tt"))
    sides = {s: tr.filter(pl.col("taker_side") == s).select("ticker", "tt", "p") for s in ("yes", "no")}
    out = x.select("eid", "ticker", "t")
    for name, (lo, hi) in (("a", WIN_A), ("b", WIN_B)):
        w = out.select("eid", "ticker", (pl.col("t") + pl.duration(seconds=hi)).alias("hi")).sort("hi")
        for s, col in (("yes", "ask_side"), ("no", "bid_side")):
            w = w.join_asof(sides[s].rename({"p": col}), left_on="hi", right_on="tt", by="ticker", strategy="backward",
                            tolerance=f"{hi - lo}s", check_sortedness=False).drop("tt")
        out = out.join(w.select("eid", "ticker", ((pl.col("ask_side") + pl.col("bid_side")) / 2).alias(f"tr_{name}")),
                       on=["eid", "ticker"], how="left")
    return out.drop("t")


def summarise(df: pl.DataFrame, by: str, order: list[str]) -> pl.DataFrame:
    rows = []
    for (key,), g in df.partition_by(by, as_dict=True).items():
        r = {by: key}
        for name, a, b, filt in (("quotes_all", "q60", "q600", None), ("quotes_fresh", "q60", "q600", "fresh"),
                                 ("trades", "tr_a", "tr_b", None)):
            gg = g.drop_nulls([a, b])
            if filt:
                gg = gg.filter(pl.col("fresh60").fill_null(False), pl.col("fresh600").fill_null(False))
            v = (gg["sgn"] * (gg[b] - gg[a])).to_numpy()
            r[f"{name}_n"] = len(v)
            r[f"{name}_mean"] = round(float(v.mean()), 2) if len(v) >= R.MIN_N else None
            r[f"{name}_t"] = round(R.clustered_t(v, gg["game_id"].to_numpy()), 1) if len(v) >= R.MIN_N else None
        rows.append(r)
    return (pl.DataFrame(rows).with_columns(pl.col(by).replace_strict(order, list(range(len(order))), default=99).alias("_o"))
            .sort("_o").drop("_o"))


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    x, con, _ = R.event_contract_rows(SportsData())
    x = x.filter(pl.col("kind").is_in(["winner", "spread", "total"]))
    m = x.join(quote_moves(x, con), on=["eid", "ticker"], how="left").join(trade_moves(x, con), on=["eid", "ticker"], how="left")
    edges = R.BINS[R.LEAGUE]
    lab = [f"<{edges[0]}"] + [f"{a}..{b}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]}"]
    m = m.with_columns(pl.when(pl.col("kind") == "winner").then(pl.lit("winner"))
                       .otherwise(pl.col("dist").cut(edges, labels=lab).cast(pl.String)).alias("distance"))
    print(f"{R.LEAGUE}: move from +60 s to +600 s, cents signed the event's way; t clustered by game")
    for kind in ("winner", "spread", "total"):
        k = m.filter(pl.col("kind") == kind)
        if k.height:
            print(f"\n######## {kind}" + (" (scoring events only)" if kind == "total" else ""))
            print(summarise(k, "distance", ["winner"] + lab))


if __name__ == "__main__":
    main()
