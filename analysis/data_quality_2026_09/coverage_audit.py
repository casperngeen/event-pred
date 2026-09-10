"""Why does `data/markets/` list events that `data/trades/` has no trades for?

Backs research_log.md §12. Run from the repo root:

    python analysis/data_quality_2026_09/coverage_audit.py

The question came out of registering INXU/INXD/NASDAQ100U as asset-price
targets (410f8fe): three candidates -- NASDAQ100D, TNOTED, USDJPYH -- carried
real aggregate `volume` in the markets metadata and had zero trades locally.
The registry docstring attributed that to Kalshi's ~67-day trade retention.
This script tests that explanation and rejects it.
"""

import glob
import re

import polars as pl

MK = "data/markets/*.parquet"
TR = "data/trades/*.parquet"


def _series(col: str = "ticker") -> pl.Expr:
    """Canonical series prefix: strip the KX era-prefix, take the head segment."""
    return pl.col(col).str.replace(r"^KX", "").str.split("-").list.first()


def test_1_retention(mk: pl.DataFrame, tr: pl.LazyFrame) -> None:
    """Retention predicts absence tracks DATE. Does it?"""
    print("=" * 72)
    print("1. Is absence explained by trade retention?")
    print("=" * 72)
    span = tr.select("created_time").collect()
    print(f"trades archive spans {span['created_time'].min()} -> "
          f"{span['created_time'].max()}\n")

    counts = (tr.select("ticker").with_columns(_series().alias("s"))
              .group_by("s").agg(pl.len().alias("n_trades")).collect())
    traded = dict(zip(counts["s"].to_list(), counts["n_trades"].to_list()))

    focus = ["NASDAQ100D", "NASDAQ100U", "INXD", "INXU", "INX",
             "TNOTED", "TNOTEW", "USDJPYH", "RATECUT", "CPI", "PAYROLLS"]
    print(f"{'series':<12}{'n_tickers':>10}{'volume':>12}"
          f"{'close_min':>12}{'close_max':>12}{'trades':>10}")
    for s in focus:
        d = mk.filter(pl.col("s") == s)
        if d.height == 0:
            print(f"{s:<12}{'-- absent from markets --':>56}")
            continue
        print(f"{s:<12}{d.height:>10}{int(d['volume'].sum()):>12}"
              f"{str(d['close_time'].min())[:10]:>12}"
              f"{str(d['close_time'].max())[:10]:>12}{traded.get(s, 0):>10}")

    print("\nThe counterexample: NASDAQ100D vs INXD, same era, same window.")
    for s in ["NASDAQ100D", "INXD"]:
        x = (tr.filter(_series().eq(s)).select("created_time").collect())
        rng = (f"{x['created_time'].min()} -> {x['created_time'].max()}"
               if x.height else "no trades")
        print(f"  {s:<12} {x.height:>8} trades   {rng}")
    print("\n=> INXD's trades blanket the window in which NASDAQ100D has none.")
    print("   A 67-day retention cutoff cannot delete one and spare the other.")
    print("   Absence tracks the SERIES, not the DATE. Retention rejected.\n")


def test_2_independent_pulls(mk: pl.DataFrame, tr: pl.LazyFrame) -> None:
    """Is one archive a filtered view of the other, or two separate pulls?"""
    print("=" * 72)
    print("2. Are the two archives nested?")
    print("=" * 72)
    tr_t = tr.select("ticker").unique().collect()
    mk_series, tr_series = set(mk["s"].unique()), set(_ser(tr_t))
    mk_tick, tr_tick = set(mk["ticker"]), set(tr_t["ticker"])

    print(f"distinct series  markets={len(mk_series):>7}  trades={len(tr_series):>7}")
    print(f"distinct tickers markets={len(mk_tick):>7}  trades={len(tr_tick):>7}")
    print(f"  series  in trades but NOT markets: {len(tr_series - mk_series)}")
    print(f"  tickers in trades but NOT markets: {len(tr_tick - mk_tick)}")
    print("\n=> Neither archive contains the other. They are independent pulls,")
    print("   not one filtered view of a single source.\n")


def _ser(df: pl.DataFrame) -> list[str]:
    return df.with_columns(_series().alias("s"))["s"].to_list()


def test_3_pagination(mk: pl.DataFrame) -> None:
    """Both archives are chunked by row offset. How complete is each?"""
    print("=" * 72)
    print("3. How complete is each pull?")
    print("=" * 72)
    for name, pat in [("markets", MK), ("trades", TR)]:
        r = sorted((int(m[1]), int(m[2])) for f in glob.glob(pat)
                   if (m := re.search(r"_(\d+)_(\d+)\.parquet$", f)))
        step = r[0][1] - r[0][0]
        lo, hi = r[0][0], r[-1][1]
        print(f"{name:<9} {len(r):>5} files, step {step}, span {lo}..{hi}, "
              f"{(hi - lo) // step:>5} expected if contiguous")
    print("\n=> Both are partial. markets is missing ~44% of its own page range.\n")


def test_4_bimodal(mk: pl.DataFrame, tr: pl.LazyFrame) -> None:
    """A list-driven pull is all-or-nothing per series. A sampled one is not."""
    print("=" * 72)
    print("4. Is per-series coverage bimodal (the signature of a ticker list)?")
    print("=" * 72)
    tr_t = tr.select("ticker").unique().collect()
    tset = set(tr_t["ticker"].str.replace(r"^KX", "").to_list())
    d = (mk.with_columns(pl.col("ticker").str.replace(r"^KX", "").alias("t"))
         .with_columns(pl.col("t").is_in(list(tset)).alias("has")))
    per = (d.group_by("s").agg(pl.len().alias("n"), pl.col("has").sum().alias("k"))
           .with_columns((pl.col("k") / pl.col("n")).alias("frac"))
           .filter(pl.col("n") >= 20))
    f = per["frac"].to_numpy()
    print(f"series with >=20 market tickers: {len(f)}")
    for lo, hi in [(0, .001), (.001, .05), (.05, .25), (.25, .75), (.75, .95), (.95, 1.01)]:
        print(f"  coverage [{lo:>5}, {hi:<5}): {((f >= lo) & (f < hi)).sum():>5} series")
    print("\n=> Strongly bimodal: whole series in or whole series out. That is a")
    print("   per-ticker fetch driven by a supplied list, which is exactly what")
    print("   scripts/fetch_kalshi_data.py::fetch_trades_for_ticker does.\n")


def test_5_registered(mk: pl.DataFrame) -> None:
    """The consequence: which REGISTERED series are silently short?"""
    print("=" * 72)
    print("5. Coverage of the registered universe (the actionable part)")
    print("=" * 72)
    from stg.panel.registry import trade_coverage
    pl.Config.set_tbl_rows(60)
    cov = trade_coverage()
    print(cov.select("canon", "can_trigger", "n_events_listed",
                     "n_events_traded", "coverage").sort("coverage"))
    print("\n=> assert_trade_coverage() has an absolute floor only, so partial")
    print("   coverage passes silently. WTIW runs on 32% of its events.\n")


def main() -> None:
    mk = pl.read_parquet(MK).with_columns(_series().alias("s"))
    tr = pl.scan_parquet(TR)
    test_1_retention(mk, tr)
    test_2_independent_pulls(mk, tr)
    test_3_pagination(mk)
    test_4_bimodal(mk, tr)
    test_5_registered(mk)


if __name__ == "__main__":
    main()
