"""Integrity checks on the Kalshi sports pull (``scripts/fetch_sports.py``).

    venv/bin/python scripts/check_sports_data.py            # whatever is on disk now
    venv/bin/python scripts/check_sports_data.py --every 90 # rerun every 90 min while the pull runs

Safe to run while the pull is writing: it only reads finished part files, and
caches per-file summaries, so a rerun only reads what is new. Reads the working
block and the holdout alike, and prints **counts only** for the holdout.

Checks
  markets   one row per ticker; open < close; finalized legs carry a result;
            game events settle exactly one leg YES; threshold ladders settle
            monotonically in the strike, and agree with the numeric
            ``expiration_value`` where it is one
  trades    unique trade_id; yes + no = 100c; price in (0, 100); count > 0;
            taker_side in {yes, no}; created within [open - 1h, close + 1h];
            contracts pulled vs Kalshi's recorded volume; a leg's trades all in
            one part file (a second copy means a resume re-pulled it)
  candles   bid <= ask; prices in [0, 100]; one row per (ticker, minute);
            inside the leg's life; complementary game legs not crossed
            (bid_A + bid_B <= 100 <= ask_A + ask_B); no nulls in bid/ask close,
            volume, open interest (the quote layer reads a null bid as 0 and a
            null ask as 100, i.e. one-sided, so a parse gap would pass silently);
            OHLC ordered (low <= open, close <= high); spacing matches the leg
            kind (1-min game-level, hourly season-long); coverage of the window
            the pull asked for, and legs marked done with no candles at all
  plausibility  prices that do not reflect fair value: (a) off-market trades --
            a trade >= SPIKE_C from the previous same-side trade (YES takers
            track the ask, NO takers the bid) with the next same-side trade back
            within REVERT_C, all within SPIKE_WINDOW (a taker sweeping an empty
            book, or a maker's stray order being lifted; the bid-ask bounce of a
            wide market is not flagged); (b) quote spikes -- a two-sided candle
            bid or ask close that jumps >= SPIKE_C for one minute and comes back;
            (c) trades against their minute's candle (ts = end of minute): the
            price must lie in [bid_low - 1, ask_high + 1]

Writes data/sports/integrity/{report.md, flags_*.parquet}; data/ is git-ignored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "data" / "sports"
HOLD = SP / "_HOLDOUT_DO_NOT_USE"
OUT = SP / "integrity"
CACHE = OUT / "cache"
SLACK = dt.timedelta(hours=1)


def markets() -> pl.DataFrame:
    frames = [pl.read_parquet(SP / "markets.parquet").with_columns(pl.lit(False).alias("holdout")),
              pl.read_parquet(HOLD / "markets.parquet").with_columns(pl.lit(True).alias("holdout"))]
    return pl.concat(frames, how="diagonal_relaxed")


def part_files(kind: str) -> list[Path]:
    # skip files still being written: the pull writes each part in one call,
    # so a file untouched for 30 s is complete
    now = time.time()
    return sorted(p for root in (SP, HOLD) for p in (root / kind).glob("*/*.parquet")
                  if now - p.stat().st_mtime > 30)


def cached(p: Path, fn) -> pl.DataFrame:
    st = p.stat()
    key = hashlib.sha1(f"{p}|{st.st_size}|{st.st_mtime_ns}|{fn.__name__}".encode()).hexdigest()[:16]
    c = CACHE / f"{key}.parquet"
    if c.exists():
        return pl.read_parquet(c)
    df = fn(p).with_columns(pl.lit(str(p.relative_to(SP))).alias("file"))
    CACHE.mkdir(parents=True, exist_ok=True)
    df.write_parquet(c)
    return df


# --------------------------------------------------------------------------- markets

STRIKE = re.compile(r"(?i)over\s+([\d.]+)")


def check_markets(mk: pl.DataFrame) -> tuple[list[str], dict[str, pl.DataFrame]]:
    lines, flags = [], {}
    dup = mk.group_by("ticker").len().filter(pl.col("len") > 1)
    bad_time = mk.filter(pl.col("open_time").is_null() | pl.col("close_time").is_null()
                         | (pl.col("close_time") <= pl.col("open_time")))
    fin_nores = mk.filter((pl.col("status") == "finalized") & ~pl.col("result").is_in(["yes", "no", "scalar"]))
    unsettled = mk.filter(pl.col("status") != "finalized")
    lines += [f"- duplicate tickers: **{dup.height}**",
              f"- missing or inverted open/close time: **{bad_time.height}**",
              f"- finalized without a yes/no/scalar result: **{fin_nores.height}**",
              f"- not finalized (status): {unsettled.group_by('status').len().rows()}"]
    flags["markets_time"] = bad_time.select("ticker", "open_time", "close_time", "holdout")

    # game events: exactly one YES among the legs
    ge = (mk.filter(pl.col("series_ticker").str.contains(r"GAME$") & (pl.col("status") == "finalized"))
            .group_by("event_ticker", "series_ticker", "holdout")
            .agg(pl.len().alias("legs"), (pl.col("result") == "yes").sum().alias("n_yes"),
                 (pl.col("result") == "scalar").sum().alias("n_scalar"), pl.col("close_time").max()))
    odd = ge.filter(pl.col("n_yes") != 1)
    lines.append(f"- game events: {ge.height:,}; not exactly one YES: **{odd.height}** "
                 f"(of which all-scalar, e.g. postponed/tied: {odd.filter(pl.col('n_scalar') == pl.col('legs')).height})")
    flags["game_events"] = odd

    # threshold ladders ("Over X ..."): monotone in X within (event, side), and vs expiration_value
    lad = (mk.filter(pl.col("result").is_in(["yes", "no"]) & pl.col("yes_sub_title").is_not_null())
             .with_columns(pl.col("yes_sub_title").str.extract(STRIKE.pattern, 1).cast(pl.Float64, strict=False).alias("strike"),
                           # one ladder = the ticker minus its trailing strike (team / player / total)
                           pl.col("ticker").str.replace(r"\d+(?:\.\d+)?$", "").alias("side"),
                           pl.col("yes_sub_title").str.contains(r"(?i)\bwins?\b").alias("is_spread"),
                           pl.col("expiration_value").cast(pl.Float64, strict=False).alias("ev_num"))
             .filter(pl.col("strike").is_not_null()))
    # yes must be a prefix in ascending strike: flag an event/side where a higher strike is YES above a NO
    mono = (lad.sort("strike").group_by("event_ticker", "side", "holdout")
               .agg(pl.col("result").alias("res"), pl.len().alias("n"))
               .with_columns(pl.col("res").list.eval((pl.element() == "yes").cast(pl.Int8).diff().fill_null(0).max())
                             .list.first().alias("rises"))
               .filter(pl.col("rises") > 0))
    # expiration_value is the realised statistic for totals/props; for spreads it is the
    # winner's margin whichever side won, so it is only comparable on non-spread legs
    evmis = lad.filter(~pl.col("is_spread") & pl.col("ev_num").is_not_null()
                       & ((pl.col("ev_num") > pl.col("strike")) != (pl.col("result") == "yes")))
    # spreads: both teams cannot win by over anything in the same game
    both = (lad.filter(pl.col("is_spread") & (pl.col("result") == "yes"))
               .group_by("event_ticker", "holdout").agg(pl.col("side").n_unique().alias("sides"))
               .filter(pl.col("sides") > 1))
    lines.append(f"- threshold ladders ('Over X'): {lad.height:,} legs in "
                 f"{lad.select('event_ticker', 'side').n_unique():,} ladders; non-monotone settlement: **{mono.height}**; "
                 f"result disagrees with numeric expiration_value: **{evmis.height}** "
                 f"(of {lad.filter(~pl.col('is_spread') & pl.col('ev_num').is_not_null()).height:,} non-spread legs with a numeric value); "
                 f"spread events with both teams YES: **{both.height}**")
    flags["spread_both_yes"] = both
    flags["ladder_monotone"] = mono.drop("res")
    flags["ladder_expval"] = evmis.select("ticker", "yes_sub_title", "strike", "expiration_value", "result", "holdout")
    return lines, flags


# --------------------------------------------------------------------------- trades

def trade_summary(p: Path) -> pl.DataFrame:
    t = pl.read_parquet(p)
    return (t.group_by("ticker").agg(
        pl.len().alias("n"), pl.col("trade_id").n_unique().alias("n_unique"),
        pl.col("count").sum().alias("contracts"),
        pl.col("created_time").min().alias("t_min"), pl.col("created_time").max().alias("t_max"),
        ((pl.col("yes_price") + pl.col("no_price") - 100).abs() > 0.01).sum().alias("bad_sum"),
        ((pl.col("yes_price") <= 0) | (pl.col("yes_price") >= 100)).sum().alias("bad_range"),
        (pl.col("count") <= 0).sum().alias("bad_count"),
        (~pl.col("taker_side").is_in(["yes", "no"])).sum().alias("bad_side"),
        pl.any_horizontal(pl.all().exclude("is_block_trade").is_null()).sum().alias("nulls")))


def check_trades(mk: pl.DataFrame) -> tuple[list[str], dict[str, pl.DataFrame]]:
    files = part_files("trades")
    if not files:
        return ["- no trade files yet"], {}
    per = pl.concat([cached(p, trade_summary) for p in files], how="diagonal_relaxed")
    multi = per.group_by("ticker").agg(pl.col("file").n_unique().alias("files")).filter(pl.col("files") > 1)
    agg = per.group_by("ticker").agg(pl.col("n", "n_unique", "contracts", "bad_sum", "bad_range", "bad_count",
                                            "bad_side", "nulls").sum(), pl.col("t_min").min(), pl.col("t_max").max())
    j = agg.join(mk.select("ticker", "open_time", "close_time", "volume", "holdout"), on="ticker", how="left")
    ratio = (pl.col("contracts") / pl.col("volume"))
    j = j.with_columns(ratio.alias("ratio"))
    band = (j.filter(pl.col("volume") > 0)
             .with_columns(pl.when(pl.col("ratio") < 0.99).then(pl.lit("< 99%"))
                           .when(pl.col("ratio") <= 1.001).then(pl.lit("99–100.1%"))
                           .otherwise(pl.lit("> 100.1%")).alias("band"))
             .group_by("holdout", "band").agg(pl.len().alias("legs"), pl.col("volume").sum().alias("volume"))
             .sort("holdout", "band"))
    early = j.filter(pl.col("t_min") < pl.col("open_time") - SLACK)
    late = j.filter(pl.col("t_max") > pl.col("close_time") + SLACK)
    unknown = j.filter(pl.col("volume").is_null())
    tot = j.select(pl.col("n").sum(), (pl.col("n") - pl.col("n_unique")).sum().alias("dups"), pl.col("bad_sum").sum(),
                   pl.col("bad_range").sum(), pl.col("bad_count").sum(), pl.col("bad_side").sum(), pl.col("nulls").sum()).row(0)
    lines = [f"- files {len(files):,}; legs {j.height:,} (holdout {j['holdout'].sum():,}); trades **{tot[0]:,}**",
             f"- duplicate trade_id within a leg: **{tot[1]:,}**; legs split across >1 file (resume re-pull): **{multi.height}**",
             f"- yes + no ≠ 100c: **{tot[2]:,}**; price outside (0, 100): **{tot[3]:,}**; count ≤ 0: **{tot[4]:,}**; "
             f"bad taker_side: **{tot[5]:,}**; rows with a null: **{tot[6]:,}**",
             f"- trades > 1 h before open: **{early.height}** legs; > 1 h after close: **{late.height}** legs; "
             f"legs not in the listing: **{unknown.height}**",
             "- contracts pulled / Kalshi volume, by leg:", "",
             "| holdout | band | legs | volume |", "|---|---|---|---|"]
    lines += [f"| {h} | {b} | {n:,} | {v:,.0f} |" for h, b, n, v in band.rows()]
    flags = {"trades_volume": j.filter((pl.col("ratio") < 0.99) | (pl.col("ratio") > 1.001))
                               .select("ticker", "contracts", "volume", "ratio", "holdout").sort("ratio"),
             "trades_timing": pl.concat([early.with_columns(pl.lit("early").alias("why")),
                                         late.with_columns(pl.lit("late").alias("why"))], how="diagonal_relaxed")
                               .select("ticker", "why", "t_min", "t_max", "open_time", "close_time", "holdout"),
             "trades_rows": j.filter(pl.sum_horizontal("bad_sum", "bad_range", "bad_count", "bad_side", "nulls") > 0)
                             .select("ticker", "n", "bad_sum", "bad_range", "bad_count", "bad_side", "nulls", "holdout"),
             "trades_multifile": multi}
    return lines, flags


# --------------------------------------------------------------------------- candles

REQUIRED_CANDLE_COLS = ["yes_bid_close", "yes_ask_close", "volume", "open_interest"]


def candle_summary_v2(p: Path) -> pl.DataFrame:
    c = pl.read_parquet(p)
    for col in REQUIRED_CANDLE_COLS:           # a missing column counts as all-null
        if col not in c.columns:
            c = c.with_columns(pl.lit(None, pl.Float64).alias(col))
    cols = c.columns
    bid, ask = ("yes_bid_close" if "yes_bid_close" in cols else None), ("yes_ask_close" if "yes_ask_close" in cols else None)
    ohlc_bad = pl.lit(0)
    for side in ("yes_bid", "yes_ask"):
        o, h, l, cl = (pl.col(f"{side}_{f}") for f in ("open", "high", "low", "close"))
        if all(f"{side}_{f}" in cols for f in ("open", "high", "low", "close")):
            ohlc_bad = ohlc_bad + ((l > o) | (l > cl) | (h < o) | (h < cl)).cast(pl.Int64)
    c = c.sort("ticker", "ts")
    ex = [pl.len().alias("n"), pl.col("ts").n_unique().alias("n_unique_ts"),
          pl.col("ts").min().alias("t_min"), pl.col("ts").max().alias("t_max"),
          pl.col("volume").sum().alias("candle_volume"),
          *[pl.col(x).null_count().alias(f"null_{x}") for x in REQUIRED_CANDLE_COLS],
          (ohlc_bad > 0).sum().alias("ohlc_bad"),
          pl.col("ts").diff().dt.total_seconds().filter(pl.col("ts").diff().dt.total_seconds() > 0).min().alias("min_gap_s"),
          (pl.col("ts").dt.second() != 0).sum().alias("off_minute")]
    if bid and ask:
        # a one-sided book shows as bid 0 / ask 100; only a two-sided book can be crossed
        ex += [((pl.col(bid) > pl.col(ask)) & (pl.col(bid) > 0) & (pl.col(ask) < 100)).sum().alias("crossed"),
               ((pl.col(bid) < 0) | (pl.col(ask) > 100)).sum().alias("bad_range")]
    return c.group_by("ticker").agg(ex)


def check_candles(mk: pl.DataFrame) -> tuple[list[str], dict[str, pl.DataFrame]]:
    files = part_files("candles")
    if not files:
        return ["- no candle files yet (the candle stage starts after the trades)"], {}
    per = pl.concat([cached(p, candle_summary_v2) for p in files], how="diagonal_relaxed")
    nulls = [f"null_{x}" for x in REQUIRED_CANDLE_COLS]
    agg = per.group_by("ticker").agg(pl.col("n", "n_unique_ts", "candle_volume", "crossed", "bad_range",
                                            "ohlc_bad", "off_minute", *nulls).sum(),
                                     pl.col("t_min").min(), pl.col("t_max").max(), pl.col("min_gap_s").min())
    j = agg.join(mk.select("ticker", "open_time", "close_time", "volume", "holdout", "event_ticker", "series_ticker"),
                 on="ticker", how="left")
    outside = j.filter((pl.col("t_min") < pl.col("open_time") - dt.timedelta(minutes=2))
                       | (pl.col("t_max") > pl.col("close_time") + dt.timedelta(minutes=2)))
    tot = j.select(pl.col("n").sum(), (pl.col("n") - pl.col("n_unique_ts")).sum().alias("dups"),
                   pl.col("crossed").sum(), pl.col("bad_range").sum()).row(0)
    lines = [f"- files {len(files):,}; legs {j.height:,}; candles **{tot[0]:,}**",
             f"- duplicate (ticker, minute): **{tot[1]:,}**; crossed two-sided book (bid > ask): **{tot[2]:,}**; "
             f"price outside [0, 100]: **{tot[3]:,}**; legs with candles outside their life: **{outside.height}**"]
    nt = j.select(pl.col(nulls).sum(), pl.col("ohlc_bad").sum(), pl.col("off_minute").sum()).row(0, named=True)
    lines.append("- nulls (must be 0; the quote layer reads a null bid/ask as one-sided): "
                 + ", ".join(f"{k[5:]} **{nt[k]:,}**" for k in nulls)
                 + f"; OHLC out of order (low > open/close or high < open/close): **{nt['ohlc_bad']:,}**; "
                 f"ts not on a minute boundary: **{nt['off_minute']:,}**")

    # spacing and coverage against the window the pull asked for (same rule as fetch_sports)
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_sports import candle_window
    now = dt.datetime.now(dt.timezone.utc)
    want = []
    for m in mk.filter(pl.col("volume") > 0).select("ticker", "event_ticker", "series_ticker", "open_time",
                                                    "close_time").iter_rows(named=True):
        w = candle_window(m, 12.0)
        if w:
            want.append((m["ticker"], w[2], (min(w[1], now) - w[0]).total_seconds() / 60 / w[2]))
    want = pl.DataFrame(want, schema=["ticker", "interval_min", "periods"], orient="row")
    jw = j.join(want, on="ticker", how="left").with_columns((pl.col("n") / pl.col("periods")).alias("coverage"))
    spacing = jw.filter(pl.col("min_gap_s").is_not_null()
                        & (pl.col("min_gap_s") != pl.col("interval_min") * 60)
                        & ~((pl.col("interval_min") == 1) & (pl.col("min_gap_s") % 60 == 0)))  # 1-min legs skip idle minutes
    hourly_wrong = jw.filter((pl.col("interval_min") == 60) & (pl.col("min_gap_s") < 3600))
    minute_wrong = jw.filter((pl.col("interval_min") == 1) & (pl.col("min_gap_s") >= 3600) & (pl.col("n") > 3))
    cov = jw.filter(pl.col("periods") >= 10)["coverage"]
    lines.append(f"- spacing: hourly legs with sub-hour candles **{hourly_wrong.height:,}**; 1-min legs spaced hourly "
                 f"**{minute_wrong.height:,}**; other irregular spacing **{spacing.height - hourly_wrong.height:,}**")
    if cov.len():
        lines.append(f"- coverage (candles / periods asked for, legs asking >= 10): median {cov.median():.2f}, "
                     f"p10 {cov.quantile(0.1):.2f}; below 5%: **{(cov < 0.05).sum():,}** of {cov.len():,} "
                     "(Kalshi skips periods before the first quote, so well below 1 is normal for thin legs)")
    done_f = SP / "state" / "candles_done.txt"
    if done_f.exists():
        done = pl.DataFrame({"ticker": list(set(done_f.read_text().split()))})
        empty = done.join(j.select("ticker"), on="ticker", how="anti")
        # part files under 30 s old are skipped above, yet their legs may already be
        # marked done: look those legs up in every file before calling them empty
        fresh = [str(p) for root in (SP, HOLD) for p in (root / "candles").glob("*/*.parquet")]
        if empty.height and fresh:
            seen = pl.scan_parquet(fresh).select("ticker").filter(
                pl.col("ticker").is_in(empty["ticker"].implode())).unique().collect()
            empty = empty.join(seen, on="ticker", how="anti")
        empty = empty.join(mk.select("ticker", "holdout", "volume", "series_ticker"), on="ticker", how="left")
        by = empty.group_by("holdout").len().sort("holdout").rows()
        lines.append(f"- legs marked done: {done.height:,}; done but no candles at all: **{empty.height:,}** "
                     f"({', '.join(f'holdout={h}: {n:,}' for h, n in by)}; median volume "
                     f"{empty['volume'].median() or 0:,.0f} -- thin legs with no quote in the window are expected)")
        flags_empty = empty
    else:
        flags_empty = None

    # complementary game legs: bid_A + bid_B <= 100 <= ask_A + ask_B, minute by minute (working block only)
    games = j.filter(pl.col("series_ticker").str.contains(r"GAME$") & ~pl.col("holdout"))
    pairs = (mk.filter(pl.col("series_ticker").str.contains(r"GAME$")).group_by("event_ticker").len()
               .filter(pl.col("len") == 2)["event_ticker"])
    games = games.filter(pl.col("event_ticker").is_in(pairs.implode()))
    viol = None
    if games.height:
        gfiles = [p for p in files if "_HOLDOUT" not in str(p) and p.parent.name.endswith("GAME")]
        c = (pl.scan_parquet(gfiles).filter(pl.col("ticker").is_in(games["ticker"].implode()))
               .select("ticker", "ts", "yes_bid_close", "yes_ask_close").collect()
               .join(games.select("ticker", "event_ticker"), on="ticker"))
        two = (c.filter((pl.col("yes_bid_close") > 0) & (pl.col("yes_ask_close") < 100))
                 .group_by("event_ticker", "ts").agg(pl.len().alias("k"), pl.col("yes_bid_close").sum().alias("bids"),
                                                     pl.col("yes_ask_close").sum().alias("asks"))
                 .filter(pl.col("k") == 2))
        viol = two.filter((pl.col("bids") > 100) | (pl.col("asks") < 100))
        lines.append(f"- complementary game legs, minutes with both books two-sided: {two.height:,}; "
                     f"crossed across the pair (bids > 100 or asks < 100): **{viol.height:,}** "
                     f"in {viol['event_ticker'].n_unique() if viol.height else 0} events")
    flags = {"candles_outside": outside.select("ticker", "t_min", "t_max", "open_time", "close_time", "holdout"),
             "candles_rows": j.filter((pl.col("crossed") > 0) | (pl.col("bad_range") > 0) | (pl.col("n") > pl.col("n_unique_ts"))
                                      | (pl.col("ohlc_bad") > 0) | (pl.col("off_minute") > 0)
                                      | pl.any_horizontal(pl.col(c) > 0 for c in nulls))
                              .select("ticker", "n", "n_unique_ts", "crossed", "bad_range", "ohlc_bad", "off_minute",
                                      *nulls, "holdout"),
             "candles_spacing": jw.filter(pl.col("ticker").is_in(spacing["ticker"].implode()))
                                  .select("ticker", "interval_min", "min_gap_s", "n", "holdout"),
             "candles_empty": flags_empty}
    if viol is not None:
        flags["candles_pair_cross"] = viol

    # cross-source: candle volume vs trade contracts per leg (both present)
    tfiles = part_files("trades")
    if tfiles:
        tv = (pl.concat([cached(p, trade_summary) for p in tfiles], how="diagonal_relaxed")
                .group_by("ticker").agg(pl.col("contracts").sum()))
        x = j.join(tv, on="ticker").filter(pl.col("contracts") > 0).with_columns(
            (pl.col("candle_volume") / pl.col("contracts")).alias("r"))
        lines.append(f"- candle volume / trade contracts, per leg: median {x['r'].median():.3f}; "
                     f"outside [0.99, 1.01]: **{x.filter((pl.col('r') < 0.99) | (pl.col('r') > 1.01)).height:,}** of {x.height:,} "
                     "(expected below 1 where candles cover only part of a leg's life)")
    return lines, flags


# --------------------------------------------------------------------------- plausibility

SPIKE_C, REVERT_C = 30.0, 10.0                # a jump of >= 30c that comes back within 10c
SPIKE_WINDOW = dt.timedelta(minutes=30)       # neighbouring same-side trades within 30 min


def offmarket_trades_v1(p: Path) -> pl.DataFrame:
    """Off-market trades in one part file (a leg's trades all sit in one file)."""
    t = pl.read_parquet(p, columns=["trade_id", "ticker", "created_time", "yes_price", "count", "taker_side"])
    by = ["ticker", "taker_side"]
    t = t.sort("ticker", "taker_side", "created_time").with_columns(
        pl.col("yes_price").shift(1).over(by).alias("prev_same"),
        pl.col("yes_price").shift(-1).over(by).alias("next_same"),
        (pl.col("created_time") - pl.col("created_time").shift(1).over(by)).alias("dt_prev"),
        (pl.col("created_time").shift(-1).over(by) - pl.col("created_time")).alias("dt_next"))
    off = t.filter(((pl.col("yes_price") - pl.col("prev_same")).abs() >= SPIKE_C)
                   & ((pl.col("next_same") - pl.col("prev_same")).abs() <= REVERT_C)
                   & (pl.col("dt_prev") <= SPIKE_WINDOW) & (pl.col("dt_next") <= SPIKE_WINDOW))
    return off.select("trade_id", "ticker", "created_time", "taker_side", "prev_same", "yes_price", "next_same",
                      "count").with_columns(pl.lit(t.height).alias("file_trades"),
                                            pl.lit(t["count"].sum()).alias("file_contracts"))


def quote_spikes_v1(p: Path) -> pl.DataFrame:
    """One-minute bid/ask spikes that revert, on two-sided 1-min candles, in one part file."""
    c = pl.read_parquet(p, columns=["ticker", "ts", "yes_bid_close", "yes_ask_close"]).sort("ticker", "ts")
    two = (pl.col("yes_bid_close") > 0) & (pl.col("yes_ask_close") < 100)
    c = c.with_columns(two.alias("two"), (pl.col("ts") - pl.col("ts").shift(1).over("ticker")).alias("gap_prev"),
                       (pl.col("ts").shift(-1).over("ticker") - pl.col("ts")).alias("gap_next"))
    out = []
    for side in ("yes_bid_close", "yes_ask_close"):
        x = c.with_columns(pl.col(side).shift(1).over("ticker").alias("prev"),
                           pl.col(side).shift(-1).over("ticker").alias("next"),
                           pl.col("two").shift(1).over("ticker").alias("two_prev"),
                           pl.col("two").shift(-1).over("ticker").alias("two_next"))
        out.append(x.filter(((pl.col(side) - pl.col("prev")).abs() >= SPIKE_C)
                            & ((pl.col("next") - pl.col("prev")).abs() <= REVERT_C)
                            & pl.col("two_prev") & pl.col("two_next")
                            & (pl.col("gap_prev") == dt.timedelta(minutes=1))
                            & (pl.col("gap_next") == dt.timedelta(minutes=1)))
                    .select("ticker", "ts", pl.lit(side[4:7]).alias("side"), "prev", pl.col(side).alias("value"), "next"))
    res = pl.concat(out)
    return res.with_columns(pl.lit(c.height).alias("file_candles"),
                            pl.lit(int(c["two"].sum())).alias("file_two_sided"))


CROSS_SAMPLE_LEGS = 3_000                   # legs per run for check (c); --full-cross checks every leg
FULL_CROSS = False


def trade_vs_candle(series_dir: Path, tickers: list[str] | None = None) -> dict:
    """Trades outside their minute's candle bid/ask range, for one series directory
    (only ``tickers`` when given: a full pass joins every trade and is heavy)."""
    root = series_dir.parents[1]
    tfiles = sorted((root / "trades" / series_dir.name).glob("*.parquet"))
    cfiles = [f for f in sorted(series_dir.glob("*.parquet")) if time.time() - f.stat().st_mtime > 30]
    if not tfiles or not cfiles:
        return {"matched": 0, "outside": 0, "rows": None}
    c = pl.scan_parquet(cfiles).select("ticker", "ts", "yes_bid_low", "yes_ask_high")
    t = pl.scan_parquet(tfiles).select("trade_id", "ticker", "created_time", "yes_price", "taker_side")
    if tickers is not None:
        c = c.filter(pl.col("ticker").is_in(tickers))
        t = t.filter(pl.col("ticker").is_in(tickers))
    t = t.with_columns((pl.col("created_time").dt.truncate("1m") + pl.duration(minutes=1)).alias("ts"))
    j = t.join(c, on=["ticker", "ts"], how="inner")
    bad = (pl.col("yes_price") < pl.col("yes_bid_low") - 1) | (pl.col("yes_price") > pl.col("yes_ask_high") + 1)
    n = j.select(pl.len()).collect(engine="streaming").item()
    rows = j.filter(bad).collect(engine="streaming")
    return {"matched": n, "outside": rows.height, "rows": rows}


def check_plausibility(mk: pl.DataFrame) -> tuple[list[str], dict[str, pl.DataFrame]]:
    lines, flags = [], {}
    hold = mk.select("ticker", "holdout", "series_ticker")

    tfiles = part_files("trades")
    if tfiles:
        per = pl.concat([cached(p, offmarket_trades_v1) for p in tfiles], how="diagonal_relaxed")
        tot_tr = per.group_by("file").agg(pl.col("file_trades").first(), pl.col("file_contracts").first())
        n_tr, n_ct = int(tot_tr["file_trades"].sum() or 0), float(tot_tr["file_contracts"].sum() or 0)
        off = per.filter(pl.col("trade_id").is_not_null()).join(hold, on="ticker", how="left")
        big = off.filter((pl.col("yes_price") - pl.col("prev_same")).abs() >= 50)
        lines.append(f"- (a) off-market trades (same-side jump >= {SPIKE_C:.0f}c that reverts within {REVERT_C:.0f}c, "
                     f"{SPIKE_WINDOW.total_seconds()/60:.0f} min): **{off.height:,}** of {n_tr:,} trades "
                     f"({off.height / max(n_tr, 1):.1e}), {off['count'].sum() / max(n_ct, 1):.1e} of contracts, "
                     f"in {off['ticker'].n_unique():,} legs; >= 50c: **{big.height:,}**; "
                     f"holdout {int(off['holdout'].sum()):,} [counts only]")
        fam = (off.filter(~pl.col("holdout")).group_by("series_ticker").agg(pl.len().alias("off_trades"))
                  .sort("off_trades", descending=True).head(8))
        if fam.height:
            lines.append("  most affected series (working block): "
                         + ", ".join(f"{s} {n:,}" for s, n in fam.rows()))
        flags["plaus_offmarket_trades"] = off.drop("file", "file_trades", "file_contracts")

    cfiles = part_files("candles")
    if cfiles:
        per = pl.concat([cached(p, quote_spikes_v1) for p in cfiles], how="diagonal_relaxed")
        tot_c = per.group_by("file").agg(pl.col("file_two_sided").first())
        n_two = int(tot_c["file_two_sided"].sum() or 0)
        sp = per.filter(pl.col("ticker").is_not_null()).join(hold, on="ticker", how="left")
        by_side = dict(sp.group_by("side").len().rows())
        lines.append(f"- (b) one-minute quote spikes (two-sided bid/ask close jumps >= {SPIKE_C:.0f}c and returns "
                     f"within {REVERT_C:.0f}c): **{sp.height:,}** over {n_two:,} two-sided candles "
                     f"({sp.height / max(n_two, 1):.1e}); bid {by_side.get('bid', 0):,}, ask {by_side.get('ask', 0):,}; "
                     f"in {sp['ticker'].n_unique():,} legs; holdout {int(sp['holdout'].sum()):,} [counts only]")
        flags["plaus_quote_spikes"] = sp.drop("file", "file_candles", "file_two_sided")

        matched = outside = 0
        rows = []
        # legs with candles, grouped by series directory; a random sample unless --full-cross
        cand = (pl.concat([cached(p, candle_summary_v2) for p in cfiles], how="diagonal_relaxed")
                  .select("ticker", "file").unique("ticker"))
        if not FULL_CROSS and cand.height > CROSS_SAMPLE_LEGS:
            cand = cand.sample(CROSS_SAMPLE_LEGS, seed=int(time.time()))
        cand = cand.with_columns(pl.col("file").map_elements(lambda f: str(Path(f).parent), return_dtype=pl.String)
                                 .alias("dir"))
        for d, tk in cand.group_by("dir").agg(pl.col("ticker")).rows():
            r = trade_vs_candle(SP / d, None if FULL_CROSS else tk)
            matched += r["matched"]; outside += r["outside"]
            if r["rows"] is not None and r["rows"].height:
                rows.append(r["rows"])
        lines.append(f"- (c) trades matched to their minute's candle "
                     f"({'every leg' if FULL_CROSS else f'random sample of {cand.height:,} legs'}): {matched:,}; outside "
                     f"[bid_low - 1, ask_high + 1]: **{outside:,}** (must be 0: trades print against the book the "
                     "candle records)")
        if rows:
            flags["plaus_trade_vs_candle"] = pl.concat(rows, how="diagonal_relaxed").join(hold, on="ticker", how="left")
    if not lines:
        lines = ["- nothing to check yet"]
    lines.append("- Flagged prices are real executions or book states, not parse errors; they are transient and "
                 "not fair value. The quote layer should drop off-market trades before using trades as bid/ask proxies.")
    return lines, flags


# --------------------------------------------------------------------------- main

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mk = markets()
    t0 = time.time()
    sections = [("Markets", check_markets(mk)), ("Trades", check_trades(mk)), ("Candles", check_candles(mk)),
                ("Plausibility", check_plausibility(mk))]
    head = [f"# Sports pull: integrity report",
            f"",
            f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC by `scripts/check_sports_data.py` "
            f"({time.time() - t0:.0f} s). Listing: {mk.height:,} legs ({mk['holdout'].sum():,} holdout). "
            f"Holdout rows are counted, never shown. Flagged rows: `flags_*.parquet` alongside."]
    body = []
    for name, (lines, flags) in sections:
        body += ["", f"## {name}", ""] + lines
        for k, df in flags.items():
            if df is not None and "holdout" in df.columns:
                df = df.filter(~pl.col("holdout"))      # holdout rows are counted above, never written
            if df is not None and df.height:
                df.write_parquet(OUT / f"flags_{k}.parquet")
            else:
                (OUT / f"flags_{k}.parquet").unlink(missing_ok=True)   # no stale flags from an earlier run
    (OUT / "report.md").write_text("\n".join(head + body) + "\n")
    print("\n".join(head + body), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, help="rerun every N minutes until the pull process exits")
    ap.add_argument("--full-cross", action="store_true",
                    help="check (c) on every leg instead of a sample (heavy: joins every trade to its candle)")
    a = ap.parse_args()
    global FULL_CROSS
    FULL_CROSS = a.full_cross
    run()
    if a.every:
        import subprocess
        while subprocess.run(["pgrep", "-f", "fetch_sports.py (trades|candles)"], capture_output=True).returncode == 0:
            time.sleep(a.every * 60)
            run()


if __name__ == "__main__":
    main()
