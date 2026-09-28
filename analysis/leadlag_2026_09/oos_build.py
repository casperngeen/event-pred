#!/usr/bin/env python
"""Build the 2026 out-of-sample inputs for the frozen test (strategy_spec §9).

    venv/bin/python analysis/leadlag_2026_09/oos_build.py --block val --tape raw   # check
    venv/bin/python analysis/leadlag_2026_09/oos_build.py --block oos              # the build

**Data only.** This writes the panels the §9 test reads and prints row counts.
It never prints anything computed from ``win``, ``result`` or a surprise's
sign -- the test is run once, by a separate script, and this must not leak it.

What it does, in the order §9 lists:

1. **Markets.** 2026 legs come from the two raw API pulls
   (``markets_api_pull/`` for Jan-Jun, ``_OOS_DO_NOT_USE_trades_2026_ext/``
   for Jun-Sep; the later fetch wins on overlap). ``data/markets/`` also lists
   2026 markets, but that snapshot was fetched 2025-11-24, before any of them
   settled, so its ``result`` is not usable.
2. **Tape** (§9 step 7). In-archive trades on the 2026 tickers (listed in 2025,
   so their early prints are in ``data/trades/``) plus the combined 2026 file.
   Exact duplicate records are dropped on ``trade_id``, then every
   ``(ticker, created_time)`` group is collapsed to one print at the
   size-weighted price, as ``spec_v2.py`` does. ``--tape raw`` skips the
   collapse; it exists only so ``--block val`` can reproduce the committed
   in-sample panel.
3. **Surprise panel** via ``build_surprise_panel`` and ``gate_panel``,
   unchanged, with true settlement values from ``expiration_value``.
4. **Leg panel** via ``build_panel.build``, unchanged.

Two deviations from running ``build_panel.py`` as-is, both needed to keep
the in-sample choices fixed:

* **The trigger and target sets are the in-sample ones**, read from
  ``out/leadlag_legs.parquet``. Re-deriving them would apply
  ``usable_triggers(..., 10)`` to nine months of data, which no monthly series
  can pass.
* **A row is out of sample when its outcome is.** The test keys on the
  target's ``close_time >= OOS_START``, not the trigger's. A December 2025
  trigger whose first target settles in January 2026 was dropped from the
  in-sample panel at the wall (``leakage_audit.py``: zero rows past it), so
  its outcome has never been seen; it belongs here. To pick that first target
  exactly as ``build_panel`` does, the block reaches back ``LEAD`` into the
  in-sample archive for triggers and for candidate targets, and the rows whose
  target settles before the wall are dropped at the end. Decided 2026-09-26,
  before the test was run.

Three series were relaunched in April 2026 under new tickers
(``KXUSGASCPI``, ``KXUSEDCARCPI``, ``KXSHELTERCPI``) and are mapped back to
their in-sample names by ``RENAMED``; see the note there. Decided 2026-09-26,
before the test was run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "stg_infra")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_panel as bp  # noqa: E402
from stg.panel._io import (_MARKET_COLS, _TRADE_COLS, _parse_settlement,  # noqa: E402
                           load_markets, load_settlement_values, scan_trades)
from stg.panel.registry import SPECS, ticker_prefixes  # noqa: E402
from stg.panel.surprise import build_surprise_panel, gate_panel  # noqa: E402
from stg.splits import OOS_START, TRAIN_VAL_SPLIT  # noqa: E402

DATA = Path("data")
PULLS = [DATA / "markets_api_pull" / "markets_api_pull_raw.jsonl",
         DATA / "_OOS_DO_NOT_USE_trades_2026_ext" / "markets_2026_ext_raw.jsonl"]
OOS_TRADES = DATA / "_OOS_DO_NOT_USE_trades_2026" / "trades_2026_oos_combined.parquet"
OOS_OUT = DATA / "_OOS_DO_NOT_USE_built_2026"
VAL_OUT = Path("analysis/leadlag_2026_09/out/oos_build_val")
IS_LEGS = Path("analysis/leadlag_2026_09/out/leadlag_legs.parquet")

# 2026 relaunches of in-sample series, mapped here and not in the registry so no
# in-sample result can move. The old ladders were on the MoM % change
# ("gas CPI rises more than 0%"); these are on the index level ("gasoline CPI
# for April above 335"). Given last month's level, which is public at entry,
# the two are a monotone map of each other, and z_surprise and the price-bucket
# terciles are unit-free, so nothing is refitted. CPIFOOD and CPIAPPAREL were
# discontinued in 2025 and have no successor.
LEAD = dt.timedelta(days=2 * bp.MAX_GAP_DAYS)   # trigger lookback + its target window

RENAMED = {"USGASCPI": "CPIGAS", "USEDCARCPI": "CPIUSEDCAR", "SHELTERCPI": "CPISHELTER"}


def _ts(s):
    return (dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            if s else None)


def _num(s):
    return None if s in (None, "") else int(round(float(s)))


def markets_2026() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(markets in the ``load_markets`` schema, event -> true settled value)."""
    recs: dict[str, dict] = {}
    for path in PULLS:                      # later file overwrites earlier
        fetched = dt.datetime.fromtimestamp(path.stat().st_mtime)
        for line in path.open():
            r = json.loads(line)
            if r.get("_empty") or not r.get("ticker"):
                continue
            ct = _ts(r.get("close_time"))
            if ct is None or ct < OOS_START:
                continue
            recs[r["ticker"]] = dict(
                ticker=r["ticker"], event_ticker=r["event_ticker"],
                market_type=r.get("market_type"), title=r.get("title"),
                yes_sub_title=r.get("yes_sub_title"),
                no_sub_title=r.get("no_sub_title"), status=r.get("status"),
                result=r.get("result") or None,
                open_time=_ts(r.get("open_time")), close_time=ct,
                volume=_num(r.get("volume_fp", r.get("volume"))),
                open_interest=_num(r.get("open_interest_fp", r.get("open_interest"))),
                _fetched_at=fetched,
                _expiration_value=r.get("expiration_value"),
            )
    mk = pl.DataFrame(list(recs.values()), schema_overrides={
        "open_time": pl.Datetime("ns", "UTC"), "close_time": pl.Datetime("ns", "UTC"),
        "_fetched_at": pl.Datetime("ns"), "volume": pl.Int64, "open_interest": pl.Int64})

    # Same parse and conflict check as stg.panel._io.load_settlement_values.
    vals: dict[str, set] = {}
    for ev, raw in zip(mk["event_ticker"], mk["_expiration_value"]):
        v = _parse_settlement(raw)
        if v is not None:
            vals.setdefault(ev, set()).add(v)
    bad = {e: s for e, s in vals.items() if len(s) > 1}
    if bad:
        raise AssertionError(f"conflicting expiration_value: {list(bad.items())[:5]}")
    sv = pl.DataFrame({"event_ticker": list(vals),
                       "resolved_value_true": [next(iter(s)) for s in vals.values()]},
                      schema={"event_ticker": pl.Utf8, "resolved_value_true": pl.Float64})

    mk = mk.select(_MARKET_COLS).with_columns(
        pl.col("ticker").str.replace(r"^KX", "").str.split("-").list.first()
        .replace(RENAMED).alias("series_raw"))
    return mk, sv


def tape(raw: pl.DataFrame, vwap: bool) -> pl.DataFrame:
    """Exact duplicates dropped; optionally ties collapsed to a VWAP print."""
    d = raw.unique(subset=["trade_id"], keep="first", maintain_order=True)
    if not vwap:
        return d
    return (d.group_by("ticker", "created_time")
            .agg(((pl.col("yes_price") * pl.col("count")).sum()
                  / pl.col("count").sum()).alias("yes_price"),
                 pl.col("count").sum(),
                 pl.col("trade_id").min())
            .with_columns((100.0 - pl.col("yes_price")).alias("no_price"),
                          pl.lit(None, pl.Utf8).alias("taker_side"))
            .select(_TRADE_COLS)
            .sort("ticker", "created_time"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["oos", "val"], required=True)
    ap.add_argument("--tape", choices=["vwap", "raw"], default="vwap")
    a = ap.parse_args()

    is_legs = pl.read_parquet(IS_LEGS)
    triggers = sorted(is_legs["trigger"].unique())
    targets = sorted(is_legs["target"].unique())
    lo, hi = (OOS_START, None) if a.block == "oos" else (TRAIN_VAL_SPLIT, OOS_START)
    out = OOS_OUT if a.block == "oos" else VAL_OUT
    out.mkdir(parents=True, exist_ok=True)
    print(f"block={a.block} [{lo.date()}, {hi.date() if hi else 'end'})  tape={a.tape}")
    print(f"frozen triggers ({len(triggers)}): {triggers}")
    print(f"frozen targets  ({len(targets)}): {targets}\n")

    if a.block == "oos":
        mk, sv = markets_2026()
        # The last LEAD of the in-sample archive, for December triggers and for
        # the target events they could match (see the module docstring).
        prefixes = [p for c in set(triggers) | set(targets) for p in ticker_prefixes(c)]
        is_mk = load_markets(is_only=True).filter(
            (pl.col("close_time") >= lo - LEAD) & pl.col("series_raw").is_in(prefixes))
        mk = pl.concat([is_mk, mk.select(is_mk.columns)], how="vertical_relaxed")
        sv = pl.concat([load_settlement_values(True), sv])
        tk = mk["ticker"].unique().to_list()
        raw = pl.concat([
            scan_trades(is_only=True).filter(pl.col("ticker").is_in(tk)).collect(),
            pl.read_parquet(OOS_TRADES).select(_TRADE_COLS),
        ])
    else:
        # Validation block: the in-sample archive, exactly as build_panel.py
        # reads it, restricted to events closing on/after TRAIN_VAL_SPLIT.
        mk = load_markets(is_only=True)
        sv = load_settlement_values(True)
        raw = scan_trades(is_only=True).collect()
    tr = tape(raw, vwap=a.tape == "vwap")
    print(f"trades: {raw.height} raw -> {tr.height} on the tape")

    # --- series coverage: which frozen series have markets in the block ---
    blk = mk.filter(pl.col("close_time") >= lo)
    if hi is not None:
        blk = blk.filter(pl.col("close_time") < hi)
    rows = []
    for s in sorted(set(triggers) | set(targets)):
        m = blk.filter(pl.col("series_raw").is_in(list(ticker_prefixes(s))))
        rows.append(dict(series=s, trigger=s in triggers, target=s in targets,
                         events=m["event_ticker"].n_unique(), legs=m.height))
    cov = pl.DataFrame(rows)
    with pl.Config(tbl_rows=40):
        print(cov)
    missing = cov.filter(pl.col("events") == 0)["series"].to_list()
    if missing:
        print(f"no markets in block for {missing}")

    # --- surprise panel: build_panel.py's main(), on this block ---
    sp_raw = build_surprise_panel(triggers, markets=mk, trades=tr.lazy(),
                                  settlements=sv, gated=False)
    sp = gate_panel(sp_raw).sort("series", "close_time")
    sp = sp.filter(pl.col("close_time") >= (lo - dt.timedelta(days=bp.MAX_GAP_DAYS)
                                            if a.block == "oos" else lo))
    if hi is not None:
        sp = sp.filter(pl.col("close_time") < hi)
    sp = (sp.with_columns((pl.col("surprise") / pl.col("implied_std")).alias("z_surprise"))
          .filter(pl.col("z_surprise").is_finite() & pl.col("s_pit").is_finite()))
    print(f"\nsurprise panel: {sp_raw.height} ungated -> {sp.height} gated, in block")
    print(sp.group_by("series").agg(pl.len().alias("events"),
                                    (pl.col("resolved_source") == "expiration_value")
                                    .sum().alias("true_value")).sort("series"))

    # --- leg panel: build_panel.build, unchanged, on the frozen sets ---
    bp.MIN_TRIGGER_EVENTS = 1   # the trigger set is fixed above, not re-derived
    tmk = mk.filter(pl.col("close_time") >= (lo - LEAD if a.block == "oos" else lo))
    if hi is not None:
        tmk = tmk.filter(pl.col("close_time") < hi)
    legs = bp.build(sp, targets, tmk, tr.lazy())
    if a.block == "oos":
        n_all = legs.height
        legs = legs.filter(pl.col("close_time") >= lo)
        print(f"\ndropped {n_all - legs.height} rows whose target settles before "
              f"{lo.date()} (already in the in-sample panel); "
              f"{legs.filter(pl.col('t_res') < lo).height} kept rows have a 2025 trigger")
    if legs.is_empty():
        print("no rows")
        return

    sp_raw.write_parquet(out / "surprise_panel_ungated.parquet")
    sp.write_parquet(out / "surprise_panel.parquet")
    legs.write_parquet(out / "leadlag_legs.parquet")
    tr.filter(pl.col("ticker").is_in(legs["target_ticker"].unique().to_list())) \
      .write_parquet(out / f"tape_{a.tape}.parquet")
    if a.block == "oos":
        mk.filter(pl.col("close_time") >= lo).write_parquet(out / "markets_2026.parquet")
        sv.write_parquet(out / "settlements_2026.parquet")

    print(f"\nleg rows: {legs.height}   trigger events: {legs['trigger_event'].n_unique()}"
          f"   target events: {legs['target_event'].n_unique()}"
          f"   pairs: {legs.select('trigger', 'target').unique().height}")
    print(f"t_res {legs['t_res'].min()} -> {legs['t_res'].max()}")
    print(f"close {legs['close_time'].min()} -> {legs['close_time'].max()}")
    print(legs.group_by("channel").agg(pl.len().alias("legs"),
                                       pl.col("target_event").n_unique().alias("tgt_events"))
          .sort("legs", descending=True))

    if a.block == "val":
        ref = is_legs.filter(pl.col("t_res") >= lo)
        key = ["trigger_event", "target_ticker"]
        both = ref.join(legs, on=key, how="inner", suffix="_new")
        print(f"\nvs committed panel on t_res >= {lo.date()}: committed {ref.height}, "
              f"rebuilt {legs.height}, matched {both.height}, "
              f"only committed {ref.join(legs, on=key, how='anti').height}, "
              f"only rebuilt {legs.join(ref, on=key, how='anti').height}")
        for c in ("target_event", "p_entry", "p0", "z_surprise", "signal"):
            n = both.filter(pl.col(c).ne_missing(pl.col(f"{c}_new"))).height
            print(f"  {c:<14} differs on {n} of {both.height}")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
