#!/usr/bin/env python
"""Build the (trigger event x target leg) settlement panel.

    venv/bin/python analysis/leadlag_2026_09/build_panel.py

The object every previous study in this repo declined to build. Stage 1,
``direction_study`` and ``channel_pooling`` all collapse a target event to **one
representative leg** and measure a **price change** on it. Both choices are
wrong for the question here:

* collapsing to one leg throws away the ladder, and the ladder is exactly where
  the initial-price dimension lives -- a 10c leg and a 90c leg of the *same
  event* are different bets on the same statistic;
* a price change is not what a hold-to-settlement trader earns.

So: one row per (trigger event, target event, target leg). Entry is the first
print strictly after the trigger resolves -- the earliest instant anyone could
have acted on the resolution -- and the outcome is the leg's settlement.

Scope
-----
**Threshold targets only.** A threshold leg settles YES iff ``X_target > K``,
so a shift in the target's central value moves every leg monotonically and the
ladder has a single coherent interpretation. Bucket legs (WTI) settle YES iff
``lo <= X <= hi``, which is not monotone in ``X``, so the same signal cannot be
applied to them without a distributional model. WTI is also the series
``TODO.md:204`` proposes dropping from the trigger set on principle (no
scheduled information release), and ``edge_economics.md`` §2(b) is the reason.
It stays out of both sides here.

**Same-release pairs excluded.** CPI and CPICORE print from one BLS release;
a "lead-lag" relation between them is simultaneity, not propagation.

The sign restriction
--------------------
Zero fitted parameters, imposed by economics, and falsifiable. Every series
carries a ``HAWKISH`` sign: does a *higher* print push the policy path up
(+1) or down (-1)? Then for an ordered pair::

    direction(trigger -> target) = HAWKISH[trigger] * HAWKISH[target]

A hawkish surprise in the trigger predicts a hawkish print in the target. Since
a threshold leg pays YES on ``X > K``, the aligned signal is::

    signal = HAWKISH[trigger] * HAWKISH[target] * z_surprise(trigger)

and a positive signal predicts the target's YES legs are *underpriced*.

``U3`` and ``JOBLESSCLAIMS`` carry -1, which is what makes this falsifiable
rather than a relabelling: CPI up predicts U3 *down*, so U3's YES legs should
get **less** likely. A model that scores those cells the same way it scores the
inflation cells is fitting noise, and ``falsification.py`` checks it.

In-sample only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import THRESHOLD, classify_contract, parse_threshold
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, is_same_release, series_filter_expr, trigger_universe
from stg.panel.surprise import usable_triggers
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/leadlag_2026_09/out")

MAX_GAP_DAYS = 60
MIN_TRIGGER_EVENTS = 10

# Does a higher print from this series push the policy path up or down?
# Copied deliberately from analysis/relations_2026_09/channel_pooling.py so the
# two studies cannot silently disagree about the economics.
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1,
    "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1,
    "FED": +1,
}

TYPE = {
    "CPI": "inflation", "CPICORE": "inflation", "CPIYOY": "inflation",
    "CPICOREYOY": "inflation", "CPIGAS": "inflation", "CPIUSEDCAR": "inflation",
    "CPISHELTER": "inflation", "CPIFOOD": "inflation", "CPIAPPAREL": "inflation",
    "PCECORE": "inflation",
    "PAYROLLS": "labour", "U3": "labour", "JOBLESSCLAIMS": "labour",
    "ADP": "labour",
    "GDP": "growth", "ISMPMI": "growth",
    "FED": "policy",
}


def _epoch_ns(ts) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(round(ts.timestamp() * 1_000_000_000))


def target_frames(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame):
    """Every threshold leg of every event of ``canon``, with its prints.

    Returns ``(legs, prints_by_ticker, event_index)``:
      legs   -- event_ticker, ticker, strike, result, close_time
      prints -- {ticker: (sorted epoch-ns array, yes_price array)}
      index  -- [(close_time, event_ticker)] in close order
    """
    m = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
         .select("event_ticker", "ticker", "yes_sub_title", "result", "close_time")
         .unique(subset=["ticker"]))
    if m.is_empty():
        return None
    kinds, strikes = [], []
    for t, s in zip(m["ticker"], m["yes_sub_title"]):
        c = classify_contract(t, s)
        kinds.append(c)
        strikes.append(parse_threshold(t, s) if c == THRESHOLD else None)
    m = (m.with_columns(pl.Series("kind", kinds),
                        pl.Series("strike", strikes, dtype=pl.Float64))
         .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()
                 & pl.col("result").is_in(["yes", "no"])))
    if m.is_empty():
        return None

    tk = m["ticker"].unique().to_list()
    tt = (tr.filter(pl.col("ticker").is_in(tk))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    if tt.is_empty():
        return None
    prints: dict = {}
    g = (tt.with_columns(pl.col("created_time").dt.epoch("ns").alias("_ns"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("_ns"), pl.col("yes_price")))
    for tkr, ns, px in zip(g["ticker"], g["_ns"], g["yes_price"]):
        prints[tkr] = (np.asarray(ns, dtype=np.int64),
                       np.asarray(px, dtype=float))
    m = m.filter(pl.col("ticker").is_in(list(prints)))
    if m.is_empty():
        return None

    ev = (m.group_by("event_ticker").agg(pl.col("close_time").min())
          .sort("close_time", "event_ticker"))
    index = list(zip(ev["close_time"].to_list(), ev["event_ticker"].to_list()))
    legs_by_event: dict = {}
    for r in m.iter_rows(named=True):
        legs_by_event.setdefault(r["event_ticker"], []).append(r)
    return legs_by_event, prints, index


def build(trigger_panel: pl.DataFrame, targets: list[str], mk, tr) -> pl.DataFrame:
    triggers = [t for t in usable_triggers(trigger_panel, MIN_TRIGGER_EVENTS)
                if t in HAWKISH]
    print(f"triggers: {len(triggers)} -> {sorted(triggers)}")
    print(f"targets:  {len(targets)} -> {sorted(targets)}\n")

    rows: list[dict] = []
    for tgt in targets:
        frames = target_frames(tgt, mk, tr)
        if frames is None:
            print(f"  {tgt:<14} no usable threshold legs")
            continue
        legs_by_event, prints, index = frames
        n_before = len(rows)

        for trig in triggers:
            if trig == tgt or is_same_release(trig, tgt):
                continue
            direction = HAWKISH[trig] * HAWKISH[tgt]
            sp = trigger_panel.filter(pl.col("series") == trig)

            for r in sp.iter_rows(named=True):
                t_res = r["close_time"]
                cut = _epoch_ns(t_res)
                # first target event closing strictly after the trigger resolves
                picked = None
                for c_close, c_event in index:
                    if c_close <= t_res:
                        continue
                    if (c_close - t_res).total_seconds() / 86400 > MAX_GAP_DAYS:
                        break
                    picked = (c_close, c_event)
                    break
                if picked is None:
                    continue
                c_close, c_event = picked
                gap = (c_close - t_res).total_seconds() / 86400

                for leg in legs_by_event[c_event]:
                    ns, px = prints[leg["ticker"]]
                    i = int(np.searchsorted(ns, cut, side="right"))
                    if i >= len(ns):
                        continue                      # never traded after t_res
                    p_entry = float(px[i])
                    if not 1.0 <= p_entry <= 99.0:
                        continue
                    p0 = float(px[i - 1]) if i > 0 else None
                    z = r["z_surprise"]
                    rows.append(dict(
                        trigger=trig, target=tgt,
                        channel=f"{TYPE[trig]}->{TYPE[tgt]}",
                        trigger_event=r["event_ticker"], target_event=c_event,
                        target_ticker=leg["ticker"], strike=leg["strike"],
                        t_res=t_res, close_time=c_close, gap_days=gap,
                        t_entry=dt.datetime.fromtimestamp(ns[i] / 1e9, dt.timezone.utc),
                        p_entry=p_entry, p0=p0, n_pre=i,
                        win=int(leg["result"] == "yes"),
                        direction=direction,
                        z_surprise=float(z),
                        s_pit=float(r["s_pit"]),
                        signal=direction * float(z),
                        signal_pit=direction * float(r["s_pit"]),
                        yr=t_res.year,
                    ))
        print(f"  {tgt:<14} +{len(rows) - n_before:>6} legs")
    return pl.DataFrame(rows)


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    # z_surprise is the unit-free trigger measure; raw surprise cannot be
    # pooled across series (relations_study_plan §1.3.2).
    sp = sp.with_columns(
        (pl.col("surprise") / pl.col("implied_std")).alias("z_surprise"))
    sp = sp.filter(pl.col("z_surprise").is_finite() & pl.col("s_pit").is_finite())

    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    cand = [c for c in trigger_universe(5, mk)
            if SPECS[c].kind == "threshold" and c in HAWKISH]

    legs = build(sp, cand, mk, tr)
    if legs.is_empty():
        print("no rows")
        return
    assert_no_oos(legs, time_col="close_time")
    assert_no_oos(legs, time_col="t_res")

    OUT.mkdir(parents=True, exist_ok=True)
    legs.write_parquet(OUT / "leadlag_legs.parquet")

    print(f"\nrows: {legs.height}")
    print(f"trigger events: {legs['trigger_event'].n_unique()}   "
          f"target events: {legs['target_event'].n_unique()}   "
          f"pairs: {legs.select(['trigger','target']).unique().height}")
    with pl.Config(tbl_rows=30, float_precision=2):
        print(legs.group_by("channel").agg(
            pl.len().alias("legs"),
            pl.col("target_event").n_unique().alias("tgt_events"),
            pl.col("p_entry").mean().alias("mean_p"),
            pl.col("win").mean().alias("hit"),
        ).sort("legs", descending=True))
    print(f"\nwrote {OUT / 'leadlag_legs.parquet'}")


if __name__ == "__main__":
    main()
