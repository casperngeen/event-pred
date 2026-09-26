#!/usr/bin/env python
"""Build the (trigger event x k-th target event x entry delay x leg) panel.

    venv/bin/python analysis/propagation_2026_09/build_panel.py

``leadlag_2026_09/build_panel.py`` pairs every trigger resolution with the
**first** target event closing after it, within 60 days, and enters at the
first print after the trigger resolves. That can only see a relation that is
priced -- or mispriced -- in the very next contract. A relation that takes
months to reach its target (A -> C through a slow channel) would sit in C's
second or third event, and would still be there after the market has had a
week or a month to look at A.

So this panel adds two dimensions:

* **horizon** -- which target event. Every event of the target series that
  settles within ``MAX_DAYS`` of the trigger's resolution is a row, and
  ``gap_days`` (resolution to settlement) is the horizon the analysis bands on.
  ``k`` (1 = the first event closing strictly after the trigger, the leadlag
  panel's choice) is kept for reference only: the archive has holes -- PCECORE
  jumps 272 and 420 days, CPI 216 and 337 -- so "the k-th event in the data" is
  not "the k-th release", and banding on ``k`` would file a nine-month-out
  contract under ``k = 1``.
* ``d`` -- entry delay. Enter at the first print strictly after
  ``t_res + d``. ``d = 0`` is the leadlag entry. If the market absorbs A
  quickly the edge is gone by ``d = 1``; slow propagation keeps it alive at
  ``d = 30``.

Entry must follow the cut within ``ENTRY_WINDOW_DAYS`` -- a print three weeks
after the cut is not an entry at ``t_res + d``. The exception is a leg that has
**never traded** before the cut (``deferred``): that is the "withhold A until a
suitable contract for C appears" case, and its entry is its first print however
late, as long as it is before the target closes.

Scope
-----
Only the pairs ``PAIRS`` names: the relations earlier studies identified, plus
a few with a strong *a priori* case, each tagged with where it came from. This
is not a sweep -- ``leadlag_findings.md`` §4b shows per-pair search at this
sample size is noise, and the (k, d) grid multiplies the cell count by 16.

Threshold targets only and the ``HAWKISH`` sign restriction, both copied from
``leadlag_2026_09/build_panel.py`` so the two studies cannot disagree about the
economics.

In-sample only.
"""

from __future__ import annotations

import bisect
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import THRESHOLD, classify_contract, parse_threshold
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/propagation_2026_09/out")

DELAYS = (0, 1, 7, 30)            # days after the trigger resolves
ENTRY_WINDOW_DAYS = 7             # first print must follow the cut within this
MAX_DAYS = 180                    # target event must settle within this

# Copied from leadlag_2026_09/build_panel.py (itself copied from
# relations_2026_09/channel_pooling.py) so the studies share one economics.
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1,
    "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1,
    "FED": +1,
}

# (trigger, target, group, source). ``source`` says why the pair is here:
#   identified -- a BH survivor in leadlag_findings.md §4b (channel level),
#                 edge_economics.md §1, or agcrn_checklist.md §3c;
#   theory     -- not identified empirically, but with a strong prior that the
#                 relation exists and could be slow.
PAIRS = [
    # labour->labour: leadlag §4b BH channel
    ("JOBLESSCLAIMS", "PAYROLLS", "labour->labour", "identified"),
    ("JOBLESSCLAIMS", "U3", "labour->labour", "identified"),
    ("JOBLESSCLAIMS", "ADP", "labour->labour", "identified"),
    ("PAYROLLS", "ADP", "labour->labour", "identified"),
    ("U3", "ADP", "labour->labour", "identified"),
    # PCE <-> CPI: leadlag §4b inflation->inflation BH channel;
    # CPIYOY->PCECORE is also an edge_economics BH survivor
    ("CPI", "PCECORE", "PCE<->CPI", "identified"),
    ("CPICORE", "PCECORE", "PCE<->CPI", "identified"),
    ("CPIYOY", "PCECORE", "PCE<->CPI", "identified"),
    ("CPICOREYOY", "PCECORE", "PCE<->CPI", "identified"),
    ("PCECORE", "CPI", "PCE<->CPI", "identified"),
    ("PCECORE", "CPICORE", "PCE<->CPI", "identified"),
    # labour->policy: leadlag §4b BH channel, event_time's one surviving channel
    ("PAYROLLS", "FED", "labour->policy", "identified"),
    ("U3", "FED", "labour->policy", "identified"),
    ("JOBLESSCLAIMS", "FED", "labour->policy", "identified"),
    # inflation->policy: CPI/CPICORE->FED are edge_economics BH survivors; the
    # rest are the theory case (PCE is the Fed's target measure)
    ("CPI", "FED", "inflation->policy", "identified"),
    ("CPICORE", "FED", "inflation->policy", "identified"),
    ("CPIYOY", "FED", "inflation->policy", "theory"),
    ("CPICOREYOY", "FED", "inflation->policy", "theory"),
    ("PCECORE", "FED", "inflation->policy", "theory"),
    # growth->policy: theory only
    ("GDP", "FED", "growth->policy", "theory"),
]


def _ns(ts) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(round(ts.timestamp() * 1_000_000_000))


def target_frames(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame):
    """Every threshold leg of every event of ``canon``, with its prints.

    Same construction as ``leadlag_2026_09/build_panel.py::target_frames``.
    Returns ``(legs_by_event, prints_by_ticker, [(close_time, event)])``.
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

    tt = (tr.filter(pl.col("ticker").is_in(m["ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    if tt.is_empty():
        return None
    prints: dict = {}
    g = (tt.with_columns(pl.col("created_time").dt.epoch("ns").alias("_ns"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("_ns"), pl.col("yes_price")))
    for tkr, ns, px in zip(g["ticker"], g["_ns"], g["yes_price"]):
        prints[tkr] = (np.asarray(ns, dtype=np.int64), np.asarray(px, dtype=float))
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


def build(sp: pl.DataFrame, mk, tr) -> pl.DataFrame:
    frames = {}
    for tgt in sorted({p[1] for p in PAIRS}):
        frames[tgt] = target_frames(tgt, mk, tr)
        n = 0 if frames[tgt] is None else len(frames[tgt][2])
        print(f"  target {tgt:<14} {n:>4} events with threshold legs")

    day = 86_400 * 1_000_000_000
    rows: list[dict] = []
    for trig, tgt, group, source in PAIRS:
        if frames[tgt] is None:
            print(f"  {trig}->{tgt}: no target frames, skipped")
            continue
        legs_by_event, prints, index = frames[tgt]
        closes = [c for c, _ in index]
        direction = HAWKISH[trig] * HAWKISH[tgt]
        n_before = len(rows)

        for r in sp.filter(pl.col("series") == trig).iter_rows(named=True):
            t_res = r["close_time"]
            t_res_ns = _ns(t_res)
            j0 = bisect.bisect_right(closes, t_res)      # first close > t_res
            for j in range(j0, len(index)):
                k = j - j0 + 1
                c_close, c_event = index[j]
                gap = (c_close - t_res).total_seconds() / 86400
                if gap > MAX_DAYS:
                    break
                close_ns = _ns(c_close)
                for d in DELAYS:
                    cut = t_res_ns + d * day
                    if cut >= close_ns:
                        continue
                    for leg in legs_by_event[c_event]:
                        ns, px = prints[leg["ticker"]]
                        i = int(np.searchsorted(ns, cut, side="right"))
                        if i >= len(ns) or ns[i] >= close_ns:
                            continue
                        deferred = i == 0          # never traded before the cut
                        lag_d = (ns[i] - cut) / day
                        if not deferred and lag_d > ENTRY_WINDOW_DAYS:
                            continue
                        p_entry = float(px[i])
                        if not 1.0 <= p_entry <= 99.0:
                            continue
                        rows.append(dict(
                            trigger=trig, target=tgt, pair=f"{trig}->{tgt}",
                            group=group, source=source, k=k, delay_d=d,
                            trigger_event=r["event_ticker"], target_event=c_event,
                            target_ticker=leg["ticker"], strike=leg["strike"],
                            t_res=t_res, close_time=c_close, gap_days=gap,
                            t_entry=dt.datetime.fromtimestamp(ns[i] / 1e9, dt.timezone.utc),
                            entry_lag_d=float(lag_d), deferred=deferred,
                            p_entry=p_entry,
                            p_prev=float(px[i - 1]) if i > 0 else None,
                            win=int(leg["result"] == "yes"),
                            direction=direction,
                            z_surprise=float(r["z_surprise"]),
                            signal=direction * float(r["z_surprise"]),
                            yr=t_res.year,
                        ))
        print(f"  {trig + '->' + tgt:<24} +{len(rows) - n_before:>6} rows")
    return pl.DataFrame(rows)


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    sp = (sp.with_columns((pl.col("surprise") / pl.col("implied_std")).alias("z_surprise"))
          .filter(pl.col("z_surprise").is_finite()))

    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    d = build(sp, mk, tr)
    assert_no_oos(d, time_col="close_time")
    assert_no_oos(d, time_col="t_entry")

    OUT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT / "depth_legs.parquet")

    print(f"\nrows: {d.height}   trigger events: {d['trigger_event'].n_unique()}   "
          f"target events: {d['target_event'].n_unique()}")
    with pl.Config(tbl_rows=120, float_precision=1, tbl_width_chars=200):
        print(d.with_columns(((pl.col("gap_days") // 30) * 30).cast(pl.Int32).alias("h_lo"))
              .group_by("group", "h_lo", "delay_d").agg(
            pl.len().alias("legs"),
            pl.col("target_event").n_unique().alias("tgt_ev"),
            pl.col("gap_days").median().alias("gap_med"),
            pl.col("entry_lag_d").median().alias("lag_med"),
            pl.col("deferred").mean().alias("deferred"),
        ).sort("group", "h_lo", "delay_d"))
    print(f"\nwrote {OUT / 'depth_legs.parquet'}")


if __name__ == "__main__":
    main()
