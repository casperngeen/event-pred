#!/usr/bin/env python
"""Event-time node panel: one row per (release instant, node series).

    venv/bin/python analysis/event_time_2026_09/build_panel.py    # writes out/event_nodes.parquet

The daily node panel (``stg/panel/nodes.py``) collapses a day of trades into its
close, so a release at 8:30 and the target repricing at 8:31 land in the same
snapshot and the lead-lag becomes simultaneity (``reports/agcrn_checklist.md``
§3b). Here the clock is the release itself.

Grid
----
A **step** is a release instant: every usable-trigger event in the surprise
panel (≥10 events, in ``HAWKISH``), grouped by ``close_time`` to the minute, so
CPI + CPICORE + CPIYOY printing together is one step.

Node state at t⁻ (strictly before the release)
----------------------------------------------
For each node series, its nearest event closing strictly after t (≤60 days).
Each threshold leg contributes its **last print before t**, if that print is at
most ``MAX_AGE_D`` old, and the ladder statistics come from
``stg.events.quantile.quantile_moments`` — interpolated crossings only, no pdf
recovery, so no tail assumption (``quantile_findings.md``). Prints sharing a
nanosecond are collapsed to their count-weighted VWAP first: ``.last()`` over a
tie is not a rule (``leadlag_2026_09/tie_handling.py``).

Features: ladder median ``q50``, width ``sigma_iqr``, Bowley ``skew_q``, 7-day
change in ``q50``, legs used, median leg age, share of legs printed in the last
24h, 7-day contract volume and signed taker flow, days to close, the lead
contract's price and age, the winsorised ``z`` surprise (on the releasing
nodes, 0 elsewhere), and a ``released`` flag.

Labels, on the node's **lead contract** (most prints on its event *before* t)
------------------------------------------------------------------------------
* ``y_imm``  immediate repricing, cents: price after the first 3 post-release
  prints (fewer if fewer arrive), all within 24h, minus the last pre-release
  print. Masked if nothing prints within 24h. Ends at that print's time.
* ``y_settle``  settlement residual, cents: 100·[settled YES] minus the first
  post-release print (the lead-lag / ``strategy_spec`` label, on one contract per
  node). Masked outside 1–99c. Ends at the event's close.

Labels are masked on nodes released at t and on nodes in the same release as
one released at t: that is simultaneity, not spillover (the lead-lag convention).

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
from stg.events.quantile import quantile_moments
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import SPECS, is_same_release, series_filter_expr, trigger_universe
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/event_time_2026_09/out")

MIN_TRIGGER_EVENTS = 10
MAX_GAP_DAYS = 60          # nearest event must close within this of t
MAX_AGE_D = 14             # a pre-release print older than this is too stale to use
IMM_TRADES, IMM_CAP_H = 3, 24
NS_H = 3_600_000_000_000
NS_D = 24 * NS_H

# strategy_spec.md §3.3 / leadlag_2026_09/build_panel.py, verbatim
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1,
    "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1,
    "FED": +1,
}


def _ns(ts) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return int(round(ts.timestamp() * 1e9))


def load_series(canon: str, mk: pl.DataFrame, tr: pl.LazyFrame):
    """Threshold legs of every event of ``canon`` and their VWAP-collapsed tape.

    Returns ``(events, tape)``: events = [(close_ns, event_ticker, [legs])] in
    close order, each leg a dict with ticker/strike/result; tape = {ticker:
    (ns, price, count, signed_count)} with ties collapsed.
    """
    m = (mk.filter(series_filter_expr(canon) & pl.col("close_time").is_not_null())
         .select("event_ticker", "ticker", "yes_sub_title", "result", "close_time")
         .unique(subset=["ticker"]))
    kinds = [classify_contract(t, s) for t, s in zip(m["ticker"], m["yes_sub_title"])]
    strikes = [parse_threshold(t, s) if k == THRESHOLD else None
               for t, s, k in zip(m["ticker"], m["yes_sub_title"], kinds)]
    m = (m.with_columns(pl.Series("kind", kinds),
                        pl.Series("strike", strikes, dtype=pl.Float64))
         .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()))
    if m.is_empty():
        return None
    tt = (tr.filter(pl.col("ticker").is_in(m["ticker"].unique().to_list()))
          .select("ticker", "yes_price", "count", "taker_side", "created_time")
          .with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"),
                        pl.when(pl.col("taker_side") == "yes").then(pl.col("count"))
                        .otherwise(-pl.col("count")).alias("signed"))
          .group_by("ticker", "ns")
          .agg(((pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum())
               .alias("px"),
               pl.col("count").sum().alias("n"), pl.col("signed").sum().alias("s"))
          .sort("ticker", "ns").collect())
    tape = {}
    for (tk,), g in tt.group_by("ticker", maintain_order=True):
        tape[tk] = (g["ns"].to_numpy(), g["px"].to_numpy().astype(float),
                    g["n"].to_numpy(), g["s"].to_numpy())
    events = []
    for (ev,), g in m.group_by("event_ticker"):
        # sorted so the lead-contract tie-break (first leg with the most prints)
        # does not depend on polars' group order
        legs = sorted((r for r in g.iter_rows(named=True) if r["ticker"] in tape),
                      key=lambda r: r["ticker"])
        if legs:
            events.append((_ns(g["close_time"].min()), ev, legs))
    return sorted(events), tape


def ladder_state(legs, tape, cut: int) -> dict:
    """As-of ladder statistics from each leg's last print strictly before ``cut``."""
    K, P, ages = [], [], []
    for leg in legs:
        ns, px, _, _ = tape[leg["ticker"]]
        i = int(np.searchsorted(ns, cut, side="left")) - 1
        if i >= 0 and cut - ns[i] <= MAX_AGE_D * NS_D:
            K.append(leg["strike"]); P.append(px[i] / 100.0); ages.append((cut - ns[i]) / NS_H)
    q = quantile_moments(np.array(K), np.array(P)) if len(K) >= 3 else {}
    return dict(q50=q.get("q50"), sigma_iqr=q.get("sigma_iqr"), skew_q=q.get("skew_q"),
                n_legs=len(K), med_age_h=float(np.median(ages)) if ages else None,
                fresh24=float(np.mean(np.array(ages) <= 24)) if ages else None)


def node_row(canon: str, events, tape, t_ns: int) -> dict | None:
    nxt = next(((c, ev, legs) for c, ev, legs in events
                if c > t_ns and c - t_ns <= MAX_GAP_DAYS * NS_D), None)
    if nxt is None:
        return None
    close_ns, ev, legs = nxt
    now = ladder_state(legs, tape, t_ns)
    wk = ladder_state(legs, tape, t_ns - 7 * NS_D)
    vol = flow = 0
    lead, lead_n = None, 0
    for leg in legs:
        ns, px, n, s = tape[leg["ticker"]]
        lo, hi = np.searchsorted(ns, t_ns - 7 * NS_D), np.searchsorted(ns, t_ns)
        vol += int(n[lo:hi].sum()); flow += int(s[lo:hi].sum())
        if hi > lead_n:                                   # prints before t: no look-ahead
            lead, lead_n = leg, hi
    row = dict(series=canon, event_ticker=ev,
               days_to_close=(close_ns - t_ns) / NS_D, **now,
               d_q50_7d=(now["q50"] - wk["q50"]) if None not in (now["q50"], wk["q50"]) else None,
               vol7=vol, flow7=flow / max(vol, 1),
               lead_ticker=None, p_lead=None, lead_age_h=None,
               y_imm=None, imm_end=None, y_settle=None, settle_end=close_ns,
               p_entry=None, t_entry=None, win=None)
    if lead is None:
        return row
    ns, px, _, _ = tape[lead["ticker"]]
    i = int(np.searchsorted(ns, t_ns, side="left"))        # first print at/after t
    j = int(np.searchsorted(ns, t_ns, side="right"))       # first print strictly after t
    p0 = float(px[i - 1])
    row.update(lead_ticker=lead["ticker"], p_lead=p0, lead_age_h=(t_ns - ns[i - 1]) / NS_H)
    post = [k for k in range(j, min(j + IMM_TRADES, len(ns)))
            if ns[k] - t_ns <= IMM_CAP_H * NS_H and ns[k] < close_ns]
    if post:
        row.update(y_imm=float(px[post[-1]]) - p0, imm_end=int(ns[post[-1]]))
    if j < len(ns) and ns[j] < close_ns and lead["result"] in ("yes", "no"):
        p_entry = float(px[j])
        if 1.0 <= p_entry <= 99.0:
            row.update(y_settle=100.0 * (lead["result"] == "yes") - p_entry,
                       p_entry=p_entry, t_entry=int(ns[j]),
                       win=int(lead["result"] == "yes"))
    return row


def main() -> None:
    sp = (pl.read_parquet(PANELS / "surprise_panel.parquet")
          .with_columns((pl.col("surprise") / pl.col("implied_std")).alias("z"))
          .filter(pl.col("z").is_finite()))
    assert_no_oos(sp, time_col="close_time")
    n_ev = dict(sp.group_by("series").len().iter_rows())
    triggers = sorted(s for s in HAWKISH if n_ev.get(s, 0) >= MIN_TRIGGER_EVENTS)
    sp = sp.filter(pl.col("series").is_in(triggers))
    cap = float(sp["z"].abs().quantile(0.99))
    sp = sp.with_columns(pl.col("z").clip(-cap, cap),
                         pl.col("close_time").dt.truncate("1m").alias("instant"))
    instants = (sp.group_by("instant").agg(pl.col("series"), pl.col("z"))
                .sort("instant"))

    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
    nodes = sorted(c for c in trigger_universe(5, mk)
                   if SPECS[c].kind == "threshold" and c in HAWKISH)
    print(f"triggers ({len(triggers)}): {triggers}\nnodes ({len(nodes)}): {nodes}")
    print(f"release instants: {instants.height}; z winsorised at ±{cap:.2f}\n")

    rows = []
    for canon in nodes:
        loaded = load_series(canon, mk, tr)
        if loaded is None:
            print(f"  {canon:<14} no threshold legs"); continue
        events, tape = loaded
        n0 = len(rows)
        for r in instants.iter_rows(named=True):
            t_ns = _ns(r["instant"])
            released = dict(zip(r["series"], r["z"]))
            row = node_row(canon, events, tape, t_ns)
            if row is None:
                row = dict(series=canon)
            spill_ok = canon not in released and not any(
                is_same_release(canon, s) for s in released)
            if not spill_ok:
                row.update(y_imm=None, y_settle=None, p_entry=None, win=None)
            rows.append(dict(instant=r["instant"], released=canon in released,
                             z=released.get(canon, 0.0), **row))
        got = sum(1 for x in rows[n0:] if x.get("q50") is not None)
        print(f"  {canon:<14} ladder at t⁻ on {got:>3}/{instants.height} instants")

    panel = pl.DataFrame(rows, infer_schema_length=None)
    assert_no_oos(panel, time_col="instant")
    OUT.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUT / "event_nodes.parquet")

    print(f"\nrows {panel.height}")
    with pl.Config(tbl_rows=30, tbl_cols=12, float_precision=2):
        print(panel.group_by("series").agg(
            pl.col("q50").is_not_null().mean().alias("has_ladder"),
            pl.col("y_imm").is_not_null().sum().alias("n_imm"),
            pl.col("y_settle").is_not_null().sum().alias("n_settle"),
            pl.col("y_imm").abs().median().alias("|y_imm| med"),
            pl.col("lead_age_h").median().alias("lead age h"),
            (pl.col("released") & pl.col("q50").is_null()).sum().alias("released w/o ladder"),
        ).sort("series"))
    print(f"\nwrote {OUT / 'event_nodes.parquet'}")


if __name__ == "__main__":
    main()
