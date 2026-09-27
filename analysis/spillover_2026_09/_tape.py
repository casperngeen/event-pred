"""Shared tape for the spillover scripts: threshold legs, per-contract price and
signed order flow, release times, and the response helpers.

Moved out of ``jumps.py`` unchanged (plus the signed-flow column) so ``flow.py``
measures responses exactly as ``jumps.py`` does. Not a script.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.events.implied import THRESHOLD, classify_contract, parse_threshold
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr

HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1, "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
TYPE = {**{s: "inflation" for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS",
                                    "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL", "PCECORE")},
        **{s: "labour" for s in ("PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP")},
        "GDP": "growth", "ISMPMI": "growth", "FED": "policy"}
M15, H, D = np.timedelta64(15, "m"), np.timedelta64(1, "h"), np.timedelta64(1, "D")
RELEASE_EXCL = 24 * H
HORIZONS = {"+1h": H, "+4h": 4 * H, "+24h": 24 * H, "+72h": 72 * H}

# ------------------------------------------------------------------ data
mk, tr = load_markets(is_only=True), scan_trades(is_only=True)
LEGS, TAPE, FLOW = {}, {}, {}
all_closes = []
for s in HAWKISH:
    m = (mk.filter(series_filter_expr(s) & pl.col("close_time").is_not_null())
         .select("event_ticker", "ticker", "yes_sub_title", "close_time").unique(subset=["ticker"]))
    if m.is_empty():
        continue
    all_closes += m["close_time"].dt.replace_time_zone(None).cast(pl.Datetime("us")).to_list()
    kinds = [classify_contract(t, y) for t, y in zip(m["ticker"], m["yes_sub_title"])]
    strikes = [parse_threshold(t, y) if k == THRESHOLD else None
               for t, y, k in zip(m["ticker"], m["yes_sub_title"], kinds)]
    m = (m.with_columns(pl.Series("kind", kinds), pl.Series("strike", strikes, dtype=pl.Float64))
         .filter((pl.col("kind") == THRESHOLD) & pl.col("strike").is_not_null()))
    if m.is_empty():
        continue
    tt = (tr.filter(pl.col("ticker").is_in(m["ticker"].to_list()))
          .select("ticker", "yes_price", "count", "taker_side",
                  pl.col("created_time").dt.replace_time_zone(None).cast(pl.Datetime("us")).alias("t"))
          .group_by("ticker", "t")
          .agg(((pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum()).alias("px"),
               pl.when(pl.col("taker_side") == "yes").then(pl.col("count"))
               .otherwise(-pl.col("count")).sum().alias("flow"))
          .sort("ticker", "t").collect())
    for (tk,), g in tt.group_by("ticker", maintain_order=True):
        TAPE[tk] = (g["t"].to_numpy(), g["px"].to_numpy().astype(float))
        FLOW[tk] = g["flow"].to_numpy().astype(float)    # signed contracts: + = takers bought YES
    # sorted by ticker: lead_contract breaks ties by list order, and polars'
    # unique() does not keep one, so an unsorted list made the lead contract
    # (and every response) change between runs
    LEGS[s] = sorted((r["ticker"], np.datetime64(r["close_time"].replace(tzinfo=None), "us"))
                     for r in m.iter_rows(named=True) if r["ticker"] in TAPE)
REL = np.sort(np.array(all_closes, dtype="datetime64[us]"))
print(f"series with threshold legs and trades: {len(LEGS)}; contracts {len(TAPE)}; "
      f"release/settlement times {len(REL)}")


def near_release(t):
    i = np.searchsorted(REL, t)
    lo = REL[max(i - 1, 0)]
    hi = REL[min(i, len(REL) - 1)]
    return min(abs(t - lo), abs(hi - t)) <= RELEASE_EXCL


def lead_contract(series, t):
    """Target's threshold contract with the most prints in the 7 days before t,
    among its events closing after t."""
    best, n_best = None, 0
    for tk, close in LEGS[series]:
        if close <= t:
            continue
        ts, _ = TAPE[tk]
        n = np.searchsorted(ts, t) - np.searchsorted(ts, t - 7 * D)
        if n > n_best:
            best, n_best = tk, n
    return best


def asof(tk, t, after=None):
    ts, px = TAPE[tk]
    i = np.searchsorted(ts, t, side="right") - 1
    if i < 0 or t - ts[i] > 7 * D or (after is not None and ts[i] <= after):
        return np.nan
    return px[i]


def response(tb, t0, t_end):
    """(same-bar move, {horizon: move after the bar}) of target contract tb."""
    p_start, p_end = asof(tb, t0), asof(tb, t_end)
    same = p_end - p_start if np.isfinite(p_start) and np.isfinite(p_end) else np.nan
    after = {h: asof(tb, t_end + dh, after=t_end) - p_end if np.isfinite(p_end) else np.nan
             for h, dh in HORIZONS.items()}
    return same, after


def placebo_time(t0, rng):
    for _ in range(20):
        t = t0 + int(rng.choice([-1, 1])) * int(rng.integers(3, 11)) * D
        if not near_release(t):
            return t
    return None
