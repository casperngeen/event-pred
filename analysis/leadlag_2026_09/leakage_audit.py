#!/usr/bin/env python
"""Look-ahead audit of the frozen specification.

    venv/bin/python analysis/leadlag_2026_09/leakage_audit.py

Walks the decision timeline and checks every quantity against the instant it is
used. Then re-runs the strategy with the leaks that CAN be removed, removed, so
their materiality is measured rather than argued.

Timeline
--------
    t_res         trigger A resolves; signal becomes computable
    t_entry       first post-resolution print in the target leg  -> p_entry (confirmation)
    t_entry2      second post-resolution print                   -> p2 (the fill)
    close_time    target settles

Anything used at a step must be knowable at that step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import scan_trades
from stg.splits import OOS_START

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT, FEE_RATE, CONTRACTS = 10000, 0.07, 100
BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]
CONFIRM_C, FLOOR_C = 2.0, 20.0


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cboot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    k = len(u)
    s = np.bincount(inv, weights=v, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float((b <= 0).mean()))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z_raw = d["z_surprise"].to_numpy()
    dirn = d["direction"].to_numpy().astype(float)
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("s").to_numpy()
    years = sorted(np.unique(yr))

    print("=== 1. hard wall ===")
    latest = d["close_time"].max()
    print(f"OOS_START = {OOS_START};  latest close_time in panel = {latest}")
    n_oos = int(d.filter(pl.col("close_time") >= OOS_START).height)
    n_oos_t = int(d.filter(pl.col("t_res") >= OOS_START).height)
    print(f"rows with close_time at/after the wall: {n_oos}")
    print(f"rows with t_res     at/after the wall: {n_oos_t}")

    print("\n=== 2. timing audit, quantity by quantity ===")
    audit = [
        ("resolved_value(A)", "t_res", "known: A has just settled", "OK"),
        ("implied_mean/std(A)", "pre-t_res ladder", "last day with >=3 fresh legs", "OK"),
        ("HAWKISH signs", "a priori", "economics, no data", "OK"),
        ("p0", "<= t_res", "last print at or before resolution", "OK"),
        ("price bucket of p0", "t_res", "boundaries fixed a priori", "OK"),
        ("tercile cutpoints", "years < Y", "refit each year on prior years only", "OK"),
        ("p_entry (confirmation)", "t_entry", "first print after t_res", "OK"),
        ("p2 (the fill)", "t_entry2", "second print; median 58.7 min later", "OK"),
        ("win", "close_time", "outcome only, never an input", "OK"),
        ("winsorisation limit", "FULL SAMPLE", "p99 of |z| over all years", "LEAK"),
        ("bucket eligibility (>=60 rows)", "FULL SAMPLE", "counts include future years", "LEAK"),
        ("leg must print after t_res", "FUTURE", "selects legs that went on to trade", "LEAK*"),
        ("leg must print TWICE", "FUTURE", "p2 requires a second future trade", "LEAK*"),
        ("confirm 2c / floor 20c", "FULL SAMPLE", "thresholds chosen in-sample", "SEARCH"),
        ("spread model 1-2c", "FULL SAMPLE", "estimated on the whole period", "MINOR"),
    ]
    with pl.Config(tbl_rows=30, tbl_width_chars=210, fmt_str_lengths=60):
        print(pl.DataFrame(audit, schema=["quantity", "known at", "note", "verdict"],
                           orient="row"))
    print("\nLEAK*  = survivorship: cannot be removed without quote data, only quantified (§4).")
    print("SEARCH = multiplicity, not look-ahead; already recorded in strategy_spec §6.")

    # ---------------------------------------------------------------- §3
    print("\n=== 3. re-run with the removable leaks removed ===")

    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=float))
            for t, a, b in zip(g["ticker"], g["s"], g["yes_price"])}
    ent = np.full(len(p0), np.nan)
    n_after = np.zeros(len(p0), dtype=int)
    for i in range(len(p0)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres[i], side="right"))
        n_after[i] = len(s_) - j
        if j + 1 < len(s_):
            ent[i] = px_[j + 1]

    def build_side(expanding: bool):
        """expanding=False reproduces the spec; True removes both leaks."""
        if not expanding:
            lim = float(np.percentile(np.abs(z_raw), 99))
            sig = dirn * np.clip(z_raw, -lim, lim)
        else:
            sig = np.zeros(len(z_raw))
            for Y in years:
                pri = np.abs(z_raw[yr < Y])
                lim_y = float(np.percentile(pri, 99)) if len(pri) >= 50 else np.inf
                m = yr == Y
                sig[m] = dirn[m] * np.clip(z_raw[m], -lim_y, lim_y)
        side = np.zeros(len(p0))
        for lo, hi in BUCKETS:
            m = (p0 >= lo) & (p0 < hi)
            if not expanding and m.sum() < 60:
                continue
            idx = np.where(m)[0]
            for Y in years[1:]:
                tr, te = idx[yr[idx] < Y], idx[yr[idx] == Y]
                if len(tr) < 60 or len(te) == 0:
                    continue
                if expanding and len(tr) < 60:
                    continue
                t = np.percentile(sig[tr], [33.3, 66.7])
                side[te] = np.where(sig[te] > t[1], 1.0,
                                    np.where(sig[te] <= t[0], -1.0, 0.0))
        return side

    rows = []
    for nm, expanding in (("as specified (both leaks present)", False),
                          ("expanding winsorisation + bucket gate", True)):
        side = build_side(expanding)
        e = np.where(side > 0, ent, 100.0 - ent)
        keep = (side != 0) & np.isfinite(ent) & (e >= FLOOR_C)
        keep &= ((pe - p0) * np.sign(side)) >= CONFIRM_C
        pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
        net = pay - e - (fee_cents(e) + spread_cents(e) / 2.0)
        o, lo, hi, p = cboot(net[keep], ev[keep])
        rows.append(dict(variant=nm, n=int(keep.sum()),
                         events=int(len(np.unique(ev[keep]))),
                         net=o, ci_lo=lo, ci_hi=hi, p_le0=p))
    with pl.Config(float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    # ---------------------------------------------------------------- §4
    print("\n=== 4. the survivorship leak that cannot be removed ===")
    print("Entry requires the leg to print again after t_res -- twice, for p2.")
    print("At t_res a trader does not know which legs will trade.\n")
    tot = len(p0)
    print(f"  legs in the panel (already require >=1 print after t_res): {tot}")
    print(f"  of those, with a SECOND print:      {int((n_after >= 2).sum())} "
          f"({(n_after >= 2).mean():.1%})")
    print(f"  median prints after t_res: {int(np.median(n_after))}   "
          f"p10 {int(np.percentile(n_after, 10))}")
    print("\n  This is the research_log §12 convenience-sample problem in another")
    print("  form: illiquid legs are absent, and they are absent for a reason")
    print("  correlated with how the event resolved. Quote data would fix it;")
    print("  data/kalshi_orderbooks.jsonl is sports-only from late 2025.")


if __name__ == "__main__":
    main()
