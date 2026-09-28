#!/usr/bin/env python
"""Intraday path panel: every target market's price on a grid of bars after each release.

    venv/bin/python -W ignore analysis/intraday_2026_09/build_paths.py \
        > analysis/intraday_2026_09/out/build_paths.txt       # ~1 min; writes out/cells.parquet, out/paths.parquet

Plan: ``reports/intraday_path_plan.md``. The event-time panel steps from one
release to the next; this panel moves the clock inside a release.

Triggers (``arm``)
------------------
* ``kalshi``    the 154 release instants of ``event_time_2026_09/build_panel.py``
                (``out/event_nodes.parquet``), with their winsorised z. A Kalshi
                market closes before its release (usually 5 min, some at
                midnight), so τ = 0 is the **economic-calendar release time**
                of the instant's series, matched within ±24 h. Instants with no
                match (the late-2025 shutdown, when markets closed but nothing
                was released) are dropped. Instants that land on the same
                release minute are merged.
* ``calendar``  the non-Kalshi calendar releases of ``spillover_2026_09/releases.py``
                (same filter and z = (actual − consensus)/sd, clipped at ±3),
                grouped by release minute into multi-source instants.
* ``*_placebo`` each instant moved 3–10 days away from any release
                (``_tape.placebo_time``), keeping its sources and z. Curves on
                these should be flat.

Targets: each of the 17 series' lead contract at t_rel (``_tape.lead_contract``,
chosen from prints before the release; for a releasing series this is its next
unresolved contract). p₀ = last print strictly before t_rel, at most 7 days old.

Role of a (instant, target) cell: ``own`` if the target's series released at the
instant (Kalshi arm), ``sibling`` if it shares a release with one that did
(``is_same_release``; not used as spillover, as in the event-time panel), else
``cross``. Signals, theory-signed (+ = the theory direction for the target):
``x_own`` = z of the target's own release; ``x_<family>`` = Σ z_a·HAWKISH[a]·HAWKISH[B]
over cross sources a of that family; ``sig`` = their sum.

Grid (τ from t_rel): −60, −30, −15 min; 0 (= p₀); 5-min bars to +2 h; 15-min to
+6 h; hourly to +24 h; daily to +7 d. Per bar: the as-of price (last print ≤
t_rel + τ, forward-filled), a bounce-robust price (mean of the last two prints),
prints since the release, whether it has traded, minutes since its last print,
cumulative signed taker flow since the release, and two truncation flags:
``trunc_k`` (past the next Kalshi instant, the contract's close, or the 2026
wall) and ``trunc_any`` (also past the next calendar release).

Trade time: the price and hours after release of the target's k-th
post-release print, k = 1, 2, 3, 5, 10, if it comes within 24 h and before
``trunc_k``.

In-sample only: nothing is read or forward-filled past 2026-01-01.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.panel.registry import is_same_release
from stg.splits import OOS_START, assert_no_oos

sys.path.insert(0, "analysis/spillover_2026_09")
from _tape import FLOW, HAWKISH, LEGS, REL, TAPE, TYPE, lead_contract, placebo_time  # noqa: E402
from calendar_triggers import TRIGGERS  # noqa: E402

OUT = Path("analysis/intraday_2026_09/out")
NODES = "analysis/event_time_2026_09/out/event_nodes.parquet"
CAL = "data/external/econ_calendar_us_2021q4_2025.parquet"
MIN, H, D = np.timedelta64(1, "m"), np.timedelta64(1, "h"), np.timedelta64(1, "D")
WALL = np.datetime64(OOS_START.replace(tzinfo=None), "us")
MAX_AGE = 7 * D
TAU = np.array([-60, -30, -15, 0, *range(5, 121, 5), *range(135, 361, 15),
                *range(420, 1441, 60), *range(2 * 1440, 7 * 1440 + 1, 1440)])  # minutes
KS = (1, 2, 3, 5, 10)
TARGET_TYPE = {s: TYPE[s] for s in LEGS}
CLOSE = {tk: c for s in LEGS for tk, c in LEGS[s]}
rng = np.random.default_rng(0)

# Calendar headline for each Kalshi series, to find the true release minute.
CPI_PAT = r"^(Core )?(Inflation Rate|CPI)"
CAL_PAT = {**{s: CPI_PAT for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS", "CPIUSEDCAR")},
           "PAYROLLS": r"^Non Farm Payrolls", "U3": r"^Unemployment Rate",
           "JOBLESSCLAIMS": r"^Initial Jobless Claims", "GDP": r"^GDP Growth Rate",
           "PCECORE": r"^Core PCE Price Index", "FED": r"^Fed Interest Rate Decision"}


def _naive(col: str) -> pl.Expr:
    return pl.col(col).dt.replace_time_zone(None).cast(pl.Datetime("us"))


# ------------------------------------------------------------------ triggers
cal_all = (pl.read_parquet(CAL).filter(pl.col("country") == "US")
           .with_columns(pl.col("date").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
                         .cast(pl.Datetime("us")).alias("t")))


def kalshi_instants() -> list[dict]:
    rel = (pl.read_parquet(NODES).filter(pl.col("released"))
           .select(_naive("instant").alias("instant"), "series", "z"))
    cal_t = {s: np.sort(cal_all.filter(pl.col("event").str.contains(p))["t"].to_numpy())
             for s, p in CAL_PAT.items()}
    merged, dropped = {}, []
    for (inst,), g in rel.group_by("instant", maintain_order=True):
        t0 = np.datetime64(inst, "us")
        best = None
        for s in g["series"]:
            ts = cal_t.get(s, np.array([], "datetime64[us]"))
            if len(ts):
                c = ts[np.abs(ts - t0).argmin()]
                if abs(c - t0) <= 24 * H and (best is None or abs(c - t0) < abs(best - t0)):
                    best = c
        if best is None:
            dropped.append(f"{inst:%Y-%m-%d %H:%M} {','.join(sorted(g['series']))}")
            continue
        m = merged.setdefault(best, {})
        for s, z in zip(g["series"], g["z"]):
            m[s] = z
    print(f"Kalshi instants: {rel['instant'].n_unique()}; matched to a calendar release time: "
          f"{rel['instant'].n_unique() - len(dropped)}, merged into {len(merged)} release minutes")
    print("  dropped (no calendar release within ±24 h):", *dropped, sep="\n    ")
    return [dict(arm="kalshi", t_rel=t,
                 sources=[dict(src=s, fam=TYPE[s], hk=HAWKISH[s], z=float(z)) for s, z in sorted(m.items())])
            for t, m in sorted(merged.items())]


def calendar_instants() -> list[dict]:
    """``releases.py``'s trigger set, verbatim filter, grouped by release minute."""
    cal = (pl.read_parquet(CAL)
           .filter((pl.col("country") == "US") & pl.col("estimate").is_not_null() & pl.col("actual").is_not_null())
           .with_columns(pl.col("event").str.replace(r"\s*\([^)]*\)\s*$", "").str.strip_chars().alias("ev"),
                         pl.col("date").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
                         .cast(pl.Datetime("us")).alias("t"))
           .filter(pl.col("ev").is_in(list(TRIGGERS)))
           .unique(["ev", "t"], keep="first")
           .with_columns((pl.col("actual") - pl.col("estimate")).alias("surp")))
    cal = cal.with_columns((pl.col("surp") / pl.col("surp").std().over("ev")).clip(-3, 3).alias("z")) \
             .filter(pl.col("z").is_finite() & (pl.col("z") != 0))

    def near_kalshi(t):
        i = np.searchsorted(REL, t)
        return min(abs(t - REL[j]) for j in (i - 1, i) if 0 <= j < len(REL)) <= H

    cal = cal.filter(pl.Series([not near_kalshi(np.datetime64(t, "us")) for t in cal["t"]])).sort("t", "ev")
    out = []
    for (t,), g in cal.group_by("t", maintain_order=True):
        out.append(dict(arm="calendar", t_rel=np.datetime64(t, "us"),
                        sources=[dict(src=r["ev"], fam=TRIGGERS[r["ev"]][0], hk=TRIGGERS[r["ev"]][1],
                                      z=float(r["z"])) for r in g.iter_rows(named=True)]))
    print(f"calendar releases kept: {cal.height}, in {len(out)} release minutes")
    return out


def with_placebos(insts: list[dict]) -> list[dict]:
    """Shifts are whole days, so two releases at the same clock time can land on
    one placebo minute; the later one is dropped."""
    out, seen = [], set()
    for r in insts:
        t = placebo_time(r["t_rel"], rng)
        if t is not None and (r["arm"], t) not in seen:
            seen.add((r["arm"], t))
            out.append(dict(r, arm=r["arm"] + "_placebo", t_rel=t, t_true=r["t_rel"]))
    return out


# ------------------------------------------------------------------ cells and paths
def build(insts: list[dict], next_k: np.ndarray, next_any: np.ndarray):
    cells, paths = [], []
    for r in insts:
        t = r["t_rel"]
        srcs = r["sources"]
        released = {s["src"] for s in srcs} if r["arm"].startswith("kalshi") else set()
        ref = r.get("t_true", t)                   # placebo: truncate as the real instant would
        nk = next_k[np.searchsorted(next_k, ref, side="right")] if (next_k > ref).any() else WALL
        na = next_any[np.searchsorted(next_any, ref, side="right")] if (next_any > ref).any() else WALL
        shift = t - ref
        big = max((s for s in srcs if s["src"] in LEGS), key=lambda s: abs(s["z"]), default=None)
        for B in LEGS:
            tb = lead_contract(B, t)
            if tb is None:
                continue
            ts, px = TAPE[tb]
            fl = FLOW[tb]
            i0 = int(np.searchsorted(ts, t, side="left"))      # first print at/after t_rel
            if i0 == 0 or t - ts[i0 - 1] > MAX_AGE:
                continue
            p0 = float(px[i0 - 1])
            if B in released:
                role = "own"
            elif any(is_same_release(B, s) for s in released):
                role = "sibling"
            else:
                role = "cross"
            x = {}
            for s in srcs:
                if s["src"] == B or s["src"] in released and is_same_release(B, s["src"]):
                    continue
                k = f"x_{s['fam']}"
                x[k] = x.get(k, 0.0) + s["z"] * s["hk"] * HAWKISH[B]
            cut_k = min(nk + shift, CLOSE[tb], WALL)
            cut_a = min(na + shift, CLOSE[tb], WALL)
            cell = dict(arm=r["arm"], t_rel=t.item(), target=B, ttype=TARGET_TYPE[B], role=role, ticker=tb,
                        p0=p0, age0_h=(t - ts[i0 - 1]) / H,
                        n_pre7=i0 - int(np.searchsorted(ts, t - 7 * D)),
                        x_own=next((s["z"] for s in srcs if s["src"] == B), 0.0) if role == "own" else 0.0,
                        sig=sum(x.values()), **x,
                        n_src=len(srcs), src_A=big["src"] if big else None,
                        cut_k_h=(cut_k - t) / H, cut_any_h=(cut_a - t) / H)
            for k in KS:
                j = i0 + k - 1
                ok = j < len(ts) and ts[j] - t <= 24 * H and ts[j] <= cut_k
                cell[f"p_k{k}"] = float(px[j]) if ok else None
                cell[f"h_k{k}"] = (ts[j] - t) / H if ok else None
            cells.append(cell)

            g = t + TAU * MIN
            idx = np.searchsorted(ts, g, side="right") - 1
            idx[TAU == 0] = i0 - 1
            ok = idx >= 0
            ii = np.maximum(idx, 0)
            cf = np.concatenate([[0.0], np.cumsum(fl)])        # cf[j] = flow of prints < j
            post = TAU > 0
            n_post = np.where(post, np.maximum(idx - i0 + 1, 0), 0)
            paths.append(dict(
                arm=np.full(len(TAU), r["arm"]), t_rel=np.full(len(TAU), t),
                target=np.full(len(TAU), B), tau_min=TAU,
                p=np.where(ok, px[ii], np.nan),
                p2=np.where(idx >= 1, (px[ii] + px[np.maximum(ii - 1, 0)]) / 2, np.nan),
                n_post=n_post, traded=n_post > 0,
                since_last_min=np.where(ok, (g - ts[ii]) / MIN, np.nan),
                flow=np.where(post, cf[ii + 1] - cf[i0], 0.0),
                trunc_k=g > cut_k, trunc_any=g > cut_a))
    cells = pl.DataFrame(cells, infer_schema_length=None)
    paths = pl.concat([pl.DataFrame(p) for p in paths]).fill_nan(None)   # no print yet → null
    return cells, paths


def main() -> None:
    kal, calr = kalshi_instants(), calendar_instants()
    next_k = np.array(sorted(r["t_rel"] for r in kal), dtype="datetime64[us]")
    next_any = np.array(sorted(r["t_rel"] for r in kal + calr), dtype="datetime64[us]")
    insts = kal + calr
    insts += with_placebos(insts)
    cells, paths = build(insts, next_k, next_any)
    for c in ("x_inflation", "x_labour", "x_growth", "x_policy", "x_activity", "x_sentiment"):
        if c not in cells.columns:
            cells = cells.with_columns(pl.lit(0.0).alias(c))
    xs = [c for c in cells.columns if c.startswith("x_") and c != "x_own"]
    cells = cells.with_columns(*(pl.col(c).fill_null(0.0) for c in xs))
    key = ["arm", "t_rel", "target"]
    p2_0 = paths.filter(pl.col("tau_min") == 0).select(*key, pl.col("p2").alias("p2_0"))
    paths = (paths.join(cells.select(*key, "p0"), on=key).join(p2_0, on=key)
             .with_columns((pl.col("p") - pl.col("p0")).alias("r"),
                           (pl.col("p2") - pl.col("p2_0")).alias("r2")).drop("p0", "p2_0"))
    for f in (cells, paths):
        assert_no_oos(f.with_columns(pl.col("t_rel").dt.replace_time_zone("UTC")), time_col="t_rel")
    OUT.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(OUT / "cells.parquet")
    paths.write_parquet(OUT / "paths.parquet")
    report(cells, paths)


# ------------------------------------------------------------------ sanity checks
def report(cells: pl.DataFrame, paths: pl.DataFrame) -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(3)
    print(f"\ncells {cells.height}, path rows {paths.height}")
    print(cells.group_by("arm").agg(pl.col("t_rel").n_unique().alias("instants"), pl.len().alias("cells"),
                                    (pl.col("role") == "own").sum().alias("own"),
                                    (pl.col("role") == "cross").sum().alias("cross"),
                                    (pl.col("cut_k_h") < 24).mean().alias("cut<24h (k)"),
                                    (pl.col("cut_any_h") < 24).mean().alias("cut<24h (any)"))
          .sort("arm"))

    print("\nCOVERAGE: share of cells with ≥ 1 post-release print by τ (real arms; untruncated at τ)")
    cov = (paths.filter(~pl.col("arm").str.ends_with("placebo") & pl.col("tau_min").is_in([5, 15, 60, 240, 1440])
                        & ~pl.col("trunc_k"))
           .join(cells.select("arm", "t_rel", "target", "role"), on=["arm", "t_rel", "target"]))
    print(cov.group_by("arm", "role", "tau_min").agg(pl.col("traded").mean())
          .pivot(on="tau_min", index=["arm", "role"], values="traded").sort("arm", "role"))
    print("\nby target series, Kalshi arm (own + cross cells)")
    print(cov.filter(pl.col("arm") == "kalshi").group_by("target", "tau_min").agg(pl.col("traded").mean())
          .pivot(on="tau_min", index="target", values="traded").sort("target")
          .join(cells.filter(pl.col("arm") == "kalshi").group_by("target")
                .agg(pl.len().alias("cells"), pl.col("age0_h").median().alias("p₀ age h (med)"),
                     pl.col("n_pre7").median().alias("prints 7d (med)"),
                     pl.col("h_k1").median().alias("1st print h (med)")), on="target"))

    print("\nOWN NEXT CONTRACT: theory-signed move sign(z)·(p − p₀), ¢, Kalshi own cells with a first print")
    own = cells.filter((pl.col("arm") == "kalshi") & (pl.col("role") == "own") & pl.col("p_k1").is_not_null())
    r24 = (paths.filter((pl.col("arm") == "kalshi") & (pl.col("tau_min") == 1440) & ~pl.col("trunc_k"))
           .select("t_rel", "target", pl.col("r").alias("r24")))
    own = own.join(r24, on=["t_rel", "target"], how="left")
    s = pl.col("x_own").sign()
    print(own.select(pl.len().alias("n"),
                     (s * (pl.col("p_k1") - pl.col("p0"))).mean().alias("k=1"),
                     (s * (pl.col("p_k3") - pl.col("p0"))).mean().alias("k=3"),
                     (s * (pl.col("p_k10") - pl.col("p0"))).mean().alias("k=10"),
                     (s * pl.col("r24")).mean().alias("+24h"),
                     pl.col("h_k1").median().alias("1st print h (med)")))

    print("\nCROSS CELLS: mean theory-signed move sign(sig)·r, ¢, real vs placebo (cells with sig ≠ 0)")
    cr = (paths.filter(pl.col("tau_min").is_in([-60, 15, 60, 240, 1440]) & ~pl.col("trunc_k"))
          .join(cells.filter((pl.col("role") == "cross") & (pl.col("sig") != 0))
                .select("arm", "t_rel", "target", "sig"), on=["arm", "t_rel", "target"])
          .with_columns((pl.col("sig").sign() * pl.col("r")).alias("sr")))
    print(cr.group_by("arm", "tau_min").agg(pl.col("sr").mean())
          .pivot(on="tau_min", index="arm", values="sr").sort("arm"))
    print(f"\nwrote {OUT / 'cells.parquet'} and {OUT / 'paths.parquet'}")


if __name__ == "__main__":
    main()
