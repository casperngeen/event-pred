#!/usr/bin/env python
"""Ablation and benchmarks for the frozen strategy (reports/strategy_spec.md).

    venv/bin/python analysis/leadlag_2026_09/ablation.py
    (needs build_panel.py)

Every variant runs on the same panel, the same cost model, the same entry
convention (second post-resolution print) and the same event-clustered
bootstrap, so the only thing that differs is the component being ablated.

The spec has three moving parts:

  SIGNAL    direction = HAWKISH[A]*HAWKISH[B] * z_surprise, walk-forward terciles
  CONFIRM   the market already moved >= 2c the signal's way
  FLOOR     entry price >= 20c

Benchmarks, in increasing order of how much they should be beaten by:

* **permuted signal through the whole spec** -- the real null. Shuffle
  z_surprise among trigger events *within trigger series*, rebuild the signal,
  refit the terciles, reapply both filters. Preserves the ladder, the outcomes,
  the prices, the filters and the CPI/CPIYOY dependence; destroys only the
  trigger->target pairing, which is the lead-lag claim itself.
* **momentum** -- trade the direction of the move, ignore the signal.
* **random side** -- coin flip, filters still applied.
* **always buy / always sell YES** -- passive, no view.
* **perfect foresight** -- the ceiling: trade the known settlement, same filters
  and same costs. What the apparatus could earn if the signal were oracular.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import scan_trades

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT, N_PERM, FEE_RATE, CONTRACTS = 10000, 200, 0.07, 100
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


def tercile_side(sig, p0, yr, years):
    """The walk-forward tercile rule, per price bucket. Never fits on year Y."""
    side = np.zeros(len(sig))
    for lo, hi in BUCKETS:
        m = (p0 >= lo) & (p0 < hi)
        if m.sum() < 60:
            continue
        idx = np.where(m)[0]
        for Y in years[1:]:
            tr = idx[yr[idx] < Y]
            te = idx[yr[idx] == Y]
            if len(tr) < 60 or len(te) == 0:
                continue
            t = np.percentile(sig[tr], [33.3, 66.7])
            side[te] = np.where(sig[te] > t[1], 1.0,
                                np.where(sig[te] <= t[0], -1.0, 0.0))
    return side


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet").drop_nulls("p0")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    zc = np.clip(z, -lim, lim)
    direction = d["direction"].to_numpy().astype(float)
    p0, pe = d["p0"].to_numpy(), d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev, yr = d["target_event"].to_numpy(), d["yr"].to_numpy()
    tk = d["target_ticker"].to_numpy()
    tres = d["t_res"].dt.epoch("s").to_numpy()
    years = sorted(np.unique(yr))

    # second post-resolution print = the executable entry
    tt = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(d["target_ticker"].unique().to_list()))
          .select("ticker", "yes_price", "created_time")
          .sort("ticker", "created_time").collect())
    g = (tt.with_columns(pl.col("created_time").dt.epoch("s").alias("s"))
         .group_by("ticker", maintain_order=True)
         .agg(pl.col("s"), pl.col("yes_price")))
    tape = {t: (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=float))
            for t, a, b in zip(g["ticker"], g["s"], g["yes_price"])}
    p2 = np.full(len(p0), np.nan)
    for i in range(len(p0)):
        arr = tape.get(tk[i])
        if arr is None:
            continue
        s_, px_ = arr
        j = int(np.searchsorted(s_, tres[i], side="right"))
        if j + 1 < len(s_):
            p2[i] = px_[j + 1]
    ok = ~np.isnan(p2)
    print(f"rows: {len(p0)}   with an executable entry: {int(ok.sum())}\n")

    def evaluate(side, use_confirm=True, use_floor=True):
        """Net per position under a given side vector and filter set."""
        m = ok & (side != 0)
        if use_confirm:
            m = m & (((pe - p0) * np.sign(side)) >= CONFIRM_C)
        ent = np.where(side > 0, p2, 100.0 - p2)
        if use_floor:
            m = m & (ent >= FLOOR_C)
        if m.sum() < 30:
            return None
        pay = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
        net = pay - ent - (fee_cents(ent) + spread_cents(ent) / 2.0)
        return net[m], ev[m], m

    sig_true = direction * zc
    side_true = tercile_side(sig_true, p0, yr, years)

    rows = []

    def add(name, side, conf, floor, note=""):
        r = evaluate(side, conf, floor)
        if r is None:
            return
        net, evs, m = r
        o, lo, hi, p = cboot(net, evs)
        rows.append(dict(variant=name, n=int(m.sum()),
                         events=int(len(np.unique(evs))),
                         net=o, ci_lo=lo, ci_hi=hi, p_le0=p, note=note))

    # ---- the spec and its ablations
    add("FULL SPEC (signal+confirm+floor)", side_true, True, True)
    add("  - drop FLOOR", side_true, True, False)
    add("  - drop CONFIRM", side_true, False, True)
    add("  - drop BOTH (signal only)", side_true, False, False)

    # ---- signal removed, apparatus kept
    mom = np.sign(pe - p0)
    add("SIGNAL REMOVED: momentum side", mom, True, True, "trade the move")
    rng = np.random.default_rng(0)
    rnd = rng.choice([-1.0, 1.0], size=len(p0))
    add("SIGNAL REMOVED: random side", rnd, True, True, "coin flip")
    add("PASSIVE: always buy YES", np.ones(len(p0)), False, True)
    add("PASSIVE: always sell YES", -np.ones(len(p0)), False, True)

    # ---- confirmation direction vs mere activity
    m_act = ok & (side_true != 0) & (np.abs(pe - p0) >= CONFIRM_C)
    ent_a = np.where(side_true > 0, p2, 100.0 - p2)
    m_act = m_act & (ent_a >= FLOOR_C)
    if m_act.sum() > 30:
        pay = np.where(side_true > 0, 100.0 * win, 100.0 * (1 - win))
        net_a = pay - ent_a - (fee_cents(ent_a) + spread_cents(ent_a) / 2.0)
        o, lo, hi, p = cboot(net_a[m_act], ev[m_act])
        rows.append(dict(variant="CONFIRM on |move| not direction",
                         n=int(m_act.sum()), events=int(len(np.unique(ev[m_act]))),
                         net=o, ci_lo=lo, ci_hi=hi, p_le0=p,
                         note="did it move at all?"))

    # ---- ceiling
    oracle = np.where(win > 0.5, 1.0, -1.0)
    add("CEILING: perfect foresight", oracle, False, True, "oracle side")

    with pl.Config(tbl_rows=25, float_precision=2, tbl_width_chars=240):
        print(pl.DataFrame(rows))

    # ---- the real null: permute the signal through the WHOLE spec
    print("\n=== block-permuted signal through the entire specification ===")
    print("z_surprise shuffled among trigger events within each trigger series;")
    print("signal rebuilt, terciles refit, both filters reapplied. Everything")
    print("except the trigger->target pairing is preserved.\n")
    trig_ev = d["trigger_event"].to_numpy()
    trig_sr = d["trigger"].to_numpy()
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    te_z = np.zeros(len(uniq_te))
    te_s = np.empty(len(uniq_te), dtype=object)
    for i, te in enumerate(uniq_te):
        j = int(np.argmax(trig_ev == te))
        te_z[i], te_s[i] = zc[j], trig_sr[j]
    groups = [np.where(te_s == s)[0] for s in np.unique(te_s)]

    rng = np.random.default_rng(1)
    null_net, null_n = [], []
    for _ in range(N_PERM):
        zp = te_z.copy()
        for gg in groups:
            zp[gg] = te_z[rng.permutation(gg)]
        sp = direction * zp[row_te]
        r = evaluate(tercile_side(sp, p0, yr, years), True, True)
        if r is None:
            continue
        null_net.append(float(r[0].mean()))
        null_n.append(int(r[2].sum()))
    null_net = np.array(null_net)
    obs = rows[0]["net"]
    print(f"observed        : {obs:+.2f}c   (n = {rows[0]['n']})")
    print(f"permuted null   : {null_net.mean():+.2f}c +/- {null_net.std():.2f}"
          f"   median n = {int(np.median(null_n))}   draws = {len(null_net)}")
    # (r+1)/(m+1) rather than r/m: with 200 draws the resolution floor is
    # 1/201, and reporting 0.0000 would overstate what this many draws can show.
    r = int((null_net >= obs).sum())
    print(f"draws >= observed: {r} of {len(null_net)}")
    print(f"p(one-sided)    : {(r + 1) / (len(null_net) + 1):.4f}   "
          f"(= (r+1)/(m+1); floor is {1 / (len(null_net) + 1):.4f} at m = {len(null_net)})")
    print(f"observed sits {(obs - null_net.mean()) / null_net.std():.1f} sd above the null mean")
    print(f"null percentiles: p5 {np.percentile(null_net,5):+.2f}  "
          f"p50 {np.percentile(null_net,50):+.2f}  "
          f"p95 {np.percentile(null_net,95):+.2f}  max {null_net.max():+.4f}")
    print(f"\nThe null MEAN is the number that matters: {null_net.mean():+.2f}c is what")
    print("the apparatus alone earns -- the two filters applied to a signal with")
    print("no information in it. The signal's marginal contribution is therefore")
    print(f"{obs:+.2f} - {null_net.mean():+.2f} = {obs - null_net.mean():+.2f}c, not the full {obs:+.2f}c.")


if __name__ == "__main__":
    main()
