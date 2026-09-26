#!/usr/bin/env python
"""Does the far-meeting labour->policy effect survive trading costs?

    venv/bin/python analysis/propagation_2026_09/economics.py
    (needs build_panel.py to have written out/depth_legs.parquet)

``profile.py`` found labour surprises (PAYROLLS, U3, JOBLESSCLAIMS) mispriced in
Fed meetings two to six months out at the first print after the release, and
gone within a week. This prices it.

The rule
--------
Zero parameters. At each labour release, for every FED threshold leg in the
horizon band, net the aligned signals of every trigger that printed at that
instant -- PAYROLLS and U3 come out of one BLS release, so counting them as two
trades would double every position and, when they disagree, take both sides of
one leg. Then take ``sign(net signal)``: YES if positive, NO if negative. Hold to
settlement.

**The horizon band is not pre-specified.** "2-6 months" was read off
``profile.py``'s table, so every number here is an in-sample upper bound on a
rule chosen after looking. The contrast rows (0-1m, 1-2m, later delays) are
there to show the shape, not to rescue it.

Costs
-----
``leadlag_2026_09/economics.py``'s model -- Kalshi taker fee on the contract
bought (7% x p(1-p), 100 contracts, rounded up) plus half the effective spread
-- with one change: the spread is **measured on the legs actually traded**.
The flat schedule (2c, 1c in the wings) comes from ``research_log`` §11.2 on
near-dated legs; far-dated FED legs trade less and could be wider. So §1
measures ``stg.direction.tradability.effective_spread`` on FED legs by
days-to-settlement and moneyness, in the 24 h after a labour release (the
window the trade is taken in) and at all times, and §2 charges half the
in-window median for the leg's cell. The far legs are too thin for the
standard 60 s pairing window, so where it yields nothing the cell falls back to
a 1 h window on all trades (more drift-contaminated -- ``effective_spread``'s
own validity check is that widening the window must not move the estimate),
and only then to the flat schedule. The charge is never below the flat
schedule: the 1 h estimate on the far wings is 0c on ~55 pairs, which is drift,
not a free market. §3 stresses the spread x2 and x3.

Controls
--------
``always YES`` and ``always NO`` take every far leg on one side regardless of
the signal. If in-sample far FED legs were simply mispriced one way (a hiking
cycle the market kept underestimating), the rule could earn by leaning that way
and the signal would be doing nothing. The rule has to beat the better of the
two, not just zero.

Capital is full notional (``p`` for YES, ``100 - p`` for NO). Returns are
**capital-weighted** (total net / total capital), not the mean of per-position
returns, which a handful of 1c positions would dominate. ``ret_yr`` annualises
by capital-days: total net / total (capital x days held / 365).

Inference: clustered bootstrap on ``target_event``; block permutation of
``z_surprise`` within trigger series, rule re-run on each draw.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.direction.tradability import effective_spread
from stg.panel._io import scan_trades
from stg.splits import assert_no_oos

OUT = Path("analysis/propagation_2026_09/out")
N_BOOT, N_PERM = 10000, 2000
FEE_RATE, CONTRACTS = 0.07, 100
GROUP = "labour->policy"
BANDS = [(0, 30, "0-1m"), (30, 60, "1-2m"), (60, 90, "2-3m"),
         (90, 120, "3-4m"), (120, 181, "4-6m")]
FAR = ("2-3m", "3-4m", "4-6m")
WINDOW_H = 24


def band_expr(col: str) -> pl.Expr:
    e = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, lab in reversed(BANDS):
        e = pl.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(lab)).otherwise(e)
    return e


def wing_expr(col: str) -> pl.Expr:
    return pl.when((pl.col(col) - 50).abs() > 40).then(pl.lit("wing")).otherwise(pl.lit("centre"))


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def schedule_spread(price_cents) -> np.ndarray:
    """``leadlag_2026_09/economics.py``'s flat schedule (research_log §11.2)."""
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cluster_boot(vals, groups, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s = np.bincount(inv, weights=vals, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


# ---------------------------------------------------------------- §1 spreads
def measure_spreads(d: pl.DataFrame) -> pl.DataFrame:
    legs = d.select("target_ticker", "close_time").unique("target_ticker")
    tr = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(legs["target_ticker"].to_list()))
          .select("ticker", "created_time", "yes_price", "taker_side").collect()
          .join(legs.rename({"target_ticker": "ticker"}), on="ticker")
          .with_columns(((pl.col("close_time") - pl.col("created_time"))
                         .dt.total_seconds() / 86400).alias("dtc")))
    assert_no_oos(tr, time_col="created_time")

    # in-window: within WINDOW_H after any labour release in the panel
    rel = np.sort(d["t_res"].dt.epoch("ns").unique().to_numpy())
    ns = tr["created_time"].dt.epoch("ns").to_numpy()
    j = np.searchsorted(rel, ns, side="right") - 1
    since = np.where(j >= 0, ns - rel[np.maximum(j, 0)], np.iinfo(np.int64).max)
    tr = tr.with_columns(pl.Series("in_window", since <= WINDOW_H * 3_600 * 10**9),
                         band_expr("dtc").alias("h"), wing_expr("yes_price").alias("m"))
    tr = tr.filter(pl.col("h").is_not_null()).with_columns(
        pl.concat_str("h", "m", separator="|").alias("cell"))

    out = []
    for nm, sub, gap in (("window", tr.filter(pl.col("in_window")), 60),
                         ("all", tr, 60), ("all_1h", tr, 3600)):
        e = effective_spread(sub, by="cell", max_gap_s=gap)
        if not e.is_empty():
            out.append(e.with_columns(pl.lit(nm).alias("sample")))
    sp = pl.concat(out).with_columns(pl.col("cell").str.split("|").list.get(0).alias("h"),
                                     pl.col("cell").str.split("|").list.get(1).alias("m"))
    print("=== §1 effective spread on FED legs, by days-to-settlement and moneyness ===")
    print(f"window = trades within {WINDOW_H}h after a labour release, 60 s pairing; "
          "all = every in-sample trade, 60 s; all_1h = every trade, 1 h pairing. "
          "Cells too thin or too drift-contaminated are withheld by effective_spread.\n")
    with pl.Config(tbl_rows=40, tbl_cols=-1, float_precision=2, tbl_width_chars=200):
        print(sp.select("sample", "h", "m", "spread_median", "spread_mean", "n_pairs",
                        "frac_negative").sort("sample", "h", "m", descending=[True, False, False]))
    # charge: in-window 60 s where it exists, else all-trades 1 h
    w = sp.filter(pl.col("sample") == "window").select("h", "m", "spread_median")
    f = (sp.filter(pl.col("sample") == "all_1h").select("h", "m", "spread_median")
         .join(w.select("h", "m"), on=["h", "m"], how="anti"))
    charged = pl.concat([w.with_columns(pl.lit("window_60s").alias("src")),
                         f.with_columns(pl.lit("all_1h").alias("src"))]).sort("h", "m")
    print("\ncharged spread per cell (else the flat schedule):")
    with pl.Config(tbl_rows=20, float_precision=2):
        print(charged)
    return charged.drop("src")


# ---------------------------------------------------------------- §2 the rule
def positions(d: pl.DataFrame) -> pl.DataFrame:
    """One position per (release instant, leg, delay): signals netted."""
    return (d.with_columns(pl.col("t_res").dt.truncate("1m").alias("release"))
            .group_by("release", "target_ticker", "delay_d")
            .agg(pl.col("signal").sum().alias("net_signal"),
                 pl.col("trigger").sort().str.join("+").alias("triggers"),
                 pl.col("target_event", "horizon", "p_entry", "win", "gap_days",
                        "entry_lag_d", "yr", "t_entry").first()))


def price(p: pl.DataFrame, sp: pl.DataFrame, spread_mult: float = 1.0,
          force_side: float | None = None) -> pl.DataFrame:
    side_e = pl.lit(force_side) if force_side is not None else pl.col("net_signal").sign()
    p = (p.filter(pl.col("net_signal") != 0)
         .with_columns(side_e.cast(pl.Float64).alias("side"), wing_expr("p_entry").alias("m"))
         .join(sp.rename({"h": "horizon"}), on=["horizon", "m"], how="left"))
    px, win, side = p["p_entry"].to_numpy(), p["win"].to_numpy().astype(float), p["side"].to_numpy()
    entry = np.where(side > 0, px, 100.0 - px)
    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    measured = p["spread_median"].to_numpy()
    measured = np.where(np.isnan(measured.astype(float)), 0.0, measured)
    spread = np.maximum(measured, schedule_spread(px)) * spread_mult
    fee = fee_cents(entry)
    hold = np.maximum((p["gap_days"] - p["delay_d"]).to_numpy(), 1.0)
    net = payoff - entry - fee - spread / 2.0
    return p.with_columns(pl.Series("cap", entry), pl.Series("gross", payoff - entry),
                          pl.Series("fee", fee), pl.Series("half_spread", spread / 2.0),
                          pl.Series("net", net), pl.Series("hold_d", hold))


def summarise(p: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    rows = []
    for k, g in p.group_by(keys, maintain_order=False):
        ev = g["target_event"].to_numpy()
        if len(np.unique(ev)) < 6:
            continue
        net, lo, hi, pneg = cluster_boot(g["net"].to_numpy(), ev)
        rows.append(dict(**dict(zip(keys, k)), n=g.height, tgt_ev=len(np.unique(ev)),
                         cap=g["cap"].mean(), gross=g["gross"].mean(), fee=g["fee"].mean(),
                         half_spr=g["half_spread"].mean(), net=net, ci_lo=lo, ci_hi=hi,
                         p_le0=pneg, ret_cap=100.0 * g["net"].sum() / g["cap"].sum(),
                         ret_yr=100.0 * g["net"].sum() / (g["cap"] * g["hold_d"] / 365.0).sum(),
                         hold_d=float(g["hold_d"].median())))
    return pl.DataFrame(rows).sort(keys)


def show(t: pl.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    with pl.Config(tbl_rows=60, tbl_cols=-1, float_precision=2, tbl_width_chars=260):
        print(t)


def perm_null(d: pl.DataFrame, sp: pl.DataFrame, mask_fn, seed=0) -> np.ndarray:
    """Shuffle z among trigger events within each trigger series; re-run the rule."""
    te = d.select("trigger_event", "trigger", "z_surprise").unique("trigger_event")
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        shuf = pl.concat([
            g.with_columns(pl.Series("z_perm", g["z_surprise"].to_numpy()[rng.permutation(g.height)]))
            for _, g in te.sort("trigger_event").group_by("trigger", maintain_order=True)])
        dp = (d.join(shuf.select("trigger_event", "z_perm"), on="trigger_event")
              .with_columns((pl.col("direction") * pl.col("z_perm")).alias("signal")))
        p = mask_fn(price(positions(dp), sp))
        null[b] = float(p["net"].mean()) if p.height else np.nan
    return null


def main() -> None:
    d = (pl.read_parquet(OUT / "depth_legs.parquet")
         .filter(pl.col("group") == GROUP)
         .with_columns(band_expr("gap_days").alias("horizon"))
         .filter(pl.col("horizon").is_not_null()))
    assert_no_oos(d, time_col="close_time")
    print(f"{GROUP}: {d.height} rows, triggers {sorted(d['trigger'].unique().to_list())}\n")

    sp = measure_spreads(d)
    pos = price(positions(d), sp)
    print(f"\npositions after netting same-release triggers: {pos.height} "
          f"(from {d.height} trigger x leg rows)")
    print("trigger sets per position:",
          dict(pos.group_by("triggers").len().sort("len", descending=True).iter_rows()))

    print("\n=== §2 net P&L, cents per contract, by horizon and delay ===")
    print("net = payoff - entry - taker fee - half the measured spread. "
          "CI clustered on target_event; p_le0 = bootstrap share <= 0.")
    show(summarise(pos, ["horizon", "delay_d"]), "all legs")
    mid = pos.filter((pl.col("p_entry") >= 10) & (pl.col("p_entry") < 75))
    show(summarise(mid, ["horizon", "delay_d"]), "10-75c legs (not pre-specified)")

    far = lambda p: p.filter(pl.col("horizon").is_in(FAR) & (pl.col("delay_d") == 0))
    far_mid = lambda p: far(p).filter((pl.col("p_entry") >= 10) & (pl.col("p_entry") < 75))
    print("\n=== §3 the headline rule: far meetings (2-6m), enter at d = 0 ===")
    print("Chosen after reading profile.py. An in-sample upper bound.")
    for nm, fn in (("all legs", far), ("10-75c legs", far_mid)):
        p = fn(pos)
        net, lo, hi, pneg = cluster_boot(p["net"].to_numpy(), p["target_event"].to_numpy())
        print(f"\n{nm}: n {p.height}  target events {p['target_event'].n_unique()}  "
              f"gross {p['gross'].mean():+.2f}c  fee {p['fee'].mean():.2f}c  "
              f"half-spread {p['half_spread'].mean():.2f}c")
        print(f"  net {net:+.2f}c  CI [{lo:+.2f}, {hi:+.2f}]  P(<=0) {pneg:.3f}   "
              f"mean capital {p['cap'].mean():.1f}c   median hold {p['hold_d'].median():.0f} d   "
              f"return on capital {100 * p['net'].sum() / p['cap'].sum():+.2f}%   "
              f"annualised {100 * p['net'].sum() / (p['cap'] * p['hold_d'] / 365).sum():+.1f}%")
        for nm2, fs in (("always YES", 1.0), ("always NO", -1.0)):
            pc = fn(price(positions(d), sp, force_side=fs))
            n5, l5, h5, pn5 = cluster_boot(pc["net"].to_numpy(), pc["target_event"].to_numpy())
            print(f"  control {nm2:<10}: net {n5:+.2f}c  CI [{l5:+.2f}, {h5:+.2f}]  P(<=0) {pn5:.3f}")
        for mult in (2.0, 3.0):
            ps = fn(price(positions(d), sp, spread_mult=mult))
            n2, l2, h2, pn2 = cluster_boot(ps["net"].to_numpy(), ps["target_event"].to_numpy())
            print(f"  spread x{mult:.0f}: net {n2:+.2f}c  CI [{l2:+.2f}, {h2:+.2f}]  P(<=0) {pn2:.3f}")
        rows = []
        yes = fn(price(positions(d), sp, force_side=1.0))
        for Y, g in p.group_by("yr"):
            if g["target_event"].n_unique() < 3:
                continue
            n3, l3, h3, pn3 = cluster_boot(g["net"].to_numpy(), g["target_event"].to_numpy())
            gy = yes.filter(pl.col("yr") == Y[0])
            rows.append(dict(year=Y[0], n=g.height, tgt_ev=g["target_event"].n_unique(),
                             net=n3, ci_lo=l3, ci_hi=h3, p_le0=pn3,
                             always_yes=float(gy["net"].mean()),
                             rule_minus_yes=n3 - float(gy["net"].mean())))
        show(pl.DataFrame(rows).sort("year"), f"{nm}, by year (always_yes = the control)")
        # the signal's contribution within each side, against that side's baseline
        no = fn(price(positions(d), sp, force_side=-1.0))
        rows = []
        for sv, base, lab in ((1.0, yes, "YES"), (-1.0, no, "NO")):
            g = p.filter(pl.col("side") == sv)
            n6, l6, h6, pn6 = cluster_boot(g["net"].to_numpy(), g["target_event"].to_numpy())
            rows.append(dict(side=lab, n=g.height, rule_net=n6, ci_lo=l6, ci_hi=h6,
                             always_side_net=float(base["net"].mean()),
                             signal_adds=n6 - float(base["net"].mean())))
        show(pl.DataFrame(rows), f"{nm}, the rule's trades on each side vs always taking that side")

    print("\n=== §4 block-permutation null on the headline net P&L ===")
    for nm, fn in (("all legs", far), ("10-75c legs", far_mid)):
        obs = float(fn(pos)["net"].mean())
        null = perm_null(d, sp, fn)
        print(f"{nm}: observed {obs:+.3f}c   null {np.nanmean(null):+.3f} +/- "
              f"{np.nanstd(null):.3f}   p(one-sided) = {np.nanmean(null >= obs):.4f}")

    pos.write_parquet(OUT / "economics_positions.parquet")
    print(f"\nwrote {OUT / 'economics_positions.parquet'}")


if __name__ == "__main__":
    main()
