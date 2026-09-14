#!/usr/bin/env python
"""Primary probes for the four Tier-3 candidates — which deserve a full study?

    venv/bin/python analysis/relations_2026_09/tier3_probes.py

Each section is a cheap first look, not the study itself. The point is to see
which candidates show a signal worth spending the remaining weeks on. Nothing
here is a finished result and none of it should be cited.

  §7  order-flow response   — replace a 98.1%-tied cent difference with signed
                              taker imbalance over the same window
  §8  belief co-movement    — a second adjacency from the node panel, ~10x obs
  §9  sequential transmission — does CPI->FED strengthen when the preceding
                              payrolls surprise agreed in sign?
  §10 uncertainty/attention — does a surprise move the target's implied_std,
                              entropy or trade intensity, independent of price?

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.splits import assert_no_oos
from stg.structure.stats import permutation_p, spearman

PANELS = Path("artifacts/panels")
HAWKISH = {"CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
           "CPIGAS": +1, "CPIUSEDCAR": +1, "PAYROLLS": +1, "ADP": +1,
           "U3": -1, "JOBLESSCLAIMS": -1, "GDP": +1, "WTI": +1, "WTIW": +1}
TSIGN = {("FED", "any"): +1, ("FEDDECISION", "hike"): +1, ("FEDDECISION", "cut"): -1}
EDGES = [("CPI", "FED", "any"), ("PAYROLLS", "FEDDECISION", "hike"),
         ("PAYROLLS", "FED", "any"), ("CPICOREYOY", "FEDDECISION", "cut")]


def tie_rate(x: np.ndarray) -> float:
    """Share of observations sharing a value with at least one other."""
    _, counts = np.unique(x, return_counts=True)
    return float((counts[counts > 1].sum()) / len(x)) if len(x) else float("nan")


def probe_orderflow(pp: pl.DataFrame, tr: pl.LazyFrame) -> None:
    print("\n" + "=" * 78)
    print("§7  ORDER-FLOW RESPONSE  — does signed taker imbalance beat cents?")
    print("=" * 78)
    need = pp.filter(pl.col("target_ticker").is_not_null()
                     & pl.col("t_exit").is_not_null())
    tks = need["target_ticker"].unique().to_list()
    t = (tr.filter(pl.col("ticker").is_in(tks))
           .select("ticker", "created_time", "count", "taker_side").collect())
    # signed size: taker lifting YES is positive pressure
    t = t.with_columns(
        pl.when(pl.col("taker_side") == "yes").then(pl.col("count"))
         .when(pl.col("taker_side") == "no").then(-pl.col("count"))
         .otherwise(0).alias("signed"))
    by = {}
    for tk, g in t.group_by("ticker"):
        key = tk[0] if isinstance(tk, tuple) else tk
        gg = g.sort("created_time")
        by[key] = (gg["created_time"].to_numpy().astype("datetime64[ns]").astype("int64"),
                   gg["signed"].to_numpy().astype(float),
                   gg["count"].to_numpy().astype(float))
    flow, vol = [], []
    for r in need.iter_rows(named=True):
        rec = by.get(r["target_ticker"])
        if rec is None:
            flow.append(np.nan); vol.append(np.nan); continue
        ts, sg, ct = rec
        a = np.datetime64(r["t0"].replace(tzinfo=None), "ns").astype("int64")
        b = np.datetime64(r["t_exit"].replace(tzinfo=None), "ns").astype("int64")
        m = (ts > a) & (ts <= b)
        flow.append(float(sg[m].sum()) if m.any() else 0.0)
        vol.append(float(ct[m].sum()) if m.any() else 0.0)
    need = need.with_columns(pl.Series("net_flow", flow), pl.Series("volume", vol)).drop_nulls("net_flow")

    resp = need["response"].to_numpy().astype(float)
    nf = need["net_flow"].to_numpy()
    print(f"rows {need.height}")
    print(f"  tie rate, price response  : {tie_rate(resp):.3f}")
    print(f"  tie rate, order flow      : {tie_rate(nf):.3f}   <- the whole point")
    print(f"  corr(price resp, net flow): {np.corrcoef(resp, nf)[0,1]:+.3f}")

    print("\n  per published edge, Spearman(surprise, response) vs (surprise, net_flow):")
    print(f"  {'edge':32s} {'n':>4s} {'rho_price':>10s} {'p':>7s} {'rho_flow':>9s} {'p':>7s}")
    for a, b, s in EDGES:
        d = need.filter((pl.col("trigger") == a) & (pl.col("target") == b) & (pl.col("side") == s))
        if d.height < 10:
            continue
        S = d["surprise"].to_numpy().astype(float)
        R = d["response"].to_numpy().astype(float); F = d["net_flow"].to_numpy()
        rp, rf = spearman(S, R), spearman(S, F)
        pp_ = permutation_p(S, R, 4000, rng=np.random.default_rng(0))
        pf = permutation_p(S, F, 4000, rng=np.random.default_rng(0))
        print(f"  {a+'->'+b+'/'+s:32s} {d.height:4d} {rp:+10.3f} {pp_:7.4f} {rf:+9.3f} {pf:7.4f}")

    # pooled channel, aligned
    d = need.filter(~pl.col("same_release") & pl.col("trigger").is_in(list(HAWKISH)))
    keep = [(g, s) in TSIGN for g, s in zip(d["target"], d["side"])]
    d = d.filter(pl.Series(keep))
    if d.height:
        sgn = np.array([np.sign(x) * HAWKISH[t_] * TSIGN[(g, s)]
                        for x, t_, g, s in zip(d["surprise"], d["trigger"], d["target"], d["side"])])
        R = d["response"].to_numpy().astype(float); F = d["net_flow"].to_numpy()
        print(f"\n  pooled data->policy, aligned sign agreement (n={d.height}):")
        print(f"    price response : {float((sgn * R > 0)[(sgn!=0)&(R!=0)].mean()):.3f}  "
              f"(usable {int(((sgn!=0)&(R!=0)).sum())})")
        print(f"    order flow     : {float((sgn * F > 0)[(sgn!=0)&(F!=0)].mean()):.3f}  "
              f"(usable {int(((sgn!=0)&(F!=0)).sum())})  <- more usable rows = the power gain")


def probe_comovement(node: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("§8  BELIEF CO-MOVEMENT GRAPH — a second adjacency from the node panel")
    print("=" * 78)
    n = node.filter(pl.col("d_implied_mean").is_not_null())
    w = n.group_by("series", "date").agg(pl.col("d_implied_mean").mean()).pivot(
        values="d_implied_mean", index="date", on="series").sort("date")
    cols = [c for c in w.columns if c != "date"]
    print(f"node-panel rows {node.height}, usable deltas {n.height}, "
          f"dates {w.height}, series {len(cols)}")
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = w.select(a, b).drop_nulls()
            if d.height < 20:
                continue
            x, y = d[a].to_numpy(), d[b].to_numpy()
            if x.std() == 0 or y.std() == 0:
                continue
            rows.append(dict(a=a, b=b, n=d.height, rho=spearman(x, y),
                             p=permutation_p(x, y, 2000, rng=np.random.default_rng(0))))
    t = pl.DataFrame(rows)
    if t.height == 0:
        print("  no pair reaches 20 shared dates — co-movement graph not viable")
        return
    print(f"  estimable pairs {t.height}; nominal p<0.05: {int((t['p']<0.05).sum())} "
          f"(expected {0.05*t.height:.1f})")
    print(f"  median shared dates per pair: {t['n'].median():.0f}  "
          f"(Stage-1 pairs run on n=10-36)")
    with pl.Config(tbl_rows=12, float_precision=3):
        print(t.sort("p").head(12))


def probe_sequential(sp: pl.DataFrame, pp: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("§9  SEQUENTIAL TRANSMISSION — does a prior payrolls surprise condition it?")
    print("=" * 78)
    pay = (sp.filter(pl.col("series") == "PAYROLLS")
           .select("close_time", pl.col("surprise").alias("pay_s")).sort("close_time"))
    for trig, tgt, side in [("CPI", "FED", "any"), ("CPICORE", "FED", "any")]:
        d = pp.filter((pl.col("trigger") == trig) & (pl.col("target") == tgt)
                      & (pl.col("side") == side)).sort("t0")
        if d.height < 12:
            continue
        d = d.join_asof(pay, left_on="t0", right_on="close_time", strategy="backward")
        d = d.drop_nulls("pay_s").with_columns(
            (np.sign(pl.col("surprise")) == np.sign(pl.col("pay_s"))).alias("agree"))
        print(f"\n  {trig}->{tgt}/{side}   n={d.height}")
        for agree in (True, False):
            g = d.filter(pl.col("agree") == agree)
            if g.height < 8:
                print(f"    agree={agree}: n={g.height} (too few)"); continue
            S = g["surprise"].to_numpy().astype(float)
            R = g["response"].to_numpy().astype(float)
            rho = spearman(S, R)
            p = permutation_p(S, R, 4000, rng=np.random.default_rng(0))
            print(f"    prior payrolls {'AGREED ' if agree else 'CONFLICTED'}: "
                  f"n={g.height:3d}  rho={rho:+.3f}  p={p:.4f}")


def probe_uncertainty(sp: pl.DataFrame, node: pl.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("§10 UNCERTAINTY / ATTENTION RESPONSE — channels other than the mean")
    print("=" * 78)
    n = (node.sort("series", "event_ticker", "date")
         .with_columns([
             (pl.col("implied_std") - pl.col("implied_std").shift(1))
               .over(["series", "event_ticker"]).alias("d_std"),
             (pl.col("implied_entropy") - pl.col("implied_entropy").shift(1))
               .over(["series", "event_ticker"]).alias("d_entropy"),
             (pl.col("recent_volume") - pl.col("recent_volume").shift(1))
               .over(["series", "event_ticker"]).alias("d_volume"),
         ]))
    trig = sp.select(pl.col("series").alias("trigger"), "close_time",
                     pl.col("surprise").abs().alias("abs_s"),
                     pl.col("surprisal"), pl.col("implied_std").alias("trig_std"))
    trig = trig.with_columns(pl.col("close_time").dt.date().alias("date"))
    out = []
    for tgt in ("FED", "FEDDECISION", "CPI", "U3"):
        tn = n.filter(pl.col("series") == tgt).select(
            "date", "d_std", "d_entropy", "d_volume")
        if tn.height < 20:
            continue
        for trg in ("CPI", "PAYROLLS", "U3", "CPICORE"):
            if trg == tgt:
                continue
            j = (trig.filter(pl.col("trigger") == trg).join(tn, on="date", how="inner")
                 .drop_nulls(["abs_s"]))
            if j.height < 12:
                continue
            for col in ("d_std", "d_entropy", "d_volume"):
                dd = j.drop_nulls(col)
                if dd.height < 12:
                    continue
                x = dd["abs_s"].to_numpy().astype(float)
                y = dd[col].to_numpy().astype(float)
                if x.std() == 0 or y.std() == 0:
                    continue
                out.append(dict(trigger=trg, target=tgt, channel=col, n=dd.height,
                                rho=spearman(x, y),
                                p=permutation_p(x, y, 2000, rng=np.random.default_rng(0))))
    if not out:
        print("  no cell reaches n>=12 — same-day node rows are too sparse")
        return
    t = pl.DataFrame(out)
    print(f"  cells {t.height}; nominal p<0.05: {int((t['p']<0.05).sum())} "
          f"(expected {0.05*t.height:.1f})")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(t.sort("p").head(15))


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    pp = pl.read_parquet(PANELS / "pair_panel_dormant.parquet")
    node = pl.read_parquet(PANELS / "node_panel_event.parquet")
    assert_no_oos(sp, time_col="close_time")
    tr = scan_trades(is_only=True)
    probe_orderflow(pp, tr)
    probe_comovement(node)
    probe_sequential(sp, pp)
    probe_uncertainty(sp, node)


if __name__ == "__main__":
    main()
