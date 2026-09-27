#!/usr/bin/env python
"""What does trading the event-time signal earn?

    venv/bin/python analysis/event_time_2026_09/backtest.py \
        > analysis/event_time_2026_09/out/backtest.txt   # needs build_panel.py, models.py

The trade, per (release, target node) the model or rule has a view on: take
the lead contract at its **first post-release print** — YES if the view is
positive, NO if negative — and **hold to settlement**. One contract per trade,
100-contract tickets for the fee. Costs are ``leadlag_2026_09/economics.py``'s:
Kalshi taker fee ``ceil(0.07·C·P(1−P))`` plus half the measured effective
spread on entry; settlement has no fee and no spread.

Strategies
----------
rule   zero-parameter: side = sign(Σ HAWKISH[a]·HAWKISH[b]·z_a); trades only
       where the signal fires. Nothing is fitted.
model  walk-forward out-of-fold predictions from ``models.py`` (settle label):
       side = sign(prediction). Scored on every cell it covers and on the cells
       where the economic signal fires (elsewhere the prediction is only the
       fitted intercept, i.e. a directional bias, not the strategy).

"all edges" is the graph fixed a priori. "BH channels" were selected in
``leadlag_2026_09/relations.py`` on these same in-sample events, so their rows
are optimistic by construction.

Inference: mean net cents per contract with a 95% CI from a bootstrap over
release instants (a release's trades share one surprise); a permutation null
shuffles the surprise vector across release instants (``N_PERM`` draws) and
reports the share of draws whose net is ≥ the observed one. In-sample only.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.panel.registry import is_same_release

OUT = "analysis/event_time_2026_09/out"
FEE_RATE, CONTRACTS, N_BOOT, N_PERM = 0.07, 100, 10000, 2000

HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
TYPE = {**{s: "inflation" for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS",
                                    "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL",
                                    "PCECORE")},
        **{s: "labour" for s in ("PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP")},
        "GDP": "growth", "ISMPMI": "growth", "FED": "policy"}
BH_CHANNELS = {("labour", "labour"), ("inflation", "inflation"), ("labour", "policy")}


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def pnl(p_yes, win, side):
    """Per-contract gross and net cents, and capital, for side ±1 at YES price p_yes."""
    entry = np.where(side > 0, p_yes, 100.0 - p_yes)
    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    gross = payoff - entry
    return gross, gross - fee_cents(entry) - spread_cents(entry) / 2.0, entry


def cboot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(u), size=(N_BOOT, len(u)))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return np.percentile(b, 2.5), np.percentile(b, 97.5), float((b <= 0).mean())


# ------------------------------------------------------------------ data
panel = pl.read_parquet(f"{OUT}/event_nodes.parquet").with_columns(
    pl.col("instant").dt.replace_time_zone(None))
released = (panel.filter(pl.col("released"))
            .group_by("instant").agg(pl.col("series"), pl.col("z")))
rel = {r["instant"]: dict(zip(r["series"], r["z"])) for r in released.iter_rows(named=True)}
inst_list = sorted(rel)
trades = panel.filter(pl.col("p_entry").is_not_null()).select(
    "instant", "series", "p_entry", "win", "t_entry")


def signal(inst, tgt, channels=None, zmap=None):
    zmap = zmap if zmap is not None else rel[inst]
    s = 0.0
    for a, z in zmap.items():
        if a == tgt or is_same_release(a, tgt):
            continue
        if channels is not None and (TYPE[a], TYPE[tgt]) not in channels:
            continue
        s += HAWKISH[a] * HAWKISH[tgt] * z
    return s


rows = trades.to_dicts()
for r in rows:
    r["sig_all"] = signal(r["instant"], r["series"])
    r["sig_bh"] = signal(r["instant"], r["series"], BH_CHANNELS)
tr = pl.DataFrame(rows)
_all = panel.filter(pl.col("p_entry").is_not_null())
y_all = (100.0 * _all["win"].to_numpy() - _all["p_entry"].to_numpy())
settle_end_all = _all["settle_end"].to_numpy().astype("datetime64[ns]")
oof = pl.read_parquet(f"{OUT}/oof_settle.parquet").with_columns(
    pl.col("instant").cast(pl.Datetime("us")))
delay_h = ((tr["t_entry"] - tr["instant"].dt.epoch("ns")) / 3.6e12).to_numpy()
print(f"candidate trades (spillover cells with a tradable first post-release print): "
      f"{tr.height} over {tr['instant'].n_unique()} releases")
print(f"entry delay after release: median {np.median(delay_h):.1f} h, "
      f"p25 {np.percentile(delay_h, 25):.1f} h, p75 {np.percentile(delay_h, 75):.1f} h")


def summarise(name, d: pl.DataFrame, side: np.ndarray, perm=None):
    m = side != 0
    if m.sum() == 0:
        print(f"{name:44} no trades"); return
    p, w, g = d["p_entry"].to_numpy()[m], d["win"].to_numpy()[m].astype(float), \
        d["instant"].to_numpy()[m]
    gross, net, cap = pnl(p, w, side[m])
    lo, hi, p0 = cboot(net, g)
    hit = np.mean(gross > 0)
    tot = net.sum() * CONTRACTS / 100.0                     # dollars at 100 contracts/trade
    roc = net.sum() / cap.sum()
    pp = ""
    if perm is not None:
        pp = f" {np.mean(perm >= net.mean()):>6.3f}"
    print(f"{name:44} {int(m.sum()):>5} {len(np.unique(g)):>5} {np.mean(side[m] < 0):>4.0%} "
          f"{hit:>5.1%} {gross.mean():>+7.2f} "
          f"{net.mean():>+7.2f} [{lo:>+6.2f}, {hi:>+6.2f}] {p0:>6.3f} {roc:>+7.1%} "
          f"{tot:>+9.0f}{pp}")
    return net, g


def past_mean_side(d: pl.DataFrame, purge_days: int = 21) -> np.ndarray:
    """Intercept-only baseline: sign of the mean settlement residual over every
    candidate trade whose contract had settled more than ``purge_days`` before
    this release. The causal version of 'the market overprices YES'."""
    t = d["instant"].to_numpy().astype("datetime64[ns]")
    ends = settle_end_all
    out = np.zeros(d.height)
    for k, tk in enumerate(t):
        m = ends < tk - np.timedelta64(purge_days, "D")
        if m.sum() >= 30:
            out[k] = np.sign(y_all[m].mean())
    return out


def perm_null(channels, d: pl.DataFrame):
    """Shuffle each release's surprise vector across releases; rule net each draw."""
    rng = np.random.default_rng(1)
    base = d.to_dicts()
    out = np.empty(N_PERM)
    for b in range(N_PERM):
        shuf = dict(zip(inst_list, rng.permutation(len(inst_list))))
        side = np.array([np.sign(signal(r["instant"], r["series"], channels,
                                        rel[inst_list[shuf[r["instant"]]]]))
                         for r in base])
        m = side != 0
        _, net, _ = pnl(d["p_entry"].to_numpy()[m], d["win"].to_numpy()[m].astype(float),
                        side[m])
        out[b] = net.mean() if m.any() else np.nan
    return out


hdr = (f"{'strategy':44} {'trades':>5} {'rel.':>5} {'NO':>4} {'hit':>5} {'gross':>7} {'net':>7} "
       f"{'95% CI (clustered)':>18} {'P(≤0)':>6} {'ROC':>7} {'$ @100':>9} {'perm p':>6}")


def block(title, d):
    print(f"\n--- {title}  ({d.height} candidate cells)")
    print(hdr); print("-" * len(hdr))
    for nm, col, ch in (("rule: all edges (a priori)", "sig_all", None),
                        ("rule: BH channels (selected in-sample)", "sig_bh", BH_CHANNELS)):
        side = np.sign(d[col].to_numpy())
        summarise(nm, d, side, perm_null(ch, d))
    for ch in sorted(BH_CHANNELS):
        s = np.array([np.sign(signal(r["instant"], r["series"], {ch})) for r in d.to_dicts()])
        summarise(f"  rule: {ch[0]}→{ch[1]}", d, s, perm_null({ch}, d))
    summarise("control: always YES", d, np.ones(d.height))
    summarise("control: always NO", d, -np.ones(d.height))
    summarise("control: walk-forward past-mean sign", d, past_mean_side(d))
    summarise("control: always the favourite (p ≥ 50 → YES)", d,
              np.where(d["p_entry"].to_numpy() >= 50, 1.0, -1.0))


print("\nnet = gross − taker fee − half spread, cents per contract. ROC = Σnet / Σcapital.")
print("$ @100 = total net dollars trading 100 contracts per signal. perm p = share of")
print("surprise-shuffled draws with net ≥ observed (rules only).")
block("ALL RELEASES (rules only; nothing is fitted)", tr)

cov = oof.select("instant", "series").unique()
trc = tr.join(cov, on=["instant", "series"], how="inner")
block("WALK-FORWARD TEST FOLDS ONLY (the releases the models are scored on)", trc)

print(f"\n--- WALK-FORWARD MODELS (side = sign of the out-of-fold prediction)")
print("(the a-priori all-edge graph fires on every candidate cell, so 'all cells' IS")
print(" the all-edge signal set; the sub-rows restrict to BH-channel cells)")
print(hdr); print("-" * len(hdr))
summarise("control: walk-forward past-mean sign", trc, past_mean_side(trc))
_bh = trc["sig_bh"].to_numpy() != 0
summarise("   …only where the BH-channel signal fires", trc, np.where(_bh, past_mean_side(trc), 0))
summarise("control: always NO, BH-channel cells", trc, np.where(_bh, -1.0, 0))
for model in oof["model"].unique(maintain_order=True).to_list():
    if model == "linear zero":
        continue
    d = trc.join(oof.filter(pl.col("model") == model), on=["instant", "series"])
    side = np.sign(d["pred_c"].to_numpy())
    summarise(model[:44], d, side)
    fire = d["sig_bh"].to_numpy() != 0
    summarise("   …only where the BH-channel signal fires", d, np.where(fire, side, 0))

print("\nby year — rule, all edges, all releases:")
for y in sorted(tr["instant"].dt.year().unique().to_list()):
    d = tr.filter(pl.col("instant").dt.year() == y)
    summarise(f"  {y}", d, np.sign(d["sig_all"].to_numpy()))
print("by year — rule, BH channels, all releases:")
for y in sorted(tr["instant"].dt.year().unique().to_list()):
    d = tr.filter(pl.col("instant").dt.year() == y)
    summarise(f"  {y}", d, np.sign(d["sig_bh"].to_numpy()))


print("\nnet cents by entry price (YES price at first post-release print), test folds:")
BK = [(1, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 99.01)]
cols = [("past-mean sign", None), ("linear own state", "linear own state"),
        ("linear econ (all)", "linear econ signal only (all)")]
print(f"{'bucket':>10} {'n':>4} " + " ".join(f"{c:>20}" for c, _ in cols))
base = trc.join(oof.filter(pl.col("model") == "linear own state").select(
    "instant", "series", pl.col("pred_c").alias("own")), on=["instant", "series"]).join(
    oof.filter(pl.col("model") == "linear econ signal only (all)").select(
        "instant", "series", pl.col("pred_c").alias("econ")), on=["instant", "series"])
sides = {"past-mean sign": past_mean_side(base), "linear own state": np.sign(base["own"].to_numpy()),
         "linear econ (all)": np.sign(base["econ"].to_numpy())}
pe, wn = base["p_entry"].to_numpy(), base["win"].to_numpy().astype(float)
for lo, hi in BK:
    m = (pe >= lo) & (pe < hi)
    cells = []
    for c, _ in cols:
        sd = sides[c][m]
        _, net, _ = pnl(pe[m], wn[m], sd)
        cells.append(f"{net[sd != 0].mean():+7.2f} (NO {np.mean(sd < 0):3.0%})" if (sd != 0).any() else "–")
    print(f"{f'{lo:g}-{hi:g}c':>10} {int(m.sum()):>4} " + " ".join(f"{c:>20}" for c in cells))
