#!/usr/bin/env python
"""Gross returns of every model's out-of-fold signal, before costs.

    venv/bin/python analysis/event_time_2026_09/returns.py \
        > analysis/event_time_2026_09/out/returns.txt   # needs models.py and ablation.py first

The trade: for each labelled (release instant, series) cell, take the lead
contract at its first post-release print (``p_entry``) on the side the model
predicts, and hold to settlement. Gross P&L per contract, in cents:

    long YES  : 100·[YES] − p_entry  =  y_settle
    long NO   : p_entry − 100·[YES]  = −y_settle

No fee, no spread (``backtest.py`` / ``exits.py`` have the costed versions).

Signals: the settle-label models' predictions, and, separately, the imm-label
models' predictions used as a direction for the same trade. Controls on the
same cells: always YES, always NO, and the zero-parameter economic sign rule
(side = sign Σ_a G[a, b]·z_a) on the BH channels and on all edges.

Always-NO earns in this sample (YES was overpriced, mostly in 2025), so a model
that is short more often earns more for that reason alone. ``vs random side``
removes it: the model's P&L minus that of a random side with the model's own
long/short mix, E = (1 − 2q)·y for short share q. That is the skill part. CIs
are a bootstrap over release instants.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "analysis/event_time_2026_09")
from _panel import G_all, G_bh, Z, instants, nodes  # noqa: E402

OUT = "analysis/event_time_2026_09/out"
N_BOOT = 2000
rng = np.random.default_rng(0)

panel = (pl.read_parquet(f"{OUT}/event_nodes.parquet")
         .with_columns(pl.col("instant").dt.replace_time_zone(None).cast(pl.Datetime("us")))
         .filter(pl.col("y_settle").is_not_null()))
# zero-parameter economic signal per (instant, series)
sig = []
for G, name in ((G_bh, "econ BH"), (G_all, "econ all")):
    S = Z @ G
    for t, inst in enumerate(instants):
        for i, n in enumerate(nodes):
            sig.append(dict(instant=inst.astype("datetime64[us]").item(), series=n, rule=name,
                            s=float(S[t, i])))
sig = pl.DataFrame(sig).pivot(on="rule", index=["instant", "series"], values="s")
cells = panel.select("instant", "series", "y_settle", "p_entry").join(
    sig, on=["instant", "series"], how="left")

frames = [pl.read_parquet(f"{OUT}/oof_all.parquet")]
if os.path.exists(f"{OUT}/oof_ablation.parquet"):
    frames.append(pl.read_parquet(f"{OUT}/oof_ablation.parquet"))
oof = pl.concat(frames, how="diagonal_relaxed").unique(["label", "instant", "series", "model"],
                                                        keep="first", maintain_order=True)


def stats(side: np.ndarray, y: np.ndarray, inst: np.ndarray) -> dict:
    t = side != 0
    side, y, inst = side[t], y[t], inst[t]
    g = side * y
    q = float((side < 0).mean())
    ex = g - (1 - 2 * q) * y
    u, inv = np.unique(inst, return_inverse=True)
    G, E, C = (np.bincount(inv, v, len(u)) for v in (g, ex, np.ones_like(g)))
    bs = [(G[b].sum() / C[b].sum(), E[b].sum() / C[b].sum())
          for b in (rng.integers(0, len(u), len(u)) for _ in range(N_BOOT))]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(n=len(g), short=q, hit=float((g > 0).mean()), gross=g.mean(), g_lo=lo[0],
                g_hi=hi[0], ex=ex.mean(), e_lo=lo[1], e_hi=hi[1])


def line(name, r):
    print(f"{name:52} {r['n']:>5} {r['short']:>6.2f} {r['hit']:>6.3f} "
          f"{r['gross']:>+7.2f} [{r['g_lo']:+6.2f},{r['g_hi']:+6.2f}] "
          f"{r['ex']:>+7.2f} [{r['e_lo']:+6.2f},{r['e_hi']:+6.2f}]", flush=True)


HEAD = (f"{'signal':52} {'n':>5} {'short':>6} {'hit':>6} {'gross ¢/trade':>23} "
        f"{'vs random side ¢/trade':>24}")

for block, restrict in (("all labelled cells", None),
                        ("cells where the BH-channel economic signal fires", "econ BH")):
    for src in ("settle", "imm"):
        d = (oof.filter(pl.col("label") == src).select("instant", "series", "model", "pred_c")
             .join(cells, on=["instant", "series"], how="inner"))
        if restrict:
            d = d.filter(pl.col(restrict).fill_null(0) != 0)
        base = d.unique(["instant", "series"]).sort("instant", "series")
        y, inst = base["y_settle"].to_numpy(), base["instant"].to_numpy()
        print("\n" + "=" * 118)
        print(f"{block} — signal from the '{src}'-label models → hold to settlement "
              f"({len(base)} cells, {len(np.unique(inst))} instants)")
        print("=" * 118)
        print(HEAD)
        line("always YES", stats(np.ones_like(y), y, inst))
        line("always NO", stats(-np.ones_like(y), y, inst))
        for rname in ("econ BH", "econ all"):
            line(f"zero-param econ sign rule ({rname.split()[1]})",
                 stats(np.sign(base[rname].fill_null(0).to_numpy()), y, inst))
        for m in d["model"].unique(maintain_order=True).to_list():
            if m == "linear zero":
                continue
            dm = base.select("instant", "series").join(
                d.filter(pl.col("model") == m).select("instant", "series", "pred_c"),
                on=["instant", "series"], how="left")
            line(m, stats(np.sign(dm["pred_c"].fill_null(0).to_numpy()), y, inst))
