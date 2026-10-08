"""Five seeds of the full model (GRU memory, 2 rounds of message passing; feed delay 10 s, horizons
30/60/120/1800 s, traded-price panel with futures) and their average.

1. R^2 (%) of the change in the traded price against no change, per seed and for the average of the five
   seeds' forecasts (same rows: every seed predicts the same contract-steps), all test folds.
2. Writes the averaged predictions for b7_maker_trades.py (only the arrays b5_longshort.rows reads):
   <out>/<L>_fold<k>_gru_d10_h30-60-120-1800_tpf_ens5_seed0.npz (+ .json)
   then: RUN_TAG=_h30-60-120-1800_tpf_ens5 b7_maker_trades.py <out> <L> 10 <panel>, and b11_ablation_pnl.py seeds.

    venv/bin/python -W ignore analysis/graph_2026_10/b13_seed_ensemble.py <dir for the averaged predictions>
    -> out/b13_seed_ensemble.txt
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

RES = Path("results/ingame")
OUT = Path(__file__).resolve().parent / "out"
SEEDS = [0, 1, 2, 3, 4]
NAME = os.environ.get("NAME", "gru_d10_h30-60-120-1800_tpf")   # NAME=gru_d10_h30-60-120-1800_tp: the model without futures (panel_v3)
HZ = ["30s", "60s", "120s", "1800s"]
LEAGUES = os.environ.get("LEAGUES", "NBA,MLB,NFL").split(",")   # LEAGUES=MLB NAME=..._tp_pitch: the MLB every-pitch model
KINDS = {0: "winner", 1: "spread", 2: "total"}
KEEP = {"step", "game", "kind", "value", "delta", "y", "has_bid", "has_ask", "entry_bid", "entry_ask"}   # what b5_longshort.rows reads


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    rows = []
    for L in LEAGUES:
        sse = {}                                  # (kind, horizon, version) -> [sse, ss_no_change]
        for k in range(1, 6):
            zs = [np.load(RES / f"{L}_fold{k}_{NAME}_seed{s}.npz") for s in SEEDS]
            z0 = zs[0]
            for z in zs[1:]:
                assert np.array_equal(z["step"], z0["step"]) and np.array_equal(z["kind"], z0["kind"]), f"{L} fold {k}: rows differ"
            deltas = [z["delta"] for z in zs]
            ens = np.mean(deltas, 0)
            tgt = z0["value_h"] - z0["value"][:, None]
            for kind, kn in KINDS.items():
                m = z0["kind"] == kind
                for hi, h in enumerate(HZ):
                    ok = m & np.isfinite(tgt[:, hi])
                    y = tgt[ok, hi]
                    for v, d in [(f"seed{s}", deltas[i]) for i, s in enumerate(SEEDS)] + [("average", ens)]:
                        a = sse.setdefault((kn, h, v), [0.0, 0.0])
                        a[0] += float(((y - d[ok, hi]) ** 2).sum())
                        a[1] += float((y ** 2).sum())
            # forecast size: the sd of the 120 s forecast on spread/total rows (averaging shrinks it)
            st = z0["kind"] >= 1
            for v, d in [(f"seed{s}", deltas[i]) for i, s in enumerate(SEEDS)] + [("average", ens)]:
                a = sse.setdefault(("sd_120s", "", v), [0.0, 0.0])
                a[0] += float((d[st, 2] ** 2).sum())
                a[1] += float(st.sum())
            name = f"{L}_fold{k}_{NAME}_ens5_seed0"
            np.savez_compressed(out / f"{name}.npz", **({c: z0[c] for c in KEEP if c != "delta"} | {"delta": ens.astype(np.float32)}))
            shutil.copy(RES / f"{L}_fold{k}_{NAME}_seed0.json", out / f"{name}.json")
        for (kn, h, v), (a, b) in sse.items():
            if kn == "sd_120s":
                rows.append(dict(league=L, kind="spread+total", horizon="forecast sd 120s (c)", version=v, value=round(np.sqrt(a / b), 2)))
            elif h != "1800s":
                rows.append(dict(league=L, kind=kn, horizon=h, version=v, value=round(100 * (1 - a / b), 2)))
        print(L, "done", flush=True)
    t = pl.DataFrame(rows).pivot(on="version", index=["league", "kind", "horizon"], values="value")
    t = t.with_columns(pl.mean_horizontal(*[f"seed{s}" for s in SEEDS]).round(2).alias("seed mean"),
                       pl.min_horizontal(*[f"seed{s}" for s in SEEDS]).alias("seed min"),
                       pl.max_horizontal(*[f"seed{s}" for s in SEEDS]).alias("seed max"))
    txt = "R^2 (%) of the change in the traded price vs no change, all test folds; last rows: forecast size\n" + str(t)
    (OUT / ("b13_seed_ensemble.txt" if NAME.endswith("_tpf") else f"b13_seed_ensemble_{NAME}.txt")).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
