"""Average the five seeds' test-fold forecasts of one model version into the "_ens5" files the trade simulator reads
(analysis/graph_2026_10/b7_maker_trades.py), on the cluster, so only one file per fold has to come back to the
laptop (5x less to copy and store). numpy only: no polars, which crashes on the cluster's login node.

For each test fold: checks every seed predicted the same rows, averages the forecast changes ("delta"), and writes
    <out>/<L>_fold<k>_<NAME>_ens5_seed0.npz   (the arrays b5_longshort.rows reads)
    <out>/<L>_fold<k>_<NAME>_ens5_seed0.json  (seed 0's settings, plus "r2_ens5" and "r2_seeds": R^2 (%) of the change
                                               in the traded price against no change, per kind and horizon)
--delete-seeds removes the per-seed .npz files once their average is written (the .json files stay).

    python cluster/average_seeds.py --league MLB --name gru_d10_h30-60-120-1800_tp_pitch_r0 [--delete-seeds]
    python cluster/average_seeds.py --league NBA --name gru_d30_h30-60-120-1800_tp
then on the laptop: rsync 'xlogin:~/event-pred/results/ingame/ens5/<L>_fold*_<NAME>_ens5_seed0.*' results/ingame/ens5/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KINDS = {0: "winner", 1: "spread", 2: "total"}
KEEP = ("step", "game", "kind", "value", "y", "has_bid", "has_ask", "entry_bid", "entry_ask")   # + delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--name", required=True, help="the run name between fold<k>_ and _seed<s>, e.g. gru_d10_h30-60-120-1800_tp")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--res", default="results/ingame")
    ap.add_argument("--out", default="results/ingame/ens5")
    ap.add_argument("--delete-seeds", action="store_true")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    for k in range(1, 6):
        files = [res / f"{a.league}_fold{k}_{a.name}_seed{s}.npz" for s in seeds]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print(f"fold {k}: missing {missing}, skipped")
            continue
        zs = [np.load(f) for f in files]
        z0 = zs[0]
        for z in zs[1:]:
            assert np.array_equal(z["step"], z0["step"]) and np.array_equal(z["kind"], z0["kind"]), f"fold {k}: rows differ"
        deltas = [z["delta"] for z in zs]
        ens = np.mean(deltas, 0)
        meta = json.loads((res / f"{a.league}_fold{k}_{a.name}_seed{seeds[0]}.json").read_text())
        hz = meta["horizons"]
        tgt = z0["value_h"] - z0["value"][:, None]
        r2 = lambda d: {kn: {h: round(100 * (1 - float(((tgt[ok, i] - d[ok, i]) ** 2).sum()) / float((tgt[ok, i] ** 2).sum())), 3)  # noqa: E731
                             for i, h in enumerate(hz) for ok in [(z0["kind"] == kc) & np.isfinite(tgt[:, i])] if ok.any()}
                        for kc, kn in KINDS.items()}
        meta["r2_ens5"] = r2(ens)
        meta["r2_seeds"] = {f"seed{s}": r2(d) for s, d in zip(seeds, deltas)}
        name = f"{a.league}_fold{k}_{a.name}_ens5_seed0"
        np.savez_compressed(out / f"{name}.npz", **{c: z0[c] for c in KEEP}, delta=ens.astype(np.float32))
        (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(f"fold {k}: {len(ens):,} rows -> {out / name}.npz; R2 120s spread {meta['r2_ens5']['spread'].get('120s')} "
              f"total {meta['r2_ens5']['total'].get('120s')}", flush=True)
        if a.delete_seeds:
            for f in files:
                f.unlink()


if __name__ == "__main__":
    main()
