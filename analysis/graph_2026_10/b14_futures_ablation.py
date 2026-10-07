"""Simplifying the model: can the teams' futures contracts be dropped from the graph (panel_v3 instead of
panel_v3f) without losing forecast accuracy on the game contracts (winner / spread / total)?

Both versions: GRU play memory, 2 rounds of message passing, feed delay 10 s, horizons 30/60/120/1800 s,
traded-price targets, seeds 0-4, test folds 1-5. The game-contract rows (winner / spread / total) are the
same contract-steps in both versions (checked), so the comparison is row for row.

1. R^2 (%) of the change in the traded price against "no change", per league x contract type x horizon:
   with futures, without, and the difference, each for the average of the five seeds' forecasts; plus the
   range of the difference over single seeds (seed s with vs seed s without).
2. A game-clustered t for the difference (5-seed averages): per game, the gain in squared error.
3. How many futures rows each league's panel_v3f has per fold (MLB has none: a pure seed-noise check).

    venv/bin/python -W ignore analysis/graph_2026_10/b14_futures_ablation.py
    -> out/b14_futures_ablation.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

RES = Path("results/ingame")
OUT = Path(__file__).resolve().parent / "out"
SEEDS = [0, 1, 2, 3, 4]
NAMES = {"with": "gru_d10_h30-60-120-1800_tpf", "without": "gru_d10_h30-60-120-1800_tp"}
HZ = ["30s", "60s", "120s", "1800s"]
KINDS = {0: "winner", 1: "spread", 2: "total"}


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(80)
    rows, per_game, fut = [], [], []
    for L in ("NBA", "MLB", "NFL"):
        acc = {}                                       # (kind, horizon) -> sums
        for k in range(1, 6):
            z = {v: [np.load(RES / f"{L}_fold{k}_{n}_seed{s}.npz") for s in SEEDS] for v, n in NAMES.items()}
            w0, o0 = z["with"][0], z["without"][0]
            g = w0["kind"] <= 2
            fut.append({"league": L, "fold": k, "futures_rows": int((~g).sum()), "game_rows": int(g.sum())})
            for v in z:
                for zz in z[v][1:]:
                    assert np.array_equal(zz["step"], z[v][0]["step"]), f"{L} fold {k} {v}: seeds' rows differ"
            assert np.array_equal(w0["step"][g], o0["step"]) and np.array_equal(w0["kind"][g], o0["kind"]), f"{L} fold {k}: rows differ"
            tgt = o0["value_h"] - o0["value"][:, None]
            assert np.allclose(np.nan_to_num(w0["value_h"][g] - w0["value"][g][:, None], nan=-9), np.nan_to_num(tgt, nan=-9)), f"{L} fold {k}: targets differ"
            dw = [zz["delta"][g] for zz in z["with"]]
            do = [zz["delta"] for zz in z["without"]]
            ew, eo = np.mean(dw, 0), np.mean(do, 0)
            game = o0["game"]
            for kind, kn in KINDS.items():
                m = o0["kind"] == kind
                for hi, h in enumerate(HZ):
                    ok = m & np.isfinite(tgt[:, hi])
                    y = tgt[ok, hi]
                    a = acc.setdefault((kn, h), {"ss": 0.0, "with": 0.0, "without": 0.0,
                                                 "sw": np.zeros(len(SEEDS)), "so": np.zeros(len(SEEDS))})
                    a["ss"] += float((y ** 2).sum())
                    sw, so = (y - ew[ok, hi]) ** 2, (y - eo[ok, hi]) ** 2
                    a["with"] += float(sw.sum())
                    a["without"] += float(so.sum())
                    for i in range(len(SEEDS)):
                        a["sw"][i] += float(((y - dw[i][ok, hi]) ** 2).sum())
                        a["so"][i] += float(((y - do[i][ok, hi]) ** 2).sum())
                    per_game.append(pl.DataFrame({"league": L, "kind": kn, "horizon": h, "fold": k, "game": game[ok],
                                                  "gain": so - sw, "ss": y ** 2})
                                    .group_by("league", "kind", "horizon", "fold", "game").agg(pl.col("gain", "ss").sum()))
        for (kn, h), a in acc.items():
            r_w, r_o = 100 * (1 - a["with"] / a["ss"]), 100 * (1 - a["without"] / a["ss"])
            seed_diff = 100 * (a["so"] - a["sw"]) / a["ss"]
            rows.append({"league": L, "kind": kn, "horizon": h, "R2_with": r_w, "R2_without": r_o, "diff": r_w - r_o,
                         "seed_diff_min": seed_diff.min(), "seed_diff_max": seed_diff.max()})
    res = pl.DataFrame(rows)
    pg = pl.concat(per_game)
    t = (pg.group_by("league", "kind", "horizon")
         .agg(n_games=pl.len(), S=pl.col("ss").sum(), G=pl.col("gain").sum(), sd=pl.col("gain").std())
         .with_columns(t=pl.col("G") / (pl.col("sd") * pl.col("n_games").sqrt()))
         .select("league", "kind", "horizon", "n_games", "t"))
    res = res.join(t, on=["league", "kind", "horizon"]).with_columns(pl.col(pl.Float64).round(2))
    order = {L: i for i, L in enumerate(["NBA", "MLB", "NFL"])} | {k: i for i, k in enumerate(KINDS.values())} | {h: i for i, h in enumerate(HZ)}
    res = res.sort(pl.col("league").replace_strict(order), pl.col("kind").replace_strict(order), pl.col("horizon").replace_strict(order))
    lines = [__doc__.split("\n\n")[0], "",
             "R^2 in %; diff = with - without (5-seed averages; negative = the simpler model is better); seed_diff = the same difference seed by seed;",
             "t = game-clustered t of the per-game squared-error gain (negative = the simpler model is better).", "",
             str(res), "", "Futures rows in panel_v3f test folds (game rows identical in both versions):",
             str(pl.DataFrame(fut).pivot(on="fold", index="league", values="futures_rows"))]
    OUT.mkdir(exist_ok=True)
    (OUT / "b14_futures_ablation.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
