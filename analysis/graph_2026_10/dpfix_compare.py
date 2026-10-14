"""Double-play fix A/B on the MLB test folds.

    venv/bin/python -W ignore analysis/graph_2026_10/dpfix_compare.py panel        # the rebuilt panel differs from panel_v3 only in the play-group columns
    venv/bin/python -W ignore analysis/graph_2026_10/dpfix_compare.py pnl OLD_TAG NEW_TAG
        TAG = the b7 file stem after "b7_maker_trades_MLB_d10", e.g. _h30-60-120-1800_tp_pitch_ens5_dec5_one_skip15-60
        -> frozen rule (k 1, through 0, 120 s hold), frozen sizing, per kind: c/contract, game t, $, and the paired
           game-level difference NEW - OLD.
"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "analysis/graph_2026_10")
D = "data/sports/derived/ingame"
OUT = "analysis/graph_2026_10/out"


def panel() -> None:
    a, b = f"{D}/MLB_pitch/panel_v3", f"{D}/MLB_pitch_dpfix/panel_v3"
    ma, mb = (json.load(open(f"{p}/manifest.json")) for p in (a, b))
    diff = {k: (ma.get(k), mb.get(k)) for k in set(ma) | set(mb) if k not in ("built",) and ma.get(k) != mb.get(k)}
    print("manifest differences:", diff)
    names = ma["play_features"]
    for f in sorted(p for p in __import__("os").listdir(a) if p.endswith(".npy")):
        x, y = np.load(f"{a}/{f}", mmap_mode="r"), np.load(f"{b}/{f}", mmap_mode="r")
        if x.shape != y.shape:
            print(f"{f}: SHAPE {x.shape} vs {y.shape}")
            continue
        if f == "plays_x.npy":
            cols = [names[j] for j in range(x.shape[1]) if not np.array_equal(np.asarray(x[:, j]), np.asarray(y[:, j]), equal_nan=True)]
            print(f"{f}: columns that differ {cols}")
            for j in range(9):
                print(f"   {names[j]:18s} old {float(np.asarray(x[:, j], np.float32).sum()):8.0f}  new {float(np.asarray(y[:, j], np.float32).sum()):8.0f}")
        else:
            same = all(np.array_equal(np.asarray(x[i:i + 2_000_000]), np.asarray(y[i:i + 2_000_000]), equal_nan=True)
                       for i in range(0, max(len(x), 1), 2_000_000))
            print(f"{f}: {'identical' if same else 'DIFFERS'}")


def pnl(old: str, new: str) -> None:
    sys.argv = sys.argv[:1] + ["results/ingame", "MLB"]
    from b7_maker_liquidity import liquidity
    from b7_maker_robust import with_net
    from b9_sizing import RULES
    size = RULES["quadratic signal x linear width (missing 1)"]

    def load(tag):
        f = pl.read_parquet(f"{OUT}/b7_maker_trades_MLB_d10{tag}_fold*.parquet").filter(
            pl.col("hold") == "120s", pl.col("k") == 1, pl.col("through") == 0, pl.col("kind").is_in(["spread", "total"]))
        return liquidity(with_net(f)).with_columns(size.alias("size")).with_columns((pl.col("size") * pl.col("net")).alias("pnl"))

    fo, fn = load(old), load(new)
    games = pl.concat([fo.select("game_id", "kind"), fn.select("game_id", "kind")]).unique()
    rows = []
    for kind in ["spread", "total"]:
        r = {"kind": kind}
        g = {}
        for name, f in (("old", fo), ("new", fn)):
            x = f.filter(pl.col("kind") == kind, pl.col("size") > 0)
            g[name] = x.group_by("game_id").agg(pl.col("pnl").sum().alias(name))
            s = g[name][name].to_numpy()
            r |= {f"{name} fills": x.height, f"{name} c/contract": round(x["pnl"].sum() / x["size"].sum(), 2),
                  f"{name} t": round(float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))), 1), f"{name} $": round(x["pnl"].sum() / 100, 1)}
        p = (games.filter(pl.col("kind") == kind).join(g["old"], on="game_id", how="left").join(g["new"], on="game_id", how="left")
             .fill_null(0.0).with_columns((pl.col("new") - pl.col("old")).alias("d")))
        d = p["d"].to_numpy()
        r |= {"new - old $": round(d.sum() / 100, 1), "paired game t": round(float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))), 1) if d.std() > 0 else 0.0}
        rows.append(r)
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_hide_dataframe_shape(True)
    print(pl.DataFrame(rows).transpose(include_header=True, column_names=["spread", "total"]))


if __name__ == "__main__":
    panel() if sys.argv[1] == "panel" else pnl(sys.argv[2], sys.argv[3])
