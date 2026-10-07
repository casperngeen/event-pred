"""Is the ladder gap (b12_ladder_gap.py) just the bid-ask bounce? And the inputs for its P&L test.

Bounce: a last trade at the bid makes a line look cheap against its neighbours, and the next trade ticks
back up without anything changing. Three measures, all at the step, each contract in its own cents:
    bounce          last trade (the model's value) - own bid/ask mid: the bounce itself
    gap (trades)    the ladder gap from last trades (b12_ladder_gap)
    gap (mids)      the ladder gap from bid/ask mids only (own and neighbours'): no trade price enters
Walk-forward least squares (no intercept; slopes fitted on earlier test folds, scored on folds 2-5) of the
change in the traded price at +60 s / +120 s, alone and on top of the full model's forecast, on the rows
where all three exist. If the gap were only bounce, "gap (mids)" would predict nothing once "bounce" is in.

Also writes the full model's predictions with beta x gap added to the 60 s and 120 s forecasts (spread and
total contracts; beta walk-forward on the full model's residual, 0 in fold 1) for b7_maker_trades.py:
    <scratch>/<L>_fold<k>_gru_d10_h30-60-120-1800_tpf_gap[mid]_seed0.npz  (+ .json)

    venv/bin/python -W ignore analysis/graph_2026_10/b12_ladder_bounce.py <dir for the adjusted predictions>
    -> out/b12_ladder_bounce.txt
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b12_ladder_gap import ARMS, OUT, RES, ladder_gap, ladders, load, r2   # noqa: E402
from stg.models.ingame import Panel                                        # noqa: E402
from stg.sports.loaders import derived_dir                                 # noqa: E402

X = {"bounce": ["bounce"], "gap (trades)": ["gap"], "gap (mids)": ["gap_mid"], "bounce + gap (mids)": ["bounce", "gap_mid"],
     "bounce + gap (trades)": ["bounce", "gap"]}
ON_FULL = {"full": [], "full + bounce": ["bounce"], "full + gap (mids)": ["gap_mid"], "full + gap (trades)": ["gap"],
           "full + bounce + gap (mids)": ["bounce", "gap_mid"]}
HZ = {"60": 1, "120": 2}
KEEP = {"step", "game", "kind", "value", "delta", "y", "has_bid", "has_ask", "entry_bid", "entry_ask"}   # what b5_longshort.rows reads


def walk_forward(g: pl.DataFrame, t: str, base: str | None, xs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Base + X b on folds 2-5, b fitted by least squares on the earlier folds; returns predictions and mean b."""
    ps, bs = [], []
    for k in range(2, 6):
        tr, te = g.filter(pl.col("fold") < k), g.filter(pl.col("fold") == k)
        p = te[base].to_numpy() if base else np.zeros(te.height)
        if xs:
            r = tr[t].to_numpy() - (tr[base].to_numpy() if base else 0)
            b = np.linalg.lstsq(tr.select(xs).to_numpy(), r, rcond=None)[0]
            p = p + te.select(xs).to_numpy() @ b
            bs.append(b)
        ps.append(p)
    return np.concatenate(ps), (np.mean(bs, 0) if bs else np.array([]))


def adjusted_preds(L: str, f: pl.DataFrame, col: str, tag: str, out: Path) -> None:
    """The full model's npz with beta x col added to the 60 s / 120 s forecasts of spread and total rows."""
    for k in range(1, 6):
        src = RES / f"{L}_fold{k}_{ARMS['full']}.npz"
        z = {c: v for c, v in np.load(src).items() if c in KEEP}
        d = z["delta"].copy()
        te = f.filter(pl.col("fold") == k, pl.col("kind") >= 1, pl.col(col).is_not_null())
        for h, hi in HZ.items():
            tr = f.filter(pl.col("fold") < k, pl.col("kind") >= 1, pl.col(col).is_not_null(), pl.col(f"t{h}").is_finite())
            for kind in (1, 2):
                a, b = tr.filter(pl.col("kind") == kind), te.filter(pl.col("kind") == kind)
                if a.height == 0:
                    continue                     # fold 1: no earlier folds, forecast unchanged
                x, r = a[col].to_numpy(), a[f"t{h}"].to_numpy() - a[f"d{h}"].to_numpy()
                beta = float((x * r).sum() / (x * x).sum())
                d[b["ri"].to_numpy(), hi] += beta * b[col].to_numpy()
        z["delta"] = d
        name = f"{L}_fold{k}_gru_d10_h30-60-120-1800_tpf_{tag}_seed0"
        np.savez_compressed(out / f"{name}.npz", **z)
        shutil.copy(src.with_suffix(".json"), out / f"{name}.json")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(60)
    rows, coefs = [], []
    for L in ("NBA", "MLB", "NFL"):
        pn = Panel(derived_dir() / "ingame" / L / "panel_v3f")
        f = ladder_gap(ladder_gap(ladders(load(L, pn, "full"), pn)), price="mid", name="gap_mid")
        f = f.with_columns((pl.col("value") - pl.col("mid")).fill_nan(None).alias("bounce"))
        adjusted_preds(L, f, "gap", "gap", out)
        adjusted_preds(L, f, "gap_mid", "gapmid", out)
        for kind in ("spread", "total"):
            k = f.filter(pl.col("kind") == {"spread": 1, "total": 2}[kind])
            for h in HZ:
                t = f"t{h}"
                a = k.filter(pl.col(t).is_finite())
                g = a.drop_nulls(["gap", "gap_mid", "bounce"])
                y = g.filter(pl.col("fold") >= 2)[t].to_numpy()
                r = dict(league=L, kind=kind, horizon=f"{h}s", rows=len(y), share_of_rows=round(g.height / a.height, 2),
                         corr_gap_bounce=round(float(np.corrcoef(g["gap"], g["bounce"])[0, 1]), 2),
                         corr_gapmid_bounce=round(float(np.corrcoef(g["gap_mid"], g["bounce"])[0, 1]), 2))
                for name, xs in X.items():
                    p, _ = walk_forward(g, t, None, xs)
                    r[name] = round(100 * r2(y, p), 2)
                for name, xs in ON_FULL.items():
                    p, b = walk_forward(g, t, f"d{h}", xs)
                    r[name] = round(100 * r2(y, p), 2)
                    if name == "full + bounce + gap (mids)":
                        coefs.append(dict(league=L, kind=kind, horizon=f"{h}s", beta_bounce=round(b[0], 2), beta_gap_mids=round(b[1], 2)))
                _, b = walk_forward(g, t, None, ["bounce", "gap_mid"])
                coefs[-1] |= dict(no_model_beta_bounce=round(b[0], 2), no_model_beta_gap_mids=round(b[1], 2))
                rows.append(r)
        print(L, "done", flush=True)
    t = pl.DataFrame(rows)
    txt = ("Ladder gap vs bid-ask bounce: R^2 (%) of the change in the traded price, folds 2-5, walk-forward slopes, rows "
           "where the bounce and both gaps exist\n" + str(t.select("league", "kind", "horizon", "rows", "share_of_rows", "corr_gap_bounce",
                                                                     "corr_gapmid_bounce", *X)) + "\n\non top of the full model\n"
           + str(t.select("league", "kind", "horizon", *ON_FULL)) + "\n\nslopes (mean over the walk-forward fits)\n" + str(pl.DataFrame(coefs)))
    (OUT / "b12_ladder_bounce.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
