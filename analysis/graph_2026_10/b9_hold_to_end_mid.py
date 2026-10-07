"""Hold to the end of the game, current model vs the fresh-mid model (Huber 5): NBA spreads, 120 s
horizon signal, feed delay 10 s, test folds 3-5 (the mid model's runs). Cut-offs: current >= 8c; mid
matched to the current model's trade count (signal = predicted future mid - last print). Entry filters:
all, fresh mid (both sides <= 30 s), width >= 9c, both. Held net = side x (100 y - entry) - maker fee at
entry; split into (a) entry discount (trade-based mid at entry - entry), (b) drift to the last print
30 min later, (c) the result vs that price.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_hold_to_end_mid.py
    -> out/b9_hold_to_end_mid.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b6_maker_exit import MAKER_RATE                     # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_candidates_by_model import entry_state          # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402

MODELS = {"current": "_h30-60-120-1800_tpf", "mid Huber 5": "_h30-60-120-1800_mid"}
FILTERS = {"all": pl.lit(True), "fresh mid": pl.col("older_s") <= 30, "width >= 9c": pl.col("width") >= 9,
           "fresh & width >= 9c": (pl.col("older_s") <= 30) & (pl.col("width") >= 9)}


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(60)
    pick = lambda m: pl.read_parquet(OUT / f"b7_maker_trades_NBA_d10{MODELS[m]}.parquet").filter(   # noqa: E731
        pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s", pl.col("fold").is_in([3, 4, 5]))
    raw = {m: pick(m) for m in MODELS}
    n = raw["current"].filter(pl.col("signal").abs() >= 8).height
    cut = {"current": 8.0, "mid Huber 5": float(raw["mid Huber 5"]["signal"].abs().sort(descending=True)[n - 1])}
    tk = pl.concat([f.select("ticker") for f in raw.values()]).unique()["ticker"]
    tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
    later = tr.select("ticker", pl.col("t").alias("t30"), pl.col("yes_price").alias("p30")).sort("t30")
    q = (pl.col("entry") / 100).clip(0, 1)
    rows, folds = [], []
    for m, f in raw.items():
        f = entry_state(with_net(f.filter(pl.col("signal").abs() >= cut[m])), tr)
        f = (f.with_columns((pl.col("t") + 1800).alias("t30")).sort("t30").join_asof(later, on="t30", by="ticker", strategy="backward")
             .with_columns(((pl.col("p_yes") + pl.col("p_no")) / 2).alias("mid0"), (MAKER_RATE * q * (1 - q)).alias("fee"))
             .with_columns((pl.col("side") * (100 * pl.col("y") - pl.col("entry")) - pl.col("fee")).alias("hold_net"),
                           (pl.col("side") * (pl.col("mid0") - pl.col("entry"))).alias("a_discount"),
                           (pl.col("side") * (pl.col("p30") - pl.col("mid0"))).alias("b_drift"),
                           (pl.col("side") * (100 * pl.col("y") - pl.col("p30"))).alias("c_result")))
        for fn, cond in FILTERS.items():
            g = f.filter(cond)
            gm = g["game_id"].to_numpy()
            t = lambda c, x=g, gg=gm: round(clustered_t(x[c].drop_nulls().to_numpy(), gg[x[c].is_not_null().to_numpy()]), 1)   # noqa: E731
            h = g["hold_net"].to_numpy()
            rows.append(dict(model=m, entry=fn, cut=round(cut[m], 1), trades=g.height, fixed_exit=round(g["net"].mean(), 2), fixed_t=t("net"),
                             hold=round(h.mean(), 2), hold_t=t("hold_net"), hold_median=round(float(np.median(h)), 1), won=round(float((h > 0).mean()), 2),
                             a_discount=round(g["a_discount"].mean(), 2), a_t=t("a_discount"), b_drift=round(g["b_drift"].mean(), 2), b_t=t("b_drift"),
                             c_result=round(g["c_result"].mean(), 2), c_t=t("c_result")))
            for (k,), x in sorted(g.group_by("fold"), key=lambda z: z[0]):
                folds.append(dict(model=m, entry=fn, fold=k, trades=x.height, fixed_exit=round(x["net"].mean(), 1), hold=round(x["hold_net"].mean(), 1)))
    text = ("NBA spreads, 10 s delay, folds 3-5: fixed exit vs holding to the result (cents per contract)\n" + str(pl.DataFrame(rows))
            + "\n\nby test fold\n" + str(pl.DataFrame(folds).pivot(on="fold", index=["model", "entry"], values=["fixed_exit", "hold"])))
    (OUT / "b9_hold_to_end_mid.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
