"""How related are the current model's and the fresh-mid model's maker signals? NBA spread, 120 s hold,
feed delay 10 s, test folds 3-5.
    A  current model (trade-price targets), |signal| >= 8c, width >= 9c
    B  mid model (Huber 5), signal = predicted future mid - last print, cut-off matched to the current
       model's trade count, fresh mid at entry (both sides <= 30 s)
1. trade overlap (same contract and decision time): both / only A / only B, side agreement, net
2. net per game and per day: correlation, and the two combined
3. signals at every decision row (every 60 s, contract with a print in the last 60 s): correlation of
   the two signals, sign agreement, and agreement among rows where either is strong

    venv/bin/python -W ignore analysis/graph_2026_10/b9_signal_overlap.py
    -> out/b9_signal_overlap.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.argv = ["x", "results/ingame", "NBA", "10"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b5_longshort as B                                    # noqa: E402
from b7_maker_robust import with_net, summ, OUT             # noqa: E402
from b9_candidates_by_model import entry_state             # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades     # noqa: E402

FOLDS = [3, 4, 5]
KIND, HOLD = "spread", "120s"
D = derived_dir() / "ingame" / "NBA"


def decision_signals(tag: str, panel: str) -> pl.DataFrame:
    B.RUN_TAG, B.PANEL = tag, D / panel
    pn = B.Panel(B.PANEL)
    games = json.loads((B.PANEL / "manifest.json").read_text())["games"]
    r = B.attach_tickers(B.rows(pn, games)).filter(pl.col("kind") == KIND)
    return r.select("game_id", "ticker", "t", pl.col(f"level_{HOLD}").alias("level"))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    lines = []
    out = lambda s: (print(s), lines.append(str(s)))   # noqa: E731
    # ---- 1. trades
    la = pl.read_parquet(OUT / "b7_maker_trades_NBA_d10_h30-60-120-1800_tpf.parquet")
    lb = pl.read_parquet(OUT / "b7_maker_trades_NBA_d10_h30-60-120-1800_mid.parquet")
    pick = lambda f: f.filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("fold").is_in(FOLDS), pl.col("kind") == KIND, pl.col("hold") == HOLD)   # noqa: E731
    la, lb = pick(la), pick(lb)
    n = la.filter(pl.col("signal").abs() >= 8).height
    cut_b = float(lb["signal"].abs().sort(descending=True)[n - 1])
    tk = pl.concat([la.select("ticker"), lb.select("ticker")]).unique()["ticker"]
    tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
    A = entry_state(with_net(la.filter(pl.col("signal").abs() >= 8)), tr).filter(pl.col("width") >= 9)
    Bt = entry_state(with_net(lb.filter(pl.col("signal").abs() >= cut_b)), tr).filter(pl.col("older_s") <= 30)
    key = ["ticker", "t"]
    both = A.join(Bt.select(*key, pl.col("side").alias("side_b"), pl.col("net").alias("net_b")), on=key, how="inner")
    out(f"A = current, >= 8c, width >= 9c: {A.height} trades; B = mid Huber 5, >= {cut_b:.1f}c, fresh mid: {Bt.height} trades")
    rows = [summ(both, group="both (A's result)") | dict(same_side=round((both["side"] == both["side_b"]).mean(), 2)),
            summ(both.with_columns(pl.col("net_b").alias("net")), group="both (B's result)"),
            summ(A.join(Bt.select(key), on=key, how="anti"), group="only A"),
            summ(Bt.join(A.select(key), on=key, how="anti"), group="only B")]
    out("\n######## 1. trade overlap (same contract, same decision time)")
    out(pl.DataFrame(rows))
    # ---- 2. per game and per day
    for unit in ("game_id", "day"):
        ga = A.with_columns(pl.col("first_play").dt.date().alias("day")).group_by(unit).agg(pl.col("net").sum().alias("a"))
        gb = Bt.with_columns(pl.col("first_play").dt.date().alias("day")).group_by(unit).agg(pl.col("net").sum().alias("b"))
        g = ga.join(gb, on=unit, how="full", coalesce=True).fill_null(0)
        both_active = g.filter((pl.col("a") != 0) & (pl.col("b") != 0))
        c = lambda x: round(float(np.corrcoef(x["a"], x["b"])[0, 1]), 2) if x.height > 2 else None   # noqa: E731
        sh = lambda v: round(float(v.mean() / v.std() * np.sqrt(len(v))), 1)   # noqa: E731
        out(f"\n######## 2. net per {'game' if unit == 'game_id' else 'day'}: {g.height} with any trade; correlation {c(g)} "
            f"(only where both traded: {c(both_active)}, n={both_active.height}); "
            f"t of the mean: A {sh(g['a'])}, B {sh(g['b'])}, A + B {sh(g['a'] + g['b'])}")
    # ---- 3. signals at every decision row
    sa = decision_signals("_h30-60-120-1800_tpf", "panel_v3f")
    sb = decision_signals("_h30-60-120-1800_mid", "panel_v4m")
    s = sa.join(sb.select("ticker", "t", pl.col("level").alias("level_b")), on=["ticker", "t"], how="inner")
    ref = tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t")
    s = (s.sort("t").join_asof(ref, on="t", by="ticker", strategy="backward").filter((pl.col("t") - pl.col("t_ref")) <= 60)
         .with_columns((pl.col("level") - pl.col("ref")).alias("sig_a"), (pl.col("level_b") - pl.col("ref")).alias("sig_b")))
    x, y = s["sig_a"].to_numpy(), s["sig_b"].to_numpy()
    out(f"\n######## 3. signals at {s.height:,} decision rows (both models, a print in the last 60 s)")
    out(f"correlation {np.corrcoef(x, y)[0, 1]:.2f}; rank correlation {pl.DataFrame({'a': x, 'b': y}).select(pl.corr('a', 'b', method='spearman')).item():.2f}; "
        f"same sign {np.mean(np.sign(x) == np.sign(y)):.2f}")
    sa_, sb_ = np.abs(x) >= 8, np.abs(y) >= cut_b
    out(f"A strong (>= 8c): {sa_.sum():,}; B strong (>= {cut_b:.1f}c): {sb_.sum():,}; both strong: {(sa_ & sb_).sum():,} "
        f"({(sa_ & sb_).sum() / max(1, sa_.sum()):.0%} of A's, {(sa_ & sb_).sum() / max(1, sb_.sum()):.0%} of B's); "
        f"same direction when both strong {np.mean(np.sign(x[sa_ & sb_]) == np.sign(y[sa_ & sb_])):.2f}")
    out(f"when A is strong, B's signal points the same way {np.mean(np.sign(y[sa_]) == np.sign(x[sa_])):.2f} (median |B| {np.median(np.abs(y[sa_])):.1f}c); "
        f"when B is strong, A agrees {np.mean(np.sign(x[sb_]) == np.sign(y[sb_])):.2f} (median |A| {np.median(np.abs(x[sb_])):.1f}c)")
    (OUT / "b9_signal_overlap.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
