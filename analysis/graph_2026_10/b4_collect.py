"""Collect in-game model results (cluster/train_ingame.py JSONs) across folds and variants.

Per league, variant (memory gru / none), contract kind and horizon: R^2 of the value change (model and
baseline reversion, against no change) and interval loss (model, no change, baseline), averaged over
test folds weighted by rows, with the per-fold spread; drift R^2; final-result log loss vs the market.

    venv/bin/python analysis/graph_2026_10/b4_collect.py results/ingame
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ingame")
    rows = []
    for f in sorted(d.glob("*_d*_seed*.json")):
        r = json.loads(f.read_text())
        for kind, m in r["metrics"].items():
            for h, v in m.items():
                rows.append(dict(league=r["league"], memory=r["memory"], delay=r.get("delay_s", 0.0), seed=r["seed"],
                                 horizons="-".join(r.get("horizons", [])),
                                 fold=r["test_fold"], kind=kind,
                                 target=h, epochs=r["epochs_run"], **v))
    t = pl.DataFrame(rows, infer_schema_length=None)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    w = lambda c: ((pl.col(c) * pl.col("rows")).sum() / pl.col("rows").sum()).round(4).alias(c)   # noqa: E731
    h = t.filter(pl.col("target").str.ends_with("s") & ~pl.col("target").str.starts_with("drift"))
    print("######## value change: R^2 against no change, and interval loss (row-weighted mean over folds)")
    print(h.group_by("league", "horizons", "memory", "delay", "kind", "target").agg(
        pl.col("fold").n_unique().alias("folds"), pl.col("rows").sum(), w("r2_model"), w("r2_baseline"),
        pl.col("r2_model").min().round(4).alias("r2_model_worst_fold"), w("interval_model"), w("interval_no_change"),
        w("interval_baseline")).sort("league", "kind", "target", "memory", "delay"))
    dr = t.filter(pl.col("target") == "drift_60_600")
    if dr.height:
        print("\n######## drift (+60 s -> +600 s)")
        print(dr.group_by("league", "memory", "delay", "kind").agg(pl.col("fold").n_unique().alias("folds"), w("r2_model"), w("corr"))
              .sort("league", "kind", "memory", "delay"))
    oc = t.filter(pl.col("target") == "outcome")
    if oc.height:
        print("\n######## final result: log loss (lower is better), row-weighted over folds")
        print(oc.group_by("league", "memory", "delay", "kind").agg(pl.col("fold").n_unique().alias("folds"), pl.col("games").sum(),
                                                          w("logloss_model"), w("logloss_market"))
              .sort("league", "kind", "memory", "delay"))


if __name__ == "__main__":
    main()
