"""Candidate strategies x models x entry filters, with robustness checks. NBA, feed delay 10 s, test folds
3-5 (the folds with fresh-mid-target runs). Models: the current one (trade-price targets) and the
fresh-mid-target models (signal = predicted future mid - last print). Each model's cut-off gives the
current model's trade count at >= 8c for that candidate (all entries); the entry filters are then applied.

Entry filters (prints before the decision only):
    fresh mid      the last YES-taker and the last NO-taker prints both at most 30 s old (as the mid
                   models' training targets)
    width >= 9c    last YES-taker price - last NO-taker price, each within 120 s
Checks: by test fold; median trade; share stuck to settlement and net with stuck trades charged 3.5c;
net without the best 5% of games; fills needing a print 2c+ through; 2c order offset; maker fee x2;
half / double the trade count (a stricter / looser cut-off).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_candidates_by_model.py
    -> out/b9_candidates_by_model.txt, .parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_robust import with_net, summ, OUT     # noqa: E402
from stg.sports.loaders import scan_trades           # noqa: E402

MODELS = {"current": "_h30-60-120-1800_tpf", "mid Huber 5": "_h30-60-120-1800_mid",
          "mid ~squared": "_h30-60-120-1800_mid_hub1000", "mid ~sq, no result": "_h30-60-120-1800_mid_hub1000_ly0"}
STRATS = [("spread", "120s"), ("spread", "60s"), ("total", "120s"), ("total", "60s")]
FOLDS = [3, 4, 5]
FILTERS = {"all": pl.lit(True), "fresh mid": pl.col("older_s") <= 30, "stale mid": pl.col("older_s") > 30,
           "width >= 9c": pl.col("width") >= 9, "fresh & width >= 9c": (pl.col("older_s") <= 30) & (pl.col("width") >= 9)}


def entry_state(f: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    # prints with the same timestamp: ties broken by price so results are reproducible
    yes = tr.filter(pl.col("taker_side") == "yes").select("ticker", "t", pl.col("t").alias("t_yes"), pl.col("yes_price").alias("p_yes")).sort("t", "p_yes")
    no = tr.filter(pl.col("taker_side") == "no").select("ticker", "t", pl.col("t").alias("t_no"), pl.col("yes_price").alias("p_no")).sort("t", "p_no")
    f = f.sort("t").join_asof(yes, on="t", by="ticker", strategy="backward").join_asof(no, on="t", by="ticker", strategy="backward")
    return f.with_columns((pl.col("t") - pl.min_horizontal("t_yes", "t_no")).fill_null(1e9).alias("older_s"),
                          pl.when(((pl.col("t") - pl.col("t_yes")) <= 120) & ((pl.col("t") - pl.col("t_no")) <= 120))
                          .then(pl.col("p_yes") - pl.col("p_no")).alias("width"))


def checks(g: pl.DataFrame, g2c: pl.DataFrame, gk2: pl.DataFrame) -> dict:
    if g.height < 20:
        return dict(trades=g.height)
    s = summ(g, x=0)
    by = g.group_by("game_id").agg(pl.col("net").sum().alias("s")).sort("s", descending=True)
    top = by.head(max(1, by.height // 20))["game_id"]
    stuck = g["how"] == "settled"
    r = dict(trades=g.height, net=s["net"], t=s["t"], median=round(g["net"].median(), 2), stuck=round(stuck.mean(), 2),
             stuck_cost_3_5=round(g.with_columns(pl.when(stuck).then(-3.5).otherwise(pl.col("net")).alias("x"))["x"].mean(), 2),
             no_best_5pct=round(g.filter(~pl.col("game_id").is_in(top.implode()))["net"].mean(), 2),
             fill_2c=round(g2c["net"].mean(), 2) if g2c.height else None, offset_2c=round(gk2["net"].mean(), 2) if gk2.height else None,
             fee_x2=round(with_net(g.drop("gross", "net"), 2.0)["net"].mean(), 2))
    for k in FOLDS:
        x = g.filter(pl.col("fold") == k)
        r[f"fold{k}"] = round(x["net"].mean(), 1) if x.height >= 10 else None
    return r


def main() -> None:
    pl.Config.set_tbl_rows(1000)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_fmt_str_lengths(24)
    raw = {m: pl.read_parquet(OUT / f"b7_maker_trades_NBA_d10{t}.parquet").filter(pl.col("fold").is_in(FOLDS)) for m, t in MODELS.items()}
    tk = pl.concat([f.select("ticker") for f in raw.values()]).unique()["ticker"]
    tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
    raw = {m: entry_state(with_net(f), tr) for m, f in raw.items()}
    rows = []
    for kind, hold in STRATS:
        cand = {m: f.filter(pl.col("kind") == kind, pl.col("hold") == hold) for m, f in raw.items()}
        base = {m: f.filter(pl.col("k") == 1, pl.col("through") == 0) for m, f in cand.items()}
        n_ref = base["current"].filter(pl.col("signal").abs() >= 8).height
        for m in MODELS:
            mags = base[m]["signal"].abs().sort(descending=True)
            cuts = {"x1": float(mags[n_ref - 1]), "half": float(mags[n_ref // 2 - 1]), "double": float(mags[min(2 * n_ref, len(mags)) - 1])}
            for fname, cond in FILTERS.items():
                r = dict(candidate=f"{kind} {hold}", model=m, entry=fname, cut=round(cuts["x1"], 1))
                sel = lambda c, k=1, thr=0: cand[m].filter(pl.col("k") == k, pl.col("through") == thr, pl.col("signal").abs() >= c, cond)   # noqa: E731
                r |= checks(sel(cuts["x1"]), sel(cuts["x1"], 1, 1), sel(cuts["x1"], 2, 0))
                for lab in ("half", "double"):
                    g = sel(cuts[lab])
                    r[f"net_{lab}_n"] = round(g["net"].mean(), 2) if g.height >= 20 else None
                rows.append(r)
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "b9_candidates_by_model.parquet")
    lines = ["NBA, 10 s delay, folds 3-5. Net cents per contract. Each model's cut-off ('cut') gives the current model's "
             "trade count at >= 8c for the candidate; then the entry filter. stuck_cost_3_5: stuck trades charged 3.5c; "
             "no_best_5pct: without the best 5% of games; fill_2c: fills need a print 2c+ through; offset_2c: orders 2c from "
             "the last print; fee_x2: maker fee doubled; net_half_n / net_double_n: half / double the trades (stricter / looser cut-off)."]
    for kind, hold in STRATS:
        lines += [f"\n================ NBA {kind}, hold {hold}",
                  str(t.filter(pl.col("candidate") == f"{kind} {hold}").drop("candidate"))]
    (OUT / "b9_candidates_by_model.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
