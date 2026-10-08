"""Why does deciding every 5 s help NBA / NFL but hurt MLB? Test folds only, the frozen model (average of 5
seeds), 120 s hold, frozen sizing, both decision rules (b19_decision_timing.py).

The MLB model's 'play' is a plate appearance (about one every 1-2 min), while the market moves on every pitch
(about one every 16 s). Between plate appearances the forecast barely updates, so its gap to the latest print
can be stale news rather than an edge. Per trade, at its decision time t:
    play_age_s       seconds since the last play the model knew (plays known after the 10 s feed delay)
    pitches_since    MLB: pitches thrown since that play (the information the model does not have)
    print_age_s      seconds since the contract's last print (b7_maker_liquidity age_s)
    markout_60s      side x (last print 60 s after the fill - entry), cents: did the price move for or against us
Net (fixed exit, hold to settlement) in cents per contract, size-weighted, by bins of each.

    venv/bin/python -W ignore analysis/graph_2026_10/b20_mlb_timing.py
    -> out/b20_mlb_timing.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b17_holdout_report import SIZE                      # noqa: E402
from b19_decision_timing import LEAGUES, RULES, SUFFIX, TAG, load          # noqa: E402
from b6_maker_exit import MAKER_RATE, sweep_sorted       # noqa: E402
from stg.models.ingame import Panel                      # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

PANELS = Path("data/sports/derived/ingame")
DELAY_S = 10.0
STRATS = [("MLB", "total"), ("MLB", "spread"), ("NBA", "spread"), ("NBA", "total"), ("NFL", "total"), ("NFL", "spread")]
RULE_SET = ["every 60 s", "every 5 s, one at a time"]


def play_age(f: pl.DataFrame, L: str) -> np.ndarray:
    """Seconds since the last play known at each trade's decision time (after the feed delay)."""
    d = PANELS / (L + ("_pitch" if "_pitch" in TAG else "")) / "panel_v3"     # the every-pitch model's plays are pitches
    pn = Panel(d)
    gidx = {g: i for i, g in enumerate(json.loads((d / "manifest.json").read_text())["games"])}
    out = np.full(f.height, np.nan)
    for (gid,), d in f.with_row_index("i").partition_by("game_id", as_dict=True).items():
        g = gidx.get(gid)
        if g is None:
            continue
        pt = pn.plays_t[pn.game_play_ptr[g]:pn.game_play_ptr[g + 1]]
        cut = d["t"].to_numpy() - DELAY_S
        k = np.searchsorted(pt, cut, side="right")
        out[d["i"].to_numpy()] = np.where(k > 0, cut - pt[np.maximum(k - 1, 0)], np.nan) + DELAY_S
    return out


def pitches_since(f: pl.DataFrame, age: np.ndarray) -> np.ndarray:
    """MLB: pitches thrown (pitch start) between the last known plate appearance and the decision time."""
    ev = (pl.read_parquet("data/sports/pbp/events_MLB.parquet", columns=["game_id", "t", "is_pitch"])
          .filter(pl.col("is_pitch"), pl.col("game_id").is_in(f["game_id"].unique().implode()))
          .select("game_id", (pl.col("t").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t")).sort("game_id", "t"))
    by = {g: d["t"].to_numpy() for (g,), d in ev.partition_by("game_id", as_dict=True).items()}
    out = np.full(f.height, np.nan)
    t, gids = f["t"].to_numpy(), f["game_id"].to_list()
    for i in range(f.height):
        p = by.get(gids[i])
        if p is not None and np.isfinite(age[i]):
            out[i] = np.searchsorted(p, t[i], side="right") - np.searchsorted(p, t[i] - age[i], side="right")
    return out


def markout(f: pl.DataFrame) -> np.ndarray:
    series = f.select(pl.col("ticker").str.extract(r"^([^-]+)")).to_series().unique().to_list()
    tr = sweep_sorted(scan_trades(series).filter(pl.col("ticker").is_in(f["ticker"].unique().implode()))
                      .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price",
                              "taker_side", "trade_id").collect())
    q = (f.with_row_index("i").select("i", "ticker", (pl.col("t_fill") + 60).alias("t")).sort("t", maintain_order=True)
         .join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("px60")).sort("t", maintain_order=True),
                    on="t", by="ticker", strategy="backward").sort("i"))
    return (f["side"] * (q["px60"] - f["entry"])).to_numpy()


def table(x: pl.DataFrame, col: str, edges: list[float], labels: list[str]) -> pl.DataFrame:
    return (x.with_columns(pl.col(col).cut(edges, labels=labels, left_closed=True).alias("bin"))
            .group_by("rule", "bin").agg(pl.len().alias("trades"), pl.col("size").sum().round(0).alias("contracts"),
                                         ((pl.col("size") * pl.col("net")).sum() / pl.col("size").sum()).round(2).alias("fixed_net_c"),
                                         ((pl.col("size") * pl.col("hold_net")).sum() / pl.col("size").sum()).round(2).alias("hold_net_c"),
                                         ((pl.col("size") * pl.col("markout_60s")).sum() / pl.col("size").sum()).round(2).alias("markout_60s_c"))
            .with_columns(pl.col("bin").cast(pl.String))
            .sort("rule", pl.col("bin").replace_strict({b: i for i, b in enumerate(labels)}, default=99)))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    q = (pl.col("entry") / 100).clip(0, 1)
    lines = [__doc__.split("\n\n")[0], ""]
    summary = []
    for L in [x for x in ("MLB", "NBA", "NFL") if x in LEAGUES]:
        logs = {r: load(L, RULES[r]) for r in RULE_SET}
        for _, kind in [s for s in STRATS if s[0] == L]:
            parts = []
            for r, f in logs.items():
                x = f.filter(pl.col("kind") == kind)
                x = x.with_columns(pl.Series("play_age_s", play_age(x, L)), pl.Series("markout_60s", markout(x)),
                                   (pl.col("side") * (100 * pl.col("y") - pl.col("entry")) - MAKER_RATE * q * (1 - q)).alias("hold_net"),
                                   pl.lit(r).alias("rule"))
                if L == "MLB":
                    x = x.with_columns(pl.Series("pitches_since", pitches_since(x, x["play_age_s"].to_numpy())))
                parts.append(x)
            x = pl.concat(parts, how="diagonal_relaxed").rename({"age_s": "print_age_s"})
            lines += [f"\n================ {L} {kind}",
                      "-- by seconds since the model's last known play",
                      str(table(x, "play_age_s", [10, 20, 40, 60, 120], ["<10", "10-20", "20-40", "40-60", "60-120", ">=120"]))]
            if L == "MLB":
                lines += ["-- by pitches since the model's last known play (information the model does not have)",
                          str(table(x, "pitches_since", [1, 2, 3, 5], ["0", "1", "2", "3-4", ">=5"]))]
            lines += ["-- by seconds since the contract's last print (age of the reference price)",
                      str(table(x, "print_age_s", [5, 15, 30, 60], ["<5", "5-15", "15-30", "30-60", ">=60"]))]
            for r in RULE_SET:
                y = x.filter(pl.col("rule") == r)
                w = y["size"]
                summary.append(dict(strategy=f"{L} {kind}", rule=r, trades=y.height,
                                    median_play_age_s=round(float(y["play_age_s"].median()), 1),
                                    median_print_age_s=round(float(y["print_age_s"].median()), 1),
                                    median_pitches_since=float(y["pitches_since"].median()) if "pitches_since" in y.columns else None,
                                    median_signal_c=round(float(y["signal"].abs().median()), 1),
                                    markout_60s_c=round(float((w * y["markout_60s"]).sum() / w.sum()), 2),
                                    fixed_net_c=round(float((w * y["net"]).sum() / w.sum()), 2),
                                    hold_net_c=round(float((w * y["hold_net"]).sum() / w.sum()), 2)))
    lines = lines[:2] + ["SUMMARY (sized means; markout = price move in our favour 60 s after the fill, before fees)",
                         str(pl.DataFrame(summary))] + lines[2:]
    (OUT / f"b20_mlb_timing{SUFFIX}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
