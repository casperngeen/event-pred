"""MLB (every-pitch model): where the returns come from, by pitches thrown since the model's last known pitch, for
every 5 s one at a time, every 60 s one at a time (grid 0), and scaling in (12 grids x 1/12; b24_scale_in.py).

    pitches_since 0   no pitch thrown since the last pitch the model knows (after the 10 s feed delay): the model is
                      up to date
    pitches_since 1   a newer pitch has been thrown that the model does not know yet: the market may already be
                      reacting to it, so the gap between forecast and price can be the model being behind
Per bin: contracts, net (fixed exit / hold to settlement, cents per contract, size-weighted), the bin's share of
the rule's net dollars, and the 60 s markout (price move in our favour 60 s after the fill, before fees).
Same setup as b24 (120 s hold, offset 1c, strictly-through fills, frozen sizing, minimum signal 5c), test folds.

    venv/bin/python -W ignore analysis/graph_2026_10/b25_mlb_pitch_bins.py
    -> out/b25_mlb_pitch_bins.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
import b20_mlb_timing as T                                # noqa: E402
from b6_maker_exit import MAKER_RATE                      # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

TAG = "_h30-60-120-1800_tp_pitch_ens5"
OFFSETS = range(0, 60, 5)


def prepare(f: pl.DataFrame, kind: str) -> pl.DataFrame:
    q = (pl.col("entry") / 100).clip(0, 1)
    x = f.filter(pl.col("kind") == kind)
    x = x.with_columns(pl.Series("play_age_s", T.play_age(x, "MLB")), pl.Series("markout_60s", T.markout(x)),
                       (pl.col("side") * (100 * pl.col("y") - pl.col("entry")) - MAKER_RATE * q * (1 - q)).alias("hold_net"))
    return x.with_columns(pl.Series("pitches_since", T.pitches_since(x, x["play_age_s"].to_numpy())))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    B.TAG = T.TAG = TAG
    grids = lambda one: pl.concat([B.load("MLB", (f"_off{o}" if o else "") + one).with_columns(pl.col("size") / 12) for o in OFFSETS],
                                  how="diagonal_relaxed")
    logs = {"every 5 s, one at a time": B.load("MLB", "_dec5_one"), "every 60 s, one at a time (grid 0)": B.load("MLB", "_one"),
            "12 grids x 1/12, one per grid": grids("_one"), "12 grids x 1/12, stacking": grids("")}
    lines = [__doc__.split("\n\n")[0], ""]
    for kind in ("total", "spread"):
        rows = []
        for rule, f in logs.items():
            x = prepare(f, kind).with_columns(pl.col("pitches_since").fill_nan(None).cast(pl.Int64).clip(upper_bound=2).cast(pl.String)
                                              .replace({"2": ">=2"}).fill_null("unknown").alias("bin"))
            tot_fix, tot_hold = (x["size"] * x["net"]).sum(), (x["size"] * x["hold_net"]).sum()
            agg = (x.group_by("bin").agg(
                pl.col("size").sum().alias("c"), (pl.col("size") * pl.col("net")).sum().alias("pf"),
                (pl.col("size") * pl.col("hold_net")).sum().alias("ph"), (pl.col("size") * pl.col("markout_60s")).sum().alias("pm"),
                pl.col("play_age_s").median().alias("age"))
                .with_columns(pl.lit(rule).alias("rule"), (100 * pl.col("c") / pl.col("c").sum()).round(1).alias("contracts_pct"),
                              (pl.col("pf") / pl.col("c")).round(2).alias("fixed_net_c"), (100 * pl.col("pf") / tot_fix).round(0).alias("fixed_share_pct"),
                              (pl.col("ph") / pl.col("c")).round(2).alias("hold_net_c"), (100 * pl.col("ph") / tot_hold).round(0).alias("hold_share_pct"),
                              (pl.col("pm") / pl.col("c")).round(2).alias("markout_60s_c"), pl.col("age").round(1).alias("median_play_age_s"))
                .select("rule", "bin", "contracts_pct", "fixed_net_c", "fixed_share_pct", "hold_net_c", "hold_share_pct",
                        "markout_60s_c", "median_play_age_s").sort("bin"))
            rows.append(agg)
        lines += [f"\n================ MLB {kind}: by pitches since the model's last known pitch", str(pl.concat(rows))]
    (OUT / "b25_mlb_pitch_bins.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
