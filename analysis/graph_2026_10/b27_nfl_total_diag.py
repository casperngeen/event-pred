"""Why NFL totals did worse on the holdout than on the test folds: forecasts and trades, side by side.
Diagnostic only: nothing here changes the frozen spec, and the holdout result stands as reported.

Forecasts (spread and total rows, average of 5 seeds, the move in the traded price over 30 / 60 / 120 s):
    R^2 against no change; slope of the realised move on the forecast (1 = right size, < 1 = too big);
    forecast and realised spread (sd) in cents; R^2 by time in the game (quarter of the game's steps), by
    the age of the contract's last print, and for a no-model rule: "the price moves to the middle of the
    bid and ask" (a share of the skill is stale prints catching up; journal 2026-10-06).
Trades (every 5 s, one at a time, fixed exit, frozen sizing): by |signal|, net and gross per contract,
how the exit ended, median width and signal.

    venv/bin/python -W ignore analysis/graph_2026_10/b27_nfl_total_diag.py test
    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b27_nfl_total_diag.py holdout
    -> out/b27_nfl_total_diag_<test|holdout>.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

MODE = sys.argv[1]
sys.argv = sys.argv[:1] + ["NFL"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b17_holdout_report import exits, load as load_holdout   # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402
from stg.models.ingame import Panel                      # noqa: E402
from stg.sports.loaders import derived_dir               # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

assert MODE in ("test", "holdout") and (MODE == "test") != sports_holdout_unsealed()
HZ = ["30s", "60s", "120s"]
KINDS = {1: "spread", 2: "total"}
D = derived_dir() / "ingame" / "NFL"


def forecasts() -> tuple[dict, Panel]:
    pn = Panel(D / "panel_v3")
    if MODE == "holdout":
        z = dict(np.load(D.parent / "results" / "NFL_fold99_gru_d10_h30-60-120-1800_tp_ens5_seed0.npz"))
    else:
        parts = []
        for k in range(1, 6):
            z0 = dict(np.load(f"results/ingame/NFL_fold{k}_gru_d10_h30-60-120-1800_tp_seed0.npz"))
            ze = np.load(f"results/ingame/ens5/NFL_fold{k}_gru_d10_h30-60-120-1800_tp_ens5_seed0.npz")
            assert np.array_equal(z0["step"], ze["step"])
            parts.append(z0 | {"delta": ze["delta"]})
        z = {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}
    return z, pn


def frame(z: dict, pn: Panel) -> pl.DataFrame:
    step = z["step"]
    first = np.r_[True, step[1:] != step[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
    node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
    feats = pn.manifest["node_market_features"]
    nx = lambda c: np.asarray(pn.nodes_x[node, feats.index(c)], dtype=np.float64)   # noqa: E731
    g = z["game"]
    a, b = pn.game_step_ptr[g], pn.game_step_ptr[g + 1]
    tgt = z["value_h"][:, :3] - z["value"][:, None]
    mid = np.where((nx("has_bid") > 0) & (nx("has_ask") > 0), 50 * (nx("bid") + nx("ask")) - z["value"], np.nan)
    return pl.DataFrame({"kind": z["kind"], "phase": np.minimum(4 * (step - a) / np.maximum(b - a, 1), 3.999).astype(int) + 1,
                         "has_print": nx("has_print") > 0, "log_print_age": nx("log_print_age"), "mid_gap": mid,
                         **{f"y_{h}": tgt[:, i] for i, h in enumerate(HZ)}, **{f"d_{h}": z["delta"][:, i] for i, h in enumerate(HZ)}})


def r2(x: pl.DataFrame, h: str, pred: str | None = None) -> float:
    y, d = x[f"y_{h}"].to_numpy(), x[pred or f"d_{h}"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(d)
    return round(100 * (1 - ((y[ok] - d[ok]) ** 2).sum() / (y[ok] ** 2).sum()), 2) if ok.sum() > 100 else float("nan")


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_column_data_types(True)
    z, pn = forecasts()
    f = frame(z, pn).filter(pl.col("kind").is_in(list(KINDS)))
    rows, by = [], []
    for kc, kn in KINDS.items():
        x = f.filter(pl.col("kind") == kc)
        y60, d60 = x["y_60s"].to_numpy(), x["d_60s"].to_numpy()
        ok = np.isfinite(y60)
        rows.append(dict(kind=kn, rows=x.height, **{f"R2_{h}": r2(x, h) for h in HZ},
                         slope_60s=round(float((y60[ok] * d60[ok]).sum() / (d60[ok] ** 2).sum()), 2),
                         forecast_sd_60s_c=round(float(d60[ok].std()), 2), move_sd_60s_c=round(float(y60[ok].std()), 2),
                         share_moved_60s_pct=round(100 * float((np.abs(y60[ok]) >= 1).mean()), 1),
                         no_model_mid_R2_60s=r2(x.with_columns(pl.col("mid_gap").alias("m")), "60s", "m"),
                         has_print_pct=round(100 * float(x["has_print"].mean()), 1)))
        for col, lab in (("phase", "quarter of the game"),):
            for v, xx in x.group_by(col):
                by.append(dict(kind=kn, split=lab, bin=str(v[0]), rows=xx.height, R2_60s=r2(xx, "60s")))
        x = x.with_columns(pl.when(~pl.col("has_print")).then(pl.lit("no print"))
                           .when(pl.col("log_print_age") < x.filter(pl.col("has_print"))["log_print_age"].median()).then(pl.lit("fresher half"))
                           .otherwise(pl.lit("older half")).alias("age"))
        for v, xx in x.group_by("age"):
            by.append(dict(kind=kn, split="last print", bin=v[0], rows=xx.height, R2_60s=r2(xx, "60s")))
    # trades
    if MODE == "holdout":
        t = load_holdout(D.parent / "results", "NFL", "_ens5")
    else:
        B.TAG = "_h30-60-120-1800_tp_ens5"
        t = B.load("NFL", "_dec5_one")
    tr = []
    for kn in KINDS.values():
        g = exits(t.filter(pl.col("kind") == kn))["fixed exit"]
        g = g.with_columns(pl.col("signal").abs().cut([8, 10, 15], labels=["5-8c", "8-10c", "10-15c", ">=15c"]).cast(pl.String).alias("sig"))
        for lab, x in [("all", g)] + sorted(g.group_by("sig"), key=lambda p: ["5-8c", "8-10c", "10-15c", ">=15c"].index(p[0][0])):
            lab = lab if isinstance(lab, str) else lab[0]
            c = x["size"].sum()
            tr.append(dict(kind=kn, signal=lab, trades=x.height, contracts=round(c, 1), gross_c=round((x["size"] * x["gross"]).sum() / c, 2),
                           net_c=round(x["pnl"].sum() / c, 2), settled_pct=round(100 * float((x["size"] * (x["how"] == "settled")).sum() / c), 1),
                           median_width=round(float(x["width"].median()), 1) if "width" in x.columns else None,
                           median_signal=round(float(x["signal"].abs().median()), 1)))
    lines = [__doc__.split("\n\n")[0], f"\n######## {MODE}", "-- forecasts", str(pl.DataFrame(rows)),
             "-- R^2 at 60 s by", str(pl.DataFrame(by).sort("kind", "split", "bin")),
             "-- trades, fixed exit", str(pl.DataFrame(tr))]
    (OUT / f"b27_nfl_total_diag_{MODE}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
