"""MLB holdout: postseason against regular season, for the two MLB candidates (every-pitch model, every 5 s,
one at a time, skip 15-60 s after the last known pitch, fixed exit, frozen sizing). Diagnostic only.

Trades: games, games traded, trades and contracts per game, median |signal| / width / contracts printed in the
last 5 min, gross and net per contract, win rate, how the exit ended, per-game P&L spread.
Forecasts: R^2 against no change at 30 / 60 / 120 s, slope of the realised move on the forecast, forecast and
realised sd, and the no-model "price moves to the middle of the bid and ask" R^2.

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b28_mlb_postseason.py
    -> out/b28_mlb_postseason.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.argv = sys.argv[:1] + ["MLB"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_stability as S                                 # noqa: E402
from b17_holdout_report import exits, load               # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402
from stg.models.ingame import Panel                      # noqa: E402
from stg.sports.loaders import derived_dir               # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

HZ = ["30s", "60s", "120s"]
KINDS = {1: "spread", 2: "total"}


def main() -> None:
    assert sports_holdout_unsealed(), "reads the holdout: set STG_SPORTS_UNSEAL=final-test"
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_hide_column_data_types(True)
    res = derived_dir() / "ingame" / "results"
    S.LEAGUE = "MLB"
    typ = dict(S.game_meta().select("game_id", "type").iter_rows())
    D = derived_dir() / "ingame" / "MLB_pitch"
    all_games = pl.read_parquet(D / "games.parquet").select("game_id").with_columns(
        pl.col("game_id").replace_strict(typ, default=None).alias("type"))
    t = load(res, "MLB", "_ens5")
    rows = []
    for kind in KINDS.values():
        g = exits(t.filter(pl.col("kind") == kind))["fixed exit"]
        for ty in ("regular", "playoffs"):
            x = g.filter(pl.col("type") == ty)
            n_games = all_games.filter(pl.col("type") == ty).height
            c = x["size"].sum()
            pg = x.group_by("game_id").agg(pl.col("pnl").sum())["pnl"].to_numpy() / 100
            rows.append(dict(kind=kind, type=ty, games=n_games, games_traded=x["game_id"].n_unique(),
                             trades_per_game=round(x.height / n_games, 1), contracts_per_game=round(c / n_games, 2),
                             median_signal=round(float(x["signal"].abs().median()), 1), median_width=round(float(x["width"].median()), 1),
                             median_vol_5m=round(float(x["vol_5m"].median())), gross_c=round((x["size"] * x["gross"]).sum() / c, 2),
                             net_c=round(x["pnl"].sum() / c, 2), win_pct_sized=round(100 * float((x["size"] * (x["pnl"] > 0)).sum() / c), 1),
                             settled_pct=round(100 * float((x["size"] * (x["how"] == "settled")).sum() / c), 1),
                             limit_filled_pct=round(100 * float((x["size"] * (x["how"] == "limit filled")).sum() / c), 1),
                             games_up_pct=round(100 * float((pg > 0).mean()), 1), best_game_usd=round(float(pg.max()), 2),
                             worst_game_usd=round(float(pg.min()), 2)))
    # forecasts
    z = np.load(res / "MLB_fold99_gru_d10_h30-60-120-1800_tp_pitch_ens5_seed0.npz")
    pn = Panel(D / "panel_v3")
    games = json.loads((D / "panel_v3" / "manifest.json").read_text())["games"]
    gtype = np.array([typ.get(g) for g in games])[z["game"]]
    step = z["step"]
    first = np.r_[True, step[1:] != step[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
    node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
    feats = pn.manifest["node_market_features"]
    nx = lambda c: np.asarray(pn.nodes_x[node, feats.index(c)], dtype=np.float64)   # noqa: E731
    mid = np.where((nx("has_bid") > 0) & (nx("has_ask") > 0), 50 * (nx("bid") + nx("ask")) - z["value"], np.nan)
    tgt = z["value_h"][:, :3] - z["value"][:, None]
    fr = []
    for kc, kn in KINDS.items():
        for ty in ("regular", "playoffs"):
            m = (z["kind"] == kc) & (gtype == ty)
            r = dict(kind=kn, type=ty, rows=int(m.sum()))
            for i, h in enumerate(HZ):
                ok = m & np.isfinite(tgt[:, i])
                y, d = tgt[ok, i], z["delta"][ok, i]
                r[f"R2_{h}"] = round(100 * (1 - ((y - d) ** 2).sum() / (y ** 2).sum()), 2)
                if h == "60s":
                    r |= dict(slope_60s=round(float((y * d).sum() / (d ** 2).sum()), 2), forecast_sd_60s_c=round(float(d.std()), 2),
                              move_sd_60s_c=round(float(y.std()), 2))
                    ok2 = ok & np.isfinite(mid)
                    y2 = tgt[ok2, i]
                    r["no_model_mid_R2_60s"] = round(100 * (1 - ((y2 - mid[ok2]) ** 2).sum() / (y2 ** 2).sum()), 2)
            fr.append(r)
    lines = [__doc__.split("\n\n")[0], "", "-- trades (fixed exit; per game = over all games of that type in the holdout)",
             str(pl.DataFrame(rows)), "", "-- forecasts (average of 5 seeds)", str(pl.DataFrame(fr))]
    (OUT / "b28_mlb_postseason.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
