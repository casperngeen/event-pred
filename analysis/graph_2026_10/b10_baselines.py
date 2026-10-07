"""Two more no-model baselines through the same maker backtest as the model (b10_simple_rule.py: same
decision rows, fills, fixed exit after 120 s, fees, feed delay 10 s, all test folds).

1. Fade the last trade only when no play happened recently: as b10_simple_rule's rule (signal = trade-
   based mid - last print, i.e. half the width against the last taker), but only if no scoring play
   (a change in the score) became known in the last W seconds (W = 30 / 60 / 120; known = play time +
   10 s delay), or no play at all in the last 30 s.
2. The structural baseline (b1_baseline: normal curve for the final margin / total anchored on the winner
   price, from the plays known after the 10 s delay; the 'base' column of the model's predictions):
   raw     signal = baseline price - last print
   scaled  signal = value + beta x (baseline - value) - last print, beta per contract type fitted on the
           earlier test folds' rows (120 s change of the traded price on baseline - value), so fold 1 is
           left out; the model is also shown on folds 2-5.
Sizes: the model's frozen sizing on the baseline's signal, hard cut-offs, and width-only sizes for rule 1.

    venv/bin/python -W ignore analysis/graph_2026_10/b10_baselines.py
    -> out/b10_baselines.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b10_simple_rule as S                                # noqa: E402  (sets the backtest to 120 s, offset 1c, strictly through)
from b10_simple_rule import B, T, last_sides, summary, MODEL_SIZE, RULE_SIZES, CANDS, TAG   # noqa: E402
from b7_maker_liquidity import liquidity                   # noqa: E402
from b7_maker_robust import with_net, OUT                  # noqa: E402
from stg.models.ingame import KIND_NAMES                   # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades     # noqa: E402

DELAY = 10.0
H120 = 2   # index of the 120 s horizon in the runs' horizons (30s, 60s, 120s, 1800s)


def rows_with_base(pn, games: list[str]) -> pl.DataFrame:
    """b5_longshort.rows plus the structural baseline, the 120 s target and the game index."""
    prs = []
    for f in sorted(B.RES.glob(f"{B.LEAGUE}_fold*_gru_d{DELAY:g}{TAG}_seed0.npz")):
        assert json.loads(f.with_suffix(".json").read_text())["horizons"][H120] == "120s"
        prs.append(dict(np.load(f)))
    pr = {k: np.concatenate([p[k] for p in prs]) for k in prs[0]}
    step = pr["step"]
    first = np.r_[True, step[1:] != step[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
    node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
    keep = ((step - pn.game_step_ptr[pr["game"]]) % (B.DECIDE_S // B.STEP) == 0) & np.isfinite(pr["y"])
    i = np.flatnonzero(keep)
    return pl.DataFrame({
        "g": pr["game"][i].astype(np.int64), "game_id": np.asarray(games)[pr["game"][i]], "t": pn.steps_t[step[i]].astype(float),
        "kind": np.asarray(KIND_NAMES)[pr["kind"][i]], "side": pn.node_side[node[i]].astype(np.int64),
        "strike": pn.node_strike[node[i]].astype(float), "y": pr["y"][i].astype(float), "value": pr["value"][i].astype(float),
        "base": pr["base"][i].astype(float), "value_h": pr["value_h"][i, H120].astype(float),
        "level_120s": (pr["value"] + pr["delta"][:, H120])[i].astype(float)})


def quiet(r: pl.DataFrame, pn) -> pl.DataFrame:
    """Seconds since the last scoring play / any play became known (play time + delay), per decision row."""
    since_score, since_play = np.full(r.height, 1e9), np.full(r.height, 1e9)
    gi, tt = r["g"].to_numpy(), r["t"].to_numpy()
    for g in np.unique(gi):
        a, b = pn.game_play_ptr[g], pn.game_play_ptr[g + 1]
        pt, tot = pn.plays_t[a:b].astype(float), pn.plays_state[a:b, 2].astype(float)
        score_t = pt[np.r_[False, np.diff(tot) != 0]] if len(pt) else pt
        m = gi == g
        now = tt[m] - DELAY
        k = np.searchsorted(pt, now, side="right")
        since_play[m] = np.where(k > 0, now - pt[np.maximum(k - 1, 0)], 1e9) if len(pt) else 1e9
        ks = np.searchsorted(score_t, now, side="right")
        since_score[m] = np.where(ks > 0, now - score_t[np.maximum(ks - 1, 0)], 1e9) if len(score_t) else 1e9
    return r.with_columns(pl.Series("since_score", since_score), pl.Series("since_play", since_play))


def run(x: pl.DataFrame, tr: pl.DataFrame, g: pl.DataFrame) -> pl.DataFrame:
    x = x.with_columns(pl.col("sig").alias("sig_120s"))
    f = T.simulate(x, tr)
    return with_net(f.join(x.select("row", "kind", "game_id", "ticker", "t", "ref", "strike", "y", "width"), on="row").join(g, on="game_id", how="left"))


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_fmt_str_lengths(48)
    lines = []
    for L in sorted({c[0] for c in CANDS}):
        B.LEAGUE, B.DELAY, B.RUN_TAG = L, DELAY, TAG
        B.TAB = derived_dir() / "ingame" / L
        B.PANEL = B.TAB / "panel_v3f"
        pn = B.Panel(B.PANEL)
        gl = json.loads((B.PANEL / "manifest.json").read_text())["games"]
        kinds = [k for l_, k in CANDS if l_ == L]
        r = B.attach_tickers(rows_with_base(pn, gl)).filter(pl.col("kind").is_in(kinds)).with_row_index("row")
        tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
              .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side")
              .collect().sort("ticker", "t", "yes_price"))
        r = (r.sort("t").join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t", "ref"),
                                   on="t", by="ticker", strategy="backward").filter((pl.col("t") - pl.col("t_ref")) <= T.REF_MAX_S))
        g = pl.read_parquet(B.PANEL.parent / "games.parquet").select("game_id", "fold", "first_play")
        r = quiet(last_sides(r, tr), pn).join(g.select("game_id", "fold"), on="game_id", how="left")
        model = liquidity(with_net(pl.read_parquet(OUT / f"b7_maker_trades_{L}_d10{TAG}.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s"))).with_columns(MODEL_SIZE.alias("size"))
        for kind in kinds:
            rk = r.filter(pl.col("kind") == kind)
            games = rk["game_id"].unique()
            games25 = rk.filter(pl.col("fold") >= 2)["game_id"].unique()
            mk = model.filter(pl.col("kind") == kind)
            rows = [summary(mk, games, baseline="MODEL, frozen sizing")]
            # ---- 1. fade the last trade when no play happened recently
            fade = rk.filter(pl.col("width") > 0).with_columns((pl.col("fade_side") * pl.col("width") / 2).alias("sig"))
            fade_trades = run(fade, tr, g).join(fade.select("row", "since_score", "since_play"), on="row")
            for cname, cond in (("no score change for 30 s", pl.col("since_score") >= 30), ("no score change for 60 s", pl.col("since_score") >= 60),
                                ("no score change for 120 s", pl.col("since_score") >= 120), ("no play for 30 s", pl.col("since_play") >= 30)):
                for sname in ("width >= 9", "width >= 15", "size = width / 10"):
                    rows.append(summary(fade_trades.filter(cond).with_columns(RULE_SIZES[sname].alias("size")), games, baseline=f"FADE, {cname}, {sname}"))
            # ---- 2. structural baseline: raw, and scaled walk-forward
            st = rk.filter(pl.col("base").is_finite()).with_columns((pl.col("base") - pl.col("ref")).alias("sig"))
            st_tr = run(st.filter(pl.col("sig").abs() >= 5), tr, g)
            rows.append(summary(st_tr.with_columns(MODEL_SIZE.alias("size")), games, baseline="STRUCTURAL raw, frozen sizing"))
            rows.append(summary(st_tr.with_columns(((pl.col("signal").abs() >= 8) & (pl.col("width") >= 9)).cast(pl.Float64).alias("size")),
                                games, baseline="STRUCTURAL raw, signal >= 8, width >= 9"))
            parts, betas = [], {}
            ok = pl.col("base").is_finite() & pl.col("value_h").is_finite()
            for k in sorted(rk["fold"].unique().to_list()):
                if k < 2:
                    continue
                past = rk.filter(pl.col("fold") < k, ok).select((pl.col("base") - pl.col("value")).alias("gap"), (pl.col("value_h") - pl.col("value")).alias("dv"))
                b = float((past["gap"] * past["dv"]).sum() / (past["gap"] ** 2).sum())
                betas[k] = round(b, 2)
                parts.append(rk.filter(pl.col("fold") == k, pl.col("base").is_finite())
                             .with_columns((pl.col("value") + b * (pl.col("base") - pl.col("value")) - pl.col("ref")).alias("sig")))
            sc = pl.concat(parts)
            sc_tr = run(sc.filter(pl.col("sig").abs() >= 5), tr, g)
            rows.append(summary(mk.filter(pl.col("fold") >= 2), games25, baseline="MODEL, frozen sizing (folds 2-5)"))
            rows.append(summary(sc_tr.with_columns(MODEL_SIZE.alias("size")), games25, baseline=f"STRUCTURAL scaled (folds 2-5), frozen sizing"))
            rows.append(summary(sc_tr.with_columns(((pl.col("signal").abs() >= 8) & (pl.col("width") >= 9)).cast(pl.Float64).alias("size")),
                                games25, baseline="STRUCTURAL scaled (folds 2-5), signal >= 8, width >= 9"))
            lines += [f"\n================ {L} {kind}, 120 s, 10 s delay, all test folds (structural scale factor by fold: {betas})",
                      str(pl.DataFrame(rows, infer_schema_length=None))]
            print(lines[-2], flush=True)
    text = ("No-model baselines vs the graph model (net cents per contract; t = mean P&L per game / standard error over all games with a "
            "decision; folds 2-5 rows: games of folds 2-5 only)\n" + "\n".join(lines))
    (OUT / "b10_baselines.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
