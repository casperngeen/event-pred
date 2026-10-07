"""Quick check, before building a market-side memory into the in-game graph model: does a contract's
recent market history explain the current model's errors?

Rows: the GRU model's test predictions (cluster/train_ingame.py --save-preds, all folds) for spread
and total contracts, every DECIDE_S seconds of game time, where a print exists at the horizon.
Target: the change in value at the horizon (value_h - value); the model's error = target - delta.

Feature sets (all as of the step: prints count from known_at; market data are not delayed):
    snapshot   what the model already sees at the step, plus its own predictions: value, bid, ask,
               gap, quote age, sides present, kind, the predicted changes at every horizon
    quotes     (option B) the contract's rebuilt value / bid / ask 15, 30 and 60 s earlier (as changes)
    prints     (option A) the contract's last N_PRINTS prints (price - value, taker side, log size,
               age), the last YES-taker and NO-taker prices - value, and over the last 60 s the print
               count, signed contracts and volume-weighted price - value

"predictions only" (the model's own predicted changes) checks whether a simple rescaling of its
output already explains the gain from the snapshot.

A quick model (LightGBM, and ridge regression as a simple check) predicts the error from each set,
walk-forward: for test fold k (3..5) it is fitted on folds < k. Score: R^2 of the change with the
model alone (delta) vs delta + predicted error. History helps only if snapshot + history beats
snapshot alone (the control for the different model type).

With B8_SETS=relative it instead compares the model's predictions plus price differences (mid - value,
value - bid, ask - value, ask - bid, last print - value; none is a network input) against the raw levels.

    [B8_SETS=relative] RUN_TAG=_h30-60-120-1800_tpf venv/bin/python -W ignore analysis/graph_2026_10/b8_memory_check.py \
        results/ingame NBA 10 data/sports/derived/ingame/NBA/panel_v3f
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_longshort import LEAGUE, DELAY, PANEL, RES, RUN_TAG, KIND_NAMES, Panel, attach_tickers   # noqa: E402
from stg.sports.loaders import scan_trades                                                   # noqa: E402

HORIZONS = ["60s", "120s"]
DECIDE_S, STEP = 30, 5
LAGS_S = [15, 30, 60]
N_PRINTS = 5
TEST_FOLDS = [3, 4, 5]
SETS = os.environ.get("B8_SETS", "memory")     # "memory" (history sets) or "relative" (price differences)


def load_rows(pn: Panel, games: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """All spread / total prediction rows (every step, for the quote lags), with decision flags."""
    parts, hz = [], None
    for f in sorted(RES.glob(f"{LEAGUE}_fold*_gru_d{DELAY:g}{RUN_TAG}_seed0.npz")):
        hz = json.loads(f.with_suffix(".json").read_text())["horizons"]
        z = dict(np.load(f))
        step = z["step"]
        first = np.r_[True, step[1:] != step[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
        node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
        keep = np.isin(z["kind"], [KIND_NAMES.index("spread"), KIND_NAMES.index("total")]) & z["live"]
        i = np.flatnonzero(keep)
        d = {"game_id": np.asarray(games)[z["game"][i]], "game": z["game"][i], "t": pn.steps_t[step[i]].astype(float),
             "decide": ((step[i] - pn.game_step_ptr[z["game"][i]]) % (DECIDE_S // STEP) == 0),
             "kind": np.asarray(KIND_NAMES)[z["kind"][i]], "side": pn.node_side[node[i]].astype(np.int64),
             "strike": pn.node_strike[node[i]].astype(float), "value": z["value"][i].astype(np.float32),
             "bid": np.where(z["has_bid"][i], z["entry_bid"][i], np.nan).astype(np.float32),
             "ask": np.where(z["has_ask"][i], z["entry_ask"][i], np.nan).astype(np.float32),
             "quote_age": z["quote_age_s"][i].astype(np.float32)}
        for hi, hn in enumerate(hz):
            d[f"delta_{hn}"] = z["delta"][i, hi].astype(np.float32)
            d[f"vh_{hn}"] = z["value_h"][i, hi].astype(np.float32)
        parts.append(pl.DataFrame(d))
    return pl.concat(parts), hz


def add_quote_lags(r: pl.DataFrame) -> pl.DataFrame:
    base = r.select("ticker", "t", "value", "bid", "ask").sort("t")
    out = r.sort("t")
    for L in LAGS_S:
        lag = base.with_columns((pl.col("t") + L).alias("t")).rename({"value": "v_l", "bid": "b_l", "ask": "a_l"})
        out = (out.join_asof(lag, on="t", by="ticker", strategy="backward", tolerance=STEP)
               .with_columns((pl.col("value") - pl.col("v_l")).alias(f"dv_{L}"), (pl.col("bid") - pl.col("b_l")).alias(f"db_{L}"),
                             (pl.col("ask") - pl.col("a_l")).alias(f"da_{L}"))
               .drop("v_l", "b_l", "a_l"))
    return out


def add_print_features(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    s = pl.when(pl.col("taker_side") == "yes").then(1).otherwise(-1)
    tr = tr.with_columns(s.alias("sgn"), pl.col("count").log1p().alias("lsz")).with_columns(
        pl.col("count").cum_sum().over("ticker").alias("cv"), (pl.col("count") * pl.col("sgn")).cum_sum().over("ticker").alias("cs"),
        (pl.col("count") * pl.col("yes_price")).cum_sum().over("ticker").alias("cn"), pl.int_range(pl.len()).over("ticker").alias("ci"))
    lagc = []
    for k in range(N_PRINTS):
        lagc += [pl.col("yes_price").shift(k).over("ticker").alias(f"p{k}"), pl.col("sgn").shift(k).over("ticker").alias(f"s{k}"),
                 pl.col("lsz").shift(k).over("ticker").alias(f"z{k}"), pl.col("t").shift(k).over("ticker").alias(f"tp{k}")]
    last = tr.select("ticker", "t", "cv", "cs", "cn", "ci", *lagc)
    yes = tr.filter(pl.col("sgn") == 1).select("ticker", "t", pl.col("yes_price").alias("p_yes"))
    no = tr.filter(pl.col("sgn") == -1).select("ticker", "t", pl.col("yes_price").alias("p_no"))
    r = r.sort("t").join_asof(last.sort("t"), on="t", by="ticker", strategy="backward")
    r = r.join_asof(yes.sort("t"), on="t", by="ticker", strategy="backward", tolerance=60.0)
    r = r.join_asof(no.sort("t"), on="t", by="ticker", strategy="backward", tolerance=60.0)
    ago = last.select("ticker", (pl.col("t") + 60).alias("t"), pl.col("cv").alias("cv60"), pl.col("cs").alias("cs60"),
                      pl.col("cn").alias("cn60"), pl.col("ci").alias("ci60"))
    r = r.join_asof(ago.sort("t"), on="t", by="ticker", strategy="backward")
    fill0 = lambda c: pl.col(c).fill_null(0)   # noqa: E731
    r = r.with_columns((pl.col("ci") - fill0("ci60")).alias("n60"), (pl.col("cs") - fill0("cs60")).alias("signed60"),
                       ((pl.col("cn") - fill0("cn60")) / (pl.col("cv") - fill0("cv60"))).alias("vwap60"))
    r = r.with_columns(pl.when(pl.col("n60") > 0).then(pl.col("vwap60") - pl.col("value")).alias("vwap60_rel"),
                       (pl.col("p_yes") - pl.col("value")).alias("yes_rel"), (pl.col("p_no") - pl.col("value")).alias("no_rel"),
                       *[(pl.col(f"p{k}") - pl.col("value")).alias(f"p{k}_rel") for k in range(N_PRINTS)],
                       *[(pl.col("t") - pl.col(f"tp{k}")).clip(0, 3600).log1p().alias(f"age{k}") for k in range(N_PRINTS)])
    return r


def main() -> None:
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    r, hz = load_rows(pn, games)
    fold = pl.read_parquet(PANEL.parent / "games.parquet").select("game_id", "fold")
    r = attach_tickers(r).join(fold, on="game_id")
    r = add_quote_lags(r).filter(pl.col("decide"))
    tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "count", "taker_side")
          .collect().sort("ticker", "t"))
    r = add_print_features(r, tr)
    r = r.with_columns((pl.col("ask") - pl.col("bid")).alias("gap"), pl.col("quote_age").log1p().alias("lage"),
                       pl.col("bid").is_not_null().cast(pl.Int8).alias("has_bid"), pl.col("ask").is_not_null().cast(pl.Int8).alias("has_ask"),
                       (pl.col("kind") == "total").cast(pl.Int8).alias("is_total"),
                       ((pl.col("bid") + pl.col("ask")) / 2 - pl.col("value")).alias("mid_rel"))
    snap = ["value", "bid", "ask", "gap", "mid_rel", "lage", "has_bid", "has_ask", "is_total"] + [f"delta_{h}" for h in hz]
    quotes = [f"{c}_{L}" for L in LAGS_S for c in ("dv", "db", "da")]
    prints = ([f"p{k}_rel" for k in range(N_PRINTS)] + [f"s{k}" for k in range(N_PRINTS)] + [f"z{k}" for k in range(N_PRINTS)]
              + [f"age{k}" for k in range(N_PRINTS)] + ["yes_rel", "no_rel", "n60", "signed60", "vwap60_rel"])
    pred = [f"delta_{h}" for h in hz]
    r = r.with_columns((pl.col("value") - pl.col("bid")).alias("v_minus_bid"), (pl.col("ask") - pl.col("value")).alias("ask_minus_v"))
    if SETS == "relative":     # are the price differences (not given to the network) what the correction uses?
        sets = {"predictions only": pred, "predictions + mid - value": pred + ["mid_rel"],
                "predictions + all differences": pred + ["mid_rel", "v_minus_bid", "ask_minus_v", "gap", "p0_rel"],
                "predictions + raw levels (no differences)": pred + ["value", "bid", "ask", "has_bid", "has_ask"],
                "snapshot": snap}
    else:
        sets = {"predictions only": pred, "snapshot": snap, "snapshot + quotes (B)": snap + quotes, "snapshot + prints (A)": snap + prints,
                "snapshot + both": snap + quotes + prints}
    print(f"{LEAGUE}, feed delay {DELAY:g} s: {r.height:,} decision rows (spread / total, every {DECIDE_S} s)")
    res = []
    for h in HORIZONS:
        d = r.filter(pl.col(f"vh_{h}").is_finite()).with_columns((pl.col(f"vh_{h}") - pl.col("value")).alias("y"))
        d = d.with_columns((pl.col("y") - pl.col(f"delta_{h}")).alias("err"))
        for k in TEST_FOLDS:
            trn, tst = d.filter(pl.col("fold") < k), d.filter(pl.col("fold") == k)
            y, base = tst["y"].to_numpy(), tst[f"delta_{h}"].to_numpy()
            sst = (y ** 2).sum()                                                       # R^2 against "no change", as the trainer reports
            r2 = lambda p: 1 - ((y - p) ** 2).sum() / sst                              # noqa: E731
            row = dict(horizon=h, test_fold=k, train_rows=trn.height, test_rows=tst.height, model_alone=round(r2(base), 4))
            for name, cols in sets.items():
                Xtr = trn.select(cols).to_numpy().astype(np.float32)
                Xte = tst.select(cols).to_numpy().astype(np.float32)
                g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
                g.fit(Xtr, trn["err"].to_numpy())
                row[f"lgbm: {name}"] = round(r2(base + g.predict(Xte)), 4)
                lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                lin.fit(np.nan_to_num(Xtr), trn["err"].to_numpy())
                row[f"ridge: {name}"] = round(r2(base + lin.predict(np.nan_to_num(Xte))), 4)
            res.append(row)
            print(row, flush=True)
    out = pl.DataFrame(res)
    out.write_parquet(Path(__file__).resolve().parent / "out" / f"b8_memory_check_{LEAGUE}_d{DELAY:g}{RUN_TAG}{'_relative' if SETS == 'relative' else ''}.parquet")
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(260)
    print(out)


if __name__ == "__main__":
    main()
