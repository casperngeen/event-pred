"""Final test, step 4: the holdout report, exactly as fixed in reports/holdout_spec.md section 5.

Reads the holdout trade logs (b7_maker_trades.py, run with the holdout unsealed, inside the holdout
folder) and forecasts (b15_holdout_predict.py). For each candidate and each exit (fixed exit, hold to
settlement), on the average of the five seeds' forecasts:
    1. before fees, after fees, after twice the fees (cents per contract)
    2. contracts, net $ (1-contract maximum per trade), game-level t, share of game days up, largest
       drawdown, Sharpe from daily P&L (unlimited capacity: relative only), months with a profit
    3. win rates (all trades / by size), average winning and losing trade, and for the fixed exit how it
       ended (limit filled / taken / settled)
    4. net per contract seed by seed
    5. by game type (MLB regular season vs postseason; NBA regular season vs playoffs)
Descriptive: forecast R^2 against "no change" per contract type and horizon. Next to each candidate,
its test-fold result (spec section 4) for reference. Leagues without holdout games are skipped.

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b17_holdout_report.py
    -> out/b17_holdout_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.argv = sys.argv[:1] + ["MLB"]       # b1_response (imported via b1_stability) reads a league from argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_stability as S                                 # noqa: E402
from b6_maker_exit import MAKER_RATE                     # noqa: E402
from b7_maker_liquidity import liquidity                # noqa: E402
from b7_maker_robust import with_net                      # noqa: E402
from b9_sizing import RULES                              # noqa: E402
from stg.sports.loaders import derived_dir               # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

NAME = "gru_d10_h30-60-120-1800_tp"
SEEDS = [0, 1, 2, 3, 4]
STRATS = [("NBA", "spread"), ("MLB", "total"), ("NBA", "total"), ("NFL", "total")]
SIZE = RULES["quadratic signal x linear width (missing 1)"]
TEST_FOLDS = {("NBA", "spread"): (5.67, 8.36), ("MLB", "total"): (7.21, 15.67), ("NBA", "total"): (2.54, 4.73),
              ("NFL", "total"): (7.44, 12.79)}          # spec section 4: net c/contract, fixed exit / hold
HZ = ["30s", "60s", "120s", "1800s"]
KINDS = {0: "winner", 1: "spread", 2: "total"}
OUT = Path(__file__).resolve().parent / "out"


def per_game(g: pl.DataFrame, games: pl.Series) -> np.ndarray:
    return (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
            .fill_null(0).sort("game_id")["pnl"].to_numpy())


def summary(g: pl.DataFrame, games: pl.Series, days: pl.DataFrame) -> dict:
    """g: trades with size, gross, fee (cents per contract) and pnl (size x net)."""
    c = g["size"].sum()
    if c == 0:
        return dict(contracts=0)
    pg = per_game(g, games)
    d = days.join(g.group_by("day").agg(pl.col("pnl").sum()), on="day", how="left").fill_null(0).sort("day")["pnl"].to_numpy() / 100
    span_y = max((days["day"].max() - days["day"].min()).days, 1) / 365.25
    cum = np.r_[0, np.cumsum(d)]
    months = g.group_by(pl.col("day").dt.strftime("%Y-%m")).agg(pl.col("pnl").sum())
    return dict(contracts=round(c), gross_c=round((g["size"] * g["gross"]).sum() / c, 2), NET_c=round(g["pnl"].sum() / c, 2),
                net_2x_fees_c=round((g["pnl"] - g["size"] * g["fee"]).sum() / c, 2), net_usd=round(g["pnl"].sum() / 100, 1),
                t_game=round(pg.mean() / pg.std(ddof=1) * np.sqrt(len(pg)), 1) if len(pg) > 1 else None,
                sharpe_ann=round(d.mean() / d.std(ddof=1) * np.sqrt(len(d) / span_y), 1) if len(d) > 1 and d.std() > 0 else None,
                days_up_pct=round(100 * (d > 0).mean()), max_dd_usd=round(float((np.maximum.accumulate(cum) - cum).max()), 1),
                months_up=f"{(months['pnl'] > 0).sum()}/{months.height}")


def wins(g: pl.DataFrame) -> dict:
    n, w = g["pnl"] / g["size"], g["size"]
    return dict(trades=g.height, win_pct=round(100 * float((n > 0).mean()), 1), win_pct_sized=round(100 * float(w.filter(n > 0).sum() / w.sum()), 1),
                avg_win_c=round(float(n.filter(n > 0).mean()), 1), avg_loss_c=round(float(n.filter(n < 0).mean()), 1))


def exits(x: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The two exits on the same entries: gross, fee and pnl columns per trade."""
    q = (pl.col("entry") / 100).clip(0, 1)
    fixed = x.with_columns((pl.col("gross") - pl.col("net")).alias("fee"), (pl.col("size") * pl.col("net")).alias("pnl"))
    hold = x.with_columns((pl.col("side") * (100 * pl.col("y") - pl.col("entry"))).alias("gross"),
                          (MAKER_RATE * q * (1 - q)).alias("fee")).with_columns(
        (pl.col("size") * (pl.col("gross") - pl.col("fee"))).alias("pnl"))
    return {"fixed exit": fixed, "hold to settlement": hold}


def load(res: Path, L: str, suffix: str) -> pl.DataFrame | None:
    f = res / f"b7_maker_trades_{L}_d10_h30-60-120-1800_tp{suffix}.parquet"
    if not f.exists():
        return None
    S.LEAGUE = L
    typ = S.game_meta().select("game_id", "type")
    return (liquidity(with_net(pl.read_parquet(f).filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")))
            .with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
            .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day"))
            .join(typ, on="game_id", how="left"))


def forecast_r2(res: Path, L: str) -> list[dict]:
    f = res / f"{L}_fold99_{NAME}_ens5_seed0.npz"
    if not f.exists():
        return []
    z = np.load(f)
    tgt = z["value_h"] - z["value"][:, None]
    out = []
    for kind, kn in KINDS.items():
        for hi, h in enumerate(HZ):
            ok = (z["kind"] == kind) & np.isfinite(tgt[:, hi])
            if ok.sum() < 100:
                continue
            y, d = tgt[ok, hi], z["delta"][ok, hi]
            out.append(dict(league=L, kind=kn, horizon=h, rows=int(ok.sum()), R2_pct=round(100 * (1 - ((y - d) ** 2).sum() / (y ** 2).sum()), 2)))
    return out


def main() -> None:
    assert sports_holdout_unsealed(), "the holdout report reads the holdout: set STG_SPORTS_UNSEAL=final-test"
    res = derived_dir() / "ingame" / "results"
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    main_rows, win_rows, how_rows, seed_rows, type_rows, r2 = [], [], [], [], [], []
    for L in ("NBA", "MLB", "NFL"):
        ens = load(res, L, "_ens5")
        if ens is None:
            print(f"{L}: no holdout trade log yet, skipped")
            continue
        r2 += forecast_r2(res, L)
        seeds = {s: load(res, L, f"_seed{s}" if s else "") for s in SEEDS}
        for _, kind in [s for s in STRATS if s[0] == L]:
            x = ens.filter(pl.col("kind") == kind)
            if x.height == 0:
                continue
            games, days = x["game_id"].unique().sort(), x.select("day").unique()
            cv = TEST_FOLDS[(L, kind)]
            for i, (ex, g) in enumerate(exits(x).items()):
                main_rows.append(dict(strategy=f"{L} {kind}", exit=ex, games=len(games), test_folds_NET_c=cv[i]) | summary(g, games, days))
                win_rows.append(dict(strategy=f"{L} {kind}", exit=ex) | wins(g))
                for typ, gt in g.group_by("type"):
                    gg = gt["game_id"].unique().sort()
                    type_rows.append(dict(strategy=f"{L} {kind}", exit=ex, type=typ[0], games=len(gg))
                                     | {k: v for k, v in summary(gt, gg, gt.select("day").unique()).items()
                                        if k in ("contracts", "gross_c", "NET_c", "t_game")})
                for s, xs in seeds.items():
                    if xs is not None:
                        gs = exits(xs.filter(pl.col("kind") == kind))[ex]
                        seed_rows.append(dict(strategy=f"{L} {kind}", exit=ex, seed=s,
                                              NET_c=round(gs["pnl"].sum() / gs["size"].sum(), 2) if gs["size"].sum() else None))
            how_rows.append(x.group_by("how").agg(pl.col("size").sum()).with_columns(
                (100 * pl.col("size") / pl.col("size").sum()).round(1).alias("pct"), pl.lit(f"{L} {kind}").alias("strategy")))
    lines = [__doc__.split("\n\n")[0], "",
             "1-2. Main results (average of 5 seeds; NET = after fees; test_folds_NET_c = the same strategy on the test folds)",
             str(pl.DataFrame(main_rows)), "",
             "3. Win rates (net > 0; by size = weighted by contracts); average winning / losing trade (c per contract)",
             str(pl.DataFrame(win_rows))]
    if how_rows:
        lines += ["", "   How the fixed exit ended (% of contracts)",
                  str(pl.concat(how_rows).pivot(on="how", index="strategy", values="pct"))]
    if seed_rows:
        lines += ["", "4. Net c/contract seed by seed",
                  str(pl.DataFrame(seed_rows).pivot(on="seed", index=["strategy", "exit"], values="NET_c"))]
    lines += ["", "5. By game type", str(pl.DataFrame(type_rows).sort("strategy", "exit", "type")),
              "", "Descriptive: forecast R^2 (%) against no change (average of 5 seeds)", str(pl.DataFrame(r2))]
    OUT.mkdir(exist_ok=True)
    (OUT / "b17_holdout_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
