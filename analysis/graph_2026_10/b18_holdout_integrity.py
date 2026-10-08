"""Final test, step 3 checks: is the holdout panel built like the training panel? No model output and no
P&L is read here, only the panels.

Per league, holdout panel (inside the holdout folder) against the training panel (panel_v3):
    1. games by type and month; plays, steps and contract rows per game
    2. per contract kind: rows, live share, share with a result, share with each horizon's target
    3. every input feature (plays, steps, contracts): mean, sd, share zero, share non-finite; flagged when
       the holdout mean moves by more than half a training sd, or non-finite values appear
    4. results against the play-by-play's final state: a winner contract should settle YES exactly when its
       team ends ahead, a spread when side x margin > strike, a total when total > strike (mismatches catch
       wrong labels, scalar / placeholder legs and play-by-play that stops early); the same rate on the
       training panel for reference
    5. as-of order: play and step times increase within each game, every game starts on or after the
       holdout start and is fold 99; no duplicate (game, kind, side, strike) contract

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b18_holdout_integrity.py MLB NFL
    -> out/b18_holdout_integrity.txt
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.models.ingame import Panel                       # noqa: E402
from stg.sports.loaders import derived_dir                # noqa: E402
from stg.splits import SPORTS_HOLDOUT_START, sports_holdout_unsealed   # noqa: E402

TRAIN = Path("data/sports/derived/ingame")                # the training panels (read directly, not through the loaders)
KINDS = {0: "winner", 1: "spread", 2: "total"}
OUT = Path(__file__).resolve().parent / "out"
SAMPLE = 400_000                                           # contract rows sampled for the feature table


def node_index(pn: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Per contract row: its step and its game."""
    step = np.repeat(np.arange(len(pn.steps_t)), np.diff(pn.step_node_ptr))
    game = np.repeat(np.arange(pn.n_games), np.diff(pn.game_step_ptr))[step]
    return step, game


def per_game(pn: Panel) -> dict:
    return dict(games=pn.n_games, plays_per_game=float(np.median(np.diff(pn.game_play_ptr))),
                steps_per_game=float(np.median(np.diff(pn.game_step_ptr))),
                rows_per_game=float(np.median(np.bincount(node_index(pn)[1], minlength=pn.n_games))))


def by_kind(pn: Panel, label: str) -> list[dict]:
    kind, y = np.asarray(pn.nodes_kind), np.asarray(pn.y, dtype=np.float32)
    live = np.asarray(pn.nodes_x[:, pn.market_node_features.index("live")], dtype=np.float32) > 0.5
    lo = np.asarray(pn.lo, dtype=np.float32)
    out = []
    for k, kn in KINDS.items():
        m = kind == k
        r = dict(panel=label, kind=kn, rows=int(m.sum()), live_pct=round(100 * live[m].mean(), 1), result_pct=round(100 * np.isfinite(y[m]).mean(), 1))
        for hi, h in enumerate(pn.manifest["horizons"]):
            if h in ("30s", "60s", "120s", "1800s"):
                r[f"target_{h}_pct"] = round(100 * np.isfinite(lo[m, hi]).mean(), 1)
        out.append(r)
    return out


def feature_stats(arr, names: list[str], rng: np.random.Generator) -> pl.DataFrame:
    n = arr.shape[0]
    idx = np.sort(rng.choice(n, min(n, SAMPLE), replace=False))
    x = np.asarray(arr[idx], dtype=np.float32)
    fin = np.isfinite(x)
    xm = np.where(fin, x, np.nan)
    return pl.DataFrame(dict(feature=names, mean=np.nanmean(xm, 0), sd=np.nanstd(xm, 0), zero_pct=100 * (x == 0).mean(0),
                             nonfinite_pct=100 * (~fin).mean(0)))


def label_check(pn: Panel) -> pl.DataFrame:
    """Each contract's result against the final margin / total in the play-by-play."""
    _, game = node_index(pn)
    df = pl.DataFrame(dict(game=game, kind=np.asarray(pn.nodes_kind).astype(np.int8), side=np.asarray(pn.node_side).astype(np.int8),
                           strike=np.asarray(pn.node_strike), y=np.asarray(pn.y, dtype=np.float32))).unique(["game", "kind", "side", "strike"])
    last = pn.game_play_ptr[1:] - 1
    has = np.diff(pn.game_play_ptr) > 0
    fin = pl.DataFrame(dict(game=np.arange(pn.n_games), margin=np.where(has, pn.plays_state[np.maximum(last, 0), 1], np.nan),
                            total=np.where(has, pn.plays_state[np.maximum(last, 0), 2], np.nan)))
    df = df.join(fin, on="game").with_columns(
        pl.when(pl.col("kind") == 0).then(pl.col("side") * pl.col("margin") > 0)
        .when(pl.col("kind") == 1).then(pl.col("side") * pl.col("margin") > pl.col("strike"))
        .otherwise(pl.col("total") > pl.col("strike")).alias("expected"))
    return (df.group_by("kind").agg(pl.len().alias("contracts"), pl.col("y").is_nan().sum().alias("no_result"),
                                    ((pl.col("y") == 1) != pl.col("expected")).filter(pl.col("y").is_not_nan()).sum().alias("mismatch"),
                                    pl.col("game").filter(((pl.col("y") == 1) != pl.col("expected")) & pl.col("y").is_not_nan())
                                    .n_unique().alias("games_with_mismatch"))
            .with_columns(pl.col("kind").replace_strict(KINDS)).sort("kind"))


def asof(pn: Panel) -> dict:
    bad_play = sum(int((np.diff(pn.plays_t[a:b]) < 0).any()) for a, b in zip(pn.game_play_ptr[:-1], pn.game_play_ptr[1:]))
    bad_step = sum(int((np.diff(pn.steps_t[a:b]) <= 0).any()) for a, b in zip(pn.game_step_ptr[:-1], pn.game_step_ptr[1:]))
    first = dt.datetime.fromtimestamp(float(pn.steps_t[pn.game_step_ptr[:-1]].min()), dt.timezone.utc)
    return dict(games_plays_out_of_order=bad_play, games_steps_out_of_order=bad_step, earliest_step=str(first),
                starts_after_holdout=first >= SPORTS_HOLDOUT_START, all_fold_99=bool((np.asarray(pn.game_fold) == 99).all()))


def main() -> None:
    assert sports_holdout_unsealed(), "reads the holdout panel: set STG_SPORTS_UNSEAL=final-test"
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_hide_column_data_types(True)
    rng = np.random.default_rng(0)
    lines = [__doc__.split("\n\n")[0]]
    for L in sys.argv[1:] or ["MLB", "NFL"]:
        hd = derived_dir() / "ingame" / L
        ho, tr = Panel(hd / "panel_v3"), Panel(TRAIN / L / "panel_v3")
        lines += [f"\n######## {L}", "1. per game (medians)", str(pl.DataFrame([dict(panel="training") | per_game(tr), dict(panel="holdout") | per_game(ho)]))]
        g = pl.read_parquet(hd / "games.parquet")
        lines += ["   holdout games by month", str(g.group_by(pl.col("first_play").dt.strftime("%Y-%m").alias("month")).len().sort("month"))]
        types = pl.read_parquet(hd / "steps.parquet", columns=["game_id"]).unique()   # games that made the table
        lines += [f"   holdout games in the table: {types.height}"]
        lines += ["2. per contract kind", str(pl.DataFrame(by_kind(tr, "training") + by_kind(ho, "holdout")))]
        tabs = []
        for name, a_tr, a_ho, cols in (("plays", tr.plays_x, ho.plays_x, tr.manifest["play_features"]),
                                       ("steps", tr.steps_x, ho.steps_x, tr.manifest["step_market_features"]),
                                       ("contracts", tr.nodes_x, ho.nodes_x, tr.manifest["node_market_features"])):
            assert a_tr.shape[1] == a_ho.shape[1], f"{name}: {a_tr.shape[1]} vs {a_ho.shape[1]} features"
            t = feature_stats(a_tr, cols, rng).join(feature_stats(a_ho, cols, rng), on="feature", suffix="_ho")
            tabs.append(t.with_columns(pl.lit(name).alias("input"),
                                       ((pl.col("mean_ho") - pl.col("mean")) / pl.col("sd").clip(1e-6)).alias("shift_sd")))
        ft = pl.concat(tabs).select("input", "feature", "mean", "mean_ho", "sd", "sd_ho", "zero_pct", "zero_pct_ho", "nonfinite_pct",
                                    "nonfinite_pct_ho", "shift_sd").with_columns(pl.col(pl.Float64, pl.Float32).round(3))
        flag = ft.filter((pl.col("shift_sd").abs() > 0.5) | (pl.col("nonfinite_pct_ho") > pl.col("nonfinite_pct")))
        lines += [f"3. input features: {ft.height}, flagged {flag.height} (|shift| > 0.5 training sd, or new non-finite values)",
                  str(flag if flag.height else ft.head(0)), "   all features:", str(ft)]
        lines += ["4. results vs the play-by-play's final state", "   training:", str(label_check(tr)), "   holdout:", str(label_check(ho))]
        con = pl.read_parquet(hd / "contracts.parquet")
        dup = con.group_by("game_id", "kind", "side", "strike").len().filter(pl.col("len") > 1).height
        lines += ["5. as-of order", str(asof(ho)), f"   duplicate (game, kind, side, strike) contracts: {dup}"]
        print("\n".join(lines[-12:]), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "b18_holdout_integrity.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {OUT / 'b18_holdout_integrity.txt'}")


if __name__ == "__main__":
    main()
