"""Checks on the in-game training table (b3_table): coverage, target alignment, and as-of inputs.

    1. sizes per fold; plays and steps per game; games with a pregame total (baseline) and a winner mid
    2. per contract kind: rows, live share, share with each target, y coverage, quote age, drift spread
    3. target alignment: value_60s at step k equals value at step k + 12 of the same contract
    4. as-of: no step's last known play happened after the step; quotes are never from the future
       (quote age >= 0)
    5. one contract's rows around a big play, for eyeballing

    venv/bin/python -W ignore analysis/graph_2026_10/b3_table_check.py NFL > analysis/graph_2026_10/out/b3_table_check_NFL.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.sports.loaders import derived_dir                 # noqa: E402

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "NFL"
D = derived_dir() / "ingame" / LEAGUE
MAX_GAMES = 300


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(230)
    games, plays, steps = (pl.read_parquet(D / f) for f in ("games.parquet", "plays.parquet", "steps.parquet"))
    con = pl.read_parquet(D / "contracts.parquet")
    # a futures contract belongs to every game of its team: join by game as well as ticker
    cols = ["k", "ticker", "bid", "ask", "quote_age_s", "value", "live", "y", "value_10s", "value_60s", "value_600s",
            "drift_60_600", "pv_60s", "pv_1800s"]
    # at most MAX_GAMES games (every n-th by kick-off), so the check fits in memory for NBA / MLB
    files = sorted((D / "contract_steps").glob("*.parquet"))
    files = files[::max(1, len(files) // MAX_GAMES)]
    cs = (pl.scan_parquet(files, include_file_paths="path", missing_columns="insert")
          .select(*cols, pl.col("path").str.extract(r"([^/]+)\.parquet$").alias("game_id")).collect()
          .join(con.select("ticker", "kind", "game_id"), on=["ticker", "game_id"]))
    cs = cs.join(games.select("game_id", "fold"), on="game_id")
    print(f"(contract-step checks on {cs['game_id'].n_unique()} of {games.height} games)")
    print(f"######## 1. {LEAGUE} sizes")
    print(games.group_by("fold").agg(pl.len().alias("games"), pl.col("L").is_not_null().sum().alias("with_pregame_total"))
          .join(steps.group_by("fold").agg(pl.len().alias("steps"), pl.col("p_home").is_not_null().mean().round(3)
                                           .alias("steps_with_winner_mid")), on="fold")
          .join(cs.group_by("fold").agg(pl.len().alias("contract_steps")), on="fold").sort("fold"))
    print(f"plays per game: {plays.group_by('game_id').len()['len'].median():.0f}; steps per game: "
          f"{steps.group_by('game_id').len()['len'].median():.0f}; plays with a dWP: {plays['dwp'].is_not_null().mean():.3f}")

    print("\n######## 2. per contract kind (in play only)")
    ip = cs.join(steps.select("game_id", "k", "after_end"), on=["game_id", "k"]).filter(~pl.col("after_end"))
    print(ip.group_by("kind").agg(
        pl.len().alias("rows"), pl.col("ticker").n_unique().alias("contracts"), pl.col("live").mean().round(3).alias("live"),
        *[pl.col(f"value_{h}").is_not_null().mean().round(3).alias(f"has_{h}") for h in ("10s", "60s", "600s")],
        *[pl.col(f"pv_{h}").is_not_null().mean().round(3).alias(f"has_trade_{h}") for h in ("60s", "1800s")],
        pl.col("y").is_not_null().mean().round(3).alias("has_y"),
        pl.col("quote_age_s").median().alias("quote_age_med_s"),
        pl.col("drift_60_600").std().round(2).alias("drift_sd"),
        pl.col("drift_60_600").mean().round(3).alias("drift_mean")).sort("kind"))

    print("\n######## 3. target alignment (value_60s at k vs value at k + 12)")
    a = cs.select("game_id", "ticker", "k", "value", "value_60s")
    b = a.join(a.select("game_id", "ticker", (pl.col("k") - 12).alias("k"), pl.col("value").alias("v_later")),
               on=["game_id", "ticker", "k"], how="inner")
    both = b.drop_nulls(["value_60s", "v_later"])
    print(f"rows compared {both.height:,}; mismatches {(both['value_60s'] != both['v_later']).sum()} "
          "(rows where the later step was dropped for having no quote do not compare)")

    print("\n######## 4. as-of checks")
    chk = steps.join(plays.select("game_id", pl.col("seq"), "t_event"), on=["game_id", "seq"], how="left")
    print(f"steps whose last known play happened after the step: {(chk['t_event'] > chk['t']).sum()}")
    print(f"contract-steps with a negative quote age: {(cs['quote_age_s'] < 0).sum()}")

    print("\n######## 5. one winner contract around the game's biggest play")
    big = plays.filter(pl.col("dwp").abs() == pl.col("dwp").abs().max()).row(0, named=True)
    k0 = steps.filter(pl.col("game_id") == big["game_id"], pl.col("t") >= big["t_event"])["k"].min()
    tk = con.filter(pl.col("game_id") == big["game_id"], pl.col("kind") == "winner", pl.col("side") == "home")["ticker"][0]
    print(f"game {big['game_id']}, play seq {big['seq']} ({big['play_group']}, dWP {big['dwp']:+.3f}); contract {tk}")
    print(cs.filter(pl.col("game_id") == big["game_id"], pl.col("ticker") == tk, pl.col("k").is_between(k0 - 3, k0 + 6))
          .select("k", "bid", "ask", "quote_age_s", "value", "live", "value_10s", "value_60s", "value_600s", "drift_60_600", "y")
          .sort("k"))


if __name__ == "__main__":
    main()
