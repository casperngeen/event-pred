"""Should the minimum signal differ by league? Walk-forward test on the test folds only.

Setup: the frozen candidates' setup (120 s hold, offset 1c, strictly-through fills, frozen sizing), decision every
60 s with stacking (dropping a trade then changes no other trade, so raising the minimum on the logged trades is
exact). MLB: the every-pitch model; NBA / NFL: the plate-appearance-era model (tp_ens5). The sizing formula is
unchanged; only trades with |signal| below the minimum are dropped.

For each test fold k >= 2 the minimum is picked from MENU using folds 1..k-1 only (most net dollars there), then
fold k is scored with it. Fold 1 has no earlier fold, so it is not scored. Four ways to pick:
    shared 5c        the frozen rule (no choice)
    one for all      one minimum for every league and contract, picked on all leagues' earlier folds
    per league       one minimum per league (spread + total together)
    per contract     one minimum per league x spread / total
Per league beats one-for-all out of sample only if leagues really differ; otherwise it is fitting noise.

    venv/bin/python -W ignore analysis/graph_2026_10/b22_league_thresholds.py
    -> out/b22_league_thresholds.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b19_decision_timing as B                          # noqa: E402
from b17_holdout_report import exits                      # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402

MENU = [5.0, 8.0, 10.0, 12.0]
TAGS = {"MLB": "_h30-60-120-1800_tp_pitch_ens5", "NBA": "_h30-60-120-1800_tp_ens5", "NFL": "_h30-60-120-1800_tp_ens5"}
KINDS = ("spread", "total")
SCHEMES = {"shared 5c": None, "one for all": (), "per league": ("league",), "per contract": ("league", "kind")}


def trades() -> pl.DataFrame:
    parts = []
    for L, tag in TAGS.items():
        B.TAG = tag
        f = B.load(L, "").filter(pl.col("kind").is_in(KINDS))
        for ex, g in exits(f).items():
            parts.append(g.select(pl.lit(L).alias("league"), "kind", pl.lit(ex).alias("exit"), "fold", "game_id",
                                  pl.col("signal").abs().alias("sig"), "size", "pnl"))
    return pl.concat(parts)


def pick(prior: pl.DataFrame, keys: tuple[str, ...]) -> dict:
    """Minimum with the most net dollars on the earlier folds, per group of `keys`."""
    rows = []
    for m in MENU:
        rows.append(prior.filter(pl.col("sig") >= m).group_by(list(keys) or pl.lit(1).alias("_all")).agg(pl.col("pnl").sum())
                    .with_columns(pl.lit(m).alias("min")))
    best = pl.concat(rows).sort("pnl", "min", descending=[True, False])
    if not keys:
        return {(): float(best["min"][0])}
    best = best.unique(list(keys), keep="first", maintain_order=True)
    return {tuple(r[k] for k in keys): r["min"] for r in best.iter_rows(named=True)}


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_hide_column_data_types(True)
    t = trades()
    scored, chosen = [], []
    for ex in ("fixed exit", "hold to settlement"):
        e = t.filter(pl.col("exit") == ex)
        for k in range(2, 6):
            prior, test = e.filter(pl.col("fold") < k), e.filter(pl.col("fold") == k)
            for name, keys in SCHEMES.items():
                if keys is None:
                    mins = {(): 5.0}
                    keys = ()
                else:
                    mins = pick(prior, keys)
                    chosen += [dict(exit=ex, scheme=name, fold=k, group=" ".join(g) or "all", min=m) for g, m in mins.items()]
                m = test.with_columns(pl.struct(list(keys) or [pl.lit(1).alias("_")]).map_elements(
                    lambda r, keys=keys, mins=mins: mins.get(tuple(r[c] for c in keys), 5.0), return_dtype=pl.Float64).alias("min")
                                      if keys else pl.lit(mins[()]).alias("min"))
                scored.append(m.filter(pl.col("sig") >= pl.col("min")).with_columns(pl.lit(name).alias("scheme")))
    s = pl.concat(scored)
    order = {n: i for i, n in enumerate(SCHEMES)}

    def table(by: list[str]) -> pl.DataFrame:
        return (s.group_by(["exit", "scheme"] + by).agg(
            pl.len().alias("trades"), pl.col("size").sum().round(0).alias("contracts"),
            (pl.col("pnl").sum() / pl.col("size").sum()).round(2).alias("net_c"), (pl.col("pnl").sum() / 100).round(1).alias("net_usd"))
            .with_columns(pl.col("scheme").replace_strict(order).alias("_o")).sort(["exit"] + by + ["_o"]).drop("_o"))

    def tstats(by: list[str]) -> pl.DataFrame:
        """Net per contract and game-level t (per-game net dollars over the games the shared 5c rule traded)."""
        rows = []
        for key, base_g in s.filter(pl.col("scheme") == "shared 5c").group_by(["exit"] + by, maintain_order=True):
            games = base_g["game_id"].unique()
            for name in SCHEMES:
                g = s.filter(pl.col("scheme") == name, *[pl.col(c) == v for c, v in zip(["exit"] + by, key)])
                pg = (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
                      .fill_null(0)["pnl"].to_numpy())
                rows.append(dict(zip(["exit"] + by, key)) | dict(scheme=name, games=len(games),
                            net_c=round(g["pnl"].sum() / g["size"].sum(), 2), t_game=round(pg.mean() / pg.std(ddof=1) * np.sqrt(len(pg)), 2),
                            win_pct=round(100 * (g["pnl"] > 0).mean(), 1)))
        return pl.DataFrame(rows)

    # game-level paired difference vs shared 5c, all leagues together (folds 2-5)
    base = s.filter(pl.col("scheme") == "shared 5c").group_by("exit", "league", "game_id").agg(pl.col("pnl").sum().alias("b"))
    diff = []
    for ex in ("fixed exit", "hold to settlement"):
        for name in list(SCHEMES)[1:]:
            g = (s.filter(pl.col("exit") == ex, pl.col("scheme") == name).group_by("league", "game_id").agg(pl.col("pnl").sum().alias("a"))
                 .join(base.filter(pl.col("exit") == ex).drop("exit"), on=["league", "game_id"], how="full", coalesce=True).fill_null(0))
            d = (g["a"] - g["b"]).to_numpy() / 100
            diff.append(dict(exit=ex, scheme=name, games=len(d), usd_vs_shared_5c=round(d.sum(), 1),
                             t_game=round(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), 2) if d.std() > 0 else 0.0))
    ch = (pl.DataFrame(chosen).pivot(on="fold", index=["exit", "scheme", "group"], values="min").sort("exit", "scheme", "group"))
    lines = [__doc__.split("\n\n")[0], "", f"Menu of minimums: {MENU} c. Scored folds 2-5 only. Net in cents per contract after fees.",
             "", "-- all six contracts together", str(table([])),
             "", "-- by contract", str(table(["league", "kind"])),
             "", "-- net per contract and game-level t, by league (spread + total)", str(tstats(["league"])),
             "", "-- net per contract and game-level t, by contract", str(tstats(["league", "kind"])),
             "", "-- change in net dollars vs shared 5c (game-level paired t, all leagues)", str(pl.DataFrame(diff)),
             "", "-- minimum chosen for each scored fold (picked on the folds before it)", str(ch)]
    (OUT / "b22_league_thresholds.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
