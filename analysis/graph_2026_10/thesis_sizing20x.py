"""Re-score Chapter 5's saved historical fills at round_half_up(20*w, .01).

Run from the repository root, separately for development and original holdout:
  venv/bin/python -W ignore analysis/graph_2026_10/thesis_sizing20x.py
  STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/thesis_sizing20x.py

Forecasts, entries, exits and historical per-unit fees are unchanged. This is a
fixed-fill sizing calculation, not a new order-book replay: it does not restore
opportunities blocked by orders whose resized quantity rounds to zero, and it
does not model quantity-dependent fills or market impact. All eligible games,
including games with no retained fills, enter game-level statistics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

CLI_ARGS = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).parent))
from b33_ablation_table import FULL, VERSIONS, OUT, SIZE, liquidity, with_net
from stg.splits import sports_holdout_unsealed
from stg.sports.loaders import derived_dir

DEST = OUT / "thesis_sizing20x"


def stat(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else None


def per_game(trades, games, value="pnl"):
    return (games.select("game_id").join(trades.group_by("game_id").agg(pl.col(value).sum()),
            on="game_id", how="left").fill_null(0)[value].to_numpy())


def summary(x, games):
    q = x["size"].sum()
    pnl = per_game(x, games)
    dated = games.select("game_id", "first_play").with_columns(
        pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day"))
    daily = (dated.join(x.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
             .fill_null(0).group_by("day").agg(pl.col("pnl").sum()))
    if not q:
        raise ValueError("No positive quantity")
    return dict(games=games.height, games_traded=x["game_id"].n_unique(), fills=x.height,
                quantity=float(q), gross_c=float((x["size"] * x["gross"]).sum() / q),
                net_c=float(x["pnl"].sum() / q), net_2x_c=float((x["size"] * (2*x["net"]-x["gross"])).sum()/q),
                usd=float(x["pnl"].sum()/100), t_game=stat(pnl), days_up=float(100*(daily["pnl"]>0).mean()),
                fill_win_pct=float(100*(x["pnl"]>0).mean()))


def refresh_fill_win_from_cache():
    """Add fill win rates from already scored fills without repeating the backtest."""
    held = sports_holdout_unsealed()
    mode = "holdout" if held else "development"
    cache = derived_dir()/"ingame/results/thesis_sizing20x" if held else DEST
    rates = []
    for league in (("MLB", "NFL") if held else ("MLB", "NBA", "NFL")):
        rates.append(pl.scan_parquet(cache/f"{mode}_{league}.parquet")
                     .group_by("source", "hold", "kind")
                     .agg((100*(pl.col("pnl")>0).mean()).alias("fill_win_pct"),
                          pl.len().alias("cached_fills"))
                     .with_columns(pl.lit(league).alias("league"))
                     .rename({"hold":"horizon"}).collect())
    path = DEST/f"{mode}_summary.csv"
    original = pl.read_csv(path).drop("fill_win_pct", strict=False)
    updated = original.join(pl.concat(rates), on=["league","source","horizon","kind"],
                            how="left", validate="1:1", maintain_order="left")
    assert updated["fill_win_pct"].null_count() == 0
    assert (updated["fills"] == updated["cached_fills"]).all()
    updated.drop("cached_fills").write_csv(path)


def main():
    held = sports_holdout_unsealed()
    mode = "holdout" if held else "development"
    DEST.mkdir(exist_ok=True)
    results, daily_rows = [], []
    for league in (("MLB", "NFL") if held else ("MLB", "NBA", "NFL")):
        frames, fallback = [], set()                     # fallback: NBA/NFL ablations still at the old 10 s delay (compared with the 10 s full model)
        def add(source, files, horizons=("120s",)):
            if not files:
                raise FileNotFoundError(f"Missing {league} {source}")
            frame = pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed").filter(
                pl.col("k")==1, pl.col("through")==0, pl.col("hold").is_in(horizons),
                pl.col("kind").is_in(("spread", "total")))
            frames.append(frame.with_columns(pl.lit(source).alias("source")))
        folder = derived_dir()/"ingame"/"results" if held else OUT
        if held:
            tag = FULL.at(league).tag(league)
            rule = FULL.at(league).rule_mlb if league == "MLB" else "_dec5_one"
            stem = f"b7_maker_trades_{league}_d{FULL.at(league).delay:g}{tag}"
            def paths(suffix, strategy=rule):
                p = folder/f"{stem}{suffix}{strategy}.parquet"
                return [p] if p.exists() else sorted(folder.glob(f"{stem}{suffix}{strategy}_part*of*.parquet"))
            add("full model (frozen)", paths("_ens5"))
            for seed in range(5):
                add(f"seed{seed}", paths(f"_seed{seed}" if seed else ""))
            if league == "NFL":
                add("original 60s decision grid", paths("_ens5", ""))
        else:
            for variant in VERSIONS:
                if league not in variant.leagues:
                    continue
                stem = variant.at(league).log_stem(league)
                if variant.group in ("A. components", "B. inputs") and variant.at(league).delay != FULL.at(league).delay:
                    fallback.add(variant.name)
                files = sorted(folder.glob(f"{stem}_fold*.parquet"))
                if not files:
                    files = sorted(folder.glob(f"{stem}.parquet"))
                if not files and variant is not FULL:
                    print(f"  (no {league} {variant.name} logs at its delay yet: left out)", flush=True)
                    continue
                assert len(files) in (1, 5), (league, variant.name, len(files))
                add(variant.name, files, ("60s", "120s") if variant is FULL else ("120s",))
            add("full model (frozen)", sorted(folder.glob(f"{FULL.at(league).log_stem(league)}_holds30s_fold*.parquet")), ("30s",))
            if league != "MLB":
                add("full model (10 s reference)", sorted(folder.glob(f"{FULL.log_stem(league)}_fold*.parquet")), ("120s",))
        raw = pl.concat(frames, how="diagonal_relaxed")
        rich = liquidity(with_net(raw)).with_columns(SIZE.alias("weight"))
        rich = rich.with_columns(((pl.col("weight")*2000+0.5).floor()/100).alias("size"))
        assert rich["size"].min() >= 0 and rich["size"].max() <= 20
        zero = rich.filter((pl.col("weight")>0)&(pl.col("size")==0)).height
        print(f"{mode} {league}: {raw.height} saved fills, {zero} positive weights round to zero", flush=True)
        rich = rich.filter(pl.col("size")>0).with_columns((pl.col("size")*pl.col("net")).alias("pnl"))
        game_folder = derived_dir()/"ingame"/("MLB_pitch" if league=="MLB" else league)
        games = pl.read_parquet(game_folder/"games.parquet").select("game_id","first_play","fold")
        if not held:
            games=games.filter(pl.col("fold")>0)
        assert set(rich["game_id"]).issubset(set(games["game_id"]))
        full = rich.filter(pl.col("source")==FULL.name,pl.col("hold")=="120s")
        ref10 = rich.filter(pl.col("source")=="full model (10 s reference)",pl.col("hold")=="120s")
        for (source, horizon, kind), x in rich.group_by("source","hold","kind",maintain_order=True):
            old_delay = source in fallback or source == "full model (10 s reference)"
            row=dict(sample=mode,league=league,source=source,horizon=horizon,kind=kind,**summary(x,games),
                     delay_s=10 if league == "MLB" or old_delay else 60)
            base=(ref10 if source in fallback else full).filter(pl.col("kind")==kind)
            row["paired_t_vs_full"]=stat(per_game(x,games)-per_game(base,games))
            row["paired_usd_vs_full"]=float((per_game(x,games)-per_game(base,games)).sum()/100)
            results.append(row)
            if source==FULL.name:
                dg=(games.join(x.group_by("game_id").agg(pl.col("pnl").sum(),
                    (pl.col("size")*(2*pl.col("net")-pl.col("gross"))).sum().alias("pnl_2x")),on="game_id",how="left")
                    .with_columns(pl.col("pnl","pnl_2x").fill_null(0),pl.lit(league).alias("league"),
                                  pl.lit(kind).alias("kind"),pl.lit(horizon).alias("horizon"),pl.lit(mode).alias("sample")))
                daily_rows.append(dg)
        # Quantity-independent trade dates and marks are retained for audit.
        cache = folder/"thesis_sizing20x" if held else DEST
        cache.mkdir(exist_ok=True)
        rich.write_parquet(cache/f"{mode}_{league}.parquet")
        print(f"Completed {mode} {league}",flush=True)
    pl.DataFrame(results,infer_schema_length=None).write_csv(DEST/f"{mode}_summary.csv")
    pl.concat(daily_rows,how="diagonal_relaxed").write_parquet(DEST/f"{mode}_per_game.parquet")


if __name__ == "__main__":
    if "--refresh-fill-win-from-cache" in CLI_ARGS:
        refresh_fill_win_from_cache()
    else:
        main()
