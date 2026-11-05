"""Add game-clustered paired loss-difference t statistics to existing holdout R2.

Run from the repository root with STG_SPORTS_UNSEAL=final-test. Reads only the
two saved original fold99 ensemble forecasts used by b17_holdout_report.py;
does not train, predict, select models, or evaluate additional games.
Positive t favours the model over no change. The row-weighted mean squared-error
gain has a cluster-robust standard error, with each game one cluster (CR1).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.splits import sports_holdout_unsealed
from stg.sports.loaders import derived_dir


def main() -> None:
    assert sports_holdout_unsealed(), "Requires STG_SPORTS_UNSEAL=final-test"
    root = derived_dir() / "ingame" / "results"
    rows = []
    # MLB is the frozen 10 s model; NFL is the 60 s model its live runner uses (NFL_DELAY=10 reproduces the 10 s rows)
    nfl_delay = os.environ.get("NFL_DELAY", "60")
    for league, tag, delay in (("MLB", "_pitch", "10"), ("NFL", "", nfl_delay)):
        path = root / f"{league}_fold99_gru_d{delay}_h30-60-120-1800_tp{tag}_ens5_seed0.npz"
        horizons = json.loads(path.with_suffix(".json").read_text())["horizons"]
        with np.load(path) as archive:
            kind, game, value = (archive[k] for k in ("kind", "game", "value"))
            target, prediction = archive["value_h"], archive["delta"]
        for code, name in enumerate(("winner", "spread", "total")):
            for horizon in ("30s", "60s", "120s"):
                hi = horizons.index(horizon)
                valid = (kind == code) & np.isfinite(target[:, hi]) & np.isfinite(value)
                valid &= np.isfinite(prediction[:, hi])
                actual = target[valid, hi].astype(np.float64) - value[valid]
                forecast = prediction[valid, hi].astype(np.float64)
                gain = actual**2 - (actual - forecast)**2
                _, groups = np.unique(game[valid], return_inverse=True)
                counts = np.bincount(groups)
                gains = np.bincount(groups, weights=gain)
                n, g = len(gain), len(counts)
                residual = gains - counts * gain.mean()
                variance_sum = g / (g - 1) * np.dot(residual, residual)
                stat = gain.sum() / np.sqrt(variance_sum) if variance_sum > 0 else np.nan
                rows.append(dict(league=league, kind=name, horizon=horizon,
                                 rows=n, games=g, R2_pct=100 * gain.sum() / np.dot(actual, actual),
                                 t_game_clustered=stat))
        print(f"Completed {league}", flush=True)
        del kind, game, value, target, prediction
    result = pl.DataFrame(rows)
    # Guard against accidentally reporting a different saved holdout snapshot.
    report = (Path(__file__).parent / "out" / "b17_holdout_report.txt").read_text()
    reference = {}
    for line in report.splitlines():
        fields = [v.strip() for v in line.strip("│ ").split("┆")]
        if len(fields) == 5 and fields[0] in ("MLB", "NFL") and fields[2] in ("30s", "60s", "120s"):
            reference[tuple(fields[:3])] = (int(fields[3]), float(fields[4]))
    for row in rows:
        if row["league"] == "NFL" and nfl_delay != "10":
            continue                    # the reference report covers the 10 s forecasts only
        n, skill = reference[row["league"], row["kind"], row["horizon"]]
        assert n == row["rows"] and abs(skill - row["R2_pct"]) < 0.011
    result.write_csv(Path(__file__).parent / "out" / ("thesis_holdout_skill_stats.csv" if nfl_delay != "10" else "thesis_holdout_skill_stats_d10.csv"))
    pl.Config.set_tbl_rows(30)
    print(result)


if __name__ == "__main__":
    main()
