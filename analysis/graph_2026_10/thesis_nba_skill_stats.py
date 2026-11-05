"""NBA development-fold counterpart to thesis_holdout_skill_stats.py.

Reads saved five-seed ensemble predictions on the five disjoint walk-forward
test folds, at the 60-second feed delay of the live NBA model (NBA_DELAY=10 for the 10 s forecasts). No training or holdout access. Pools squared
errors across valid rows and clusters paired loss differences by game.
Run from the repository root with venv/bin/python.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    delay = os.environ.get("NBA_DELAY", "60")
    accum = {}
    seen = set()
    panel = Path("data/sports/derived/ingame/NBA/panel_v3")
    manifest = json.loads((panel / "manifest.json").read_text())
    pointers = np.load(panel / "step_node_ptr.npy", mmap_mode="r")
    panel_kind = np.load(panel / "nodes_kind.npy", mmap_mode="r")
    panel_value = np.load(panel / "value.npy", mmap_mode="r")
    panel_target = np.load(panel / "value_h.npy", mmap_mode="r")
    for fold in range(1, 6):
        path = Path("results/ingame/ens5") / f"NBA_fold{fold}_gru_d{delay}_h30-60-120-1800_tp_ens5_seed0.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as archive:
            game, kind, value = (archive[k] for k in ("game", "kind", "value"))
            pred = archive["delta"]
            step = archive["step"]
        # Compact ensemble archives omit targets. Recover their panel rows using
        # the same within-step mapping as b5_longshort.rows, and verify identity.
        first = np.r_[True, step[1:] != step[:-1]]
        starts = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
        nodes = pointers[step] + np.arange(len(step)) - starts
        assert np.array_equal(panel_kind[nodes], kind)
        assert np.allclose(panel_value[nodes], value, equal_nan=True)
        columns = [manifest["horizons"].index(h) for h in meta["horizons"]]
        target = panel_target[nodes][:, columns]
        games = set(np.unique(game))
        assert not seen.intersection(games), "Test games overlap across folds"
        seen.update(games)
        for code, name in enumerate(("winner", "spread", "total")):
            for horizon in ("30s", "60s", "120s"):
                hi = meta["horizons"].index(horizon)
                valid = (kind == code) & np.isfinite(value) & np.isfinite(target[:, hi]) & np.isfinite(pred[:, hi])
                actual = target[valid, hi].astype(np.float64) - value[valid]
                forecast = pred[valid, hi].astype(np.float64)
                gain = actual**2 - (actual - forecast)**2
                assert len(gain) == meta["metrics"][name][horizon]["rows"]
                _, groups = np.unique(game[valid], return_inverse=True)
                item = accum.setdefault((name, horizon), dict(counts=[], gains=[], ss0=0.0))
                item["counts"].append(np.bincount(groups))
                item["gains"].append(np.bincount(groups, weights=gain))
                item["ss0"] += np.dot(actual, actual)
        print(f"Completed NBA fold {fold}", flush=True)
        del game, kind, value, target, pred
    rows = []
    for (name, horizon), item in accum.items():
        counts, gains = np.concatenate(item["counts"]), np.concatenate(item["gains"])
        g = len(counts)
        mean = gains.sum() / counts.sum()
        residual = gains - counts * mean
        variance_sum = g / (g - 1) * np.dot(residual, residual)
        rows.append(dict(league="NBA", kind=name, horizon=horizon,
                         rows=int(counts.sum()), games=g,
                         R2_pct=100 * gains.sum() / item["ss0"],
                         t_game_clustered=gains.sum() / np.sqrt(variance_sum)))
    result = pl.DataFrame(rows)
    result.write_csv(Path(__file__).parent / "out" / ("thesis_nba_skill_stats.csv" if delay != "10" else "thesis_nba_skill_stats_d10.csv"))
    print(result)


if __name__ == "__main__":
    main()
