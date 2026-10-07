"""Quick stand-in for a better-trained model: correct the GRU model's predicted changes with a ridge
regression on its own predictions plus mid - value (where the last price sits relative to the middle
of the rebuilt bid / ask: an input the network already has, as raw levels), fitted walk-forward, and
write the corrected predictions as new prediction files for the unchanged backtests.

For test fold k (2..5) the ridge for each horizon is fitted on folds 1..k-1 (their out-of-sample
predictions), on rows with a target, one row per 30 s of game time; fold 1 has nothing earlier and
is not written. Features: the predicted changes at every horizon, mid - value (0 if one side is
missing), a both-sides flag, and the contract kind. The correction is applied to every row.

    venv/bin/python -W ignore analysis/graph_2026_10/b8_corrected_preds.py results/ingame NBA 10
    -> results/ingame_ridge/<L>_fold<k>_gru_d<delay>_h30-60-120-1800_tpf_ridge_seed0.{npz,json}
    then: RUN_TAG=_h30-60-120-1800_tpf_ridge b7_maker_trades.py results/ingame_ridge <L> <delay> <panel>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES = Path(sys.argv[1])
LEAGUE, DELAY = sys.argv[2], float(sys.argv[3])
TAG = "_h30-60-120-1800_tpf"
OUT = Path("results/ingame_ridge")
SUB = 6                     # one row per 30 s (6 steps) for fitting


def feats(z: dict) -> np.ndarray:
    both = z["has_bid"] & z["has_ask"]
    mid_rel = np.where(both, (z["entry_bid"] + z["entry_ask"]) / 2 - z["value"], 0.0)
    k = z["kind"]
    return np.column_stack([z["delta"], mid_rel, both, k == 0, k == 2]).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = {k: dict(np.load(RES / f"{LEAGUE}_fold{k}_gru_d{DELAY:g}{TAG}_seed0.npz")) for k in range(1, 6)}
    meta = {k: json.loads((RES / f"{LEAGUE}_fold{k}_gru_d{DELAY:g}{TAG}_seed0.json").read_text()) for k in range(1, 6)}
    X = {k: feats(z[k]) for k in z}
    nh = z[1]["delta"].shape[1]
    for k in range(2, 6):
        zk = dict(z[k])
        delta = zk["delta"].copy()
        rep = []
        for h in range(nh):
            xs, ys = [], []
            for j in range(1, k):
                y = z[j]["value_h"][:, h] - z[j]["value"]
                sel = np.isfinite(y) & z[j]["live"] & (z[j]["step"] % SUB == 0)
                xs.append(X[j][sel])
                ys.append((y - z[j]["delta"][:, h])[sel])
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(np.concatenate(xs), np.concatenate(ys))
            delta[:, h] = zk["delta"][:, h] + m.predict(X[k]).astype(np.float32)
            y = zk["value_h"][:, h] - zk["value"]
            ok = np.isfinite(y) & zk["live"]
            r2 = lambda p: 1 - ((y[ok] - p[ok]) ** 2).sum() / (y[ok] ** 2).sum()   # noqa: E731
            rep.append(f"{meta[k]['horizons'][h]}: {r2(zk['delta'][:, h]):.3f} -> {r2(delta[:, h]):.3f}")
        zk["delta"] = delta
        name = f"{LEAGUE}_fold{k}_gru_d{DELAY:g}{TAG}_ridge_seed0"
        np.savez_compressed(OUT / f"{name}.npz", **zk)
        (OUT / f"{name}.json").write_text(json.dumps(meta[k] | {"correction": "ridge on predictions + mid - value, folds < k"}, indent=1))
        print(f"{LEAGUE} d{DELAY:g} fold {k}: R^2 on live rows with a target  " + "; ".join(rep), flush=True)


if __name__ == "__main__":
    main()
