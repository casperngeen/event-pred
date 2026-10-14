"""Train and evaluate the in-game graph model (stg.models.ingame) on one walk-forward fold.

Runs anywhere: CUDA if present (the cluster), else Apple MPS, else CPU. The panel directory is the
output of analysis/graph_2026_10/b4_export.py (data/sports/derived/ingame/<league>/panel_v2/), the only
data that may be copied to the cluster.

    python cluster/train_ingame.py --panel data/sports/derived/ingame/NFL/panel_v2 --test-fold 1 --memory gru
    python cluster/train_ingame.py --panel ... --test-fold 1 --memory none      # the no-memory ablation
    python cluster/train_ingame.py --panel ... --test-fold 1 --memory transformer [--tf-window 30]   # v2 memory
    python cluster/train_ingame.py --panel ... --test-fold 1 --delay 15         # plays reach the model 15 s late
    python cluster/train_ingame.py --panel ... --test-fold 1 --horizons 30s,60s,120s   # train on 30 s - 2 min only
    python cluster/train_ingame.py --panel .../panel_v4m --test-fold 3 --huber 1000 --lambda-y 0   # mid target, ~squared error

Writes <out>/<league>_fold<k>_<memory>_d<delay>[_h<horizons>]_seed<s>.json (settings, epochs, and the test metrics) and
prints a short table; with --save-preds also the test predictions (.npz) for backtests.

The final model for the holdout test (--final instead of --test-fold): trained the same way on every game in the
panel (the last 15% by date for early stopping), no test set; --save-model writes the weights (.pt) next to the
json, named <league>_final_<memory>_d<delay>[_h…]_seed<s>:

    python cluster/train_ingame.py --panel panels/MLB_v3 --final --save-model --delay 10 --horizons 30s,60s,120s,1800s \
        --games-per-batch 16 --steps-per-game 256 --seed 0 --out results/ingame/final
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

# load the model file on its own: importing the stg.models package would pull in polars, which the
# trainer does not need (and which crashes on the cluster's login node CPU)
_spec = importlib.util.spec_from_file_location(
    "ingame", Path(__file__).resolve().parents[1] / "stg_infra" / "stg" / "models" / "ingame.py")
_ingame = importlib.util.module_from_spec(_spec)
sys.modules["ingame"] = _ingame
_spec.loader.exec_module(_ingame)
Panel, train = _ingame.Panel, _ingame.train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    fold = ap.add_mutually_exclusive_group(required=True)
    fold.add_argument("--test-fold", type=int)
    fold.add_argument("--final", action="store_true", help="train on every game (the holdout model); no test set")
    ap.add_argument("--memory", choices=["gru", "transformer", "none"], default="gru")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tf-window", type=int, default=0, help="transformer memory: attend to the last N plays only (0 = all)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--games-per-batch", type=int, default=8)
    ap.add_argument("--steps-per-game", type=int, default=128)
    ap.add_argument("--lambda-y", type=float, default=0.1)
    ap.add_argument("--huber", type=float, default=5.0, help="loss switch from squared to linear error (cents); large = squared")
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0, help="live-feed delay in seconds applied to the play-by-play")
    ap.add_argument("--horizons", default="", help="comma list to train / evaluate on, e.g. 30s,60s,120s (default: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/ingame")
    ap.add_argument("--save-preds", action="store_true", help="also write the test predictions (npz) for backtests")
    ap.add_argument("--save-model", action="store_true", help="also write the trained weights (.pt)")
    a = ap.parse_args()
    t0 = time.time()
    hz = [h for h in a.horizons.split(",") if h]
    pn = Panel(a.panel, horizons=hz or None)
    got = train(pn, None if a.final else a.test_fold, memory=a.memory, hidden=a.hidden, n_rounds=a.rounds, dropout=a.dropout, lr=a.lr,
                   epochs=a.epochs, patience=a.patience, games_per_batch=a.games_per_batch,
                   steps_per_game=a.steps_per_game, lambda_y=a.lambda_y, eval_stride=a.eval_stride, seed=a.seed, delay_s=a.delay, tf_window=a.tf_window, huber=a.huber,
                   device=a.device, log=lambda s: print(s, flush=True), return_preds=a.save_preds and not a.final)
    res = got[0]
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if not hz else "_h" + "-".join(h.rstrip("s") for h in hz)
    if pn.manifest.get("targets") == "trade prints":         # panel_v3: targets are traded prices
        tag += "_tpf" if pn.manifest.get("futures") else "_tp"     # _tpf: panel_v3f, with futures nodes
    elif pn.manifest.get("targets") == "fresh mid":         # panel_v4m: targets are fresh bid / ask mids
        tag += "_mid"
    tag += (f"_hub{a.huber:g}" if a.huber != 5.0 else "") + (f"_ly{a.lambda_y:g}" if a.lambda_y != 0.1 else "")
    tag += f"_r{a.rounds}" if a.rounds != 2 else ""          # _r0: no message passing (graph ablation)
    tag += "_pitch" if pn.manifest.get("play_unit") == "pitch" else ""   # MLB with every pitch as a play
    tag += "_dpfix" if pn.manifest.get("mlb_groups") == "dpfix" else ""  # double / triple plays coded as such
    res["targets"] = pn.manifest.get("targets", "quotes")
    mem = a.memory + (f"w{a.tf_window}" if a.memory == "transformer" and a.tf_window else "")
    split = "final" if a.final else f"fold{a.test_fold}"
    f = out / f"{res['league']}_{split}_{mem}_d{a.delay:g}{tag}_seed{a.seed}.json"
    res["args"] = vars(a)
    f.write_text(json.dumps(res, indent=1))
    if a.save_model:
        _ingame.save_model(f.with_suffix(".pt"), got[1], pn,
                           _ingame.model_config(pn, hidden=a.hidden, memory=a.memory, n_rounds=a.rounds, dropout=a.dropout,
                                                tf_window=a.tf_window), res)
    if a.save_preds and not a.final:
        import numpy as np
        np.savez_compressed(f.with_suffix(".npz"), **got[2])
    print(f"\nwrote {f} ({res['seconds']} s)")
    for kind, r in res["metrics"].items():
        for h, m in r.items():
            if "r2_model" in m and h != "drift_60_600":
                print(f"{kind:7s} {h:5s} rows {m['rows']:>9,}  R2 model {m['r2_model']:+.3f}  baseline {m['r2_baseline']:+.3f}  "
                      f"interval model {m['interval_model']:.2f} / no change {m['interval_no_change']:.2f} / "
                      f"baseline {m['interval_baseline']:.2f}")
            elif h == "drift_60_600":
                print(f"{kind:7s} drift 60->600 s  R2 model {m['r2_model']:+.3f}  corr {m['corr']:+.3f}")
            elif h == "outcome":
                print(f"{kind:7s} final result  log loss model {m['logloss_model']:.4f}  market {m['logloss_market']:.4f}  "
                      f"({m['games']} games)")


if __name__ == "__main__":
    main()
