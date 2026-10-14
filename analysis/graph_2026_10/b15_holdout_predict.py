"""Final test, step 3 (laptop only): the frozen model's forecasts on the holdout panel.

Loads each seed's final weights (cluster/train_ingame.py --final --save-model, copied back from the
cluster), checks that the holdout panel has the same features, scales and horizons as the training
panel, predicts every step of every holdout game at the frozen feed delay, and writes per seed and
for the average of the seeds, in the format the backtests read (b5_longshort.rows, b7_maker_trades):

    <out>/<L>_fold99_<NAMES[L]>_seed<s>.npz (+ .json)
    <out>/<L>_fold99_<NAMES[L]>_ens5_seed0.npz (+ .json)

Runs only with the holdout unsealed (STG_SPORTS_UNSEAL=final-test) on a holdout panel (b4_export with
the same switch); <out> defaults to a folder inside the holdout. --smoke runs on a training panel
instead (no holdout data), writing to the given --out, to test the chain before the real run.

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b15_holdout_predict.py MLB \
        --weights results/ingame/final
Descriptive feed-delay runs (spec section 5): --delay 0 forecasts with plays 0 s after they are logged, with weights
trained at --weights-delay (default the same delay; 10 = the frozen models, a mismatch: outputs tagged _w10).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.models.ingame import Panel, load_model, pick_device, predict   # noqa: E402
from stg.sports.loaders import derived_dir                              # noqa: E402
from stg.splits import SPORTS_HOLDOUT_DIR, sports_holdout_unsealed      # noqa: E402

TAGS = {"MLB": "_h30-60-120-1800_tp_pitch", "NFL": "_h30-60-120-1800_tp", "NBA": "_h30-60-120-1800_tp"}
NAMES = {L: f"gru_d10{t}" for L, t in TAGS.items()}
# the frozen models (reports/holdout_spec.md; MLB: every pitch a play, section 9); feed delay 10 s
SEEDS = [0, 1, 2, 3, 4]
DELAY_S = 10.0
FOLD = 99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("league", choices=["MLB", "NFL", "NBA"])
    ap.add_argument("--weights", default="results/ingame/final", help="folder with <L>_final_<NAMES[L]>_seed<s>.pt")
    ap.add_argument("--panel", help="default: the holdout panel, <derived>/ingame/<L>/panel_v3")
    ap.add_argument("--out", help="default: <derived>/ingame/results (inside the holdout folder)")
    ap.add_argument("--smoke", action="store_true", help="test on a training panel (needs --panel and --out)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--delay", type=float, default=DELAY_S, help="feed delay when forecasting (descriptive runs; frozen: 10)")
    ap.add_argument("--weights-delay", type=float, help="feed delay the weights were trained with (default: --delay)")
    a = ap.parse_args()
    wd = a.delay if a.weights_delay is None else a.weights_delay
    wname = f"gru_d{wd:g}{TAGS[a.league]}"                                   # the weights
    oname = f"gru_d{a.delay:g}{TAGS[a.league]}" + (f"_w{wd:g}" if wd != a.delay else "")   # the forecasts
    if a.smoke:
        assert a.panel and a.out and not sports_holdout_unsealed(), "--smoke: a training panel, an --out, and the holdout sealed"
    else:
        assert sports_holdout_unsealed(), "the final test reads the holdout: set STG_SPORTS_UNSEAL=final-test"
    panel = Path(a.panel) if a.panel else derived_dir() / "ingame" / (a.league + ("_pitch" if a.league == "MLB" else "")) / "panel_v3"
    out = Path(a.out) if a.out else derived_dir() / "ingame" / "results"
    pn = Panel(panel, horizons=["30s", "60s", "120s", "1800s"])
    if not a.smoke:
        assert pn.manifest.get("holdout_panel") and SPORTS_HOLDOUT_DIR in panel.resolve().parts, f"{panel}: not a holdout panel"
        assert SPORTS_HOLDOUT_DIR in out.resolve().parts, f"{out}: holdout forecasts stay inside the holdout folder"
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    games = list(range(pn.n_games))
    print(f"{a.league}: {pn.n_games} games, panel {panel}, device {dev}", flush=True)
    preds = []
    for s in SEEDS:
        w = Path(a.weights) / f"{a.league}_final_{wname}_seed{s}.pt"
        model, ck = load_model(w, pn, dev)
        assert ck["info"]["delay_s"] == wd and ck["info"]["seed"] == s, f"{w}: not the expected settings"
        pr = predict(model, pn, games, dev, stride=1, delay_s=a.delay)
        preds.append(pr)
        base = out / f"{a.league}_fold{FOLD}_{oname}_seed{s}"
        np.savez_compressed(base.with_suffix(".npz"), **pr)
        base.with_suffix(".json").write_text(json.dumps(ck["info"] | {"weights": str(w), "panel": str(panel)}, indent=1))
        print(f"  seed {s}: {len(pr['delta']):,} contract-steps -> {base}.npz", flush=True)
    for pr in preds[1:]:
        assert np.array_equal(pr["step"], preds[0]["step"]) and np.array_equal(pr["kind"], preds[0]["kind"]), "seeds' rows differ"
    ens = preds[0] | {"delta": np.mean([p["delta"] for p in preds], 0).astype(np.float32),
                      "corr": np.mean([p["corr"] for p in preds], 0).astype(np.float32)}
    base = out / f"{a.league}_fold{FOLD}_{oname}_ens5_seed0"
    np.savez_compressed(base.with_suffix(".npz"), **ens)
    base.with_suffix(".json").write_text(json.dumps({"horizons": pn.horizons, "seeds": SEEDS, "panel": str(panel)}, indent=1))
    print(f"  average of {len(SEEDS)} seeds -> {base}.npz")


if __name__ == "__main__":
    main()
