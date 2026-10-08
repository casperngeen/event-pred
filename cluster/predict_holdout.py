"""The frozen model's forecasts on a holdout panel, on the cluster (the laptop runs out of memory on MLB).

Same output as analysis/graph_2026_10/b15_holdout_predict.py, in the format the backtests read
(b5_longshort.rows, b7_maker_trades, b17_holdout_report), but one seed at a time (the average of the seeds
is kept as a running sum) and only the fields those scripts read:

    <out>/<L>_fold99_<NAMES[L]>_seed<s>.npz (+ .json)
    <out>/<L>_fold99_<NAMES[L]>_ens5_seed0.npz (+ .json)

The holdout panel is copied to the cluster only for this run, into holdout_panels/ (never panels/, which
training jobs read), and deleted with the outputs once they are copied back (cluster/README.md).

    python cluster/predict_holdout.py --league MLB --panel holdout_panels/MLB --weights results/ingame/final \\
        --out results/holdout
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# load the model file on its own (as train_ingame.py): the stg package pulls in polars
_spec = importlib.util.spec_from_file_location(
    "ingame", Path(__file__).resolve().parents[1] / "stg_infra" / "stg" / "models" / "ingame.py")
_ingame = importlib.util.module_from_spec(_spec)
sys.modules["ingame"] = _ingame
_spec.loader.exec_module(_ingame)

NAMES = {"MLB": "gru_d10_h30-60-120-1800_tp_pitch", "NFL": "gru_d10_h30-60-120-1800_tp", "NBA": "gru_d10_h30-60-120-1800_tp"}
# the frozen models (reports/holdout_spec.md; MLB: every pitch a play, section 9)
HZ = ["30s", "60s", "120s", "1800s"]
DELAY_S = 10.0
FOLD = 99
KEEP = ("step", "game", "kind", "value", "delta", "value_h", "y", "has_bid", "has_ask", "entry_bid", "entry_ask")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=["MLB", "NFL", "NBA"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--weights", default="results/ingame/final")
    ap.add_argument("--out", default="results/holdout")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    panel, out = Path(a.panel), Path(a.out)
    assert "panels" not in panel.parts, "keep holdout panels out of panels/ (training jobs read it)"
    pn = _ingame.Panel(panel, horizons=HZ)
    assert pn.manifest.get("holdout_panel") and pn.manifest["league"] == a.league, f"{panel}: not the {a.league} holdout panel"
    out.mkdir(parents=True, exist_ok=True)
    dev = _ingame.pick_device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]
    games = list(range(pn.n_games))
    print(f"{a.league}: {pn.n_games} games, device {dev}, seeds {seeds}", flush=True)
    total, first = None, None
    for s in seeds:
        w = Path(a.weights) / f"{a.league}_final_{NAMES[a.league]}_seed{s}.pt"
        model, ck = _ingame.load_model(w, pn, dev)
        assert ck["info"]["delay_s"] == DELAY_S and ck["info"]["seed"] == s, f"{w}: not the frozen settings"
        pr = _ingame.predict(model, pn, games, dev, stride=1, delay_s=DELAY_S)
        pr = {k: pr[k] for k in KEEP}
        if first is None:
            first = {k: v for k, v in pr.items() if k != "delta"}
            total = pr["delta"].astype(np.float64)
        else:
            assert np.array_equal(pr["step"], first["step"]) and np.array_equal(pr["kind"], first["kind"]), "seeds' rows differ"
            total += pr["delta"]
        base = out / f"{a.league}_fold{FOLD}_{NAMES[a.league]}_seed{s}"
        np.savez_compressed(base.with_suffix(".npz"), **pr)
        base.with_suffix(".json").write_text(json.dumps(ck["info"] | {"weights": str(w), "panel": str(panel)}, indent=1))
        print(f"  seed {s}: {len(pr['delta']):,} contract-steps -> {base}.npz", flush=True)
        del pr, model
    base = out / f"{a.league}_fold{FOLD}_{NAMES[a.league]}_ens5_seed0"
    np.savez_compressed(base.with_suffix(".npz"), **first, delta=(total / len(seeds)).astype(np.float32))
    base.with_suffix(".json").write_text(json.dumps({"horizons": HZ, "seeds": seeds, "panel": str(panel)}, indent=1))
    print(f"  average of {len(seeds)} seeds -> {base}.npz")


if __name__ == "__main__":
    main()
