"""Deployment panel: the training panel and the (already read) holdout panel joined end to end, so a final model
can be retrained on every game so far for paper trading. Not for testing: nothing trained on it can be scored on
these games.

Both panels come from the same code with the same fixed scales; this checks that their identity fields (features,
scales, horizons, play unit), array types and widths match, that no game is in both, and that every holdout game
starts after the last training step. Arrays are copied in order with the pointer arrays shifted:
    per game        game_L, game_fold (holdout games keep fold 99)
    game pointers   game_play_ptr, game_step_ptr  (+ the training panel's play / step counts)
    per play        plays_x, plays_t, plays_state
    per step        steps_t, steps_x;  step_node_ptr (+ the training panel's contract-row count)
    per row         nodes_x, nodes_kind, node_strike, node_side, node_sid, value, lo, hi, value_h, y
Training (cluster/train_ingame.py --final) keeps the most recent 15% of games for early stopping, which here are
the latest holdout games.

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b32_deploy_panel.py MLB NFL
    -> data/sports/derived/ingame/deploy/<MLB_pitch | NFL>/panel_v3/
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.models.ingame import PANEL_IDENTITY             # noqa: E402
from stg.splits import sports_holdout_unsealed          # noqa: E402

TRAIN = Path("data/sports/derived/ingame")
HOLD = Path("data/sports/_HOLDOUT_DO_NOT_USE/derived/ingame")
OUT = TRAIN / "deploy"
DIRS = {"MLB": "MLB_pitch", "NFL": "NFL", "NBA": "NBA"}
CHUNK = 2_000_000


def copy_cat(out: Path, a: np.ndarray, b: np.ndarray, shift: int = 0, drop_first_b: bool = False) -> None:
    b0 = 1 if drop_first_b else 0
    n = len(a) + len(b) - b0
    m = np.lib.format.open_memmap(out, mode="w+", dtype=a.dtype, shape=(n,) + a.shape[1:])
    for i in range(0, len(a), CHUNK):
        part = np.asarray(a[i:i + CHUNK])
        m[i:i + len(part)] = part
    for i in range(b0, len(b), CHUNK):
        j = len(a) + i - b0
        part = np.asarray(b[i:i + CHUNK])
        m[j:j + len(part)] = part + shift if shift else part
    m.flush()
    del m


def build(L: str) -> None:
    d = DIRS[L]
    pa, pb, po = TRAIN / d / "panel_v3", HOLD / d / "panel_v3", OUT / d / "panel_v3"
    ma, mb = (json.loads((p / "manifest.json").read_text()) for p in (pa, pb))
    bad = {k: (ma.get(k), mb.get(k)) for k in PANEL_IDENTITY if ma.get(k) != mb.get(k)}
    assert not bad, f"{L}: panels differ: {bad}"
    assert not mb.get("holdout_panel") is False and ma.get("holdout_panel") is False, "expected a training and a holdout panel"
    assert not set(ma["games"]) & set(mb["games"]), "a game in both panels"
    ld = lambda p, k: np.load(p / f"{k}.npy", mmap_mode="r")   # noqa: E731
    assert ld(pb, "steps_t")[0] > ld(pa, "steps_t")[-1], "holdout games must come after the training games"
    names = sorted(f.stem for f in pa.glob("*.npy"))
    assert names == sorted(f.stem for f in pb.glob("*.npy")), "different array sets"
    G, P, S, N = (len(ld(pa, k)) for k in ("game_fold", "plays_t", "steps_t", "nodes_kind"))
    po.mkdir(parents=True, exist_ok=True)
    for k in names:
        a, b = ld(pa, k), ld(pb, k)
        assert a.dtype == b.dtype and a.shape[1:] == b.shape[1:], f"{k}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        if k == "game_play_ptr":
            copy_cat(po / f"{k}.npy", a, b, shift=P, drop_first_b=True)
        elif k == "game_step_ptr":
            copy_cat(po / f"{k}.npy", a, b, shift=S, drop_first_b=True)
        elif k == "step_node_ptr":
            copy_cat(po / f"{k}.npy", a, b, shift=N, drop_first_b=True)
        else:
            assert len(a) in (G, P, S, N), f"{k}: length {len(a)} matches no unit"
            copy_cat(po / f"{k}.npy", a, b)
    man = ma | {"games": ma["games"] + mb["games"], "rows": ma["rows"] + mb["rows"], "holdout_panel": False, "deploy_panel": True,
                "deploy_sources": [str(pa), str(pb)], "built": dt.datetime.now(dt.timezone.utc).isoformat()}
    (po / "manifest.json").write_text(json.dumps(man, indent=1))
    # checks on the result
    gp, gs, sn = (np.load(po / f"{k}.npy") for k in ("game_play_ptr", "game_step_ptr", "step_node_ptr"))
    n_g = len(np.load(po / "game_fold.npy"))
    assert len(gp) == len(gs) == n_g + 1 == len(man["games"]) + 1
    assert gp[-1] == len(np.load(po / "plays_t.npy", mmap_mode="r")) and gs[-1] == len(np.load(po / "steps_t.npy", mmap_mode="r"))
    assert sn[-1] == len(np.load(po / "nodes_kind.npy", mmap_mode="r")) and (np.diff(gp) >= 0).all() and (np.diff(sn) >= 0).all()
    first = np.load(po / "steps_t.npy", mmap_mode="r")[gs[:-1]]
    assert first[len(ma["games"]):].min() > first[:len(ma["games"])].max(), "a holdout game starts before a training game"
    print(f"{L}: {len(ma['games'])} + {len(mb['games'])} = {n_g} games, {gp[-1]:,} plays, {gs[-1]:,} steps, {sn[-1]:,} contract rows -> {po}")


def main() -> None:
    assert sports_holdout_unsealed(), "reads the holdout panels: set STG_SPORTS_UNSEAL=final-test"
    for L in sys.argv[1:] or ["MLB", "NFL"]:
        build(L)


if __name__ == "__main__":
    main()
