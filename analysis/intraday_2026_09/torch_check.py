#!/usr/bin/env python
"""Is the AGCRN's predict-zero in ``models.py`` a finding or a training failure?

    venv/bin/python -W ignore analysis/intraday_2026_09/torch_check.py \
        > analysis/intraday_2026_09/out/torch_check.txt     # ~40 min; imports models.py's tensors

A first full-batch run of ``models.py`` scored both AGCRN rungs at R² = 0.0000:
full-batch Adam takes one step per epoch, so early stopping after ~20 epochs
left the zero-initialised head at predict-zero, and the training loss barely
moved. ``models.py`` now trains on minibatches of 32 releases. Questions, on two
walk-forward folds (the middle and the last), for the GRU and the adaptive AGCRN:

  1  does the training loss fall (can the model fit its training bars)?
  2  does the early-stop loss beat predict-zero, and by how much?
  3  do the batch size and learning rate change that?
  4  does training only on bars where the target trades in the window (80% of
     1 h labels are an exact 0 because nothing trades) change it?

Each run is scored on the fold's test releases (R² vs 0 on all bars and on
traded bars, both labels) next to rung 3 (linear path state) on the same bars.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "analysis/intraday_2026_09")
import models as mdl  # noqa: E402

SETTINGS = {
    "full batch (as first run)": dict(lr=3e-3, patience=15, max_epochs=150, batch=10_000),
    "default: batch 32, lr 1e-3": dict(),
    "batch 32, lr 3e-4, patience 20": dict(lr=3e-4, patience=20),
    "batch 32, lr 3e-3": dict(lr=3e-3),
    "default, traded bars only": dict(traded_only=True),
}
RUNGS = {k: mdl.MODELS[k][1] for k in ("4 GRU per target", "5a AGCRN adaptive")}
FOLDS = (4, 7)


def r2(pred, te, lab, traded):
    m = mdl.YM[lab] & te[:, None, None] & np.isfinite(pred) & (mdl.TRADED[lab] if traded else True)
    y = (mdl.Y[lab] / mdl.LSD[lab])[m]
    return 1 - ((y - pred[m]) ** 2).sum() / (y ** 2).sum()


def line(name, pred, te):
    return f"{name:44} " + " ".join(f"{r2(pred[lab], te, lab, t):>+9.4f}" for lab in mdl.LABELS
                                    for t in (False, True))


for f, tr, te, fit_m, es_m in mdl.folds():
    if f not in FOLDS:
        continue
    print(f"\n{'=' * 118}\nfold {f}: train {tr.sum()} releases (fit {fit_m.sum()}, early-stop {es_m.sum()}), "
          f"test {te.sum()}\n{'=' * 118}")
    print(f"{'':44} {'1h all':>9} {'1h trd':>9} {'24h all':>9} {'24h trd':>9}   (test R² vs 0)")
    print(line("3 linear path state", mdl.rung_ridge(tr, te, True), te), flush=True)
    for rung, make in RUNGS.items():
        for sname, kw in SETTINGS.items():
            pred, h = mdl.rung_torch(make, tr, te, fit_m, es_m, 0, **kw)
            tr_l, es_l = np.array(h["train"]), np.array(h["es"])
            b = h["best_epoch"]
            print(line(f"{rung.split()[0]} {sname}", pred, te)
                  + f"   | epochs {h['epochs']:>3}, best {b:>3}; train loss {tr_l[0]:.4f} → {tr_l[b]:.4f} "
                    f"(end {tr_l[-1]:.4f}); early-stop loss best {es_l[b]:.4f} vs zero {h['es_zero']:.4f} "
                    f"({100 * (h['es_zero'] - es_l[b]) / h['es_zero']:+.2f}%)", flush=True)
