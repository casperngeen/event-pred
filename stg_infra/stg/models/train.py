"""Walk-forward training harness — mirrors ``agcrn_complexity.py`` T5.

8 expanding folds inside the in-sample block. Each fold: train on windows whose
target date is before the fold cut, hold out the next slice, with a
``PURGE_DAYS`` gap so a target's label window cannot straddle the boundary. Val
predictions are pooled across folds. The 2026 wall is never approached — inputs
come from the IS-only node panel.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from stg.models.tensors import (
    apply_feature_scaler, apply_label_scaler, fit_feature_scaler,
)
from stg.splits import PURGE_DAYS

PURGE = np.timedelta64(PURGE_DAYS, "D")


def _fold_cuts(dates: np.ndarray, n_folds: int, start_frac: float = 0.4) -> list:
    d = np.sort(np.unique(dates))
    return [d[min(len(d) - 1, int(len(d) * (start_frac + (1 - start_frac) * i / n_folds)))]
            for i in range(n_folds)] + [d[-1] + np.timedelta64(1, "D")]


def evaluate(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> dict:
    yt, yp = y[mask], yhat[mask]
    if yt.size == 0:
        return dict(n=0, mae=np.nan, rmse=np.nan, r2_vs_zero=np.nan, dir_acc=np.nan)
    err = yt - yp
    sse0 = float((yt ** 2).sum())
    nz = yp != 0
    da = float((np.sign(yp[nz]) == np.sign(yt[nz])).mean()) if nz.any() else np.nan
    return dict(
        n=int(yt.size),
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err ** 2).mean())),
        r2_vs_zero=float(1 - (err ** 2).sum() / sse0) if sse0 > 0 else np.nan,
        dir_acc=da,
    )


# ----------------------------------------------------------------- linear
def run_linear(make_model: Callable, win: dict, label_sd: np.ndarray, *,
               n_folds: int = 8) -> dict:
    dates = win["dates"]
    cuts = _fold_cuts(dates, n_folds)
    preds = np.zeros_like(win["y"])       # z-space (belief_z) / raw (atm_cents)
    y_eval = np.zeros_like(win["y"])
    covered = np.zeros(win["y"].shape[0], bool)
    for i in range(n_folds):
        tr = dates < (cuts[i] - PURGE)
        te = (dates >= cuts[i]) & (dates < cuts[i + 1])
        if tr.sum() < 30 or te.sum() == 0:
            continue
        wtr = _subset(win, tr)
        wtr["y"] = apply_label_scaler(wtr["y"], label_sd)
        model = make_model().fit(wtr)
        preds[te] = model.predict(_subset(win, te))
        y_eval[te] = apply_label_scaler(win["y"][te], label_sd)
        covered[te] = True
    return _pooled(win, preds, y_eval, covered)


# ----------------------------------------------------------------- torch
def run_torch(make_model: Callable, win: dict, label_sd: np.ndarray, *,
              n_folds: int = 8, seeds=(0,), max_epochs: int = 200,
              patience: int = 15, lr: float = 1e-3, wd: float = 1e-4,
              device: str = "cpu") -> dict:
    dates = win["dates"]
    cuts = _fold_cuts(dates, n_folds)
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        preds = np.zeros_like(win["y"])
        y_eval = np.zeros_like(win["y"])
        covered = np.zeros(win["y"].shape[0], bool)
        for i in range(n_folds):
            tr = dates < (cuts[i] - PURGE)
            te = (dates >= cuts[i]) & (dates < cuts[i + 1])
            if tr.sum() < 40 or te.sum() == 0:
                continue
            # inner early-stop split: last 15% of train by date
            tdates = np.sort(dates[tr])
            es_cut = tdates[int(len(tdates) * 0.85)]
            fit_m = tr & (dates < es_cut)
            es_m = tr & (dates >= es_cut)
            if fit_m.sum() < 20 or es_m.sum() < 5:
                fit_m, es_m = tr, tr

            mu, sd_f = fit_feature_scaler(win["Xs"][fit_m], win["Ms"][fit_m])
            sd_y = label_sd

            def prep(m):
                return (
                    torch.tensor(apply_feature_scaler(win["Xs"][m], mu, sd_f), device=device),
                    torch.tensor(win["Ms"][m], device=device),
                    torch.tensor(apply_label_scaler(win["y"][m], sd_y), dtype=torch.float32, device=device),
                    torch.tensor(win["ym"][m], device=device),
                )
            Xf, Mf, yf, ymf = prep(fit_m)
            Xe, Me, ye, yme = prep(es_m)
            Xt, Mt, _, _ = prep(te)

            model = make_model().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            best, best_state, bad = np.inf, None, 0
            for _ in range(max_epochs):
                model.train()
                opt.zero_grad()
                out = model(Xf, Mf).squeeze(-1)
                loss = _masked_huber(out, yf, ymf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                model.eval()
                with torch.no_grad():
                    ev = _masked_huber(model(Xe, Me).squeeze(-1), ye, yme).item()
                if ev < best - 1e-5:
                    best, best_state, bad = ev, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                preds[te] = model(Xt, Mt).squeeze(-1).cpu().numpy()
            y_eval[te] = apply_label_scaler(win["y"][te], sd_y)
            covered[te] = True
        per_seed.append(_pooled(win, preds, y_eval, covered))
    return _agg_seeds(per_seed)


# ----------------------------------------------------------------- helpers
def _masked_huber(pred, y, m):
    d = (pred - y)[m]
    return torch.nn.functional.huber_loss(d, torch.zeros_like(d), delta=1.0) if d.numel() else pred.sum() * 0


def _subset(win: dict, m: np.ndarray) -> dict:
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == m.shape[0] else v)
            for k, v in win.items()}


def _pooled(win: dict, preds: np.ndarray, y_eval: np.ndarray,
            covered: np.ndarray) -> dict:
    ym = win["ym"] & covered[:, None]
    return {**evaluate(y_eval, preds, ym), "_preds": preds, "_y": y_eval, "_mask": ym}


def _agg_seeds(runs: list[dict]) -> dict:
    keys = ("mae", "rmse", "r2_vs_zero", "dir_acc")
    out = {"n": runs[0]["n"], "n_seeds": len(runs)}
    for k in keys:
        vals = np.array([r[k] for r in runs], float)
        out[k] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals))
    out["_preds"] = np.mean([r["_preds"] for r in runs], axis=0)
    out["_y"] = runs[0]["_y"]
    out["_mask"] = runs[0]["_mask"]
    return out
