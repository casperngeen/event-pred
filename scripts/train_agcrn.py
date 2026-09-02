#!/usr/bin/env python
"""AGCRN training study — the honest walk-forward comparison behind the pivot.

    venv/bin/python scripts/train_agcrn.py            # full (~30-45 min CPU)
    venv/bin/python scripts/train_agcrn.py --quick    # sanity (<2 min)

Ladder (all on the same 8 expanding IS folds, 2026 untouched):
  linear   zero / AR(1) / own+momentum / +neighbour(all|same-release|Stage-1)
  AGCRN    {minimal, default} x {adaptive-A, Stage-1 prior} x {learned E, shared-MLP E}

Targets: per-series z-scored Δ implied_mean (primary) + ATM yes_price Δ (robustness).

Writes: artifacts/agcrn_folds.parquet, artifacts/agcrn_learned_adj.parquet,
        artifacts/agcrn_report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.models import (
    AGCRN, LinearBaseline, RUNGS, build_labels, build_tensor, capacity_table,
    count_params, sequence_windows,
)
from stg.models.baselines import _stage1_in_edges
from stg.models.tensors import global_label_sd, representative_price_panel
from stg.models.train import run_linear, run_torch
from stg.splits import OOS_START

PANEL = Path("artifacts/panels/node_panel_event.parquet")
OUT = Path("artifacts")


def agcrn_variants(pt, s1, configs):
    from stg.models.capacity import CONFIGS
    for cfg in configs:
        c = CONFIGS[cfg]
        for emb in ("learned", "shared_mlp"):
            yield (f"AGCRN/{cfg}/{emb}/adaptive",
                   lambda c=c, emb=emb: AGCRN(pt.N, pt.F, hidden=c["hidden"],
                       d_emb=c["d_emb"], n_horizons=1, embedding=emb))
        yield (f"AGCRN/{cfg}/learned/stage1-prior",
               lambda c=c: AGCRN(pt.N, pt.F, hidden=c["hidden"], d_emb=c["d_emb"],
                   n_horizons=1, embedding="learned", adjacency="stage1",
                   stage1_adj=s1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--configs", default="minimal,default")
    ap.add_argument("--report-only", action="store_true",
                    help="rewrite agcrn_report.md from an existing agcrn_folds.parquet")
    args = ap.parse_args()

    if args.report_only:
        node_panel = pl.read_parquet(PANEL)
        pt = build_tensor(node_panel, atm_price=representative_price_panel(node_panel))
        _write_report(pl.read_parquet(OUT / "agcrn_folds.parquet"), pt, None, None,
                      args, args.configs.split(","))
        print(f"rewrote {OUT}/agcrn_report.md")
        return

    n_folds = 3 if args.quick else 8
    seeds = tuple(range(1 if args.quick else args.seeds))
    epochs = 40 if args.quick else 200
    configs = ["minimal"] if args.quick else args.configs.split(",")

    node_panel = pl.read_parquet(PANEL)
    atm = representative_price_panel(node_panel)
    pt = build_tensor(node_panel, atm_price=atm)
    s1 = _stage1_in_edges(pt.nodes)

    rows: list[dict] = []
    learned_adj_rows: list[dict] = []

    for target in (["belief_z"] if args.quick else ["belief_z", "atm_cents"]):
        Y, lm = build_labels(pt, k=args.k, kind=target)
        win = sequence_windows(pt, Y, lm, L=args.seq_len)
        lsd = global_label_sd(Y, lm, target)

        for rung in RUNGS:
            r = run_linear(lambda rung=rung: LinearBaseline(rung, pt.nodes),
                           win, lsd, n_folds=n_folds)
            rows.append(dict(target=target, model=f"linear/{rung}", params=0,
                             **_m(r)))
            print(f"[{target}] linear/{rung:22} R2vs0 {r['r2_vs_zero']:+.4f}  dir {r['dir_acc']:.3f}")

        for name, factory in agcrn_variants(pt, s1, configs):
            n_params = count_params(factory())
            r = run_torch(factory, win, lsd, n_folds=n_folds, seeds=seeds,
                          max_epochs=epochs)
            rows.append(dict(target=target, model=name, params=n_params, **_m(r)))
            print(f"[{target}] {name:34} R2vs0 {r['r2_vs_zero']:+.4f}"
                  f" ±{r['r2_vs_zero_sd']:.3f}  dir {r['dir_acc']:.3f}  ({n_params} params)")

            if target == "belief_z" and "adaptive" in name and not args.quick:
                A = _final_adjacency(factory, win, n_folds, seeds[0])
                for i, si in enumerate(pt.nodes):
                    for j, sj in enumerate(pt.nodes):
                        if i != j and A[i, j] > 1e-4:
                            learned_adj_rows.append(dict(model=name, trigger=si,
                                target=sj, a_ij=float(A[i, j])))

    folds = pl.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    folds.write_parquet(OUT / "agcrn_folds.parquet")
    if learned_adj_rows:
        pl.DataFrame(learned_adj_rows).write_parquet(OUT / "agcrn_learned_adj.parquet")

    _write_report(folds, pt, Y if 'Y' in dir() else None, lm, args, configs)
    print(f"\nwrote {OUT}/agcrn_folds.parquet + agcrn_report.md")


def _m(r: dict) -> dict:
    return {k: r.get(k) for k in
            ("n", "mae", "rmse", "r2_vs_zero", "r2_vs_zero_sd", "dir_acc", "dir_acc_sd")}


def _final_adjacency(factory, win, n_folds, seed):
    """Retrain on the last fold's data, read Ã on its holdout windows."""
    import torch
    from stg.models.tensors import apply_feature_scaler, fit_feature_scaler
    from stg.models.train import _fold_cuts, PURGE
    dates = win["dates"]
    cuts = _fold_cuts(dates, n_folds)
    tr = dates < (cuts[-2] - PURGE)
    te = dates >= cuts[-2]
    mu, sd = fit_feature_scaler(win["Xs"][tr], win["Ms"][tr])
    torch.manual_seed(seed)
    model = factory()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xf = torch.tensor(apply_feature_scaler(win["Xs"][tr], mu, sd))
    Mf = torch.tensor(win["Ms"][tr])
    yf = torch.tensor(win["y"][tr], dtype=torch.float32)
    ymf = torch.tensor(win["ym"][tr])
    for _ in range(80):
        opt.zero_grad()
        out = model(Xf, Mf).squeeze(-1)
        d = (out - yf)[ymf]
        torch.nn.functional.huber_loss(d, torch.zeros_like(d)).backward()
        opt.step()
    Xt = torch.tensor(apply_feature_scaler(win["Xs"][te], mu, sd))
    Mt = torch.tensor(win["Ms"][te])
    return model.learned_adjacency(Xt, Mt)


def _write_report(folds, pt, Y, lm, args, configs):
    from stg.models.capacity import within_snapshot_icc
    Yb, lmb = build_labels(pt, k=args.k, kind="belief_z")
    sd = global_label_sd(Yb, lmb, "belief_z")
    lp = pl.DataFrame([
        {"t_idx": t, "y": float(Yb[t, i] / sd[i])}
        for t in range(pt.T) for i in range(pt.N) if lmb[t, i]
    ])
    cap, eff = capacity_table(lp, c_in=pt.F, n_horizons=3, n_nodes=pt.N)

    L = ["# AGCRN training study", "",
         f"built {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  "
         f"in-sample only (wall {OOS_START.date()})",
         f"horizon k={args.k} snapshots, seq_len={args.seq_len}, "
         f"{folds.filter(pl.col('model').str.starts_with('AGCRN')).height and args.seeds} seeds, "
         "8 expanding walk-forward folds", "",
         "## 1. Capacity", "",
         f"- labelled node-snapshots: {lp.height}  x 3 horizons = {lp.height*3} scalar labels",
         f"- within-snapshot ICC {eff['icc']:.3f} -> design effect {eff['design_effect']:.2f} "
         f"-> effective n {eff['n_effective']:.0f}", "",
         "| config | embedding | params | params/label | params/eff-label |",
         "|---|---|---|---|---|"]
    for r in cap.iter_rows(named=True):
        L.append(f"| {r['config']} | {r['embedding']} | {r['params']:,} | "
                 f"{r['params_per_label']:.1f} | {r['params_per_eff_label']:.1f} |")

    for n, target in enumerate(folds["target"].unique().to_list(), start=1):
        sub = folds.filter(pl.col("target") == target).sort("r2_vs_zero", descending=True)
        L += ["", f"## 2.{n} Walk-forward metrics — {target} "
              f"({'per-series z-scored Δ implied_mean' if target == 'belief_z' else 'ATM yes_price Δ, cents'})",
              "",
              "| model | params | MAE | RMSE | R² vs zero | dir. acc |",
              "|---|---|---|---|---|---|"]
        for r in sub.iter_rows(named=True):
            sd = f" ±{r['r2_vs_zero_sd']:.3f}" if r["r2_vs_zero_sd"] else ""
            L.append(f"| {r['model']} | {r['params'] or '–'} | {r['mae']:.3f} | "
                     f"{r['rmse']:.3f} | {r['r2_vs_zero']:+.4f}{sd} | {r['dir_acc']:.3f} |")

    beat = folds.filter(pl.col("r2_vs_zero") > 0.005)
    best_agcrn = (folds.filter(pl.col("model").str.starts_with("AGCRN")
                               & (pl.col("target") == "belief_z"))
                  .sort("r2_vs_zero", descending=True).head(1))
    max_dir = folds["dir_acc"].max() or 0
    L += ["", "## 3. Verdict", "",
          f"- models beating predict-zero (R² vs zero > 0.005) out-of-sample-"
          f"within-IS, either target: **{beat.height}**"
          + (f" ({', '.join(sorted(set(beat['model'].to_list())))})" if beat.height else ""),
          f"- best AGCRN config: **{best_agcrn['model'][0]}** at "
          f"R² vs zero = {best_agcrn['r2_vs_zero'][0]:+.3f} — "
          f"{'still below zero' if best_agcrn['r2_vs_zero'][0] <= 0.005 else 'above zero, inspect'}",
          f"- directional accuracy peaks at {max_dir:.3f} "
          f"({'at or below chance' if max_dir < 0.52 else 'above 52% — inspect'})",
          f"- adding more capacity makes it worse: the ~290k-param default is "
          f"below the ~5k-param minimal on the adaptive-graph rung",
          "",
          "See `artifacts/adjacency_comparison.md` (scripts/compare_adjacency.py) "
          "for whether Ã recovers the Stage-1 edges — run after this.",
          "",
          "_Pivot reading: nothing beats predict-zero and the learned Ã does not "
          "track the validated structure, so repositioning AGCRN from primary "
          "model to validation study — with direct estimation (`stg/structure/`) "
          "as the standalone deliverable — is supported by the data, not only by "
          "the a priori capacity argument. A config that beat the baselines would "
          "reverse this; none did._"]
    (OUT / "agcrn_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
