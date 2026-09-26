"""
score_forecast.py

SCORE A TRAINED FORECAST CHECKPOINT ON A HELD-OUT SPLIT. ONCE.

train_forecast.py only ever evaluates val, because val is what it selects
and early-stops on. That makes the val number a SELECTION statistic, not
an estimate: taking the minimum over N epochs of a metric that swings
+/-0.13 between epochs is optimistically biased by roughly that swing. The
MECE run's val_mece_ratio went 0.836, 0.964, 0.824, 0.896, 0.796, 0.807 --
the 0.796 that got checkpointed is the best of thirteen noisy draws, not a
converged value.

So the reported result has to come from a split the selection never
touched. This script does that and nothing else: load best.pt, rebuild the
model, score one split, print.

WHY THERE IS NO --epochs, NO TUNING, AND NO RETRY. Every time you look at
test and then change something, test becomes part of the selection loop and
stops being held out. Running this twice with different flags and keeping
the better number is the same bias the val split already has, just hidden.
Run it once, on the checkpoint val already chose, and write down whatever
it says.

WHAT IT PRINTS, all against baselines computed on the same positions:
  - node loss vs persistence, POOLED and split by regime
  - per-horizon ratios
  - per-mechanism ratios vs persistence AND vs the optimal shrinkage
    oracle (alpha fitted on the scored split, so deliberately optimistic)

READ THE 'moved' STRATUM, NOT THE POOLED NODE RATIO. ~68% of supervised
positions do not move between t and t+h. On those, persistence has an MSE
of exactly zero and any model output scores worse, so no forecast can win
there and including them guarantees a pooled ratio above 1.0 however good
the model is where it matters. On the MECE run the pooled node ratio of
1.008x decomposes exactly into 0.991x on 497,865 movers and an unavoidable
0.00018 penalty on 1,048,605 flats.

READ THE SHRINKAGE COLUMN, NOT THE PERSISTENCE ONE. A model that merely
discovers "this quantity decays toward its mean" beats persistence without
having learned anything about this market. The shrinkage oracle already
does that optimally, so it is the baseline a claim has to survive. On the
MECE run: 0.807x vs persistence but 0.862x vs shrink(alpha=0.870).

Standardization statistics were fitted on TRAIN and are registered buffers,
so they travel inside the checkpoint's state_dict. Nothing is refitted here
and no training data is touched -- which is the point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.forecast_head import ResidualForecastHead  # noqa: E402
from model.forecast_objective import (  # noqa: E402
    MechanismForecastObjective, format_horizons, format_mechanisms, format_strata,
)
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone  # noqa: E402
from train_forecast import PILOT_MONTHS, _accum, _chunk_loss, _finish  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_mece/best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--supervise", choices=["active", "all"], default="active")
    ap.add_argument("--w-node", type=float, default=1.0)
    ap.add_argument("--w-ladder", type=float, default=1.0)
    ap.add_argument("--w-mece", type=float, default=1.0)
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_absolute():
        for c in (_THIS_DIR / ck_path, _REPO_ROOT / ck_path, Path.cwd() / ck_path):
            if c.exists():
                ck_path = c
                break
    if not ck_path.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    cfg = ck["config"]
    horizons = ck.get("horizons") or list(getattr(cfg, "horizons", (1, 2, 3, 6, 12)))

    print("=" * 96)
    print(f"SCORING {ck_path.name} ON THE {args.split.upper()} SPLIT")
    print("=" * 96)
    print(f"  checkpoint epoch : {ck.get('epoch')}")
    print(f"  selected by      : {ck.get('selected_by')}  "
          f"(val_loss={ck.get('val_loss')}, val_ratio={ck.get('val_ratio')})")
    print(f"  horizons         : {horizons}")
    if args.split == "test":
        print()
        print("  This is the held-out split. Whatever it prints is the result --")
        print("  including if it is worse than val. Changing a setting and rerunning")
        print("  folds test into the selection loop and there is no second held-out")
        print("  split left to recover with.")
    print()

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _REPO_ROOT / cache
    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS, cache),
                               verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=cfg.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks in months {args.months or PILOT_MONTHS}")
    print(f"  {len(chunks)} chunk(s), {sum(c.length for c in chunks)} snapshot(s)\n")

    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=cfg.chunk_len)
    head = ResidualForecastHead(embed_dim=cfg.embed_dim, horizons=horizons)
    # strict=True on purpose: a silently-missing buffer here would mean
    # scoring with unfitted standardization, which looks like a bad model
    # rather than like a loading bug.
    model.load_state_dict(ck["model"], strict=True)
    head.load_state_dict(ck["head"], strict=True)
    model.eval()
    head.eval()

    # Both mechanisms are scored whatever their training weights were --
    # this is measurement, not optimisation, so there is no reason to
    # leave one of them blank.
    objective = MechanismForecastObjective(
        horizons=horizons, huber_beta=getattr(cfg, "huber", 0.0),
        w_node=args.w_node, w_ladder=args.w_ladder, w_mece=args.w_mece,
        monitor_unweighted=True)

    tot_l = tot_p = 0.0
    tot_n = 0
    node_l = 0.0
    acc_h, acc_m, acc_s = {}, {}, {}
    with torch.no_grad():
        for pos, c in enumerate(chunks):
            ct = store.materialize_chunk(c)
            out = _chunk_loss(model, head, objective, ct,
                              f"{args.split} {pos + 1}/{len(chunks)}", args.supervise)
            if out["n"]:
                tot_l += float(out["loss"]) * out["n"]
                tot_p += out["persistence"] * out["n"]
                node_l += out["node_loss"] * out["n"]
                tot_n += out["n"]
                _accum(acc_h, out["per_horizon"])
                _accum(acc_m, out.get("mechanisms", {}))
                _accum(acc_s, out.get("strata", {}))
            del ct

    if not tot_n:
        raise SystemExit("nothing supervised on this split")

    per_h, mech, strata = _finish(acc_h), _finish(acc_m), _finish(acc_s)
    node = node_l / tot_n
    pers = tot_p / tot_n
    ratio = node / pers if pers > 0 else float("nan")

    print()
    print("=" * 96)
    print(f"{args.split.upper()} RESULT")
    print("=" * 96)
    print(f"  node (pooled)  : {node:.5f} vs persist {pers:.5f} = {ratio:.3f}x  "
          f"(n={tot_n:,})")
    print(f"  combined obj   : {tot_l / tot_n:.5f}   (no single baseline; "
          f"reported for comparability with training logs only)")
    if per_h:
        print(f"  per-horizon    : {format_horizons(per_h)}")
    if strata:
        print(f"  by regime      : {format_strata(strata)}")
    if mech:
        print(f"  mechanisms     : {format_mechanisms(mech)}")

    print()
    print("-" * 96)
    moved = strata.get("moved") or {}
    flat = strata.get("flat") or {}
    if moved:
        n_m, n_f = moved.get("n", 0), flat.get("n", 0)
        share = 100.0 * n_f / max(n_m + n_f, 1)
        print(f"  HEADLINE (node): {moved['ratio']:.3f}x on the {n_m:,} positions that "
              f"MOVED.")
        print(f"  The pooled {ratio:.3f}x includes {n_f:,} flat positions ({share:.0f}%) "
              f"where persistence")
        print(f"  is exact by construction and no forecast can do better than tie.")
    for name, v in sorted(mech.items()):
        vs = v.get("vs_shrink")
        if vs is not None and vs == vs:
            verdict = ("BEATS the shrinkage oracle" if vs < 1.0
                       else "does NOT beat the shrinkage oracle")
            print(f"  {name.upper():<7}: {vs:.3f}x vs shrink(alpha={v['alpha']:.3f}) "
                  f"-- {verdict}.  ({v['ratio']:.3f}x vs persistence, n={v['n']:,})")
        else:
            print(f"  {name.upper():<7}: {v['ratio']:.3f}x vs persistence, n={v['n']:,} "
                  f"(no shrinkage control available)")
    print("-" * 96)
    print()
    print("  An alpha well below 1 means the quantity genuinely mean-reverts and there")
    print("  is something for a forecaster to find; an alpha near 1 means it is a random")
    print("  walk and beating persistence on it would be surprising. That difference --")
    print("  MECE alpha ~ 0.87 against ladder alpha ~ 0.99 -- is the structural finding,")
    print("  and it follows from the constraints themselves: a MECE basket is pinned to")
    print("  a LEVEL (the legs must sum to 1), while a ladder pair is only bounded by an")
    print("  INEQUALITY (gap <= 0), which an entire half-line satisfies. There is no")
    print("  level for a ladder gap to revert to, so there is nothing to forecast.")


if __name__ == "__main__":
    main()