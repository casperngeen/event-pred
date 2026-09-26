"""
train_forecast.py

Trains the STGAT to FORECAST prices at t+h, so that dislocations become
predictable quantities rather than structurally impossible ones.

WHAT CHANGED FROM run_real_training.py, AND WHY EACH CHANGE WAS FORCED

  1. TARGET. Reconstruction at t -> forecast at t+h. The old objective
     asked "what is this leg worth now"; when the printed price is a
     stale quote, the training LABEL is that stale quote, so the model was
     rewarded for reproducing the very lag it was meant to detect. That is
     why the leg's own last price beats the trained model 4.4x on test.

  2. HEAD. MeceOutputHead's grouped softmax forces predicted legs to sum
     to exactly $1.00, making a predicted dislocation structurally
     impossible and the model's basket deviation identically equal to the
     naive rule's. LadderOutputHead's fair_a >= fair_b guarantee does the
     same for monotonicity violations. Both are replaced by one
     unconstrained ResidualForecastHead; the mechanism stays in the GRAPH
     (MECE and ladder edges), which is where a spatio-temporal attention
     model is supposed to carry it.

  3. PARAMETERISATION. predicted = observed_now + delta, with the final
     layer zero-initialised, so the model STARTS as the persistence
     baseline and can only move away from it by reducing loss.

  4. REPORTING. Persistence loss is computed on identical positions and
     printed every epoch. The ratio model/persistence is the number that
     decides whether the run is worth anything; the raw loss is not.

WHAT IS DELIBERATELY UNCHANGED: the backbone (type-specific projection ->
two-channel spatial attention -> causal temporal attention), the monthly
bundle store, the chunking, and the train-only feature standardization.
The existing reconstruction pipeline is untouched -- this is a parallel
entry point, not a replacement, so the earlier results stay reproducible.

Usage:
    python train_forecast.py --months 2025-05 2025-06 2025-07 2025-08 2025-09 \\
        --epochs 30 --horizons 1 2 3 6 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges, describe_chunks  # noqa: E402
from model.forecast_head import ResidualForecastHead  # noqa: E402
from model.forecast_objective import (  # noqa: E402
    MechanismForecastObjective, format_horizons, format_mechanisms, format_strata,
    _shrinkage_mse,
)
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.run_guards import (  # noqa: E402
    BaselineFingerprint, check_drop_edges, check_history_columns, check_hub_slot,
    check_ladder_persistence_decomposition, check_month_schema,
    check_split_integrity, require_finite_selection,
)
from model.train import (  # noqa: E402
    STGATBackbone, TrainingConfig, _compute_train_feature_stats, _rss_mb, true_prices,
)

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]


VOLUME_SLOT = 4      # window_volume, see model/data_validation.py
SLOT_LEGS_TOTAL = 3  # basket hub's leg count, see stg/nodes/kalshi.py

# Mechanisms that have columns in history.csv. This tuple is the single
# source of truth for BOTH the header and every row, so the two cannot
# drift -- and check_history_columns fails the run if a mechanism is
# computed and printed without appearing here. ladder_moved spent a whole
# run in that state: it was the metric being selected on, and the chosen
# epoch's value was unrecoverable once the terminal closed.
MECH_COLUMNS = ("ladder", "ladder_moved", "mece")
_MECH_FIELDS = (("", "model"), ("_persist", "persistence"), ("_ratio", "ratio"),
                ("_n", "n"), ("_shrink", "shrink"), ("_alpha", "alpha"),
                ("_vs_shrink", "vs_shrink"))


def _supervision_masks(features, mask, adj, mode):
    """Which (t, node) positions are allowed to supervise the loss.

    WHY THIS EXISTS, AND WHY 'active' IS THE DEFAULT. Measured on this
    dataset, 90% of consecutive snapshot observations show NO price change
    at all, and most ticker nodes belong to no MECE basket or ladder pair.
    Supervising every observed position therefore builds a training
    distribution in which the loss-minimising answer is literally
    delta = 0 -- the persistence baseline. A model trained that way
    converges to persistence and then gets reported as "not beating
    persistence", which is a statement about the sampling, not the market.

    'active' keeps a position only when the leg is:
      - a ticker (basket hubs carry sum_cents in slot 0, not a price);
      - a MEMBER of a MECE basket or ladder pair this snapshot, since
        nothing else is what this project is about;
      - LIVE, i.e. window_volume > 0, so its printed price reflects a
        trade in this window rather than a day-old quote. This is read at
        t and at t+h, both observable, so it is a filter and not
        look-ahead.

    Returns (eligible, mask_sup): eligible gates the node term, mask_sup
    replaces `mask` for target selection so stale endpoints cannot supply
    a target in either the node or the mechanism terms.
    """
    is_ticker = features[..., -1] == 0.0
    if mode == "all":
        return is_ticker, mask

    T, N = mask.shape
    member = torch.zeros(T, N, dtype=torch.bool, device=mask.device)
    for key in ("mece_leg_to_basket", "ladder_monotonic"):
        lst = adj.get(key)
        if lst is None:
            continue
        for t in range(T):
            e = lst[t]
            if e.edge_index.numel() == 0:
                continue
            member[t, e.edge_index[0]] = True
            if key == "ladder_monotonic":
                member[t, e.edge_index[1]] = True

    live = features[..., VOLUME_SLOT] > 0.0 if features.shape[-1] > VOLUME_SLOT \
        else torch.ones_like(mask)
    return is_ticker & member, mask & live


def _chunk_loss(model, head, objective, ct, label="", supervise="active",
                drop_edges=()):
    """One chunk: backbone -> head -> horizon loss. Shared by train and
    val so the two cannot drift apart in what they compute.

    ``drop_edges`` removes edge types from the graph the BACKBONE reads,
    while leaving the loss terms untouched. That separation is the whole
    point: it asks whether an edge type earns its place as CONTEXT for
    other predictions, independently of whether its own derived quantity
    is forecastable.

    This matters for the ladder. Its gap is not forecastable (see
    claude/ladder-conclusion-overturned.md) -- but a ladder edge still
    tells the model that two markets are two views of the same underlying,
    which is information about each leg's price whether or not the gap
    between them can be predicted. Dropping the edges and retraining is
    the only way to find out; the alternative is asserting that the graph
    structure helps because it ought to.
    """
    features, mask = ct["features"], ct["mask"]
    prices = true_prices(features)

    adj = ct["adjacency_by_type"]
    # The BACKBONE still sees everything -- stale neighbours are useful
    # context even when they are useless targets. Only the LOSS is
    # restricted.
    eligible, mask_sup = _supervision_masks(features, mask, adj, supervise)
    adj_in = {k: v for k, v in adj.items() if k not in drop_edges} if drop_edges else adj
    h_final = model(features, mask, adj_in)
    pred = head(h_final, prices)                     # (T, N, H)
    # Ladder and MECE structure enters TWICE and deliberately: as edges the
    # spatial attention reads (how a shock propagates), and here as direct
    # supervision of the derived quantity (gap, basket sum). The first
    # teaches the model the relationship; the second makes it optimise the
    # number the strategy is actually read from.
    return objective.compute_loss(
        pred, prices, mask_sup, eligible,
        ladder_adj=adj.get("ladder_monotonic"),
        mece_adj=adj.get("mece_leg_to_basket"))


def _accum(dst, src, key_n="n"):
    """Merge per-horizon / per-mechanism stats across chunks, weighting by
    the number of supervised positions. Overwriting instead of merging is
    how an earlier version reported only the LAST chunk -- a 2-day
    fragment -- as if it were the whole split."""
    for k, v in (src or {}).items():
        n = v.get(key_n, 0)
        if not n:
            continue
        d = dst.setdefault(k, {"n": 0, "model": 0.0, "persistence": 0.0,
                               "sxx": 0.0, "sxy": 0.0, "syy": 0.0, "_shrink": False})
        d["n"] += n
        d["model"] += v["model"] * n
        d["persistence"] += v["persistence"] * n
        # Shrinkage sums add directly (they are sums, not means). Keeping them
        # lets _finish refit one alpha over the whole split rather than
        # averaging per-chunk alphas, which no single scalar could achieve.
        if "sxx" in v:
            d["_shrink"] = True
            d["sxx"] += v["sxx"]
            d["sxy"] += v["sxy"]
            d["syy"] += v["syy"]


def _finish(dst):
    out = {}
    for k, d in dst.items():
        if not d["n"]:
            continue
        m = d["model"] / d["n"]
        p = d["persistence"] / d["n"]
        row = {"n": d["n"], "model": m, "persistence": p,
               "ratio": m / p if p > 0 else float("nan")}
        if d.get("_shrink") and d["sxx"] > 0:
            shrink, alpha = _shrinkage_mse(d["sxx"], d["sxy"], d["syy"], d["n"])
            row.update(shrink=shrink, alpha=alpha,
                       vs_shrink=(m / shrink if shrink > 0 else float("nan")))
        out[k] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk-len", type=int, default=168)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--huber", type=float, default=0.0,
                    help="smooth-L1 beta; 0 = plain MSE. Price jumps are heavy tailed, so "
                         "a small value (e.g. 0.02) stops a few large moves dominating.")
    ap.add_argument("--w-node", type=float, default=1.0,
                    help="weight on per-node price forecasting.")
    ap.add_argument("--w-ladder", type=float, default=1.0,
                    help="weight on the LADDER GAP at t+h. Set 0 to train on node prices "
                         "only and derive gaps at inference.")
    ap.add_argument("--w-mece", type=float, default=1.0,
                    help="weight on the MECE basket dislocation (sum of legs - 1) at t+h.")
    ap.add_argument("--drop-edges", nargs="*", default=[],
                    choices=["ladder_monotonic", "mece_leg_to_basket",
                             "mece_basket_to_leg"],
                    help="ABLATION: remove these edge types from the graph the backbone "
                         "reads, leaving the loss unchanged. Use it to ask whether an "
                         "edge type helps as CONTEXT even when its own derived quantity "
                         "is not forecastable -- the open question for the ladder. "
                         "These are the exact keys model/spatial_attention.py reads; "
                         "dropping the whole spatial mechanism needs ALL THREE, since "
                         "the MECE channel is built from two directed spoke sets.")
    ap.add_argument("--ladder-supervise", choices=["all", "moved", "weighted"],
                    default="all",
                    help="which ladder pairs the LOSS is computed on. 'moved' keeps only "
                         "pairs where a leg repriced between t and t+h -- ~10%% of them. "
                         "The other 90%% have gap(t+h)==gap(t) exactly, so training on "
                         "them scores the model on cases whose answer is 'unchanged'. "
                         "NOTE: 'moved' is an ORACLE condition (not knowable at t), so it "
                         "is legitimate for training but its metric is NOT deployable. "
                         "'weighted' keeps ALL pairs but upweights movers by --moved-weight, which is the safer choice: a model trained on movers ALONE never sees a pair that stays put and then predicts movement everywhere. "
                         "Both populations are reported whatever this is set to.")
    ap.add_argument("--moved-weight", type=float, default=9.0,
                    help="weight on moved ladder pairs under --ladder-supervise weighted. 9.0 roughly equalises the two populations at the measured ~10%% move rate.")
    ap.add_argument("--monitor-unweighted", action="store_true",
                    help="also SCORE mechanisms whose weight is 0, without training on "
                         "them. Without this a --w-ladder 0 run logs nan for ladder and "
                         "cannot show whether optimising one mechanism degraded the "
                         "other. Costs a forward-only pass over that mechanism's pairs.")
    ap.add_argument("--supervise", choices=["active", "all"], default="active",
                    help="'active' (default) supervises only mechanism-member legs that "
                         "actually traded in the window. 'all' reverts to every observed "
                         "ticker, where ~90% of targets are 'unchanged' and the loss "
                         "optimum is the persistence baseline itself.")
    ap.add_argument("--select-on", choices=["ladder", "mece", "node", "total"],
                    default="ladder",
                    help="which val quantity decides the best checkpoint AND drives early "
                         "stopping. Set it to the mechanism you are actually studying: "
                         "selecting on one that cannot improve stops the run while another "
                         "is still descending, and saves the wrong epoch.")
    ap.add_argument("--baseline-ref", default=None,
                    help="path to another run's baselines.json. The persistence and "
                         "shrinkage baselines depend only on the data, so two runs "
                         "scored on the same split MUST produce identical ones; this "
                         "asserts it at startup instead of leaving a paired ablation "
                         "to be verified by eye afterwards. It is also what catches a "
                         "training weight leaking into the REPORTED metric, which has "
                         "happened once and survived three runs.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default="checkpoints_forecast")
    args = ap.parse_args()

    months = args.months or PILOT_MONTHS
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache)
    if not cache_dir.is_absolute():
        cache_dir = _REPO_ROOT / cache_dir
    store = MonthlyBundleStore(build_month_paths(months, cache_dir), verbose=True)
    print(store.summary(), flush=True)
    check_month_schema(store)

    cfg = TrainingConfig(embed_dim=args.embed_dim, n_heads=args.n_heads,
                         chunk_len=args.chunk_len, lr=args.lr, epochs=args.epochs,
                         seed=args.seed, patience=args.patience)

    ranges = build_split_ranges(store.timestamps)
    chunks = chunk_ranges(ranges, chunk_len=cfg.chunk_len, min_chunk_len=cfg.min_chunk_len)
    print("Chunked into:\n" + describe_chunks(chunks), flush=True)
    # Before anything is loaded or trained: no overlap, no chunk straddling a
    # split, and strictly forward-in-time train -> val -> test. This is the
    # guard whose failure would invalidate every result at once rather than
    # one table, and it is the one that breaks when MONTHS ARE ADDED --
    # data_windows.py splits by string comparison against TRAIN_END/VAL_END,
    # which appending months does not move.
    check_split_integrity(store.timestamps, chunks)
    train_chunks = [c for c in chunks if c.split == "train"]
    val_chunks = [c for c in chunks if c.split == "val"]
    if not train_chunks:
        raise SystemExit("No train chunks -- check months against data_windows.py boundaries.")
    if not val_chunks:
        print("\n*** WARNING: no validation chunks. Nothing can be selected or early "
              "stopped; the final epoch will be saved and labelled as such. ***\n", flush=True)

    max_h = max(args.horizons)
    if cfg.chunk_len <= max_h:
        raise SystemExit(f"--chunk-len {cfg.chunk_len} must exceed the longest horizon "
                         f"({max_h}), or no position can be supervised.")

    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=cfg.chunk_len)
    print("Fitting feature standardization on TRAIN chunks only ...", flush=True)
    t0 = time.time()
    feat_mean, feat_std, n_obs = _compute_train_feature_stats(
        store.materialize_chunk, train_chunks, store.feature_width)
    model.fit_standardization(feat_mean, feat_std)
    print(f"  fitted on {n_obs:,} observed positions in {time.time() - t0:.1f}s", flush=True)

    head = ResidualForecastHead(embed_dim=cfg.embed_dim, horizons=args.horizons,
                                dropout=args.dropout)
    objective = MechanismForecastObjective(
        horizons=args.horizons, huber_beta=args.huber,
        w_node=args.w_node, w_ladder=args.w_ladder, w_mece=args.w_mece,
        monitor_unweighted=args.monitor_unweighted,
        ladder_supervise=args.ladder_supervise,
        moved_weight=args.moved_weight)
    # VERIFY THE ABLATION ACTUALLY ABLATES, against the CACHE rather than
    # against the hand-written argparse choices list -- those are different
    # things and only the first is evidence. Same probe chunk also checks the
    # basket hub's leg-count slot, which every coverage fraction divides by.
    _probe = store.materialize_chunk(train_chunks[0])
    check_drop_edges(args.drop_edges, _probe["adjacency_by_type"].keys())
    check_hub_slot(_probe, SLOT_LEGS_TOTAL)
    del _probe
    print(f"Objective weights: node={args.w_node} ladder={args.w_ladder} "
          f"mece={args.w_mece} | supervise={args.supervise} "
          f"| select_on={args.select_on} "
          f"| monitor_unweighted={args.monitor_unweighted}\n", flush=True)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                                 lr=cfg.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = _REPO_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hist_path = ckpt_dir / "history.csv"
    # EVERY ratio goes in the file, not just the node one. An earlier
    # version logged node columns only, so a run where node loss rose
    # while the mechanism terms fell -- the intended trade-off under a
    # multi-task objective -- was indistinguishable in history.csv from a
    # run that was simply diverging.
    #
    # The SHRINKAGE columns matter more than the persistence ones and were
    # missing until now. Persistence is the weak baseline: any model that
    # merely learns "the derived quantity decays toward its mean" beats it
    # without having learned anything about this market. The optimal
    # shrinkage control (alpha fitted on the scored split, so deliberately
    # an oracle) is the one a result has to survive, and it was being
    # computed every epoch, printed to stdout, and then dropped on the
    # floor -- leaving history.csv, the file any write-up actually plots,
    # showing only the flattering comparison.
    # THE HEADER IS NOT WRITTEN HERE ANY MORE. It is derived from the row
    # dict at epoch 0, so header and row are the same object and cannot
    # disagree -- the earlier version maintained two long parallel literals
    # by hand, and a metric that was computed, printed, and selected on
    # (ladder_moved) was for one whole run absent from the file, which made
    # the chosen epoch unrecoverable once the terminal closed.
    baselines = BaselineFingerprint(ckpt_dir / "baselines.json",
                                    ref_path=args.baseline_ref)

    best_val = float("inf")
    since_improve = 0
    history: List[Dict] = []

    for epoch in range(cfg.epochs):
        te = time.time()
        rng = random.Random(cfg.seed + epoch)
        by_month: Dict[str, List[int]] = {}
        for i, c in enumerate(train_chunks):
            by_month.setdefault(store.month_of(c), []).append(i)
        order = []
        month_order = list(by_month)
        rng.shuffle(month_order)
        for m in month_order:
            idxs = by_month[m]
            rng.shuffle(idxs)
            order.extend(idxs)

        model.train()
        head.train()
        tr_l, tr_p, tr_n = 0.0, 0.0, 0
        acc_h, acc_m, acc_s = {}, {}, {}
        tr_node, tr_node_p = 0.0, 0.0
        for pos, idx in enumerate(order):
            c = train_chunks[idx]
            lbl = f"e{epoch} train {pos + 1}/{len(order)}"
            ct = store.materialize_chunk(c)
            out = _chunk_loss(model, head, objective, ct, lbl, args.supervise,
                              drop_edges=tuple(args.drop_edges))
            if out["n"] == 0:
                del ct
                continue
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            tr_l += float(out["loss"].detach()) * out["n"]
            tr_p += out["persistence"] * out["n"]
            tr_n += out["n"]
            _accum(acc_h, out["per_horizon"])
            _accum(acc_m, out.get("mechanisms", {}))
            _accum(acc_s, out.get("strata", {}))
            tr_node += out["node_loss"] * out["n"]
            tr_node_p += out["persistence"] * out["n"]
            print(f"  [{lbl}] n={out['n']:,} loss={float(out['loss'].detach()):.5f} "
                  f"persist={out['persistence']:.5f} ratio={out['ratio']:.3f}x "
                  f"rss={_rss_mb():.0f}MB", flush=True)
            del ct

        model.eval()
        head.eval()
        va_l, va_p, va_n = 0.0, 0.0, 0
        vacc_h, vacc_m, vacc_s = {}, {}, {}
        va_node, va_node_p = 0.0, 0.0
        with torch.no_grad():
            for pos, c in enumerate(val_chunks):
                ct = store.materialize_chunk(c)
                out = _chunk_loss(model, head, objective, ct,
                                  f"e{epoch} val {pos + 1}", args.supervise,
                                  drop_edges=tuple(args.drop_edges))
                if out["n"]:
                    va_l += float(out["loss"]) * out["n"]
                    va_p += out["persistence"] * out["n"]
                    va_n += out["n"]
                    _accum(vacc_h, out["per_horizon"])
                    _accum(vacc_m, out.get("mechanisms", {}))
                    _accum(vacc_s, out.get("strata", {}))
                    va_node += out["node_loss"] * out["n"]
                    va_node_p += out["persistence"] * out["n"]
                del ct

        # Headline ratio compares LIKE WITH LIKE: node forecasting loss
        # against node persistence. The combined objective (node + ladder
        # + mece) is what the optimiser minimises and is reported as
        # `total`, but it has no single baseline to divide by.
        trl = tr_node / tr_n if tr_n else float("nan")
        trp = tr_node_p / tr_n if tr_n else float("nan")
        val = va_node / va_n if va_n else float("nan")
        vap = va_node_p / va_n if va_n else float("nan")
        tr_total = tr_l / tr_n if tr_n else float("nan")
        va_total = va_l / va_n if va_n else float("nan")
        val_per_h, val_mech = _finish(vacc_h), _finish(vacc_m)
        last_per_h, last_mech = _finish(acc_h), _finish(acc_m)
        val_strata, last_strata = _finish(vacc_s), _finish(acc_s)
        tr_ratio = trl / trp if trp else float("nan")
        va_ratio = val / vap if vap else float("nan")
        secs = time.time() - te

        print(f"\nEPOCH {epoch}  NODE train {trl:.5f} vs persist {trp:.5f} = {tr_ratio:.3f}x"
              f"  |  val {val:.5f} vs persist {vap:.5f} = {va_ratio:.3f}x  ({secs:.0f}s)")
        print(f"  combined objective (node+ladder+mece): train {tr_total:.5f} "
              f"val {va_total:.5f}  -- minimised, but has no single baseline")
        if val_per_h:
            print(f"  val per-horizon: {format_horizons(val_per_h)}")
            print(f"  val mechanisms : {format_mechanisms(val_mech)}")
            print(f"  val by regime  : {format_strata(val_strata)}")
        elif last_per_h:
            print(f"  train per-horizon: {format_horizons(last_per_h)}")
            print(f"  train mechanisms : {format_mechanisms(last_mech)}")
            print(f"  train by regime  : {format_strata(last_strata)}")
        # The mechanism ratios are the violation-timing claim. Node-level
        # loss can fall while the gap ratio sits at 1.0, which would mean
        # the model got better at prices and no better at violations.
        _mech = val_mech or last_mech
        for _k, _v in sorted(_mech.items()):
            if _v["ratio"] == _v["ratio"] and _v["ratio"] >= 1.0:
                print(f"  >>> {_k.upper()}: not beating persistence on the derived quantity.")
        # THE ONLY NUMBER THAT MATTERS. A ratio at or above 1.0 means a
        # one-line rule -- "the price stays where it is" -- is doing the
        # model's job better than the model, whatever the raw loss shows.
        #
        # ...but the POOLED node ratio is the wrong place to read that, and
        # this warning used to print "the graph is contributing nothing"
        # directly underneath a mechanism line showing 0.807x. Two reasons
        # it misfires:
        #
        #   1. It is hardcoded to the node metric regardless of
        #      --select-on, so a run optimising MECE gets judged on a
        #      quantity it was told not to optimise.
        #   2. ~68% of supervised positions do not move between t and t+h.
        #      On those, persistence has an MSE of exactly 0 and ANY model
        #      output scores worse -- no forecast can win there, so
        #      including them guarantees a pooled ratio above 1.0 however
        #      good the model is on the positions that actually move.
        #
        # The 'moved' stratum is the honest headline, so the warning now
        # reads that when it exists and says which quantity it is judging.
        _strata = val_strata or last_strata or {}
        _moved, _flat = _strata.get("moved") or {}, _strata.get("flat") or {}
        _n_moved, _n_flat = _moved.get("n", 0), _flat.get("n", 0)
        _flat_pct = 100.0 * _n_flat / max(_n_moved + _n_flat, 1)
        _hl_r = _moved.get("ratio", va_ratio)
        _hl_n = "node/moved" if _moved else "node/pooled"
        if _hl_r == _hl_r and _hl_r >= 1.0:
            print(f"  >>> NOT BEATING PERSISTENCE on {_hl_n} ({_hl_r:.3f}x). Read the "
                  f"mechanism line above before concluding anything -- --select-on is "
                  f"what this run is optimising, and it may not be this quantity.")
        elif _moved:
            print(f"  >>> beating persistence on {_hl_n}: {_hl_r:.3f}x (n={_n_moved:,}). "
                  f"Pooled node ratio is {va_ratio:.3f}x only because {_flat_pct:.0f}% of "
                  f"positions do not move, and persistence is exact on those.")
        print(flush=True)

        def _m(name, field, default=float("nan")):
            v = (val_mech or {}).get(name)
            return v.get(field, default) if v else default

        # A mechanism that is computed and printed but has no column here is
        # evidence that survives only as long as the scrollback does.
        check_history_columns(val_mech, MECH_COLUMNS, known=MECH_COLUMNS)

        # Persistence and shrinkage are functions of the DATA, not of the
        # model, so they must be identical at every epoch -- and identical
        # to any run this one is paired with. Movement means the supervised
        # population is drifting, or a training weight has reached the
        # reported metric.
        baselines.check(epoch, val_mech, vap, va_n)
        check_ladder_persistence_decomposition(val_mech)

        # ONE ordered dict feeds both the header and the row.
        row: Dict[str, str] = {
            "epoch": f"{epoch}",
            "train_node": f"{trl:.6f}", "train_node_persist": f"{trp:.6f}",
            "train_node_ratio": f"{tr_ratio:.6f}",
            "val_node": f"{val:.6f}", "val_node_persist": f"{vap:.6f}",
            "val_node_ratio": f"{va_ratio:.6f}",
            "train_total": f"{tr_total:.6f}", "val_total": f"{va_total:.6f}",
        }
        for mech in MECH_COLUMNS:
            for suffix, field in _MECH_FIELDS:
                v = _m(mech, field, 0 if field == "n" else float("nan"))
                row[f"val_{mech}{suffix}"] = (f"{v:.0f}" if field == "n"
                                              else f"{float(v):.6f}")
        row["seconds"] = f"{secs:.1f}"

        if epoch == 0 or not hist_path.exists():
            hist_path.write_text(",".join(row) + "\n")
        with hist_path.open("a") as f:
            f.write(",".join(row.values()) + "\n")
        history.append({"epoch": epoch, "train": trl, "val": val, "val_ratio": va_ratio})

        payload = {
            "model": model.state_dict(), "head": head.state_dict(),
            "config": cfg, "horizons": list(args.horizons), "epoch": epoch,
            "val_loss": val, "val_persistence": vap, "val_ratio": va_ratio,
            "objective": "horizon_forecast_residual",
        }
        # Save unconditionally: an earlier version of the reconstruction
        # trainer wrote nothing when a run had no val split, and a
        # 40-minute run produced no checkpoint at all.
        torch.save({**payload, "selected_by": "last_epoch"}, ckpt_dir / "last.pt")
        # SELECT ON WHAT THE PROJECT TRADES. Under a multi-task objective
        # the node loss is not the thing being optimised, so selecting on
        # it can checkpoint a model that is getting worse at gaps. When a
        # ladder ratio is available, select on it; fall back to node loss
        # only when there is no mechanism signal at all.
        # WHICH QUANTITY DECIDES "BEST". Getting this wrong does two kinds of
        # damage at once: it checkpoints the wrong epoch, and it early-stops
        # on a metric that is not the one improving. An earlier run selected
        # on the ladder gap, which is a random walk here (alpha=0.995) and
        # never improved -- so it halted after 5 epochs and saved epoch 0,
        # while the MECE term was still descending 1.011 -> 0.891 -> 0.887.
        _pref = args.select_on
        # Select on the population actually being trained, or early stopping
        # watches a metric the optimiser is not moving -- the failure that
        # once saved epoch 0 while MECE was still descending.
        if _pref == "ladder" and args.ladder_supervise == "moved":
            _pref = "ladder_moved"
        _sel = _sel_name = None
        if _pref in ("ladder", "ladder_moved", "mece"):
            v = (val_mech or {}).get(_pref, {}).get("model")
            if v is not None:
                _sel, _sel_name = v, f"val_{_pref}"
        elif _pref == "total":
            _sel, _sel_name = va_total, "val_total"
        if _sel is None:
            _sel, _sel_name = val, "val_node"
        # A NaN here loses every comparison silently: no checkpoint is ever
        # written and the run stops at the patience limit looking exactly
        # like a model that could not learn. Fail now and say why.
        _sel = require_finite_selection(_sel_name, _sel)
        if va_n and _sel < best_val:
            best_val = _sel
            since_improve = 0
            torch.save({**payload, "selected_by": _sel_name}, ckpt_dir / "best.pt")
            print(f"  checkpoint: new best {_sel_name} {_sel:.5f} -> "
                  f"{ckpt_dir / 'best.pt'}", flush=True)
        else:
            since_improve += 1
            if cfg.patience and va_n and since_improve >= cfg.patience:
                print(f"  early stop: {since_improve} epochs without improvement", flush=True)
                break

    final = ckpt_dir / ("best.pt" if (ckpt_dir / "best.pt").exists() else "last.pt")
    print(f"\nDone. Checkpoint: {final}")
    print(f"History: {hist_path}")
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()