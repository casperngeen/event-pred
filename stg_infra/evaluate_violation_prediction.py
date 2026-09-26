"""
evaluate_violation_prediction.py

CAN THE STGAT IDENTIFY ARBITRAGE THAT ARITHMETIC CANNOT?

This is the decisive test of the project's central claim. Everything before
it asked whether the model could improve on a naive rule that was assumed to
be correct. `diagnose_staleness.py` showed that assumption is false: the
naive rule flags 2,489 violations of which 98.3% are artifacts of comparing
two stale last-trade prints. The comparison operator is not a strong
baseline on the real task -- it only appeared strong because it was scored
against the same stale data that produced it.

That is the opening for a model. At time t most legs are stale, so the
printed price is not the current value. The STGAT estimates what each rung
is worth from the REST OF THE GRAPH. A violation in those estimates is a
prediction that a genuine violation exists -- one the arithmetic cannot
reliably see.

And the market eventually adjudicates. When both rungs finally trade, a
CONFIRMED violation is observed on fresh prices. So:

    signal at t : model's estimated gap   vs   the stale arithmetic gap
    label at t+k: violation computed only from freshly REPRICED legs
    question    : which signal better predicts the confirmed violation?

Unobservable target, non-trivial baseline, exact ground truth. Prediction
markets make this cleaner than equities: the structural relation is a
logical identity, so a confirmed violation is unambiguous rather than a
residual from an estimated cointegration relationship.

THE ARCHITECTURAL OBSTACLE, AND THE WORKAROUND. LadderOutputHead guarantees
fair_a >= fair_b, so within one forward pass the model's gap is <= 0 BY
CONSTRUCTION and it can never predict a violation. The workaround is to read
each rung from a pass in which only THAT rung is hidden: fair_A from a pass
masking A, fair_B from a pass masking B. The monotonicity constraint binds
within each pass, not across them, so the cross-pass gap can be positive.
Each number remains a genuine estimate of a hidden leg from its neighbours.

FAIRNESS OF THE COMPARISON. The two signals must be compared at MATCHED
ALERT COUNTS, not at a shared threshold -- a signal that simply fires more
often will otherwise look better on recall and worse on precision for
reasons that have nothing to do with skill. Top-N by each signal, same N,
same events. This control exists because its absence produced three false
conclusions earlier in this project.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.inference import masked_forward  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, TrainingConfig, true_prices  # noqa: E402
from model.training_objective import MaskedLegReconstructionObjective  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
TICK = 0.01
# Ticker feature layout (stg/nodes/kalshi.py, documented in model/data_validation.py):
#   0 last_yes_price  1 yes_vwap  2 price_return  3 price_std  4 window_volume
#   5 net_flow  6 buy_ratio  7 trade_intensity  8 time_to_close
# net_flow is SIGNED order-flow imbalance in [-1, 1] -- the market's own
# order-flow momentum, available without any model. Features are RAW here
# (true_prices only divides slot 0 by 100); standardization happens inside
# the backbone, so this slot can be read directly.
NET_FLOW_SLOT = 5


def _pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--threshold", type=float, default=TICK,
                    help="gap in dollars above which a CONFIRMED violation counts")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--passes", type=int, default=6,
                    help="masking passes per chunk. A rung is only read in passes that "
                         "hide it, so coverage of PAIRS (both rungs read) needs several.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-eps", type=float, default=TICK / 2)
    ap.add_argument("--sweep-horizon", type=int, default=None,
                    help="which horizon the budget sweep and bootstrap use. Defaults to the "
                         "first available, which is usually the WEAKEST -- set it explicitly.")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="resamples for the confidence interval on lift. Confirmed-event "
                         "counts here are small (50-150), so a point estimate of lift is "
                         "not enough to claim anything.")
    ap.add_argument("--task", choices=["confirm", "emerge", "both"], default="both",
                    help="'confirm' = T1, does an apparent violation survive adjudication. "
                         "'emerge' = T2, does a violation APPEAR where the stale prints show "
                         "none -- the anticipatory task, and the only one where resting a "
                         "limit order makes sense.")
    ap.add_argument("--emerge-budget", type=float, default=0.05,
                    help="alert budget for T2, as a fraction of events. T2 has no natural "
                         "budget because the stale rule does not fire at all by construction.")
    ap.add_argument("--momentum-lag", type=int, default=3,
                    help="snapshots back for the momentum baseline: gap(t) - gap(t-lag).")
    ap.add_argument("--require-both-fresh", action="store_true", default=True,
                    help="a confirmed violation requires BOTH rungs to have repriced")
    ap.add_argument("--either-fresh", dest="require_both_fresh", action="store_false",
                    help="relax to: at least one rung repriced (larger sample, weaker label)")
    args = ap.parse_args()

    months = args.months or PILOT_MONTHS
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    store = MonthlyBundleStore(build_month_paths(months, _REPO_ROOT / "cache"), verbose=False)
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg: TrainingConfig = ckpt.get("config", TrainingConfig())
    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=max(cfg.chunk_len, args.chunk_len))
    model.load_state_dict(ckpt["model"])
    model.eval()
    objective = MaskedLegReconstructionObjective(raw_feature_dim=cfg.raw_feature_width,
                                                 mask_ratio=cfg.mask_ratio)
    objective.load_state_dict(ckpt["objective"])

    print(f"VIOLATION PREDICTION | checkpoint {ckpt_path.name} "
          f"(epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f})")
    print(f"Split: {args.split.upper()} | mask_ratio {args.mask_ratio} x {args.passes} passes")
    print(f"Confirmed violation: gap > ${args.threshold:.3f} with "
          f"{'BOTH rungs' if args.require_both_fresh else 'at least one rung'} freshly repriced\n")

    t_index = {ts: i for i, ts in enumerate(store.timestamps)}
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len,
                                      min_chunk_len=cfg.min_chunk_len) if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")

    px = defaultdict(dict)                 # ticker -> {gt: observed price}
    fair = defaultdict(dict)               # ticker -> {gt: model fair value (leg hidden)}
    pair_seen = defaultdict(set)           # (ta, tb) -> {gt} where the edge exists
    flow = defaultdict(dict)               # ticker -> {gt: net_flow (raw, -1..1)}

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)
        ladder_adj = adj.get("ladder_monotonic")
        node_ids = ct.get("node_ids")
        if ladder_adj is None or not node_ids:
            del ct
            continue

        for t in range(features.shape[0]):
            gt = t_index.get(ct["timestamps"][t])
            edges = ladder_adj[t]
            if gt is None or edges.edge_index.numel() == 0:
                continue
            for k in range(edges.edge_index.shape[1]):
                a, b = int(edges.edge_index[0, k]), int(edges.edge_index[1, k])
                if not (bool(mask[t, a]) and bool(mask[t, b])):
                    continue
                ta, tb = str(node_ids[a]), str(node_ids[b])
                px[ta][gt] = float(prices[t, a])
                px[tb][gt] = float(prices[t, b])
                if features.shape[-1] > NET_FLOW_SLOT:
                    flow[ta][gt] = float(features[t, a, NET_FLOW_SLOT])
                    flow[tb][gt] = float(features[t, b, NET_FLOW_SLOT])
                pair_seen[(ta, tb)].add(gt)

        is_ticker = features[..., -1] == 0.0
        for p in range(args.passes):
            gen = torch.Generator().manual_seed(args.seed + 1000 * p + ci)
            target_mask = (mask & is_ticker) & (
                torch.rand(mask.shape, generator=gen) < args.mask_ratio)
            with torch.no_grad():
                h = masked_forward(model, features, mask, adj, target_mask,
                                   objective.mask_token, cfg.raw_feature_width)
                ladder_out = model.ladder_head(h, ladder_adj)
            for t in range(features.shape[0]):
                gt = t_index.get(ct["timestamps"][t])
                res = ladder_out.get(t)
                if gt is None or res is None:
                    continue
                edges = ladder_adj[t]
                if edges.edge_index.numel() == 0:
                    continue
                acc = defaultdict(list)
                for k in range(edges.edge_index.shape[1]):
                    a, b = int(edges.edge_index[0, k]), int(edges.edge_index[1, k])
                    # ONLY read a rung from a pass that HID it: an unmasked rung's
                    # "estimate" is just its own input price echoed back.
                    if bool(target_mask[t, a]):
                        acc[a].append(float(res["fair_a"][k]))
                    if bool(target_mask[t, b]):
                        acc[b].append(float(res["fair_b"][k]))
                for idx, vals in acc.items():
                    fair[str(node_ids[idx])][gt] = sum(vals) / len(vals)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} done", flush=True)

    # ------------------------------------------------------------------
    # Events: at t both rungs are quoted and the model has read BOTH of
    # them (each from a pass that hid it). At t+h the pair reprices, and
    # the fresh prices adjudicate whether a violation was really there.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # T2: EMERGENCE. The anticipatory task, and the one that matches a
    # resting-order strategy: at t the stale prints show NO violation, so
    # nothing is visible to trade. Does a violation APPEAR on fresh prices
    # at t+k?
    #
    # Why this is the sharper version of the project's claim: T1 asks the
    # model to adjudicate something already flagged, where the stale gap at
    # least points at the pair. Here the arithmetic has nothing to say --
    # by construction every event has stale gap <= threshold. If the model
    # ranks these at all, it is seeing something the tape has not printed.
    #
    # BASELINES NAMED FIRST, as the results document's section 5 pattern
    # demands. The obvious rules here are (a) proximity -- pairs nearest to
    # crossing are likeliest to cross, which is just the stale gap again,
    # and (b) momentum -- the gap has been trending toward a crossing.
    # Both are computable with no model. If the STGAT cannot beat those,
    # it adds nothing, however far it beats the random base rate.
    #
    # There is no natural alert budget here (the stale rule never fires),
    # so all signals are compared at the SAME fixed budget.
    # ------------------------------------------------------------------
    if args.task in ("emerge", "both"):
        print("=" * 100)
        print(f"T2: EMERGENCE -- violations that do NOT yet exist in the stale prints")
        print(f"    (alert budget {100 * args.emerge_budget:.0f}% of events, "
              f"momentum lag {args.momentum_lag})")
        print("=" * 100)
        print(f"{'h':>3} {'events':>8} {'emerged':>9} {'base':>7}  "
              f"{'proximity':>10} {'momentum':>10} {'ordflow':>10} {'model':>10}  "
              f"{'model/best':>11}")
        print("-" * 100)
        emerge_ev = {}
        for h in args.horizons:
            ev2 = []
            for (ta, tb), times in pair_seen.items():
                for gt in times:
                    pa, pb = px[ta].get(gt), px[tb].get(gt)
                    fa, fb = fair[ta].get(gt), fair[tb].get(gt)
                    if None in (pa, pb, fa, fb):
                        continue
                    if (pb - pa) > args.threshold:
                        continue                     # already visible -> that is T1
                    pa1, pb1 = px[ta].get(gt + h), px[tb].get(gt + h)
                    if pa1 is None or pb1 is None:
                        continue
                    a_fresh = abs(pa1 - pa) > args.move_eps
                    b_fresh = abs(pb1 - pb) > args.move_eps
                    fresh = ((a_fresh and b_fresh) if args.require_both_fresh
                             else (a_fresh or b_fresh))
                    if not fresh:
                        continue
                    lag = args.momentum_lag
                    p0a, p0b = px[ta].get(gt - lag), px[tb].get(gt - lag)
                    mom = ((pb - pa) - (p0b - p0a)) if (p0a is not None and p0b is not None) \
                        else float("-inf")           # no history -> ranks last, never dropped
                    # ORDER-FLOW MOMENTUM, model-free. A violation is p_B > p_A,
                    # so a crossing is driven by BUYING pressure on B and SELLING
                    # pressure on A: net_flow(B) - net_flow(A). This is the
                    # arithmetic version of "predict where order flow is heading",
                    # and it is the baseline the model must beat -- not the base rate.
                    nfa, nfb = flow[ta].get(gt), flow[tb].get(gt)
                    fl = (nfb - nfa) if (nfa is not None and nfb is not None) else float("-inf")
                    ev2.append(((pb - pa), mom, fl, (fb - fa),
                                (pb1 - pa1) > args.threshold, ta.split("-", 1)[0], gt))
            emerge_ev[h] = ev2
            n2 = len(ev2)
            n_em = sum(1 for e in ev2 if e[4])
            if n2 < 50 or n_em < 5:
                print(f"{h:>3} {n2:>8,} {n_em:>9,} {'(too few emergences to judge)':>48}")
                continue
            k2 = max(10, int(args.emerge_budget * n2))
            precs = []
            for idx in (0, 1, 2, 3):          # proximity, momentum, order flow, model
                top = sorted(ev2, key=lambda e: -e[idx])[:k2]
                precs.append(sum(1 for e in top if e[4]) / len(top))
            best_base = max(precs[0], precs[1], precs[2])
            rel = (precs[3] / best_base) if best_base > 0 else float("nan")
            print(f"{h:>3} {n2:>8,} {n_em:>9,} {100 * n_em / n2:>6.1f}%  "
                  f"{100 * precs[0]:>9.1f}% {100 * precs[1]:>9.1f}% {100 * precs[2]:>9.1f}% "
                  f"{100 * precs[3]:>9.1f}%  {rel:>10.2f}x")
        print("-" * 100)
        print("  proximity = rank by the stale gap (closest to crossing). Model-free.")
        print("  momentum  = rank by gap(t) - gap(t-lag). Model-free.")
        print("  ordflow   = rank by net_flow(B) - net_flow(A), the SIGNED order-flow")
        print("              imbalance already in the feature vector (slot 5). This is")
        print("              'predict where order flow momentum is heading', done with")
        print("              arithmetic. It is the hardest baseline here.")
        print("  model/best = model precision divided by the BEST of the three baselines.")
        print("               Beating the base rate is not enough; beat the obvious rules.")

        hE = args.sweep_horizon if args.sweep_horizon in emerge_ev else (
            args.horizons[-1] if args.horizons else None)
        ev2 = emerge_ev.get(hE) or []
        n2 = len(ev2)
        n_em = sum(1 for e in ev2 if e[4])
        if n2 >= 50 and n_em >= 5:
            import random as _r2
            rng2 = _r2.Random(args.seed)
            k2 = max(10, int(args.emerge_budget * n2))
            rels = []
            for _ in range(args.bootstrap):
                samp = [ev2[rng2.randrange(n2)] for _ in range(n2)]
                pr = []
                for idx in (0, 1, 2, 3):
                    top = sorted(samp, key=lambda e: -e[idx])[:k2]
                    pr.append(sum(1 for e in top if e[4]) / len(top))
                bb = max(pr[0], pr[1], pr[2])
                if bb > 0:
                    rels.append(pr[3] / bb)
            rels.sort()

            def _q2(p):
                return rels[min(len(rels) - 1, max(0, int(p * len(rels))))] if rels else float("nan")
            print()
            print(f"  Bootstrap at h={hE} ({args.bootstrap} resamples, budget {k2}):")
            print(f"    model / best baseline   5th {_q2(0.05):.2f}x   median {_q2(0.50):.2f}x"
                  f"   95th {_q2(0.95):.2f}x")
            if _q2(0.05) > 1.0:
                print("    -> interval excludes 1.0 UPWARD: the model anticipates emergences")
                print("       that proximity, price momentum and raw ORDER FLOW all miss.")
                print("       THIS is predictive arbitrage, and it is the project's claim.")
            elif _q2(0.95) < 1.0:
                print("    -> interval excludes 1.0 DOWNWARD: the model is RELIABLY WORSE than")
                print("       the model-free rules here. Not ambiguity -- a negative result,")
                print("       and a more informative one than a wide interval would be.")
            else:
                print("    -> interval includes 1.0: not distinguishable from the model-free")
                print("       rules at this sample size.")

        # ---- T2 per-series -------------------------------------------
        # T1 showed the whole test-split advantage came from ONE family
        # (KXBTCD), and that val contains almost none of that family. If
        # T2 shows the same composition, the test/val divergence is about
        # WHICH MARKETS each period contains, not about the model working
        # in one period and failing in another -- a different claim with a
        # different remedy (more months, analysed per family).
        if ev2:
            per = defaultdict(list)
            for e in ev2:
                per[e[5]].append(e)
            rows = [(k_, v) for k_, v in per.items()
                    if len(v) >= 150 and sum(1 for e in v if e[4]) >= 8]
            if rows:
                print(f"  Per-series at h={hE} (>=150 events, >=8 emergences):")
                print(f"    {'series':<22} {'events':>7} {'emerg':>6} {'base':>7} "
                      f"{'prox':>7} {'mom':>7} {'flow':>7} {'model':>7} {'m/best':>8}")
                for k_, v in sorted(rows, key=lambda r: -len(r[1])):
                    kk = max(10, int(args.emerge_budget * len(v)))
                    pr = []
                    for idx in (0, 1, 2, 3):
                        top = sorted(v, key=lambda e: -e[idx])[:kk]
                        pr.append(sum(1 for e in top if e[4]) / len(top))
                    bb = max(pr[0], pr[1], pr[2])
                    rl = (pr[3] / bb) if bb > 0 else float("nan")
                    print(f"    {k_:<22} {len(v):>7,} {sum(1 for e in v if e[4]):>6} "
                          f"{100 * sum(1 for e in v if e[4]) / len(v):>6.1f}% "
                          f"{100 * pr[0]:>6.1f}% {100 * pr[1]:>6.1f}% {100 * pr[2]:>6.1f}% "
                          f"{100 * pr[3]:>6.1f}% {rl:>7.2f}x")
                print("    -> compare the SAME family across splits. A family present in only")
                print("       one split cannot be replicated there, and its absence is not")
                print("       evidence against the model.")
        print()

    if args.task == "emerge":
        return

    print("\n" + "=" * 100)
    print(f"PREDICTING CONFIRMED VIOLATIONS -- {args.split.upper()} split")
    print("=" * 100)
    print(f"{'h':>3} {'events':>8} {'confirmed':>11} {'base rate':>10}  "
          f"{'stale prec':>11} {'model prec':>11} {'lift':>7}  {'stale rec':>10} {'model rec':>10}")
    print("-" * 100)

    summary = {}
    for h in args.horizons:
        ev = []
        for (ta, tb), times in pair_seen.items():
            for gt in times:
                pa, pb = px[ta].get(gt), px[tb].get(gt)
                fa, fb = fair[ta].get(gt), fair[tb].get(gt)
                if None in (pa, pb, fa, fb):
                    continue
                pa1, pb1 = px[ta].get(gt + h), px[tb].get(gt + h)
                if pa1 is None or pb1 is None:
                    continue
                a_fresh = abs(pa1 - pa) > args.move_eps
                b_fresh = abs(pb1 - pb) > args.move_eps
                fresh = (a_fresh and b_fresh) if args.require_both_fresh else (a_fresh or b_fresh)
                if not fresh:
                    continue
                confirmed = (pb1 - pa1) > args.threshold
                ev.append((pb - pa, fb - fa, confirmed))   # stale signal, model signal, label
        if len(ev) < 30:
            print(f"{h:>3} {len(ev):>8} {'(too few events -- raise --passes or use --either-fresh)':>60}")
            summary[h] = None
            continue

        n = len(ev)
        n_conf = sum(1 for _, _, c in ev if c)
        base = n_conf / n

        # MATCHED ALERT COUNTS. Fire the same number of alerts from each
        # signal (its top-N), so precision is comparable. N is the number
        # the stale rule would naturally fire, i.e. gaps over threshold.
        n_alert = max(1, sum(1 for s, _, _ in ev if s > args.threshold))
        by_stale = sorted(ev, key=lambda e: -e[0])[:n_alert]
        by_model = sorted(ev, key=lambda e: -e[1])[:n_alert]
        sp = sum(1 for _, _, c in by_stale if c) / len(by_stale)
        mp = sum(1 for _, _, c in by_model if c) / len(by_model)
        sr = (sum(1 for _, _, c in by_stale if c) / n_conf) if n_conf else 0.0
        mr = (sum(1 for _, _, c in by_model if c) / n_conf) if n_conf else 0.0
        lift = (mp / sp) if sp > 0 else float("nan")
        summary[h] = (n, base, sp, mp, lift)
        print(f"{h:>3} {n:>8,} {n_conf:>11,} {100 * base:>9.1f}%  "
              f"{100 * sp:>10.1f}% {100 * mp:>10.1f}% {lift:>7.2f}x  "
              f"{100 * sr:>9.1f}% {100 * mr:>9.1f}%")

    # ------------------------------------------------------------------
    # PRECISION AT SEVERAL ALERT BUDGETS.
    #
    # The table above fires as many alerts as the stale rule naturally
    # would. That is the operationally meaningful count, but it makes the
    # comparison hostage to one number: if the stale rule alerts on almost
    # everything, both precisions collapse to the base rate and the test
    # says nothing. Sweeping the budget removes that dependence -- at a
    # fixed number of alerts, whichever signal RANKS better wins, which is
    # the question actually being asked.
    # ------------------------------------------------------------------
    h0 = args.sweep_horizon if args.sweep_horizon in summary and summary.get(
        args.sweep_horizon) else next((h for h in args.horizons if summary.get(h)), None)
    if h0 is not None:
        ev = []
        for (ta, tb), times in pair_seen.items():
            for gt in times:
                pa, pb = px[ta].get(gt), px[tb].get(gt)
                fa, fb = fair[ta].get(gt), fair[tb].get(gt)
                if None in (pa, pb, fa, fb):
                    continue
                pa1, pb1 = px[ta].get(gt + h0), px[tb].get(gt + h0)
                if pa1 is None or pb1 is None:
                    continue
                a_fresh = abs(pa1 - pa) > args.move_eps
                b_fresh = abs(pb1 - pb) > args.move_eps
                fresh = (a_fresh and b_fresh) if args.require_both_fresh else (a_fresh or b_fresh)
                if not fresh:
                    continue
                # series and timestamp carried so the concentration check can
                # ask whether an apparent edge is really one family on one day
                ev.append((pb - pa, fb - fa, (pb1 - pa1) > args.threshold,
                           ta.split("-", 1)[0], gt))
        n = len(ev)
        n_conf = sum(1 for e in ev if e[2])
        if n >= 30 and n_conf:
            print()
            print("=" * 100)
            print(f"PRECISION AT MATCHED ALERT BUDGETS (h={h0}, base rate "
                  f"{100 * n_conf / n:.1f}%)")
            print("=" * 100)
            print(f"{'budget':>10} {'k':>7} {'stale prec':>12} {'model prec':>12} "
                  f"{'lift':>8} {'random':>9}")
            print("-" * 100)
            st = sorted(ev, key=lambda e: -e[0])
            mo = sorted(ev, key=lambda e: -e[1])
            for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
                k = int(frac * n)
                if k < 10:
                    continue
                sp = sum(1 for e in st[:k] if e[2]) / k
                mp = sum(1 for e in mo[:k] if e[2]) / k
                lift = (mp / sp) if sp > 0 else float("nan")
                print(f"{100 * frac:>9.0f}% {k:>7,} {100 * sp:>11.1f}% {100 * mp:>11.1f}% "
                      f"{lift:>7.2f}x {100 * n_conf / n:>8.1f}%")
            print("-" * 100)
            print("  At a fixed alert budget the better RANKER wins. A signal that only")
            print("  matches 'random' is carrying no information about the outcome.")
            print("  WATCH THE TIGHTEST BUDGETS. A real ranker is BEST at the top; a signal")
            print("  that only wins at loose budgets and collapses at 1-2% is over-scaling")
            print("  its extreme predictions, not ranking well.")

            # ---- bootstrap the lift ----------------------------------
            # Confirmed events number in the dozens. A point estimate of
            # lift over ~100 alerts turns on a handful of events, and this
            # project has already produced three effects that vanished
            # under a control. Resample the EVENTS (not the alerts) so the
            # interval reflects the sampling variability that matters.
            import random as _random
            rng = _random.Random(args.seed)
            n_alert = max(1, sum(1 for e in ev if e[0] > args.threshold))
            lifts, diffs = [], []
            for _ in range(args.bootstrap):
                samp = [ev[rng.randrange(n)] for _ in range(n)]
                s_ = sorted(samp, key=lambda e: -e[0])[:n_alert]
                m_ = sorted(samp, key=lambda e: -e[1])[:n_alert]
                spb = sum(1 for e in s_ if e[2]) / len(s_)
                mpb = sum(1 for e in m_ if e[2]) / len(m_)
                diffs.append(mpb - spb)
                if spb > 0:
                    lifts.append(mpb / spb)
            lifts.sort()
            diffs.sort()

            # ---- concentration: one family on one day, or a real edge? ----
            # An advantage carried by 26 alerts can be a genuine capability or
            # a single ladder family having an unusual week. Leave-one-series-out
            # is the cheap discriminator: if removing the biggest contributor
            # collapses the lift, the "capability" was that one family.
            from collections import Counter as _Counter
            top_k = max(10, int(0.01 * n))
            top_model = sorted(ev, key=lambda e: -e[1])[:top_k]
            ser = _Counter(e[3] for e in top_model)
            days = len({e[4] // 12 for e in top_model})   # ~12 snapshots per day
            print()
            print(f"  Concentration of the model's top {top_k} alerts:")
            print(f"    distinct series {len(ser)} | distinct days ~{days}")
            for name, cnt in ser.most_common(5):
                hits = sum(1 for e in top_model if e[3] == name and e[2])
                print(f"    {name:<22} {cnt:>4} alerts  {hits:>3} confirmed "
                      f"({_pct(hits, cnt)})")
            biggest = ser.most_common(1)[0][0] if ser else None
            if biggest:
                rest = [e for e in ev if e[3] != biggest]
                if len(rest) >= 30 and any(e[2] for e in rest):
                    na = max(1, sum(1 for e in rest if e[0] > args.threshold))
                    s_ = sorted(rest, key=lambda e: -e[0])[:na]
                    m_ = sorted(rest, key=lambda e: -e[1])[:na]
                    sp2 = sum(1 for e in s_ if e[2]) / len(s_)
                    mp2 = sum(1 for e in m_ if e[2]) / len(m_)
                    l2 = (mp2 / sp2) if sp2 > 0 else float("nan")
                    print(f"    excluding '{biggest}' ({len(ev) - len(rest):,} events "
                          f"dropped): stale {100 * sp2:.1f}%  model {100 * mp2:.1f}%  "
                          f"lift {l2:.2f}x")
                    print("    -> if the lift collapses here, the edge was that one family.")

            # ---- per-series lift -----------------------------------------
            # If an edge is confined to one family, that may be an artifact OR
            # a real capability limited to where the INPUTS are fresh: the most
            # liquid families reprice often, so the graph the model reads is
            # current rather than a field of day-old prints. The two readings
            # are separable only by checking the SAME family on the other
            # split, so print every family big enough to judge.
            by_series = defaultdict(list)
            for e in ev:
                by_series[e[3]].append(e)
            rows = [(s, v) for s, v in by_series.items()
                    if len(v) >= 150 and sum(1 for e in v if e[2]) >= 8]
            if rows:
                print()
                print(f"  Per-series, h={h0} (families with >=150 events and >=8 confirmed):")
                print(f"    {'series':<22} {'events':>7} {'conf':>6} {'base':>7} "
                      f"{'stale':>7} {'model':>7} {'lift':>7}")
                for s, v in sorted(rows, key=lambda r: -len(r[1])):
                    nb = sum(1 for e in v if e[2]) / len(v)
                    na = max(1, sum(1 for e in v if e[0] > args.threshold))
                    s_ = sorted(v, key=lambda e: -e[0])[:na]
                    m_ = sorted(v, key=lambda e: -e[1])[:na]
                    spx = sum(1 for e in s_ if e[2]) / len(s_)
                    mpx = sum(1 for e in m_ if e[2]) / len(m_)
                    lf = (mpx / spx) if spx > 0 else float("nan")
                    print(f"    {s:<22} {len(v):>7,} {sum(1 for e in v if e[2]):>6} "
                          f"{100 * nb:>6.1f}% {100 * spx:>6.1f}% {100 * mpx:>6.1f}% "
                          f"{lf:>6.2f}x")
                print("    -> a family that wins on BOTH splits is a narrow but real result.")
                print("       A family that wins on only one is noise, however large the lift.")

            def _q(v, p):
                return v[min(len(v) - 1, max(0, int(p * len(v))))] if v else float("nan")

            win = sum(1 for d in diffs if d > 0) / len(diffs) if diffs else float("nan")
            print()
            print(f"  Bootstrap over events ({args.bootstrap} resamples, "
                  f"alert budget {n_alert}):")
            print(f"    lift            5th {_q(lifts, 0.05):.2f}x   median "
                  f"{_q(lifts, 0.50):.2f}x   95th {_q(lifts, 0.95):.2f}x")
            print(f"    precision diff  5th {100 * _q(diffs, 0.05):+.1f}pp  median "
                  f"{100 * _q(diffs, 0.50):+.1f}pp  95th {100 * _q(diffs, 0.95):+.1f}pp")
            print(f"    model beats stale in {100 * win:.1f}% of resamples")
            if _q(lifts, 0.05) > 1.0:
                print("    -> lift interval excludes 1.0. The effect survives resampling.")
            else:
                print("    -> lift interval INCLUDES 1.0. Not yet distinguishable from chance")
                print("       at this sample size; extend the window before claiming it.")

    print("-" * 100)
    print("  events     = both rungs quoted at t, model read BOTH (each hidden in some pass),")
    print("               and the pair repriced by t+h so the outcome is adjudicated")
    print("  confirmed  = fresh prices at t+h still show a violation")
    print("  base rate  = what you get by alerting at random. The floor.")
    print("  stale prec = precision of the ARITHMETIC rule on stale prints at t")
    print("  model prec = precision of the MODEL's estimated gap, at the SAME alert count")
    print("  lift       = model precision / stale precision. >1 means the model")
    print("               identifies real arbitrage that arithmetic cannot see.")

    ok = [h for h, s in summary.items() if s and s[3] > s[2] and s[3] > s[1]]
    print()
    if not ok:
        print("  VERDICT: the model does NOT beat stale arithmetic at identifying violations")
        print("  that the market later confirms. Combined with the staleness result, the")
        print("  honest conclusion is that these violations are neither tradeable nor")
        print("  predictable from this graph, and the contribution is the measurement.")
    elif len(ok) >= max(2, len(args.horizons) // 2):
        print(f"  The model beats stale arithmetic at {len(ok)} of {len(summary)} horizons.")
        print("  THIS IS NOT YET A RESULT. Winning on horizons is necessary, not sufficient.")
        print("  Three further conditions, all of which have failed at least once here:")
        print("    1. the budget sweep must be BEST at the tightest budget, not only at loose ones;")
        print("    2. the leave-one-series-out lift must survive -- an edge carried by a single")
        print("       family is an artifact unless that same family wins on the other split;")
        print("    3. it must REPLICATE on the split not used to form the hypothesis.")
        print("  Report precision at matched alert counts, never raw counts.")
    else:
        print(f"  VERDICT: the model wins at only {len(ok)} of {len(summary)} horizons -- not")
        print("  a stable effect. Treat as suggestive, extend the test window before claiming.")


if __name__ == "__main__":
    main()