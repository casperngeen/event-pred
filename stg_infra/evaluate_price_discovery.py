"""
evaluate_price_discovery.py

WHEN A STALE LEG FINALLY TRADES, DOES IT MOVE TOWARD THE MODEL'S FAIR VALUE?

This is the first genuine FORECASTING evaluation in the project, and it needs
no retraining -- the existing checkpoint already produces the estimate.

WHY THIS TASK, AND WHY NOW. `diagnose_staleness.py` established that 90% of
consecutive snapshot observations show no price change at all, and that 98.3%
of "surviving" ladder violations survive only because neither print updated.
The printed price of a typical leg is therefore not its current value; it is
the last value at which somebody happened to trade, possibly a day ago.

That kills the arbitrage framing (you cannot fill against a price nobody is
posting) but it creates a well-posed prediction problem:

    at time t the leg is stale. The model, which cannot see the leg's own
    price, estimates its value from the rest of the graph. At t+k the leg
    finally trades. Does the new traded price move TOWARD the model's
    estimate?

Every property a defensible ML result needs is present:

  - the target (next traded price) is NOT observable at t, so no arithmetic
    rule can produce it -- unlike the arbitrage-selection task, which the
    results document showed reduces to a calculator;
  - the baseline is brutal and honest: PERSISTENCE, "it will reprint where
    it last printed", which is right 90% of the time by construction;
  - the model's one proven capability (fair value, 4.2x the best baseline on
    reconstruction) is exactly what is being tested;
  - and it is economically actionable in the one way that survives the
    staleness finding: you cannot take liquidity at a stale print, but you
    CAN rest a limit order at your own estimate and let the market come to
    you. Maker, not taker -- which matters because the taker spread cost is
    what made every strategy in the results document unprofitable.

WHAT IS MEASURED

  1. DIRECTION. Of legs that were stale at t and repriced by t+k, how often
     does the model call the sign of the move? 50% is a coin flip; the
     persistence baseline cannot play this game at all (it predicts no move),
     so the comparison is against chance and against a sign-of-last-move
     momentum rule.
  2. MAGNITUDE. MAE of the model's estimate against the realised next traded
     price, versus the persistence baseline's MAE. Persistence is the number
     to beat.
  3. CALIBRATION. Bucketed by how far the model disagrees with the stale
     print, what is the realised mean move? If the model says a leg is 5c
     cheap, does it in fact reprice ~5c higher? A model can have good
     direction and useless magnitude, and only magnitude is tradeable.

IMPORTANT CAVEAT, STATED IN THE CODE BECAUSE IT BELONGS IN THE WRITE-UP. The
model's estimate is built from the leg's neighbours, and those neighbours may
themselves be stale. This does not invalidate the test -- the estimate is
still formed without the leg's own price, and the target is still a future
traded price -- but it means the model is not being handed a clean view of
"current value". If it beats persistence anyway, it does so despite that.
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


def _median(v):
    if not v:
        return float("nan")
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    ap.add_argument("--mask-ratio", type=float, default=0.20,
                    help="fraction of ticker legs hidden per pass. Only hidden legs get a "
                         "fair-value read, so more passes cover more legs.")
    ap.add_argument("--passes", type=int, default=3,
                    help="independent masking passes per chunk, different seed each time. "
                         "Raises coverage without masking so much at once that the model "
                         "loses the neighbours it prices from.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-eps", type=float, default=TICK / 2,
                    help="a leg counts as having REPRICED if it moved more than this")
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

    print(f"PRICE DISCOVERY | checkpoint {ckpt_path.name} "
          f"(epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f})")
    print(f"Split: {args.split.upper()} | mask_ratio {args.mask_ratio} x {args.passes} passes\n")

    t_index = {ts: i for i, ts in enumerate(store.timestamps)}
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len,
                                      min_chunk_len=cfg.min_chunk_len) if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")

    px = defaultdict(dict)          # ticker -> {global_t: observed price}
    fair = defaultdict(dict)        # ticker -> {global_t: model fair value}

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)
        ladder_adj = adj.get("ladder_monotonic")
        node_ids = ct.get("node_ids")
        if ladder_adj is None or not node_ids:
            del ct
            continue

        # Record observed prices for every ladder member this chunk.
        for t in range(features.shape[0]):
            gt = t_index.get(ct["timestamps"][t])
            if gt is None:
                continue
            edges = ladder_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            for k in range(edges.edge_index.shape[1]):
                for idx in (int(edges.edge_index[0, k]), int(edges.edge_index[1, k])):
                    if bool(mask[t, idx]):
                        px[str(node_ids[idx])][gt] = float(prices[t, idx])

        is_ticker = features[..., -1] == 0.0
        for p in range(args.passes):
            gen = torch.Generator().manual_seed(args.seed + 1000 * p + ci)
            target_mask = (mask & is_ticker) & (
                torch.rand(mask.shape, generator=gen) < args.mask_ratio)
            with torch.no_grad():
                h = masked_forward(model, features, mask, adj, target_mask,
                                   objective.mask_token, cfg.raw_feature_width)
                ladder_out = model.ladder_head(h, ladder_adj)

            # A node can sit in several ladder edges; average its reads.
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
                    if bool(target_mask[t, a]):
                        acc[a].append(float(res["fair_a"][k]))
                    if bool(target_mask[t, b]):
                        acc[b].append(float(res["fair_b"][k]))
                for idx, vals in acc.items():
                    fair[str(node_ids[idx])][gt] = sum(vals) / len(vals)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} done", flush=True)

    # ------------------------------------------------------------------
    # Build the event set: STALE at t (did not move from t-1 to t) and
    # REPRICED by t+h. Staleness is judged BACKWARD -- a leg that updates
    # at t+1 was stale at t, and using the forward move to define staleness
    # would leak the answer into the question.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"PRICE DISCOVERY -- {args.split.upper()} split")
    print("=" * 100)
    print(f"{'h':>3} {'events':>8} {'model MAE':>11} {'persist MAE':>12} {'improve':>9} "
          f"{'dir acc':>9} {'momentum':>9}")
    print("-" * 100)

    per_h = {}
    for h in args.horizons:
        ev = []
        for ticker, obs in px.items():
            f = fair.get(ticker)
            if not f:
                continue
            for gt, p_now in obs.items():
                p_prev = obs.get(gt - 1)
                if p_prev is None or abs(p_now - p_prev) > args.move_eps:
                    continue                      # not stale at t
                p_next = obs.get(gt + h)
                if p_next is None or abs(p_next - p_now) <= args.move_eps:
                    continue                      # never repriced within h
                fv = f.get(gt)
                if fv is None:
                    continue
                ev.append((p_now, fv, p_next, p_prev))
        if len(ev) < 20:
            print(f"{h:>3} {len(ev):>8} {'(too few events)':>55}")
            per_h[h] = None
            continue

        model_mae = sum(abs(fv - pn) for _, fv, pn, _ in ev) / len(ev)
        pers_mae = sum(abs(p0 - pn) for p0, _, pn, _ in ev) / len(ev)
        dir_hit = sum(1 for p0, fv, pn, _ in ev
                      if (fv - p0) * (pn - p0) > 0) / len(ev)
        # momentum control: predict the next move has the same sign as the
        # last one. Needs a prior move, so it is scored on its own subset.
        mom = [(p0, pn, pp) for p0, _, pn, pp in ev if abs(p0 - pp) > args.move_eps]
        mom_hit = (sum(1 for p0, pn, pp in mom if (p0 - pp) * (pn - p0) > 0) / len(mom)
                   if mom else float("nan"))
        improve = 100.0 * (pers_mae - model_mae) / pers_mae if pers_mae else 0.0
        per_h[h] = (ev, model_mae, pers_mae, dir_hit)
        print(f"{h:>3} {len(ev):>8,} ${model_mae:>10.4f} ${pers_mae:>11.4f} "
              f"{improve:>8.1f}% {100 * dir_hit:>8.1f}% "
              f"{(100 * mom_hit if mom_hit == mom_hit else float('nan')):>8.1f}%")

    print("-" * 100)
    print("  events    = leg was STALE at t (no move t-1 -> t) and REPRICED by t+h")
    print("  persist   = baseline 'it reprints where it last printed'. THE NUMBER TO BEAT.")
    print("  dir acc   = model called the sign of the move (50% = coin flip)")
    print("  momentum  = control: 'next move has the same sign as the last move'")

    # ---- calibration: is the magnitude usable, not just the sign? -----
    ref = next((per_h[h] for h in args.horizons if per_h.get(h)), None)
    if ref:
        h0 = next(h for h in args.horizons if per_h.get(h) is ref)
        ev = ref[0]
        print("\n" + "=" * 100)
        print(f"CALIBRATION at h={h0}: does a bigger predicted move mean a bigger real one?")
        print("=" * 100)
        print(f"{'predicted move (c)':>22} {'n':>8} {'mean realised (c)':>19} {'ratio':>8}")
        print("-" * 100)
        buckets = [(-99, -0.05), (-0.05, -0.02), (-0.02, -0.005), (-0.005, 0.005),
                   (0.005, 0.02), (0.02, 0.05), (0.05, 99)]
        for lo, hi in buckets:
            sel = [(fv - p0, pn - p0) for p0, fv, pn, _ in ev if lo <= (fv - p0) < hi]
            if len(sel) < 10:
                continue
            mp = sum(s[0] for s in sel) / len(sel)
            mr = sum(s[1] for s in sel) / len(sel)
            ratio = mr / mp if abs(mp) > 1e-9 else float("nan")
            print(f"{100 * mp:>22.2f} {len(sel):>8,} {100 * mr:>19.2f} {ratio:>8.2f}")
        print("-" * 100)
        print("  ratio 1.0 = perfectly calibrated: predicted 5c move, realised 5c move.")
        print("  ratio near 0 with good direction = the SIGN is informative but the")
        print("  SIZE is not, and only size is tradeable -- the same failure mode as")
        print("  the coverage extension in section 4 of the results document.")
        print("  A ratio that is consistently positive and rising across buckets is the")
        print("  result that would justify resting limit orders at the model's estimate.")


if __name__ == "__main__":
    main()