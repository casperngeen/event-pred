"""
evaluate_baselines.py

Scores a trained STGAT against predictors that use no learning at all, on
the EXACT SAME masked positions, so a loss number becomes interpretable:
is the model actually pricing legs from their neighbours, or matching
something you could have written in one line?

FAIR-COMPARISON PROTOCOL -- THE POINT OF THE REWRITE. An earlier version
of this script masked EVERY eligible leg simultaneously (via
inference.generate_deviation_signals, whose job is signal generation, not
scoring). For the MECE mechanism that is survivable, because a basket's
HUB node is not a ticker leg and so is never masked -- it still carries
the basket's aggregate. For the LADDER mechanism it is fatal: both legs
of a pair are ticker legs, so both got masked, and the model was asked to
price leg A with leg B also hidden -- while the `partner_price` baseline
was handed leg B's TRUE OBSERVED PRICE. That hands the baseline exactly
the information the model was denied, and it is not a real finding about
the model: the same checkpoint scored 0.026 on the ladder term during
training's own validation pass and 0.208 under that protocol.

So masking here mirrors training: a RANDOM SUBSET of eligible legs, at
--mask-ratio (default 0.10, lower than training's 0.15 to keep
double-masked pairs rare). On top of that, every baseline and the model
are scored on IDENTICAL positions, and a target is only scored for a
mechanism when the context that mechanism's baseline needs is itself
unmasked:

  ladder: scored only when the PARTNER leg is not masked, since that is
          exactly the condition under which partner_price is defined.
  mece:   the complement baseline is scored only when every sibling leg
          in the basket is unmasked, for the same reason.

Excluded targets are counted and reported rather than silently dropped.

THE BASELINES

  constant_0.5   always predict 0.50. The dumbest possible predictor.
  train_mean     always predict the TRAIN split's mean observed price.
                 Beating constant_0.5 but not this means the model has
                 learned the unconditional price level and nothing else.
  partner_price  (ladder) predict this leg's price to be its partner
                 rung's observed price -- i.e. "assume no violation".
                 Strong: adjacent rungs are usually close.
  basket_uniform (MECE) predict 1/k for a k-leg basket. Uses the
                 sum-to-1 constraint and nothing else.
  complement     (MECE) predict 1 - sum(observed sibling prices). The
                 naive arithmetic the rule-based strategy effectively
                 does, and the hardest of these to beat.

WHAT THIS DOES NOT DO: compute PnL, apply execution costs, or decide
tradeability -- that is the separate evaluation against baselines.py's
rule-based strategies. This answers the strictly prior question of
whether the fair-value estimates are better than guessing.

Run from stg_infra/:
    python evaluate_baselines.py --split val
    python evaluate_baselines.py --split test --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))  # data_windows.py lives here in this repo layout

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.inference import eligible_targets, masked_forward  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, TrainingConfig, true_prices  # noqa: E402
from model.training_objective import MaskedLegReconstructionObjective  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]


class Accum:
    """Squared-error accumulator, one per (mechanism, predictor)."""

    def __init__(self):
        self.sse = 0.0
        self.n = 0

    def add(self, pred: float, true: float):
        self.sse += (pred - true) ** 2
        self.n += 1

    @property
    def mse(self):
        return self.sse / self.n if self.n else float("nan")


def _train_mean_price(store, train_chunks) -> float:
    total, count = 0.0, 0
    for c in train_chunks:
        ct = store.materialize_chunk(c)
        prices = true_prices(ct["features"])[ct["mask"]]
        total += float(prices.sum())
        count += prices.numel()
        del ct
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--mask-ratio", type=float, default=0.10,
                    help="fraction of eligible legs hidden per chunk. Lower keeps "
                         "double-masked ladder pairs rare; 0 would score nothing.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    months = args.months or PILOT_MONTHS
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    store = MonthlyBundleStore(build_month_paths(months, _REPO_ROOT / "cache"), verbose=False)
    print(store.summary(), "\n")

    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg: TrainingConfig = ckpt.get("config", TrainingConfig())
    print(f"Loaded {ckpt_path.name}: epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss')} selected_by={ckpt.get('selected_by')!r}")

    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=max(cfg.chunk_len, args.chunk_len))
    model.load_state_dict(ckpt["model"])
    model.eval()
    objective = MaskedLegReconstructionObjective(raw_feature_dim=cfg.raw_feature_width,
                                                 mask_ratio=cfg.mask_ratio)
    objective.load_state_dict(ckpt["objective"])

    ranges = build_split_ranges(store.timestamps)
    chunks = chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=cfg.min_chunk_len)
    target_chunks = [c for c in chunks if c.split == args.split]
    train_chunks = [c for c in chunks if c.split == "train"]
    if not target_chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")
    print(f"{len(target_chunks)} {args.split} chunk(s), {len(train_chunks)} train chunk(s)")
    print(f"mask_ratio={args.mask_ratio} (targets are a random subset, as in training)\n")

    tm = _train_mean_price(store, train_chunks)
    print(f"train-split mean observed price: {tm:.4f}\n")

    acc = {
        ("mece", "model"): Accum(), ("mece", "constant_0.5"): Accum(),
        ("mece", "train_mean"): Accum(), ("mece", "basket_uniform"): Accum(),
        ("mece", "complement"): Accum(),
        # PAIRED accumulators, filled ONLY on the positions where
        # `complement` is defined (every sibling unmasked). Without these
        # the headline comparison is unsound: `complement` is scored on a
        # strict SUBSET of the model's positions, so "model 0.0157 beats
        # complement 0.0168" compares different samples and a 6% edge
        # could be an artifact of which targets each one had to attempt.
        # Any claim that the model beats the naive arithmetic must come
        # from these, not from the full-coverage row.
        ("mece", "model@complement"): Accum(),
        ("mece", "basket_uniform@complement"): Accum(),
        ("ladder", "model"): Accum(), ("ladder", "constant_0.5"): Accum(),
        ("ladder", "train_mean"): Accum(), ("ladder", "partner_price"): Accum(),
        # the leg's own previous price, plus the matched-subset partners so
        # model and baselines are compared on identical positions
        ("mece", "last_value"): Accum(), ("mece", "model@last_value"): Accum(),
        ("ladder", "last_value"): Accum(), ("ladder", "model@last_value"): Accum(),
        ("ladder", "partner@last_value"): Accum(),
    }
    excluded_ladder_both = 0   # partner also masked -> not comparable, counted not hidden
    excluded_mece_sibs = 0
    excluded_mece_nolast = 0   # no prior observation -> last_value undefined
    excluded_ladder_nolast = 0

    gen = torch.Generator().manual_seed(args.seed)

    for ci, c in enumerate(target_chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)

        # Random subset of mechanism-eligible legs -- the training-like
        # protocol, so most partners/siblings stay visible.
        elig = eligible_targets(features, mask, adj)
        target_mask = elig & (torch.rand(elig.shape, generator=gen) < args.mask_ratio)
        if int(target_mask.sum()) == 0:
            del ct
            continue

        # ------------------------------------------------------------------
        # THE BASELINE THAT WAS MISSING: the leg's own last observed price.
        #
        # Masking here is a RANDOM SUBSET of (t, leg) positions, so a leg
        # hidden at t is almost always visible at t-1. Its own previous price
        # is therefore in the model's input -- and available to a baseline.
        # `diagnose_staleness.py` measured that 90% of consecutive snapshot
        # observations show NO price change at all, which makes "it is still
        # worth what it was last worth" an extremely strong predictor in this
        # market, and the obvious one to beat.
        #
        # Omitting it meant the model was only ever compared against
        # cross-sectional rules (partner, complement, uniform) and never
        # against the temporal one. carry-forward below uses only positions
        # that are BOTH observed and NOT masked, so the baseline sees exactly
        # what the model sees -- never a value hidden from the model.
        # ------------------------------------------------------------------
        visible = mask & (~target_mask)
        last_px = torch.full_like(prices, float("nan"))
        carry = torch.full((prices.shape[1],), float("nan"), dtype=prices.dtype)
        for t_ in range(prices.shape[0]):
            last_px[t_] = carry
            carry = torch.where(visible[t_], prices[t_], carry)

        with torch.no_grad():
            h = masked_forward(model, features, mask, adj, target_mask,
                               objective.mask_token, cfg.raw_feature_width)
            mece_adj = adj.get("mece_leg_to_basket")
            ladder_adj = adj.get("ladder_monotonic")
            mece_out = model.mece_head(h, mece_adj) if mece_adj is not None else {}
            ladder_out = model.ladder_head(h, ladder_adj) if ladder_adj is not None else {}

        T = features.shape[0]
        for t in range(T):
            tgt = target_mask[t].nonzero(as_tuple=True)[0]
            if tgt.numel() == 0:
                continue
            tgt_set = set(tgt.tolist())

            # ---- MECE ----
            res = mece_out.get(t)
            if res is not None:
                leg_idx, hub_idx, fair = res["leg_idx"], res["hub_idx"], res["fair_price"]
                for leg in tgt.tolist():
                    hits = (leg_idx == leg).nonzero(as_tuple=True)[0]
                    if hits.numel() == 0:
                        continue
                    obs = float(prices[t, leg])
                    acc[("mece", "model")].add(float(fair[hits[0]]), obs)
                    acc[("mece", "constant_0.5")].add(0.5, obs)
                    acc[("mece", "train_mean")].add(tm, obs)
                    lv = float(last_px[t, leg])
                    if lv == lv:                      # not NaN: a prior observation exists
                        acc[("mece", "last_value")].add(lv, obs)
                        acc[("mece", "model@last_value")].add(float(fair[hits[0]]), obs)
                    else:
                        excluded_mece_nolast += 1

                    hub = hub_idx[hits[0]]
                    sib_pos = (hub_idx == hub).nonzero(as_tuple=True)[0]
                    k = sib_pos.numel()
                    if k > 0:
                        acc[("mece", "basket_uniform")].add(1.0 / k, obs)
                    siblings = [int(leg_idx[p]) for p in sib_pos.tolist() if int(leg_idx[p]) != leg]
                    if siblings and not any(s in tgt_set for s in siblings):
                        comp = 1.0 - sum(float(prices[t, s]) for s in siblings)
                        acc[("mece", "complement")].add(comp, obs)
                        # record the model and the structural baseline on
                        # THESE SAME positions, for a like-for-like read
                        acc[("mece", "model@complement")].add(float(fair[hits[0]]), obs)
                        if k > 0:
                            acc[("mece", "basket_uniform@complement")].add(1.0 / k, obs)
                    else:
                        excluded_mece_sibs += 1

            # ---- LADDER ----
            res = ladder_out.get(t)
            if res is not None:
                a_idx, b_idx = res["leg_a_idx"], res["leg_b_idx"]
                fa, fb = res["fair_a"], res["fair_b"]
                for leg in tgt.tolist():
                    for side, own_idx, other_idx, own_fair in (
                        ("a", a_idx, b_idx, fa), ("b", b_idx, a_idx, fb),
                    ):
                        for p in (own_idx == leg).nonzero(as_tuple=True)[0].tolist():
                            partner = int(other_idx[p])
                            if partner in tgt_set:
                                # partner hidden too -> partner_price undefined,
                                # so scoring here would compare unlike things
                                excluded_ladder_both += 1
                                continue
                            obs = float(prices[t, leg])
                            acc[("ladder", "model")].add(float(own_fair[p]), obs)
                            acc[("ladder", "constant_0.5")].add(0.5, obs)
                            acc[("ladder", "train_mean")].add(tm, obs)
                            acc[("ladder", "partner_price")].add(float(prices[t, partner]), obs)
                            lv = float(last_px[t, leg])
                            if lv == lv:              # not NaN
                                acc[("ladder", "last_value")].add(lv, obs)
                                acc[("ladder", "model@last_value")].add(float(own_fair[p]), obs)
                                acc[("ladder", "partner@last_value")].add(
                                    float(prices[t, partner]), obs)
                            else:
                                excluded_ladder_nolast += 1
        del ct
        print(f"  chunk {ci + 1}/{len(target_chunks)} done", flush=True)

    print("\n" + "=" * 80)
    print(f"{args.split.upper()} SPLIT -- MSE against observed price (lower is better)")
    print("Model and every baseline scored on IDENTICAL positions.")
    print("=" * 80)
    print(f"{'mechanism':<10} {'predictor':<16} {'MSE':>10} {'RMSE(cents)':>13} {'n':>10}")
    print("-" * 80)

    beaten = []
    for mech, names in (("mece", ["model", "constant_0.5", "train_mean", "basket_uniform",
                                  "last_value"]),
                        ("ladder", ["model", "constant_0.5", "train_mean", "partner_price",
                                    "last_value"])):
        m = acc[(mech, "model")]
        if m.n == 0:
            print(f"{mech:<10} (no scored targets -- try a larger --mask-ratio or more chunks)")
            print("-" * 80)
            continue
        for name in names:
            a = acc[(mech, name)]
            if a.n == 0:
                continue
            flag = ""
            if name != "model" and a.mse < m.mse:
                flag = "  <-- BEATS THE MODEL"
                beaten.append((mech, name))
            print(f"{mech:<10} {name:<16} {a.mse:>10.4f} {100 * (a.mse ** 0.5):>13.1f} {a.n:>10,}{flag}")
        print("-" * 80)

    # --- the like-for-like LAST_VALUE comparison, on identical positions ---
    # The table above scores `model` on every target but `last_value` only on
    # targets that HAVE a prior observation, so those two rows are not
    # comparable and the "BEATS THE MODEL" flag there can mislead. This block
    # restricts both to the same positions.
    for mech, extra in (("mece", None), ("ladder", "partner@last_value")):
        lv = acc[(mech, "last_value")]
        mlv = acc[(mech, "model@last_value")]
        if not lv.n or not mlv.n:
            continue
        print()
        print("=" * 80)
        print(f"{mech.upper()}, MATCHED SUBSET -- only positions where the leg has a PRIOR")
        print("observation, so `last_value` (its own previous price) is computable.")
        print("90% of consecutive observations show no price change, so this is the")
        print("single strongest naive predictor in this market and the one to beat.")
        print("=" * 80)
        print(f"{'predictor':<26} {'MSE':>10} {'RMSE(cents)':>13} {'n':>10}")
        print("-" * 80)
        rows = [("model@last_value", mlv), ("last_value", lv)]
        if extra and acc[(mech, extra)].n:
            rows.append((extra, acc[(mech, extra)]))
        for label, a in rows:
            flag = ""
            if label != "model@last_value" and a.mse < mlv.mse:
                flag = "  <-- BEATS THE MODEL"
                beaten.append((mech, label))
            print(f"{label:<26} {a.mse:>10.4f} {100 * (a.mse ** 0.5):>13.1f} {a.n:>10,}{flag}")
        print("-" * 80)
        if lv.mse < mlv.mse:
            print("  The leg's own previous price predicts it better than the model does.")
            print("  The model's advantage over cross-sectional baselines (partner, complement)")
            print("  does NOT extend to the temporal one, and the headline fair-value claim")
            print("  must be restated against this baseline.")
        else:
            print(f"  Model beats last_value by "
                  f"{100 * (lv.mse - mlv.mse) / lv.mse:.1f}% MSE on identical positions.")

    # --- the like-for-like MECE comparison, on identical positions ---
    comp = acc[("mece", "complement")]
    if comp.n:
        print()
        print("=" * 80)
        print("MECE, MATCHED SUBSET -- only positions where every sibling is unmasked,")
        print("so `complement` (1 - sum of observed siblings) is actually computable.")
        print("This is the comparison that counts: same positions, same information.")
        print("=" * 80)
        print(f"{'predictor':<26} {'MSE':>10} {'RMSE(cents)':>13} {'n':>10}")
        print("-" * 80)
        mm = acc[("mece", "model@complement")]
        for label, a in (("model", mm),
                         ("complement", comp),
                         ("basket_uniform", acc[("mece", "basket_uniform@complement")])):
            if a.n == 0:
                continue
            flag = ""
            if label != "model" and a.mse < mm.mse:
                flag = "  <-- BEATS THE MODEL"
                beaten.append(("mece(matched)", label))
            print(f"{label:<26} {a.mse:>10.4f} {100 * (a.mse ** 0.5):>13.1f} {a.n:>10,}{flag}")
        print("-" * 80)
        if mm.n and comp.n:
            rel = 100.0 * (comp.mse - mm.mse) / comp.mse
            print(f"model vs complement on identical positions: "
                  f"{rel:+.1f}% {'better' if rel > 0 else 'WORSE'}")

    print(f"\nexcluded (partner also masked, ladder):      {excluded_ladder_both:,}")
    print(f"excluded (a sibling also masked, mece comp): {excluded_mece_sibs:,}")

    print()
    if beaten:
        print("VERDICT: a trivial predictor beats the model on: "
              + ", ".join(f"{m}/{n}" for m, n in beaten))
        print("The model has not learned to price those legs from their neighbours, so a PnL\n"
              "result built on them would not be meaningful yet.")
    else:
        print("VERDICT: the model beats every trivial predictor on both mechanisms, on\n"
              "identical positions with identical information. That clears the gate to\n"
              "evaluate it as an arbitrage signal (selection and coverage against\n"
              "baselines.py), which is a separate question from this one.")


if __name__ == "__main__":
    main()