"""
evaluate_arbitrage_ladder.py

The LADDER counterpart to evaluate_arbitrage.py, on the held-out test
split, with the same cost model and the same metric definitions so both
mechanisms' numbers sit in one comparable frame.

WHY THIS MECHANISM IS THE ONE WORTH TESTING. The MECE evaluation reached
a negative result for a structural reason that does NOT apply here:
MeceOutputHead's softmax forces predicted leg prices to sum to exactly 1,
which makes the model's basket-level deviation identically equal to the
naive rule's, so the model can add nothing at that level by construction.
LadderOutputHead has no such constraint -- fair_a and fair_b are produced
independently (subject only to fair_a >= fair_b), so the model's view of
a pair is genuinely its own and CAN disagree with the observed prices.

Three further differences all point the same way:
  - 2 legs per position instead of ~4.1, so roughly half the per-leg
    friction that made MECE unprofitable.
  - The project's own findings report ~48-50% same-side violation rates
    in weather ladders -- a much denser signal than MECE's.
  - The model is 4.2x better than the strongest baseline on ladder
    fair-value accuracy (MSE 0.0217 vs partner_price 0.0915), against
    only ~1.25x on MECE.

ECONOMIC LOGIC (mirrors pairwise_monotonicity_pnl_backtest.py). For
adjacent rungs A (lower threshold) and B (higher threshold) in the same
family, crossing B implies crossing A, so no-arbitrage requires
yes_price(A) >= yes_price(B). A violation is yes_price(A) < yes_price(B).
The position "buy YES-A, buy NO-B" costs price_A + (100 - price_B) cents
and pays out:
    outcome < A:       NO-B pays $1, YES-A pays $0  -> $1
    A <= outcome < B:  NO-B pays $1, YES-A pays $1  -> $2
    outcome >= B:      NO-B pays $0, YES-A pays $1  -> $1
Payout is ALWAYS >= $1, so the guaranteed minimum profit per contract is
    100 - (price_A + (100 - price_B)) = price_B - price_A = the gap.
That guaranteed minimum (not the optional upside from landing between the
strikes) is the conservative PnL basis, exactly as the project's own
ladder backtest does it.

FEES ARE CHARGED ON WHAT YOU ACTUALLY BUY: the YES-A contract at price_A,
and the NO-B contract at (100 - price_B). Kalshi's fee is a function of
the traded contract's own price, so using price_B for the second leg
would misprice the fee on every position.

WHAT THE MODEL DECIDES: which flagged violations to trade. Both
strategies always trade BOTH legs of any pair they take -- that is what
makes the payoff guaranteed, and the MECE work already showed what
happens when a subset is traded while whole-position profit is still
credited. The criterion is whether the model AGREES the observed prices
are near fair. A violation where the model's fair value for one leg is
far from that leg's last print looks like a stale or unrepresentative
quote rather than a real dislocation -- and a stale quote is not
tradeable at the price being differenced.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from evaluate_arbitrage import (  # noqa: E402
    CONTRACTS_PER_OPPORTUNITY, PER_LEG_HAIRCUT_CENTS, PRICE_SCALE,
    Strategy, deduplicate_opportunities, taker_fee_dollars,
)
from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.inference import masked_forward  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, TrainingConfig, true_prices  # noqa: E402
from model.training_objective import MaskedLegReconstructionObjective  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
GAP_THRESHOLD = 0.01   # minimum violation gap in dollars to trade


class LadderStrategy(Strategy):
    """Ladder positions are always exactly 2 legs, and the SECOND leg is
    bought as NO at (100 - price_B), so its fee differs from the MECE
    case. Overridden here rather than bent into the base class."""

    def add_pair(self, gap_dollars: float, price_a_cents: float, price_b_cents: float,
                 contracts: float, haircut_cents: float = PER_LEG_HAIRCUT_CENTS,
                 haircut_b_cents: float = None):
        """haircut_b_cents defaults to haircut_cents, giving the original flat
        2-leg model. Passing both separately is what lets each leg carry its OWN
        measured spread instead of one assumed constant for the whole market."""
        gross = gap_dollars * contracts
        fee = (taker_fee_dollars(price_a_cents, contracts)          # buy YES-A at price_A
               + taker_fee_dollars(100.0 - price_b_cents, contracts))  # buy NO-B at 100-price_B
        hb = haircut_cents if haircut_b_cents is None else haircut_b_cents
        haircut = ((haircut_cents + hb) / PRICE_SCALE) * contracts
        self.idealized.append(gross)
        self.fees.append(fee)
        self.haircuts.append(haircut)
        self.legs_traded.append(2)


class SpreadLookup:
    """Per-leg measured spreads, with an explicit and REPORTED fallback chain.

    The file is keyed by (leg_a, leg_b) but its spread columns are sparse --
    ~2% of rows carry a value. A join that silently falls back to a series
    median would report an assumed constant under the name 'measured spread',
    which is the same class of error as the flat 1c haircut it replaces. So
    every lookup returns its SOURCE, and the caller prints the split.

    Resolution order, most specific first:
      exact   -- this very pair, both legs' spreads present
      ticker  -- each leg's own median spread, measured on other pairs
      series  -- the market family's median (e.g. all KXBTCD)
      none    -- no information; caller decides (default: flat assumption)
    """

    def __init__(self, pair, ticker, series, global_median):
        self.pair, self.ticker, self.series = pair, ticker, series
        self.global_median = global_median

    @classmethod
    def load(cls, path, verbose=True):
        import pandas as pd
        from collections import defaultdict

        df = pd.read_parquet(path, columns=["leg_a", "leg_b", "spread_a", "spread_b"])
        n_rows = len(df)
        df = df.dropna(subset=["spread_a", "spread_b"])
        if verbose:
            print(f"  spreads file: {n_rows:,} rows, "
                  f"{len(df):,} with BOTH legs' spreads present "
                  f"({100.0 * len(df) / max(n_rows, 1):.1f}%)")

        pair, per_ticker = {}, defaultdict(list)
        for a, b, sa, sb in zip(df["leg_a"].astype(str), df["leg_b"].astype(str),
                                df["spread_a"], df["spread_b"]):
            pair[(a, b)] = (float(sa), float(sb))
            per_ticker[a].append(float(sa))
            per_ticker[b].append(float(sb))

        def _med(v):
            v = sorted(v)
            n = len(v)
            return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0

        ticker = {t: _med(v) for t, v in per_ticker.items()}
        per_series = defaultdict(list)
        for t, v in ticker.items():
            per_series[t.split("-", 1)[0]].append(v)
        series = {s: _med(v) for s, v in per_series.items()}
        gm = _med(list(ticker.values())) if ticker else None
        if verbose and gm is not None:
            print(f"  distinct tickers with a spread: {len(ticker):,} "
                  f"across {len(series):,} series; global median {gm:.3f}c/leg")
        return cls(pair, ticker, series, gm)

    def get(self, a: str, b: str):
        """-> (spread_a_cents, spread_b_cents, source)"""
        hit = self.pair.get((a, b))
        if hit is not None:
            return hit[0], hit[1], "exact"
        hit = self.pair.get((b, a))
        if hit is not None:
            return hit[1], hit[0], "exact"
        sa, sb = self.ticker.get(a), self.ticker.get(b)
        if sa is not None and sb is not None:
            return sa, sb, "ticker"
        ssa = self.series.get(a.split("-", 1)[0])
        ssb = self.series.get(b.split("-", 1)[0])
        if ssa is not None and ssb is not None:
            return ssa, ssb, "series"
        return None, None, "none"


def _haircuts(o, measured: bool, multiplier: float, flat_cents: float):
    """Per-leg haircut in cents. Under `measured`, each leg pays its own
    spread times `multiplier` (0.5 = cross half the spread from mid).
    Opportunities with no spread information keep the flat assumption and
    are counted separately by the caller."""
    if measured and o.get("spread_a") is not None:
        return multiplier * o["spread_a"], multiplier * o["spread_b"]
    return flat_cents, flat_cents


def build_ladder_strategies(opportunities, threshold: float, haircut_cents: float,
                            mode: str = "excess", measured: bool = False,
                            multiplier: float = 0.5):
    """naive = trade every flagged violation; stgat = trade only those the
    model agrees are real (both legs priced close to the model's own fair
    value). Both always trade BOTH legs."""
    naive = LadderStrategy("naive_all_violations")
    stgat = LadderStrategy("stgat_selected_violations")
    for o in opportunities:
        ha, hb = _haircuts(o, measured, multiplier, haircut_cents)
        naive.add_pair(o["deviation"], o["price_a"], o["price_b"],
                       CONTRACTS_PER_OPPORTUNITY, ha, hb)
        stat = o.get("excess") if mode == "excess" else o.get("model_gap")
        if stat is not None and stat <= threshold:
            stgat.add_pair(o["deviation"], o["price_a"], o["price_b"],
                           CONTRACTS_PER_OPPORTUNITY, ha, hb)
    return naive, stgat


def _subset_strategy(opps, haircut_cents, label="subset",
                     measured: bool = False, multiplier: float = 0.5):
    """Score an arbitrary pre-selected list of opportunities under the same
    cost model, so model-chosen and control-chosen subsets are comparable."""
    s = LadderStrategy(label)
    for o in opps:
        ha, hb = _haircuts(o, measured, multiplier, haircut_cents)
        s.add_pair(o["deviation"], o["price_a"], o["price_b"],
                   CONTRACTS_PER_OPPORTUNITY, ha, hb)
    return s


def _random_band(pool, n_select, haircut_cents, n_iter=2000, seed=0):
    """Sampling distribution of each metric for a RANDOM subset of the same
    size, drawn from the same pool.

    This is the noise floor. At n=141 out of ~2,500, a subset's median and
    Sharpe move around a great deal by chance alone. Any model-selected
    result that sits inside this band is indistinguishable from drawing at
    random, however clean its sweep looks.
    """
    import random as _random
    rng = _random.Random(seed)
    acc = {"mean_pnl_usd": [], "median_pnl_usd": [], "win_rate": [],
           "cross_sectional_sharpe": []}
    for _ in range(n_iter):
        sub = rng.sample(pool, n_select)
        m = _subset_strategy(sub, haircut_cents).metrics("realistic", 1)
        for k in acc:
            v = m.get(k)
            if v is not None:
                acc[k].append(v)
    out = {}
    for k, vals in acc.items():
        if not vals:
            out[k] = None
            continue
        vals.sort()
        def _p(q):
            return vals[min(len(vals) - 1, max(0, int(q * len(vals))))]
        out[k] = (_p(0.05), _p(0.50), _p(0.95))
    return out


def _row(m, label, base_n=None):
    if not m.get("n_opportunities"):
        return f"{label:<26} (no opportunities)"
    sh = f"{m['cross_sectional_sharpe']:.3f}" if m["cross_sectional_sharpe"] else "n/a"
    kept = f"{100 * m['n_opportunities'] / base_n:>6.1f}%" if base_n else "      -"
    return (f"{label:<26} {m['n_opportunities']:>7} {kept} ${m['total_pnl_usd']:>12,.2f} "
            f"${m['mean_pnl_usd']:>9,.2f} ${m['median_pnl_usd']:>9,.2f} "
            f"{100 * m['win_rate']:>6.1f}% {sh:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--threshold", type=float, default=GAP_THRESHOLD)
    ap.add_argument("--max-model-gap", type=float, default=0.05,
                    help="trade a flagged violation only if BOTH legs are within this "
                         "distance (dollars) of the model's own fair value")
    ap.add_argument("--mask-ratio", type=float, default=0.10,
                    help="fraction of ticker legs hidden per chunk. Low keeps a pair's PARTNER "
                         "visible, which is what lets the model price a rung from its neighbour. "
                         "Only pairs with exactly ONE hidden leg carry a usable model signal.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter-mode", choices=["excess", "agreement"], default="excess",
                    help="'excess' (default) filters on model_gap - gap/2, removing the floor "
                         "the monotonicity constraint forces; 'agreement' filters on model_gap "
                         "itself, which provably tracks gap size and discards the best trades.")
    ap.add_argument("--dedup", choices=["first", "last", "max", "none"], default="first")
    ap.add_argument("--spreads", default=None,
                    help="path to pairwise_monotonicity_taker_side_results_corrected.parquet. "
                         "When given, an additional results block prices every trade with ITS "
                         "OWN measured spread instead of the flat haircut.")
    ap.add_argument("--spread-multiplier", type=float, default=0.5,
                    help="fraction of the measured spread paid per leg. 0.5 = cross half the "
                         "spread from mid (the usual taker assumption); 1.0 = pay the full "
                         "quoted spread, the pessimistic bound. Both are reported.")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the gap-size and random-subset controls. The controls are on by "
                         "default because without them a filter that merely selects larger gaps "
                         "is indistinguishable from one that selects better trades.")
    ap.add_argument("--control-iters", type=int, default=2000,
                    help="random draws for the noise band.")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep --max-model-gap. Tune on VAL, never on test.")
    ap.add_argument("--sweep-friction", action="store_true")
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

    print(f"LADDER MECHANISM | checkpoint {ckpt_path.name} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss'):.4f})")
    print(f"Split: {args.split.upper()}  |  gap threshold ${args.threshold:.3f}  "
          f"|  max model gap ${args.max_model_gap:.3f}")
    print(f"Cost model: {CONTRACTS_PER_OPPORTUNITY:.0f} contracts/leg, "
          f"{PER_LEG_HAIRCUT_CENTS:.1f}c haircut/leg, 2 legs per position\n")

    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len,
                                      min_chunk_len=cfg.min_chunk_len) if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")

    spreads = None
    from collections import Counter as _Counter
    spread_src_counts = _Counter()
    if args.spreads:
        sp = Path(args.spreads)
        if not sp.is_absolute():
            for cand in (_THIS_DIR / sp, _REPO_ROOT / sp, Path.cwd() / sp):
                if cand.exists():
                    sp = cand
                    break
        if not sp.exists():
            raise SystemExit(f"spreads file not found: {args.spreads}")
        print("Loading measured spreads...")
        spreads = SpreadLookup.load(sp)
        print()

    opportunities = []
    n_pairs_seen = n_violations = n_scored = 0
    gen = torch.Generator().manual_seed(args.seed)

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)
        ladder_adj = adj.get("ladder_monotonic")
        if ladder_adj is None:
            del ct
            continue

        # MASK A SMALL RANDOM SUBSET, NOT EVERY LEG.
        #
        # An earlier version used `mask & is_ticker`, hiding EVERY ticker
        # leg at once. For a ladder pair that hides BOTH rungs, so the
        # model had to price each one without its partner -- the single
        # most informative neighbour it has. Its fair values then landed
        # ~15-20c from observed (consistent with the 0.208 MSE that
        # protocol produced during scoring, versus 0.0217 when partners
        # are visible), and the agreement filter rejected 100% of
        # opportunities. The model looked useless when in fact it had
        # been blindfolded.
        #
        # Here a small fraction is hidden, so a hidden leg is priced with
        # its partner and the rest of the graph VISIBLE. Only pairs with
        # EXACTLY ONE hidden leg are scored: if neither is hidden the
        # model can see the price it is being asked to judge (the gap
        # collapses to noise around zero and means nothing), and if both
        # are hidden we are back to the blindfolded case.
        is_ticker = features[..., -1] == 0.0
        target_mask = (mask & is_ticker) & (torch.rand(mask.shape, generator=gen) < args.mask_ratio)
        with torch.no_grad():
            h = masked_forward(model, features, mask, adj, target_mask,
                               objective.mask_token, cfg.raw_feature_width)
            ladder_out = model.ladder_head(h, ladder_adj)

        node_ids = ct.get("node_ids")
        for t in range(features.shape[0]):
            edges = ladder_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            res = ladder_out.get(t)
            a_idx, b_idx = edges.edge_index[0], edges.edge_index[1]
            ts = ct["timestamps"][t]
            day = getattr(ts, "date", lambda: ts)()

            for k in range(a_idx.shape[0]):
                a, b = int(a_idx[k]), int(b_idx[k])
                if not (bool(mask[t, a]) and bool(mask[t, b])):
                    continue
                n_pairs_seen += 1
                pa, pb = float(prices[t, a]), float(prices[t, b])
                # no-arbitrage requires price_A >= price_B; a violation is the reverse
                gap = pb - pa
                if gap <= args.threshold:
                    continue
                n_violations += 1

                # Model signal only from the leg that was actually hidden
                # (see the masking note above). Exactly one hidden -> usable.
                model_gap = excess = None
                a_hidden, b_hidden = bool(target_mask[t, a]), bool(target_mask[t, b])
                if res is not None and (a_hidden != b_hidden):
                    if a_hidden:
                        model_gap = abs(pa - float(res["fair_a"][k]))
                    else:
                        model_gap = abs(pb - float(res["fair_b"][k]))
                    # EXCESS above the structurally forced floor. Because
                    # LadderOutputHead guarantees fair_a >= fair_b, a violation
                    # of size g forces model_gap >= g/2 (verified: min ratio
                    # 0.5014 over 400k random cases). Filtering on model_gap
                    # itself therefore filters on gap SIZE -- it throws away the
                    # most profitable trades, which is exactly the monotonic
                    # degradation the first sweep showed.
                    #
                    # The excess is what is left after removing that floor:
                    #   excess = model_gap - g/2   >= 0
                    # excess ~ 0  -> the model's only quarrel with these prices IS
                    #                the violation; it reads as genuine.
                    # excess large -> the model thinks one rung is wrong well
                    #                BEYOND the violation, the signature of a
                    #                stale or unrepresentative last print.
                    # Measured correlation with gap: -0.145, versus +0.425 for
                    # model_gap -- so this is a genuinely different statistic and
                    # not a relabelling of the same one.
                    excess = model_gap - gap / 2.0
                    n_scored += 1

                ticker_a = str(node_ids[a]) if node_ids else None
                ticker_b = str(node_ids[b]) if node_ids else None
                spread_a = spread_b = None
                spread_src = "none"
                if spreads is not None and ticker_a and ticker_b:
                    spread_a, spread_b, spread_src = spreads.get(ticker_a, ticker_b)
                    spread_src_counts[spread_src] += 1

                opportunities.append({
                    "deviation": gap,
                    "price_a": pa * PRICE_SCALE,
                    "price_b": pb * PRICE_SCALE,
                    "model_gap": model_gap,
                    "excess": excess if model_gap is not None else None,
                    "hub_id": (node_ids[a], node_ids[b]) if node_ids else (a, b),
                    "spread_a": spread_a,
                    "spread_b": spread_b,
                    "spread_src": spread_src,
                    "day": day,
                })
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} done", flush=True)

    raw_n = len(opportunities)
    opportunities = deduplicate_opportunities(opportunities, args.dedup)
    print(f"\npairs seen: {n_pairs_seen:,} | violations over ${args.threshold:.3f}: {n_violations:,}")
    if args.dedup != "none":
        print(f"Deduplication ('{args.dedup}'): {raw_n:,} sightings -> {len(opportunities):,} "
              f"tradeable opportunities")
    if not opportunities:
        raise SystemExit("No ladder opportunities found -- nothing to score.")
    print(f"mean |gap|: ${sum(o['deviation'] for o in opportunities) / len(opportunities):.4f}")
    n_with_signal = sum(1 for o in opportunities if o.get("model_gap") is not None)
    print(f"opportunities carrying a usable model signal (exactly one leg hidden): "
          f"{n_with_signal:,} of {len(opportunities):,} "
          f"({100 * n_with_signal / max(len(opportunities), 1):.1f}%)")
    if n_with_signal == 0:
        print("  WARNING: no model signal -- raise --mask-ratio.")
    print()

    naive, stgat = build_ladder_strategies(opportunities, args.max_model_gap,
                                           PER_LEG_HAIRCUT_CENTS, args.filter_mode)

    print("=" * 100)
    print(f"LADDER ARBITRAGE -- {args.split.upper()} split. Both strategies trade BOTH legs.")
    print("=" * 100)
    print(f"{'strategy':<26} {'n':>7} {'kept%':>7} {'total_pnl':>13} "
          f"{'mean':>10} {'median':>10} {'win%':>7} {'sharpe':>8}")
    print("-" * 100)
    for scen in ("idealized", "fees_only", "realistic"):
        print(_row(naive.metrics(scen, 1), f"naive / {scen}"))
        print(_row(stgat.metrics(scen, 1), f"stgat / {scen}", base_n=naive.n))
        print("-" * 100)

    if not args.no_control:
        # ------------------------------------------------------------------
        # THE CONTROL THAT DECIDES WHETHER THE MODEL ADDS ANYTHING.
        #
        # Under realistic costs a ladder trade loses when its gap is smaller
        # than the friction it pays. So ANY filter that happens to select
        # larger gaps will show a higher win rate and a higher median --
        # with or without a model. And `excess` does tilt that way
        # (corr(excess, gap) = -0.145), so the model-selected subset is not
        # a random draw on gap size.
        #
        # Three controls, all scored on exactly the same cost model and at
        # exactly the same n as the model's selection:
        #   gap-top-N (signal pool) -- can a pure gap-size rule match the
        #       model WITHIN the subset the model is able to judge? This is
        #       the scientific question: does `excess` carry information
        #       beyond gap size?
        #   gap-top-N (all)         -- can a pure gap-size rule match the
        #       model with no model at all? This is the practical question:
        #       is the STGAT load-bearing?
        #   random (band)           -- the noise floor at this n.
        #
        # If the model's row does not clear all three, the selection claim
        # does not survive, and the section-3 negative result stands after all.
        # ------------------------------------------------------------------
        key = "excess" if args.filter_mode == "excess" else "model_gap"
        signal_pool = [o for o in opportunities if o.get(key) is not None]
        selected = [o for o in signal_pool if o[key] <= args.max_model_gap]
        n_sel = len(selected)

        print()
        print("=" * 100)
        print(f"CONTROL: is {key} doing more than selecting large gaps?  "
              f"(n={n_sel}, realistic costs)")
        print("=" * 100)
        if n_sel < 2 or len(signal_pool) <= n_sel:
            print("  Not enough selected opportunities (or pool too small) to control against.")
        else:
            gap_pool_rows = sorted(signal_pool, key=lambda o: -o["deviation"])[:n_sel]
            gap_all_rows = sorted(opportunities, key=lambda o: -o["deviation"])[:n_sel]

            def _mg(rows):
                return sum(r["deviation"] for r in rows) / len(rows)

            print(f"{'subset':<26} {'n':>7} {'kept%':>7} {'total_pnl':>13} "
                  f"{'mean':>10} {'median':>10} {'win%':>7} {'sharpe':>8}  mean_gap")
            print("-" * 100)
            print(_row(naive.metrics("realistic", 1), "naive (all violations)")
                  + f"  ${_mg(opportunities):.4f}")
            print("-" * 100)
            for rows, label in ((selected, f"model: {key}<={args.max_model_gap:.2f}"),
                                (gap_pool_rows, "control: gap-top-N (signal)"),
                                (gap_all_rows, "control: gap-top-N (all)")):
                s = _subset_strategy(rows, PER_LEG_HAIRCUT_CENTS)
                print(_row(s.metrics("realistic", 1), label, base_n=naive.n)
                      + f"  ${_mg(rows):.4f}")
            print("-" * 100)

            band = _random_band(signal_pool, n_sel, PER_LEG_HAIRCUT_CENTS,
                                n_iter=args.control_iters, seed=args.seed)
            print(f"random subsets of the signal pool, n={n_sel}, "
                  f"{args.control_iters} draws -- 5th / 50th / 95th percentile:")
            for k, lab in (("mean_pnl_usd", "mean"), ("median_pnl_usd", "median"),
                           ("win_rate", "win%"), ("cross_sectional_sharpe", "sharpe")):
                b = band.get(k)
                if b is None:
                    continue
                scale = 100.0 if k == "win_rate" else 1.0
                sel_m = _subset_strategy(selected, PER_LEG_HAIRCUT_CENTS).metrics("realistic", 1)
                got = sel_m.get(k)
                got_s = f"{got * scale:.3f}" if got is not None else "n/a"
                inside = (got is not None and b[0] <= got <= b[2])
                verdict = "INSIDE band (= noise)" if inside else "outside band"
                print(f"   {lab:<8} {b[0] * scale:>9.3f} {b[1] * scale:>9.3f} "
                      f"{b[2] * scale:>9.3f}   model={got_s:>9}   {verdict}")
            print("-" * 100)
            print("  A model row that beats both gap controls AND sits above the random 95th")
            print("  percentile is a real selection edge. Anything less is not.")

    if spreads is not None:
        # ------------------------------------------------------------------
        # THE DECISIVE RUN: every trade pays ITS OWN measured spread.
        #
        # The flat 1c haircut is the single load-bearing assumption behind
        # every PnL number above, and it is most wrong exactly where the
        # gap-size control concentrates its capital: a 39c monotonicity
        # violation on an illiquid rung is far more plausibly a stale quote
        # than a fillable dislocation, and the measured spread on such a leg
        # runs to 26c at the 95th percentile.
        #
        # So this block is not a robustness check. It decides between:
        #   (a) gap-top-N is a real strategy and model selection is worthless
        #   (b) gap-top-N is an artifact of the flat haircut, and the model's
        #       ordinary-gap picks are what survives contact with execution
        # ------------------------------------------------------------------
        key = "excess" if args.filter_mode == "excess" else "model_gap"
        signal_pool = [o for o in opportunities if o.get(key) is not None]
        selected = [o for o in signal_pool if o[key] <= args.max_model_gap]
        n_sel = len(selected)

        print()
        print("=" * 100)
        print("MEASURED SPREADS -- each leg pays its own, not a flat assumption")
        print("=" * 100)
        total = max(len(opportunities), 1)
        print("Spread source for the traded opportunities:")
        for src in ("exact", "ticker", "series", "none"):
            n_src = sum(1 for o in opportunities if o.get("spread_src") == src)
            note = {"exact": "this very pair",
                    "ticker": "each leg's own median, from other pairs",
                    "series": "market-family median -- an ASSUMPTION, not a measurement",
                    "none": f"no data; falls back to the flat {PER_LEG_HAIRCUT_CENTS:.1f}c"}[src]
            print(f"  {src:<8} {n_src:>7,}  {100.0 * n_src / total:>5.1f}%   {note}")
        have = [o for o in opportunities if o.get("spread_a") is not None]
        if have:
            sp_all = sorted(s for o in have for s in (o["spread_a"], o["spread_b"]))
            n_sp = len(sp_all)
            def _q(p):
                return sp_all[min(n_sp - 1, int(p * n_sp))]
            print(f"\n  spread per leg over traded legs (c): p25={_q(0.25):.2f}  "
                  f"med={_q(0.50):.2f}  p75={_q(0.75):.2f}  p95={_q(0.95):.2f}")
        if not have:
            print("\n  No measured spreads matched. Nothing to report here.")
        else:
            gap_pool_rows = sorted(signal_pool, key=lambda o: -o["deviation"])[:n_sel]
            gap_all_rows = sorted(opportunities, key=lambda o: -o["deviation"])[:n_sel]

            def _msp(rows):
                v = [s for o in rows if o.get("spread_a") is not None
                     for s in (o["spread_a"], o["spread_b"])]
                return sum(v) / len(v) if v else float("nan")

            for mult in sorted({args.spread_multiplier, 1.0}):
                lab = ("half spread (cross from mid)" if abs(mult - 0.5) < 1e-9
                       else "full spread (pessimistic)" if abs(mult - 1.0) < 1e-9
                       else f"{mult:g} x spread")
                print()
                print(f"--- cost = {mult:g} x measured spread per leg -- {lab} ---")
                print(f"{'subset':<26} {'n':>7} {'kept%':>7} {'total_pnl':>13} "
                      f"{'mean':>10} {'median':>10} {'win%':>7} {'sharpe':>8}  mean_spread")
                print("-" * 100)
                # THE EX-ANTE RULE. gap-top-N ranks across the whole test
                # month, which is not available at 3am when the trade must be
                # taken. This rule uses ONLY quantities known at decision
                # time -- the two prices, the fee schedule, and the expected
                # spread -- and has NO free parameter to tune, so there is
                # nothing to fit on val and nothing to overfit on test:
                #
                #     take the trade iff  gap x C  >  fees + haircut
                #
                # It is the strategy the arithmetic of section 3d implies.
                ex_ante = []
                for o in opportunities:
                    ha, hb = _haircuts(o, True, mult, PER_LEG_HAIRCUT_CENTS)
                    cost = (taker_fee_dollars(o["price_a"], CONTRACTS_PER_OPPORTUNITY)
                            + taker_fee_dollars(100.0 - o["price_b"], CONTRACTS_PER_OPPORTUNITY)
                            + ((ha + hb) / PRICE_SCALE) * CONTRACTS_PER_OPPORTUNITY)
                    if o["deviation"] * CONTRACTS_PER_OPPORTUNITY > cost:
                        ex_ante.append(o)

                rows_to_show = [(opportunities, "naive (all violations)", None),
                                (ex_ante, "EX-ANTE: gap > own cost", naive.n)]
                if n_sel >= 2:
                    rows_to_show += [
                        (selected, f"model: {key}<={args.max_model_gap:.2f}", naive.n),
                        (gap_pool_rows, "control: gap-top-N (signal)", naive.n),
                        (gap_all_rows, "control: gap-top-N (all)", naive.n),
                    ]
                for rows, label, base in rows_to_show:
                    s = _subset_strategy(rows, PER_LEG_HAIRCUT_CENTS, measured=True,
                                         multiplier=mult)
                    print(_row(s.metrics("realistic", 1), label, base_n=base)
                          + f"  {_msp(rows):>10.2f}c")
                print("-" * 100)
            print("\n  READ THE EX-ANTE ROW CAREFULLY. Under this cost model realized PnL is a")
            print("  DETERMINISTIC function of gap, prices and spread, so 'take iff gross > cost'")
            print("  selects exactly the profitable trades and will show a 100% win rate BY")
            print("  CONSTRUCTION. That is not a forecasting result. It is the upper bound on")
            print("  what any selection rule can achieve GIVEN PERFECT KNOWLEDGE OF THE SPREAD,")
            print("  and the spread here is a period aggregate, not a live quote. The gap between")
            print("  this row and reality is exactly the error in forecasting execution cost --")
            print("  which is the only unobservable term in the equation, and therefore the only")
            print("  place a learned model can contribute anything at all.")
            print()
            print("  If the gap-top-N rows go negative here while the model row survives,")
            print("  the large-gap 'opportunities' were spread artifacts and selection matters.")
            print("  If every row goes negative, the strategy does not clear real execution")
            print("  costs and THAT is the finding -- consistent with the ~1% exploitation")
            print("  rate reported for apparent arbitrage on zero-fee Polymarket.")

    if args.sweep:
        print()
        print("=" * 100)
        print(f"MODEL FILTER SWEEP (mode={args.filter_mode!r}) -- realistic. Tune on VAL, never test.")
        print("Judge on MEAN/MEDIAN per opportunity: the filter trades fewer, so total falls anyway.")
        print("=" * 100)
        base = naive.metrics("realistic", 1)
        print(_row(base, "naive (all violations)"))
        print("-" * 100)
        for mg in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.00):
            _, s = build_ladder_strategies(opportunities, mg, PER_LEG_HAIRCUT_CENTS, args.filter_mode)
            label = ("excess" if args.filter_mode == "excess" else "model_gap")
            print(_row(s.metrics("realistic", 1), f"{label}<={mg:.2f}", base_n=naive.n))
        print("-" * 100)
        print(f"  Beat on MEAN: ${base['mean_pnl_usd']:,.2f}  (median ${base['median_pnl_usd']:,.2f})")

    if args.sweep_friction:
        print()
        print("=" * 100)
        print("EXECUTION-FRICTION SENSITIVITY -- the haircut is an ASSUMPTION; breakeven is the result")
        print("=" * 100)
        print(f"{'haircut(c)':>11} {'naive_total':>15} {'stgat_total':>15} "
              f"{'naive_median':>14} {'stgat_median':>14}")
        print("-" * 100)
        nb = sb = None
        for hc in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            n2, s2 = build_ladder_strategies(opportunities, args.max_model_gap, hc, args.filter_mode)
            nm, sm = n2.metrics("realistic", 1), s2.metrics("realistic", 1)
            nt = nm.get("total_pnl_usd", 0.0) or 0.0
            st = sm.get("total_pnl_usd", 0.0) or 0.0
            if nb is None and nt < 0:
                nb = hc
            if sb is None and st < 0:
                sb = hc
            print(f"{hc:>11.2f} ${nt:>14,.2f} ${st:>14,.2f} "
                  f"${nm.get('median_pnl_usd', 0.0):>13,.2f} ${sm.get('median_pnl_usd', 0.0):>13,.2f}")
        print("-" * 100)
        print(f"  naive turns negative at roughly {nb if nb else '>3'}c per leg")
        print(f"  STGAT turns negative at roughly {sb if sb else '>3'}c per leg")
        print("  Kalshi's minimum tick is 1c. A breakeven above 1c means the strategy survives")
        print("  paying a full tick of slippage on every leg; below it, it does not.")


if __name__ == "__main__":
    main()