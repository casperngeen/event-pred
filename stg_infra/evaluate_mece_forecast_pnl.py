"""
evaluate_mece_forecast_pnl.py

DOES THE MECE FORECASTING EDGE SURVIVE TRADING COSTS?

score_forecast.py established the forecast result: 0.853x the optimal
shrinkage oracle on held-out test (val 0.851x), with the model's
incremental gain over the oracle holding at ~14.8% across two splits that
offered different amounts of mean reversion. That is an MSE result. MSE is
not money, and the ladder side already showed how far apart those two can
be: measured spreads turned +$12,618 into +$948 / -$15,700.

This script closes that gap for MECE.

THE TRADE BEING EVALUATED. A MECE basket's legs must sum to $1 at
resolution. The forecast model predicts each leg's price at t+h, so it
predicts the basket sum at t+h, so it predicts the CHANGE in dislocation:

    dev(t)      = sum of observed leg prices at t, minus 1
    dev_pred    = sum of predicted leg prices at t+h, minus 1
    delta_pred  = dev_pred - dev(t)          <- the tradeable claim
    delta_real  = dev(t+h) - dev(t)          <- what happened

If the model says the sum will RISE, buy the basket at t and sell at t+h;
if it says the sum will FALL, sell at t and buy back. PnL per contract is
therefore sign(delta_pred) * delta_real, and the position is a round trip:
it crosses the spread on every leg TWICE.

WHY THAT MATTERS MORE THAN THE FORECAST QUALITY. The classic static basket
arbitrage -- sell a basket priced at 1.06, hold to resolution, collect 6c
-- crosses each leg ONCE, because settlement is free. The round-trip
forecast trade pays entry and exit on every leg, so a 5-leg basket crosses
10 times. At the measured median 4.67c per leg that is a large fixed cost
against a dislocation change usually worth a few cents. The static arb is
therefore the benchmark this strategy has to beat, not zero, and it is
computed here alongside for exactly that reason.

THE BASELINES, WHICH ARE THE POINT. alpha < 1 means basket dislocation
mean-reverts, so "bet on reversion" is already a profitable-looking rule
that needs NO MODEL AT ALL:

  always_revert   direction = -sign(dev(t)). One line. No model, no alpha.
  oracle_shrink   predicts dev(t+h) = alpha*dev(t) with alpha fitted ON THE
                  SCORED SPLIT, so it is an oracle and deliberately
                  optimistic. Its direction is also -sign(dev(t)) whenever
                  alpha < 1; it differs from always_revert only in WHICH
                  opportunities clear the entry threshold.
  random          same trade count, coin-flip direction. Isolates pure cost
                  drag: if every strategy loses the same amount, costs are
                  the finding and the signal is irrelevant.
  static_arb      |dev(t)| held to resolution, one-way costs. The existing
                  strategy this has to improve on.

A model strategy that is profitable but does not beat always_revert has
demonstrated that dislocation mean-reverts, which diagnose_ladder_reversion
already told us, and nothing about the STGAT.

COSTS. Spreads are MEASURED per leg via SpreadLookup's ticker/series
fallback chain, and every lookup reports its source so an assumed constant
cannot masquerade as a measurement -- the mistake that once set the sign of
the ladder headline. Fees use Kalshi's published taker formula,
0.07 * C * P * (1-P), charged per leg at both entry and exit prices.

Confidence intervals bootstrap over BASKETS, not over opportunities: the
same basket sighted at 2-hourly intervals is one bet repeated, not many
independent ones.

Read-only with respect to the checkpoint. Trains nothing.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from evaluate_arbitrage_ladder import SpreadLookup  # noqa: E402
from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.forecast_head import ResidualForecastHead  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, true_prices  # noqa: E402
from train_forecast import PILOT_MONTHS  # noqa: E402

PRICE_SCALE = 100.0
TAKER_FEE_RATE = 0.07
CONTRACTS_PER_LEG = 100.0
SLOT_LEGS_TOTAL = 3          # basket hub feature slots, stg/nodes/kalshi.py
VOLUME_SLOT = 4              # window_volume, per leg per 2-hour window
FLAT_SPREAD_CENTS = 1.0      # used only where no spread was ever measured


def taker_fee_dollars(price_cents: float, contracts: float) -> float:
    """Kalshi taker fee, per leg, rounded UP to the cent-hundredth.

    The round() before the ceil is not cosmetic. 0.07 * 100 * 0.5 * 0.5 is
    exactly 1.75, but in binary floating point it evaluates to
    1.7500000000000002, so a bare ceil(raw * 10000) returns 17501 and the
    fee comes back as 1.7501. The error is 0.01c per leg -- immaterial to
    any conclusion, but it applies to EVERY leg of every trade and it is
    the kind of thing that reads as carelessness in an appendix.
    evaluate_arbitrage.py has the same bare-ceil formula; the discrepancy
    is ~0.006% of a fee, so previously published numbers do not need
    recomputing, but new work should not inherit it.
    """
    p = max(0.0, min(1.0, price_cents / PRICE_SCALE))
    raw = TAKER_FEE_RATE * contracts * p * (1.0 - p)
    return math.ceil(round(raw * 10000, 6)) / 10000


class LegSpreads:
    """Per-leg spread in cents, from SpreadLookup's ticker/series medians.

    SpreadLookup is keyed by PAIRS because the ladder side is pairwise. A
    MECE basket has N independent legs, so only the per-ticker and
    per-series tables apply. Reusing them rather than inventing a second
    cost model keeps the two mechanisms comparable, which matters because
    the whole point of this project is contrasting them.
    """

    def __init__(self, lookup: SpreadLookup | None):
        self.lk = lookup
        self.sources = defaultdict(int)

    def get(self, ticker: str) -> float:
        if self.lk is None:
            self.sources["assumed_flat"] += 1
            return FLAT_SPREAD_CENTS
        v = self.lk.ticker.get(ticker)
        if v is not None:
            self.sources["ticker"] += 1
            return float(v)
        v = self.lk.series.get(str(ticker).split("-", 1)[0])
        if v is not None:
            self.sources["series"] += 1
            return float(v)
        if self.lk.global_median is not None:
            self.sources["global"] += 1
            return float(self.lk.global_median)
        self.sources["assumed_flat"] += 1
        return FLAT_SPREAD_CENTS


class Book:
    """Accumulates one strategy's trades. Everything in dollars."""

    def __init__(self, name: str, note: str = ""):
        self.name, self.note = name, note
        self.gross, self.fees, self.spreads = [], [], []
        self.baskets, self.wins = [], 0

    def add(self, gross: float, fees: float, spreads: float, basket_id):
        self.gross.append(gross)
        self.fees.append(fees)
        self.spreads.append(spreads)
        self.baskets.append(basket_id)
        if gross > 0:
            self.wins += 1

    @property
    def n(self):
        return len(self.gross)

    def totals(self):
        g, f, s = sum(self.gross), sum(self.fees), sum(self.spreads)
        return g, f, s, g - f - s

    def net_per_trade(self):
        return [g - f - s for g, f, s in zip(self.gross, self.fees, self.spreads)]


def _cluster_bootstrap(vals, clusters, n_boot=2000, seed=0):
    """Mean net per trade with a 95% CI resampled over BASKETS."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    idx = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    k = len(idx)
    sums = torch.zeros(k, dtype=torch.float64)
    cnts = torch.zeros(k, dtype=torch.float64)
    ci = torch.tensor([idx[c] for c in clusters], dtype=torch.long)
    sums.index_add_(0, ci, torch.tensor(vals, dtype=torch.float64))
    cnts.index_add_(0, ci, torch.ones(len(vals), dtype=torch.float64))
    g = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, k, (n_boot, k), generator=g)
    means = (sums[draws].sum(1) / cnts[draws].sum(1).clamp(min=1.0)).sort().values
    return (float(sums.sum() / cnts.sum()),
            float(means[int(0.025 * n_boot)]),
            float(means[min(n_boot - 1, int(0.975 * n_boot))]))


def _gap_bootstrap(book_a, book_b, n_boot=2000, seed=0):
    """95% CI on (mean net/trade of A) - (mean net/trade of B).

    "Does the model beat the baselines" is a question about a DIFFERENCE,
    and a difference needs its own interval. Comparing two separately
    computed CIs by eye is not the same test: overlapping intervals can
    still differ significantly, and non-overlapping ones can fail to.

    The strategies here trade overlapping but not identical event sets,
    since each has its own entry rule. So the resampling unit is the
    BASKET, drawn once per replicate and applied to both books: the same
    baskets are in or out for A and B together, which is what makes the
    difference comparable replicate by replicate rather than two
    independent noise draws subtracted from each other.
    """
    ids = sorted({*book_a.baskets, *book_b.baskets})
    if not ids or not book_a.n or not book_b.n:
        return float("nan"), float("nan"), float("nan")
    idx = {c: i for i, c in enumerate(ids)}
    k = len(ids)

    def sums(book):
        s = torch.zeros(k, dtype=torch.float64)
        n = torch.zeros(k, dtype=torch.float64)
        ci = torch.tensor([idx[c] for c in book.baskets], dtype=torch.long)
        s.index_add_(0, ci, torch.tensor(book.net_per_trade(), dtype=torch.float64))
        n.index_add_(0, ci, torch.ones(book.n, dtype=torch.float64))
        return s, n

    sa, na = sums(book_a)
    sb, nb = sums(book_b)
    g = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, k, (n_boot, k), generator=g)
    ma = sa[draws].sum(1) / na[draws].sum(1).clamp(min=1e-9)
    mb = sb[draws].sum(1) / nb[draws].sum(1).clamp(min=1e-9)
    d = (ma - mb).sort().values
    point = (float(sa.sum()) / float(na.sum())) - (float(sb.sum()) / float(nb.sum()))
    return point, float(d[int(0.025 * n_boot)]), float(d[min(n_boot - 1, int(0.975 * n_boot))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_mece/best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--spreads",
                    default="pairwise_monotonicity_taker_side_results_corrected.parquet",
                    help="parquet carrying measured per-leg spreads. Pass 'none' to "
                         "fall back to a flat assumption -- and then do not call the "
                         "result a measured one.")
    ap.add_argument("--spread-multiplier", type=float, default=0.5,
                    help="0.5 = cross half the spread from mid, per crossing.")
    ap.add_argument("--horizon", type=int, default=1,
                    help="which trained horizon to trade, in snapshots (2h each).")
    ap.add_argument("--entry-threshold", type=float, default=0.01,
                    help="minimum |predicted change in basket sum|, in dollars.")
    ap.add_argument("--contracts", type=float, default=CONTRACTS_PER_LEG)
    ap.add_argument("--max-participation", type=float, default=0.0,
                    help="cap each trade at this FRACTION of the thinner side's window "
                         "volume, per leg, at entry and exit. 0 (default) disables the "
                         "cap and assumes --contracts always fills, which is what every "
                         "figure in this project assumed until now. 0.10 is a "
                         "conventional participation limit. This is the closest this "
                         "dataset can get to the executable-size analysis the "
                         "order-book literature performs, since it carries traded "
                         "volume but no depth.")
    ap.add_argument("--n-boot", type=int, default=2000)
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
    if args.horizon not in horizons:
        raise SystemExit(f"horizon {args.horizon} was not trained; available: {horizons}")
    hi = horizons.index(args.horizon)
    h = args.horizon

    print("=" * 108)
    print(f"MECE FORECAST PnL -- {ck_path.name} on the {args.split.upper()} split, "
          f"h={h} ({h * 2}h round trip)")
    print("=" * 108)
    print(f"  checkpoint epoch {ck.get('epoch')}, selected by {ck.get('selected_by')}")
    print(f"  {args.contracts:.0f} contracts/leg, taker fee {TAKER_FEE_RATE}, "
          f"spread multiplier {args.spread_multiplier} per crossing, "
          f"entry |delta| > ${args.entry_threshold:.3f}")

    spreads = None
    if args.spreads.lower() != "none":
        sp = Path(args.spreads)
        if not sp.is_absolute():
            for c in (_THIS_DIR / sp, _REPO_ROOT / sp, Path.cwd() / sp):
                if c.exists():
                    sp = c
                    break
        if sp.exists():
            spreads = SpreadLookup.load(sp)
        else:
            print(f"  WARNING: spreads file not found ({args.spreads}); "
                  f"falling back to a FLAT {FLAT_SPREAD_CENTS}c assumption.")
    legspread = LegSpreads(spreads)

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _REPO_ROOT / cache
    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS, cache),
                               verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=cfg.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=cfg.chunk_len)
    head = ResidualForecastHead(embed_dim=cfg.embed_dim, horizons=horizons)
    model.load_state_dict(ck["model"], strict=True)
    head.load_state_dict(ck["head"], strict=True)
    model.eval()
    head.eval()

    # ---- pass 1: collect every tradeable basket-snapshot -------------------
    # Held in memory so alpha can be fitted over the WHOLE split before any
    # strategy trades. Fitting it per chunk would give the oracle a
    # different alpha in each window -- a baseline no real scalar rule could
    # achieve, and therefore an unbeatable strawman rather than a control.
    events = []
    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        mece_adj = adj.get("mece_leg_to_basket")
        if mece_adj is None:
            del ct
            continue
        prices = true_prices(features)
        node_ids = ct.get("node_ids")
        with torch.no_grad():
            pred = head(model(features, mask, adj), prices)[..., hi]   # (T, N)

        T = prices.shape[0]
        for t in range(T - h):
            edges = mece_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            leg_g, hub_g = edges.edge_index[0], edges.edge_index[1]
            for hub in torch.unique(hub_g).tolist():
                legs = leg_g[(hub_g == hub).nonzero(as_tuple=True)[0]]
                if legs.numel() == 0:
                    continue
                total = float(features[t, hub, SLOT_LEGS_TOTAL])
                # A basket sum only means anything when EVERY leg is
                # present. A partial sum is below 1 for a trivial reason and
                # would read as a permanent, fictitious arbitrage.
                both = mask[t, legs] & mask[t + h, legs]
                if total <= 0 or int(both.sum()) < int(round(total)):
                    continue
                L = legs[both]
                sum_t = float(prices[t, L].sum())
                sum_h = float(prices[t + h, L].sum())
                sum_p = float(pred[t, L].sum())
                events.append({
                    "dev_t": sum_t - 1.0,
                    "delta_real": sum_h - sum_t,
                    "delta_pred": (sum_p - 1.0) - (sum_t - 1.0),
                    "entry_cents": [float(prices[t, l]) * PRICE_SCALE for l in L.tolist()],
                    "exit_cents": [float(prices[t + h, l]) * PRICE_SCALE for l in L.tolist()],
                    "tickers": [node_ids[l] for l in L.tolist()] if node_ids else
                               [str(int(l)) for l in L.tolist()],
                    "basket": node_ids[hub] if node_ids else int(hub),
                    # window_volume at both ends. It is the only observable
                    # this dataset carries that bears on EXECUTABLE SIZE --
                    # there is no order-book depth here -- and a trade needs
                    # to get in at t and out at t+h, so both matter.
                    "vol_in": [float(features[t, l, VOLUME_SLOT]) for l in L.tolist()],
                    "vol_out": [float(features[t + h, l, VOLUME_SLOT]) for l in L.tolist()],
                })
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned  ({len(events):,} basket-snapshots)",
              flush=True)

    if not events:
        raise SystemExit("no complete baskets observed at both t and t+h on this split")

    # ---- the shrinkage oracle's alpha, pooled over the whole split --------
    x = torch.tensor([e["dev_t"] for e in events], dtype=torch.float64)
    y = torch.tensor([e["dev_t"] + e["delta_real"] for e in events], dtype=torch.float64)
    sxx = float((x * x).sum())
    alpha = float((x * y).sum()) / sxx if sxx > 0 else 1.0

    print(f"\n  {len(events):,} complete basket-snapshots, "
          f"{len({e['basket'] for e in events}):,} distinct baskets")
    print(f"  shrinkage oracle alpha (fitted on this split) = {alpha:.4f}")

    # ---- pass 2: run every strategy over the SAME events ------------------
    books = {
        "model": Book("model", "sign of the STGAT's predicted change"),
        "oracle_shrink": Book("oracle_shrink", f"predicts alpha*dev(t), alpha={alpha:.3f}"),
        "always_revert": Book("always_revert", "direction = -sign(dev(t)), no model"),
        "random": Book("random", "coin flip, same entry rule as model"),
        "static_arb": Book("static_arb", "|dev(t)| held to resolution, one-way costs"),
    }
    rng = torch.Generator().manual_seed(0)

    sizes = []
    n_sized_out = 0
    for e in events:
        n_legs = len(e["entry_cents"])
        sp_c = [legspread.get(tk) for tk in e["tickers"]]
        # EXECUTABLE SIZE. Every leg must fill at entry AND at exit, so the
        # binding constraint is the thinnest leg-side in the basket. With the
        # cap off this is just --contracts, which is the assumption the rest
        # of this project has been making implicitly.
        C = args.contracts
        if args.max_participation > 0.0 and e.get("vol_in"):
            cap = min(min(vi, vo) for vi, vo in zip(e["vol_in"], e["vol_out"]))
            C = min(C, args.max_participation * cap)
        sizes.append(C)
        if C < 1.0:
            # Below one contract there is no trade to book. Counting these as
            # zero-PnL trades would flatter every strategy equally; dropping
            # them is reported instead.
            n_sized_out += 1
            continue
        spread_one_way = sum(sp_c) * args.spread_multiplier / PRICE_SCALE * C
        fee_in = sum(taker_fee_dollars(p, C) for p in e["entry_cents"])
        fee_out = sum(taker_fee_dollars(p, C) for p in e["exit_cents"])

        def book_round_trip(name, direction, _C=None):
            if direction == 0:
                return
            books[name].add(direction * e["delta_real"] * (_C if _C else C),
                            fee_in + fee_out, 2.0 * spread_one_way, e["basket"])

        if abs(e["delta_pred"]) > args.entry_threshold:
            d = 1 if e["delta_pred"] > 0 else -1
            book_round_trip("model", d)
            book_round_trip("random", 1 if torch.rand(1, generator=rng).item() > 0.5 else -1)

        d_shrink = (alpha - 1.0) * e["dev_t"]
        if abs(d_shrink) > args.entry_threshold:
            book_round_trip("oracle_shrink", 1 if d_shrink > 0 else -1)

        if abs(e["dev_t"]) > args.entry_threshold:
            book_round_trip("always_revert", -1 if e["dev_t"] > 0 else 1)
            # Static arbitrage: the edge is dev(t) itself, realised at
            # settlement, so there is no exit trade and no exit cost.
            books["static_arb"].add(abs(e["dev_t"]) * C,
                                    fee_in, spread_one_way, e["basket"])
        _ = n_legs

    # ---- report ------------------------------------------------------------
    print("\n  spread sources: " + ", ".join(f"{k}={v:,}" for k, v in
                                             sorted(legspread.sources.items())))
    print("\n" + "=" * 108)
    print(f"{'strategy':<15} {'trades':>8} {'win%':>6} {'gross $':>12} {'fees $':>11} "
          f"{'spreads $':>12} {'NET $':>13} {'net/trade':>11} {'95% CI':>22}")
    print("-" * 108)
    for name in ("model", "oracle_shrink", "always_revert", "random", "static_arb"):
        b = books[name]
        if not b.n:
            print(f"{name:<15} {'0':>8}   (no trades cleared the entry threshold)")
            continue
        g, f, s, net = b.totals()
        per = b.net_per_trade()
        mean, lo, hi_ = _cluster_bootstrap(per, b.baskets, args.n_boot)
        print(f"{name:<15} {b.n:>8,} {100*b.wins/b.n:>5.1f}% {g:>12,.0f} {f:>11,.0f} "
              f"{s:>12,.0f} {net:>13,.0f} {mean:>11.2f} "
              f"{f'[{lo:.2f}, {hi_:.2f}]':>22}")
    print("-" * 108)

    # ---- executable size, which the order-book literature reports and we
    # ---- have so far assumed away -------------------------------------------
    if sizes:
        ss = sorted(sizes)
        q = lambda f: ss[min(len(ss) - 1, int(f * len(ss)))]   # noqa: E731
        print("\n" + "=" * 108)
        print("EXECUTABLE SIZE")
        print("=" * 108)
        if args.max_participation <= 0.0:
            print(f"  NOT CAPPED. Every trade below assumes {args.contracts:.0f} contracts "
                  f"per leg fill at the")
            print("  touch, at both entry and exit. This dataset carries no order-book")
            print("  depth, so that assumption is untested. For scale: the median observed")
            print("  leg trades a few hundred contracts per 2-HOUR window, and a basket")
            print("  trade needs its full size on EVERY leg, twice.")
            print("  Rerun with --max-participation 0.10 to see the effect.")
            print("  Every cost here is linear in size, so the dollar figures scale")
            print("  proportionally and the ratios between strategies do not move at all.")
        else:
            print(f"  capped at {100*args.max_participation:.0f}% of the thinnest "
                  f"leg-side's window volume")
            print(f"  requested size        : {args.contracts:.0f} contracts/leg")
            print(f"  achievable, median    : {q(0.50):.1f}")
            print(f"  achievable, p25 / p75 : {q(0.25):.1f} / {q(0.75):.1f}")
            print(f"  achievable, p90       : {q(0.90):.1f}")
            full = 100.0 * sum(1 for x in ss if x >= args.contracts) / len(ss)
            print(f"  reached full size     : {full:.1f}% of opportunities")
            print(f"  dropped below 1 contract : {n_sized_out:,} of {len(ss):,} "
                  f"({100*n_sized_out/max(len(ss),1):.1f}%)")
            print()
            print("  This is the closest analogue available here to the executable-size")
            print("  analysis order-book studies run -- the Polymarket NBA study found")
            print("  76.9% of its opportunities capped near 14.8 shares. Traded volume is")
            print("  a WEAKER proxy than depth: it says how much changed hands over two")
            print("  hours, not how much sits at the touch right now, and it is therefore")
            print("  likely to be OPTIMISTIC.")

    # ---- model vs each baseline, as a difference with its own interval ----
    mb_ = books["model"]
    if mb_.n:
        print("\n" + "=" * 108)
        print("MODEL minus BASELINE  (the comparison the supervisor's question asks for)")
        print("=" * 108)
        print(f"{'baseline':<16} {'model $/trd':>12} {'base $/trd':>12} {'gap':>9} "
              f"{'95% CI':>22}  verdict")
        print("-" * 108)
        m_pt = mb_.totals()[3] / mb_.n
        for nm in ("random", "oracle_shrink", "always_revert", "static_arb"):
            b = books[nm]
            if not b.n:
                continue
            p, lo, hi_ = _gap_bootstrap(mb_, b, args.n_boot)
            # THREE outcomes. A one-sided test that only asks "is lo > 0?"
            # reports a significantly NEGATIVE gap as a null, which reads as
            # "no difference" when it is evidence the model is worse.
            if lo > 0:
                v = "model BEATS it"
            elif hi_ < 0:
                v = "model is significantly WORSE"
            else:
                v = "not distinguishable"
            print(f"{nm:<16} {m_pt:>12.2f} {b.totals()[3]/b.n:>12.2f} {p:>+9.2f} "
                  f"{f'[{lo:+.2f}, {hi_:+.2f}]':>22}  {v}")
        print("-" * 108)
        print("  'random' is the causal control: identical entry rule, direction thrown")
        print("  away. It is the only row where nothing but the forecast differs, so it")
        print("  is the one that isolates the model's contribution.")
        print()
        print("  The other rows are competing STRATEGIES with their own entry rules and")
        print("  their own trade sets. Losing to one of them can mean the signal is worse")
        print("  OR that the model picks costlier trades -- compare gross/trade against")
        print("  cost/trade before deciding which.")
        print()
        print("  And when every strategy loses money, 'which loses least' is a weak")
        print("  ranking: not trading at all scores $0 and beats all of them.")
        print("-" * 108)
        print(f"{'strategy':<16} {'gross/trade':>13} {'cost/trade':>13} {'net/trade':>12}")
        for nm in ("model", "random", "oracle_shrink", "always_revert", "static_arb"):
            b = books[nm]
            if not b.n:
                continue
            g_, f_, s_, net_ = b.totals()
            print(f"{nm:<16} {g_/b.n:>13.2f} {(f_+s_)/b.n:>13.2f} {net_/b.n:>12.2f}")

    # ---- the break-even bar, which is what actually decides this ---------
    # Kalshi's fee is 0.07 * C * p * (1-p) per leg. Summed over a basket
    # whose legs must total 1, sum(p_i * (1 - p_i)) = 1 - sum(p_i^2), which
    # for N roughly equal legs is (1 - 1/N). So round-trip fees are about
    #     2 * 0.07 * C * (1 - 1/N)
    # -- $9.33 per 100 contracts on a 3-leg basket, rising toward $14 as N
    # grows, and almost INDEPENDENT of the leg prices. That is a fixed toll
    # on the basket sum, and the dislocation has to move further than it in
    # 2h for any forecast, however accurate, to pay.
    mb = books["model"]
    if mb.n:
        cost_pt = (sum(mb.fees) + sum(mb.spreads)) / mb.n
        need = cost_pt / args.contracts
        moves = [abs(e["delta_real"]) for e in events]
        moves_sorted = sorted(moves)
        p90 = moves_sorted[int(0.90 * len(moves_sorted))]
        print("\n" + "=" * 108)
        print("BREAK-EVEN")
        print("=" * 108)
        print(f"  mean round-trip cost per trade      : ${cost_pt:,.2f}  "
              f"(fees ${sum(mb.fees)/mb.n:,.2f} + spreads ${sum(mb.spreads)/mb.n:,.2f})")
        print(f"  basket sum must therefore move      : {100*need:>6.2f}c  "
              f"in {h * 2}h, in the predicted direction")
        print(f"  it actually moves (mean |delta|)    : {100*sum(moves)/len(moves):>6.2f}c")
        print(f"  90th percentile |delta|             : {100*p90:>6.2f}c")
        if need > p90:
            print()
            print("  The bar is above the 90th percentile of what the basket sum actually")
            print("  does. Fewer than 10% of opportunities could clear costs even with a")
            print("  PERFECT forecast, so no improvement in forecast accuracy fixes this.")
            print("  The constraint is the round trip: entry and exit on every leg.")

    if books["oracle_shrink"].n == 0 and books["always_revert"].n:
        print("\n  NOTE: oracle_shrink made no trades. That is not a bug -- it predicts a")
        print(f"  move of only (alpha-1)*dev = {abs(alpha-1):.3f}*dev, so it needs "
              f"|dev| > ${args.entry_threshold/max(abs(alpha-1),1e-9):.3f} to clear the")
        print("  same entry threshold. always_revert takes the same DIRECTION on a looser")
        print("  entry rule, so treat it as the live no-model comparator here.")

    print("\n" + "=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    m, ar = books["model"], books["always_revert"]
    if m.n and ar.n:
        mm, aa = m.totals()[3] / m.n, ar.totals()[3] / ar.n
        mg, ag = m.totals()[0] / m.n, ar.totals()[0] / ar.n
        mc = (m.totals()[1] + m.totals()[2]) / m.n
        ac = (ar.totals()[1] + ar.totals()[2]) / ar.n
        print(f"  vs the no-model reversion rule:")
        print(f"    net/trade   model ${mm:>7.2f}   revert ${aa:>7.2f}")
        print(f"    gross/trade model ${mg:>7.2f}   revert ${ag:>7.2f}")
        print(f"    cost/trade  model ${mc:>7.2f}   revert ${ac:>7.2f}")
        print()
        if mm <= aa and mg > ag:
            # The earlier version of this message concluded from net alone
            # that "the tradeable content is 'dislocation mean-reverts', a
            # property of the market, not a finding about the STGAT". The
            # gross column refutes that whenever the model's gross edge is
            # the larger one: the signal IS better, and the net gap comes
            # from trade selection. Reading net without gross gets the
            # attribution exactly backwards.
            print(f"  The model loses on NET but wins on GROSS (${mg:.2f} vs ${ag:.2f}, "
                  f"{mg/ag if ag else float('nan'):.1f}x).")
            print("  So this is NOT 'the signal is no better than mean reversion'. The")
            print(f"  signal is better; the trades cost ${mc - ac:+.2f} more. The entry gate")
            print("  fires on predicted dislocation, dislocation grows with leg count, and")
            print("  so do fees and spread crossings -- signal strength and execution cost")
            print("  share a common cause, so selecting on one selects for the other.")
            print("  Report it as a trade-SELECTION failure, not a forecasting failure.")
        elif mm <= aa:
            print("  The model loses on net AND on gross, so the signal itself is not")
            print("  beating mean reversion. That is a forecasting result, not a cost one.")
        else:
            print("  The model beats the no-model rule on net. Use the gap CI above rather")
            print("  than comparing the two rows' own intervals by eye, and check 'random'")
            print("  is clearly worse -- if every strategy lands in the same place, costs")
            print("  dominate and the direction signal is irrelevant either way.")
    print()
    print("  If NET is negative everywhere while gross is positive, the forecast edge is")
    print("  real and smaller than the cost of acting on it. That is a legitimate and")
    print("  publishable finding -- it is the same shape as the ladder result, and it is")
    print("  the honest answer to 'is this strategy profitable'.")
    print()
    print("  Compare model against static_arb, not against zero. static_arb crosses each")
    print("  leg ONCE because settlement is free; the round trip crosses twice. A forecast")
    print("  strategy that loses to holding the static arbitrage has not earned its")
    print("  complexity even if its own net is positive.")


if __name__ == "__main__":
    main()