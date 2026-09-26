"""
evaluate_mece_maker_pnl.py

THE SAME FORECAST, EXECUTED AS A MAKER INSTEAD OF A TAKER.

evaluate_mece_forecast_pnl.py established that the model has a real
directional edge on MECE basket dislocation -- $6,918 gross against
random's $18 on the SAME 2,366 events, 51.1% wins against a 34.1% null --
and that a taker round trip destroys it: $33.51 of cost per trade against
a basket sum that moves 2.26c on average.

Two of those cost components change completely if the orders are POSTED
rather than CROSSED:

  fees    taker 0.07 * C * p * (1-p) per leg, maker 0.0175 -- 4x cheaper,
          and zero on some series entirely.
  spread  a taker PAYS half the spread on every leg on every side. A maker
          EARNS it. That is not a discount, it is a sign flip, and on this
          data it is the difference between -$26.76 and +$26.76 per trade.

So the maker version is not a marginal improvement on the taker version;
it is a different sign. Which is exactly why it must not be evaluated by
assuming the orders fill.

THE ONLY PART OF THIS THAT MATTERS: THE FILL MODEL.

A resting order does not fill on demand. It fills when somebody chooses to
trade against it -- and they choose to when they think you are wrong. That
is adverse selection, and a maker backtest that ignores it manufactures
profit out of nothing. Whelan's makers, who really did pay no fees at all
before April 2025, still averaged -9.64%.

This script never assumes a fill. It derives one from the order-flow data
already in the feature tensor:

    a posted BID fills when sellers arrive   -> net_flow < 0
    a posted OFFER fills when buyers arrive  -> net_flow > 0
    in both cases                            -> net_flow * side < 0

The adverse selection then falls out of the data rather than being bolted
on as a fudge factor: you fill precisely when flow is running against your
position, and flow predicts where the price goes next. Nothing in this
model forces that to be costly -- if it turns out not to be, the numbers
will say so -- but it is the mechanism by which maker strategies die, and
it is measured here rather than waved at. The report prints the edge on
FILLED events beside the edge on ALL events, which is the size of the
adverse selection in dollars.

PARTIAL FILLS ARE A REAL RISK, NOT AN INCONVENIENCE. A MECE basket trade
needs every leg. Three legs of a four-leg basket is not an arbitrage with
a rounding error, it is a naked directional position with none of the
structural protection the whole strategy rests on. Two honest policies:

  all_or_nothing    skip unless every leg fills. Clean, conservative,
                    and the one to quote.
  cross_remainder   cross the unfilled legs as a taker, paying the spread
                    and the taker fee on them. What a desk would actually
                    do, and strictly worse than all_or_nothing per trade.

EXIT IS NOT SYMMETRIC WITH ENTRY. Entry is patient -- if the order does not
fill, nothing happens and nothing is lost. Exit is not: at t+h the position
exists and has to go. A resting exit that does not fill leaves you holding
risk past your horizon. maker_taker (post the entry, cross the exit) is
therefore the realistic configuration, and it is spread-neutral by
construction. maker_maker is reported as the optimistic bound, not as a
result.

THE EXECUTION ALGEBRA, which the smoke test checks rather than trusts:

    PnL = d * delta_real * C  +  sigma * sum(spread) * C

with sigma = -1 per crossed side and +1 per posted side, halved per side:

    taker / taker   ->  -sum(spread)        matches the existing evaluator
    maker / taker   ->   0                  spread-neutral
    maker / maker   ->  +sum(spread)

DEFAULTS ARE DELIBERATELY UNKIND. The maker fee is charged on every series
unless --maker-fee-free-series says otherwise, even though Kalshi exempts
some (KXBTCD among them). Claiming a profit that rests on an unverified fee
exemption would be the same error as the flat-1c spread that once set the
sign of the ladder headline.

Read-only. Trains nothing.
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

from evaluate_arbitrage_ladder import SpreadLookup  # noqa: E402
from evaluate_mece_forecast_pnl import (  # noqa: E402
    Book, LegSpreads, PRICE_SCALE, SLOT_LEGS_TOTAL, _cluster_bootstrap,
    _gap_bootstrap, taker_fee_dollars,
)
from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.forecast_head import ResidualForecastHead  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, true_prices  # noqa: E402
from train_forecast import PILOT_MONTHS  # noqa: E402

VOLUME_SLOT = 4
NET_FLOW_SLOT = 5
MAKER_FEE_RATE = 0.0175


def maker_fee_dollars(price_cents, contracts, rate=MAKER_FEE_RATE):
    """Same shape as the taker fee at a lower rate. Rounded up, with the
    round() guard against binary-float drift (0.07*100*0.25 evaluates to
    1.7500000000000002, and a bare ceil turns that into 1.7501)."""
    import math
    p = max(0.0, min(1.0, price_cents / PRICE_SCALE))
    return math.ceil(round(rate * contracts * p * (1.0 - p) * 10000, 6)) / 10000


def fills(flow: float, side: int, volume: float, contracts: float, model: str) -> bool:
    """Does a resting order on `side` (+1 buy, -1 sell) get filled?

    `optimistic` is included so the cost of the assumption can be SHOWN,
    not because it is defensible. It is the number a maker backtest
    produces when it forgets that somebody has to take the other side.
    """
    if model == "optimistic":
        return True
    if volume < contracts:              # nobody traded enough size to fill you
        return False
    if model == "volume":
        return True
    return flow * side < 0              # 'flow': filled only when flow opposes you


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_mece/best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--spreads",
                    default="pairwise_monotonicity_taker_side_results_corrected.parquet")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--entry-threshold", type=float, default=0.01)
    ap.add_argument("--contracts", type=float, default=100.0)
    ap.add_argument("--fill-model", choices=["flow", "volume", "optimistic"], default="flow",
                    help="'flow' (default) fills a posted order only when order flow "
                         "runs AGAINST it, which is how adverse selection actually "
                         "arises. 'optimistic' assumes every order fills and is an "
                         "upper bound, not a result.")
    ap.add_argument("--partial", choices=["all_or_nothing", "cross_remainder"],
                    default="all_or_nothing")
    ap.add_argument("--maker-fee-rate", type=float, default=MAKER_FEE_RATE)
    ap.add_argument("--maker-fee-free-series", nargs="*", default=[],
                    help="series prefixes Kalshi exempts from maker fees (e.g. KXBTCD). "
                         "Empty by default: a profit that depends on an unverified "
                         "exemption is not a profit.")
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
        raise SystemExit(f"horizon {args.horizon} not trained; available: {horizons}")
    hi, h = horizons.index(args.horizon), args.horizon
    free = tuple(args.maker_fee_free_series)

    print("=" * 112)
    print(f"MECE MAKER-SIDE PnL -- {ck_path.name}, {args.split.upper()} split, "
          f"h={h} ({h * 2}h)")
    print("=" * 112)
    print(f"  fill model = {args.fill_model}   partial policy = {args.partial}   "
          f"maker fee = {args.maker_fee_rate}")
    print(f"  maker-fee-free series: {list(free) or 'none (charging every series)'}")

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

    # ---- collect events, carrying the flow/volume the fill model needs ----
    events = []
    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        mece_adj = adj.get("mece_leg_to_basket")
        if mece_adj is None:
            del ct
            continue
        prices = true_prices(features)
        flow, vol = features[..., NET_FLOW_SLOT], features[..., VOLUME_SLOT]
        node_ids = ct.get("node_ids")
        with torch.no_grad():
            pred = head(model(features, mask, adj), prices)[..., hi]

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
                both = mask[t, legs] & mask[t + h, legs]
                if total <= 0 or int(both.sum()) < int(round(total)):
                    continue
                L = legs[both].tolist()
                sum_t = float(prices[t, L].sum())
                events.append({
                    # dev_t is what the no-model reversion rule needs, and
                    # n_legs is what exposes whether the fill filter is
                    # quietly selecting a different KIND of basket.
                    "dev_t": sum_t - 1.0,
                    "n_legs": len(L),
                    "delta_real": float(prices[t + h, L].sum()) - sum_t,
                    "delta_pred": float(pred[t, L].sum()) - sum_t,
                    "entry_c": [float(prices[t, l]) * PRICE_SCALE for l in L],
                    "exit_c": [float(prices[t + h, l]) * PRICE_SCALE for l in L],
                    "flow_in": [float(flow[t, l]) for l in L],
                    "flow_out": [float(flow[t + h, l]) for l in L],
                    "vol_in": [float(vol[t, l]) for l in L],
                    "vol_out": [float(vol[t + h, l]) for l in L],
                    "tickers": ([node_ids[l] for l in L] if node_ids
                                else [str(l) for l in L]),
                    "basket": node_ids[hub] if node_ids else int(hub),
                })
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned ({len(events):,} events)", flush=True)

    if not events:
        raise SystemExit("no complete baskets observed at both t and t+h")

    print(f"\n  {len(events):,} complete basket-snapshots, "
          f"{len({e['basket'] for e in events}):,} distinct baskets")

    # ---- run each execution configuration over the same events -----------
    #
    # THE CONTROL ROWS EXIST BECAUSE THE FILL FILTER IS NOT NEUTRAL. It keeps
    # only events where flow opposed the position on EVERY leg at once, which
    # on this data is ~4% of signals. That is a strong, non-random filter, and
    # a filter can carry an edge of its own: if the surviving events are
    # simply better events, a coin-flip direction would profit on them too,
    # and the "maker edge" would belong to the filter rather than the model.
    #
    # So the same execution and the same filter are run with two directions
    # the model had no hand in. Only the GAP between the model row and these
    # rows is attributable to the forecast.
    DIRS = {
        "model": lambda e, r: (0 if abs(e["delta_pred"]) <= args.entry_threshold
                               else (1 if e["delta_pred"] > 0 else -1)),
        # same entry gate as the model, direction thrown away -> isolates the filter
        "random": lambda e, r: (0 if abs(e["delta_pred"]) <= args.entry_threshold
                                else (1 if torch.rand(1, generator=r).item() > 0.5 else -1)),
        # a real alternative strategy that needs no model at all
        "revert": lambda e, r: (0 if abs(e["dev_t"]) <= args.entry_threshold
                                else (-1 if e["dev_t"] > 0 else 1)),
    }
    RUNS = [
        ("taker_taker", False, False, "model", "cross both sides (the published baseline)"),
        ("maker_taker", True, False, "model", "post entry, cross exit -- the realistic one"),
        ("maker_maker", True, True, "model", "post both -- optimistic bound, not a result"),
        ("mk_tk[random]", True, False, "random",
         "CONTROL: same gate, same filter, coin-flip direction"),
        ("mk_tk[revert]", True, False, "revert",
         "CONTROL: same filter, no-model reversion direction"),
    ]
    books = {n: Book(n, note) for n, _, _, _, note in RUNS}
    edge_all, edge_filled = [], []
    n_attempt = n_filled = n_partial = 0
    # what the filter selected, filled vs all: legs, price level, signal size
    prof = {"all": [], "filled": []}

    C = args.contracts
    rng = torch.Generator().manual_seed(0)
    for e in events:
        sp_c = [legspread.get(tk) for tk in e["tickers"]]
        half = [s / 2.0 / PRICE_SCALE * C for s in sp_c]          # dollars, per leg per side

        def leg_fee(price_c, tk, is_maker):
            if is_maker:
                if free and str(tk).split("-", 1)[0] in free:
                    return 0.0
                return maker_fee_dollars(price_c, C, args.maker_fee_rate)
            return taker_fee_dollars(price_c, C)

        for name, mk_in, mk_out, dname, _ in RUNS:
            d = DIRS[dname](e, rng)
            if d == 0:
                continue
            gross = d * e["delta_real"] * C

            # --- entry ---
            if mk_in:
                filled = [fills(e["flow_in"][i], d, e["vol_in"][i], C, args.fill_model)
                          for i in range(len(sp_c))]
            else:
                filled = [True] * len(sp_c)

            if name == "maker_taker":       # fill stats once, on the live config
                n_attempt += 1
                row = (e["n_legs"], sum(e["entry_c"]) / max(len(e["entry_c"]), 1),
                       abs(e["delta_pred"]), abs(e["delta_real"]))
                prof["all"].append(row)
                if all(filled):
                    n_filled += 1
                    edge_filled.append(gross)
                    prof["filled"].append(row)
                elif any(filled):
                    n_partial += 1
                edge_all.append(gross)

            if not all(filled):
                if args.partial == "all_or_nothing" or not any(filled):
                    continue
                # cross_remainder: unfilled legs are taken instead
            # --- exit: a live position must go, so an unfilled maker exit
            #     is crossed rather than carried past the horizon ---
            if mk_out:
                out_filled = [fills(e["flow_out"][i], -d, e["vol_out"][i], C,
                                    args.fill_model) for i in range(len(sp_c))]
            else:
                out_filled = [False] * len(sp_c)

            spread_pnl = 0.0
            fee = 0.0
            for i, (tk, pin, pout) in enumerate(zip(e["tickers"], e["entry_c"], e["exit_c"])):
                in_maker = mk_in and filled[i]
                out_maker = mk_out and out_filled[i]
                spread_pnl += (half[i] if in_maker else -half[i])
                spread_pnl += (half[i] if out_maker else -half[i])
                fee += leg_fee(pin, tk, in_maker) + leg_fee(pout, tk, out_maker)
            # Book stores costs as POSITIVE numbers subtracted from gross, so a
            # spread EARNED is a negative cost.
            books[name].add(gross, fee, -spread_pnl, e["basket"])

    # ---- report -----------------------------------------------------------
    print(f"\n  spread sources: " + ", ".join(f"{k}={v:,}" for k, v in
                                              sorted(legspread.sources.items())))
    print("\n" + "=" * 112)
    # 'baskets' is not decoration. Every CI here is resampled over baskets,
    # and a cluster bootstrap estimates its variance from BETWEEN-basket
    # spread -- so with only a handful of baskets there is almost nothing to
    # estimate it from and the interval COLLAPSES rather than widening. It
    # under-covers silently, in the reassuring direction. A tight CI on a row
    # backed by few baskets is not evidence, it is the bootstrap running out
    # of independent units.
    print(f"{'configuration':<15} {'trades':>8} {'baskets':>8} {'win%':>6} {'gross $':>11} "
          f"{'fees $':>10} {'spread $':>11} {'NET $':>12} {'net/trade':>10} {'95% CI':>20}")
    print("-" * 122)
    _low_cluster = []
    for name, _, _, _, note in RUNS:
        b = books[name]
        if not b.n:
            print(f"{name:<15} {'0':>8}   (no trades survived the fill model)")
            continue
        g, f, s, net = b.totals()
        nb_ = len(set(b.baskets))
        if nb_ < 10:
            _low_cluster.append((name, nb_))
        mean, lo, hi_ = _cluster_bootstrap(b.net_per_trade(), b.baskets, args.n_boot)
        print(f"{name:<15} {b.n:>8,} {nb_:>8,} {100*b.wins/b.n:>5.1f}% {g:>11,.0f} "
              f"{f:>10,.0f} {s:>11,.0f} {net:>12,.0f} {mean:>10.2f} "
              f"{f'[{lo:.2f}, {hi_:.2f}]':>20}")
    print("-" * 122)
    print("  a NEGATIVE 'spread $' means the spread was EARNED, not paid.")
    for name, _, _, _, note in RUNS:
        print(f"    {name:<15} {note}")
    if _low_cluster:
        print()
        for name, nb_ in _low_cluster:
            print(f"  WARNING: {name} rests on only {nb_} distinct basket(s). Its CI is")
            print("  computed from between-basket spread and there is too little of it to")
            print("  measure -- the interval will look tight because the bootstrap has run")
            print("  out of independent units, not because the estimate is precise.")

    # ---- the number that decides whether any of this is real -------------
    print("\n" + "=" * 112)
    print("ADVERSE SELECTION")
    print("=" * 112)
    if n_attempt:
        print(f"  entry signals            : {n_attempt:,}")
        print(f"  fully filled             : {n_filled:,} ({100*n_filled/n_attempt:.1f}%)")
        print(f"  partially filled         : {n_partial:,} ({100*n_partial/n_attempt:.1f}%)")
    if edge_all and not edge_filled:
        # Not a crash and not an empty result -- it is the strongest possible
        # form of the finding, and it must not be printed as a blank.
        print()
        print("  NOTHING FILLED. Every entry signal wanted to trade in the direction")
        print("  order flow was already going, so no posted order was ever hit. That is")
        print("  adverse selection in its limiting form: the trades you want are exactly")
        print("  the trades nobody will give you.")
        print()
        print("  Confirm it is the fill rule and not a plumbing fault by rerunning with")
        print("  --fill-model optimistic. If that fills 100% and turns a profit, the")
        print("  difference between the two runs IS the adverse selection, and the")
        print("  optimistic number is what a maker backtest reports when it forgets")
        print("  that somebody has to take the other side.")
        if n_partial:
            print()
            print(f"  {n_partial:,} signals filled on SOME legs ({100*n_partial/max(n_attempt,1):.1f}%). "
                  f"--partial cross_remainder would")
            print("  trade those by crossing the rest, which is what a desk would do --")
            print("  and it pays the taker spread on precisely the legs that ran away.")
    elif edge_all and edge_filled:
        a = sum(edge_all) / len(edge_all)
        fl = sum(edge_filled) / len(edge_filled)
        print(f"  gross edge, ALL signals  : ${a:>8.2f} / trade  (n={len(edge_all):,})")
        print(f"  gross edge, FILLED only  : ${fl:>8.2f} / trade  (n={len(edge_filled):,})")
        print(f"  adverse selection        : ${fl - a:>8.2f} / trade "
              f"({100*(fl-a)/abs(a) if a else float('nan'):+.1f}%)")
        print()
        if fl < a:
            print("  The orders that filled are worse than the signals overall. That is")
            print("  adverse selection doing exactly what it does: you get filled when")
            print("  the flow is against you. Any maker profit below has already paid")
            print("  this, which is the point of modelling fills instead of assuming them.")
        else:
            print("  Filled orders are BETTER than the signal average. That is FAVOURABLE")
            print("  selection, the opposite of what a maker should expect, and it is the")
            print("  single most suspicious thing this script can print. It means the fill")
            print("  filter is not merely thinning the sample -- it is choosing a better")
            print("  one. The control rows are what decide whether that belongs to the")
            print("  model or to the filter:")
            mt = books.get("maker_taker")
            rc = books.get("mk_tk[random]")
            if mt and rc and mt.n and rc.n:
                mm = mt.totals()[3] / mt.n
                rr = rc.totals()[3] / rc.n
                print(f"    maker_taker  (model direction) : ${mm:+.2f}/trade over {mt.n:,} trades")
                print(f"    mk_tk[random] (coin flip)      : ${rr:+.2f}/trade over {rc.n:,} trades")
                if rr > 0:
                    print("    A COIN FLIP IS ALSO PROFITABLE through this filter. The edge is")
                    print("    in the filter, not in the forecast. Whatever the model row says,")
                    print("    it cannot be reported as a forecasting result.")
                else:
                    print("    The coin flip loses money through the same filter, so the model")
                    print("    row is not just the filter. Report the GAP between the two as")
                    print("    the model's contribution, not the model row on its own.")

    # ---- the model's contribution, with its own interval ------------------
    mt, rc, rv = (books.get("maker_taker"), books.get("mk_tk[random]"),
                  books.get("mk_tk[revert]"))
    if mt and mt.n:
        print("\n" + "=" * 112)
        print("THE MODEL'S CONTRIBUTION  (model minus control, same execution, "
              "same fill rule)")
        print("=" * 112)
        print("  Not the model row. The GAP. The model row includes whatever the entry")
        print("  gate and the fill filter contribute on their own, and those need no")
        print("  forecast; only the difference from a control that uses the same gate and")
        print("  the same filter is attributable to the STGAT.")
        print()
        for label, ctrl in (("vs coin flip", rc), ("vs no-model reversion", rv)):
            if ctrl and ctrl.n:
                p, lo, hi_ = _gap_bootstrap(mt, ctrl, args.n_boot)
                # THREE outcomes, not two. A one-sided test that only asks
                # "is lo > 0?" reports a significantly NEGATIVE gap as
                # "includes 0", which reads as a null when it is in fact
                # evidence the model is WORSE. That misreading has happened
                # in this project before and cost a rewrite; it does not get
                # to happen twice.
                if lo > 0:
                    verdict = "excludes 0 -- a real contribution"
                elif hi_ < 0:
                    verdict = "excludes 0 the WRONG WAY -- the model is significantly WORSE"
                else:
                    verdict = "includes 0 -- not distinguishable from the control"
                print(f"  {label:<24} {p:>+8.2f} $/trade   95% CI "
                      f"[{lo:+.2f}, {hi_:+.2f}]   {verdict}")
        print()
        print("  The two rows are NOT the same kind of comparison:")
        print("    vs coin flip  -- identical entry gate, identical fill filter, only the")
        print("                     direction differs. This is the causal control, and it")
        print("                     is the one that isolates the forecast.")
        print("    vs reversion  -- a different strategy with its own entry gate and its")
        print("                     own trade set. Losing to it says the model picks more")
        print("                     EXPENSIVE trades, not that its signal is worse; check")
        print("                     gross/trade against cost/trade before concluding.")
        print()
        print("  Resampled over baskets drawn once per replicate and applied to BOTH")
        print("  books, so the two means move together and the difference is a real")
        print("  interval rather than two independent noise draws subtracted.")

    # ---- what the filter actually selected --------------------------------
    if prof["filled"] and prof["all"]:
        print("\n" + "=" * 112)
        print("WHAT THE FILL FILTER SELECTED")
        print("=" * 112)
        print("  Requiring flow to oppose the position on EVERY leg is a strong condition.")
        print("  If the surviving baskets differ systematically from the rest, the maker")
        print("  result is about that subpopulation, not about maker execution.")
        print()
        cols = ("legs/basket", "mean leg price c", "mean |delta_pred|", "mean |delta_real|")
        print(f"  {'population':<12} {'n':>8} " + " ".join(f"{c:>18}" for c in cols))
        print("  " + "-" * 96)
        for tag in ("all", "filled"):
            rows = prof[tag]
            k = len(rows)
            m = [sum(r[i] for r in rows) / k for i in range(4)]
            print(f"  {tag:<12} {k:>8,} {m[0]:>18.2f} {m[1]:>18.2f} "
                  f"{m[2]:>18.4f} {m[3]:>18.4f}")
        print("  " + "-" * 96)
        print("  Divergence in 'legs/basket' or 'mean leg price' is a composition shift:")
        print("  fees scale with both, so a filter that prefers cheaper or smaller baskets")
        print("  lowers costs for reasons that have nothing to do with the forecast.")

    print("\n" + "=" * 112)
    print("HOW TO READ THIS")
    print("=" * 112)
    print("  Quote maker_taker. Entry is patient -- an unfilled post costs nothing --")
    print("  but the exit is not optional, so crossing it is the honest assumption.")
    print("  maker_maker assumes you can also leave whenever you like, which is the")
    print("  assumption that makes every maker backtest look good.")
    print()
    print("  Compare against taker_taker in the SAME table rather than against the")
    print("  earlier run: same events, same fill model, one difference.")
    print()
    print("  If maker_taker is positive, check four things before believing it:")
    print("    1. the GAP, not the row. The model row includes whatever the entry gate")
    print("       and the fill filter earn on their own, and neither needs a forecast.")
    print("    2. the basket count behind the CI. Below ~10 baskets a cluster bootstrap")
    print("       collapses instead of widening, so a tight interval there means the")
    print("       resampling ran out of independent units, not that the estimate is good.")
    print("    3. the fill rate, as capacity. 4% of signals over a five-month split is")
    print("       ~100 trades: possibly real, certainly not a deployable strategy.")
    print("    4. the composition profile. If |delta_real| rises on the filled subset")
    print("       while |delta_pred| does not, the filter found volatile periods rather")
    print("       than confident forecasts, and the extra profit is the market's, not")
    print("       the model's.")
    print()
    print("  NOT a valid check: 'optimistic should be materially BETTER, which proves the")
    print("  fill rule was binding'. That was this script's original advice and this data")
    print("  refuted it -- optimistic came back WORSE, because the fill filter selects")
    print("  favourably here rather than adversely. A fill rule can be strongly binding")
    print("  and still improve the result. Use the control rows to decide, not the")
    print("  direction of the optimistic comparison.")


if __name__ == "__main__":
    main()