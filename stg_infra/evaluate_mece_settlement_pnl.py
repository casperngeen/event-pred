"""
evaluate_mece_settlement_pnl.py

HOLD-TO-SETTLEMENT MECE ARBITRAGE, WITH THE MODEL USED TO TIME ENTRY.

WHY THIS EXISTS. evaluate_mece_forecast_pnl.py evaluates a ROUND TRIP:
enter at t, exit at t+h. That pays fees and crosses the spread twice on
every leg, and the measured result is -$30.59/trade against a break-even
of 33.5c on a 2.26c mean move. The arithmetic forbids it.

The same script's static_arb row loses only $9.10 -- not because it
forecasts better (it makes no forecast at all) but because it HOLDS TO
SETTLEMENT and therefore pays one-way costs. That cost asymmetry is worth
about $21 a trade, which dwarfs every modelling decision made anywhere
else in this project.

So this script asks the question that structure raises: with one-way
costs, is there a configuration that clears, and does the model
contribute to it?

WHAT THE MODEL CAN AND CANNOT DO HERE -- READ THIS BEFORE INTERPRETING
ANYTHING BELOW. The legs of a MECE basket must sum to $1 at resolution.
So a position held to settlement captures |dev(t)| DETERMINISTICALLY: it
does not matter what the price path does in between. There is therefore
nothing to forecast about WHETHER a dislocation converges -- it always
does.

That kills the obvious framing ("use the model to pick which
opportunities will converge"), which is vacuous here. What is left is
genuinely forecastable and is what this script tests:

  ENTRY TIMING. A dislocation that is about to WIDEN is worth waiting
  for -- entering one step later captures a larger |dev|. One that is
  about to NARROW should be taken now. The model predicts the change in
  dev, so it can make that call; a coin flip cannot.

The control is the same machinery with the timing decision replaced by a
coin flip, so costs and the trade population are identical and the only
difference is the forecast. That is the same design as `random` in the
round-trip script, for the same reason.

THE ASSUMPTION THIS STRATEGY MAKES, STATED LOUDLY. Holding to settlement
ties up capital until the event resolves, and this dataset does not say
when a position was actually closed out -- it says what the price was.
The script therefore reports the distribution of time_to_close at entry
so the lockup is visible rather than implicit. A strategy that clears
only by holding a position for three weeks is not the same product as one
that clears in two hours, and the reader should be able to see which one
this is.

WHAT IS NOT MODELLED: partial fills across legs, queue position, the
capital cost of the lockup, and any change in the fee schedule between
entry and settlement. Maker mode models fill from observed order flow but
still requires ALL legs to fill, which is optimistic.

USAGE
    python evaluate_mece_settlement_pnl.py --split val          # tune here
    python evaluate_mece_settlement_pnl.py --split test         # once
    python evaluate_mece_settlement_pnl.py --execution maker --max-participation 0.10

SELECT AND TUNE ON VALIDATION. The entry-timing rule below came from
reading test results in an earlier analysis. Fitting it on test and then
reporting test is the bias this project has already documented twice.
Run --split val first, fix the configuration, then spend one test run.
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

from model.chunking import build_split_ranges, chunk_ranges          # noqa: E402
from model.forecast_head import ResidualForecastHead                 # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.run_guards import format_verdict                          # noqa: E402
from model.train import STGATBackbone, true_prices                   # noqa: E402
from train_forecast import PILOT_MONTHS                              # noqa: E402

# Reuse the round-trip script's primitives so the two cannot disagree about
# what a fee, a spread or a bootstrap is.
from evaluate_mece_forecast_pnl import (                             # noqa: E402
    FLAT_SPREAD_CENTS, PRICE_SCALE, SLOT_LEGS_TOTAL, VOLUME_SLOT, Book,
    LegSpreads, _gap_bootstrap, taker_fee_dollars,
)
from evaluate_arbitrage_ladder import SpreadLookup                   # noqa: E402
# IMPORT the maker fee rather than redefining it. An earlier draft of this
# file defined a FLAT per-contract maker fee while evaluate_mece_maker_pnl.py
# uses the same p(1-p) shape as the taker fee at a quarter of the rate. On a
# 20c leg those differ by 1.6x and on a 5c leg by 5.3x, in opposite
# directions -- exactly the kind of silent disagreement between two scripts
# that this project has been bitten by before.
from evaluate_mece_maker_pnl import MAKER_FEE_RATE, maker_fee_dollars  # noqa: E402

NET_FLOW_SLOT = 5
TIME_TO_CLOSE_SLOT = 8

# NOTE ON THE MAKER FEE SHAPE -- WORTH VERIFYING AGAINST KALSHI'S SCHEDULE.
# The imported maker_fee_dollars applies rate * C * p * (1-p) with
# rate = 0.0175, i.e. the taker formula at a quarter of the rate, which makes
# maker fees uniformly 4x cheaper. If Kalshi's maker fee is in fact a FLAT
# per-contract charge, the picture changes sharply for MECE baskets, whose
# legs average 1/N dollars and are therefore cheap: a flat 0.0175/contract is
# 1.6x the TAKER fee on a 20c leg and 5.3x on a 5c leg. Since this determines
# whether maker execution helps or hurts, it should be checked against the
# published schedule rather than inherited.


def collect_events(store, chunks, model, head, hi, h):
    """One event per complete basket-snapshot.

    Completeness is enforced exactly as in the round-trip script: every
    declared leg must be observed at t AND t+h. A partial basket sums below
    1 for a trivial reason and would read as a permanent fictitious
    arbitrage.
    """
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
                L = legs[both]
                idx = L.tolist()
                sum_t = float(prices[t, L].sum())
                sum_h = float(prices[t + h, L].sum())
                sum_p = float(pred[t, L].sum())
                events.append({
                    "dev_t": sum_t - 1.0,
                    "dev_h": sum_h - 1.0,
                    "delta_pred": sum_p - sum_t,
                    "entry_cents": [float(prices[t, l]) * PRICE_SCALE for l in idx],
                    "later_cents": [float(prices[t + h, l]) * PRICE_SCALE for l in idx],
                    "tickers": [node_ids[l] for l in idx] if node_ids
                               else [str(int(l)) for l in idx],
                    "basket": node_ids[hub] if node_ids else int(hub),
                    "vol_in": [float(features[t, l, VOLUME_SLOT]) for l in idx],
                    "vol_later": [float(features[t + h, l, VOLUME_SLOT]) for l in idx],
                    "flow_in": [float(features[t, l, NET_FLOW_SLOT]) for l in idx],
                    "ttc": float(features[t, L, TIME_TO_CLOSE_SLOT].max()),
                })
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned  ({len(events):,} events)",
              flush=True)
    return events


MEASURED_SOURCES = ("ticker", "series")


def spread_coverage(legspread) -> str:
    """One-line 'x% measured' summary of where the leg spreads came from."""
    src = dict(legspread.sources)
    total = sum(src.values())
    if total == 0:
        return "no legs costed"
    measured = sum(src.get(k, 0) for k in MEASURED_SOURCES)
    return f"{measured / total:.1%} measured per-leg, {1 - measured / total:.1%} fallback"


def warn_spread_coverage(legspread, args, floor: float = 0.50) -> None:
    """Fail loudly when the cost model is not the one the run claims to use.

    This exists because the opposite happened silently. The default
    --spreads used to be the LADDER results file, which contains only
    crypto/financials pairs, so every weather and sports leg missed on
    ticker AND series and fell through to the global median. The run still
    printed 'spread sources: global=27,386' -- technically complete, easy
    to read past -- and the figures were described as using measured
    per-leg spreads for weeks.

    The gate's 100% win rate is DEFINITIONAL (it enters only when |dev|
    exceeds modelled cost), so the entire result rests on the cost model
    rather than on sampling. A silently substituted cost model is
    therefore the single largest threat to the headline, which is why this
    warns rather than merely reporting.
    """
    src = dict(legspread.sources)
    total = sum(src.values())
    if total == 0:
        return
    measured = sum(src.get(k, 0) for k in MEASURED_SOURCES) / total
    if measured >= floor:
        return
    print()
    print("  " + "!" * 86)
    print(f"  !! ONLY {measured:.1%} of legs were costed with a MEASURED spread "
          f"(floor {floor:.0%}).")
    print(f"  !! {src.get('global', 0):,} legs fell back to the GLOBAL median and "
          f"{src.get('assumed_flat', 0):,} to a flat assumption.")
    print(f"  !! spreads file: {args.spreads}")
    if measured == 0.0:
        print("  !! NOT ONE leg matched. The file almost certainly covers a "
              "different population")
        print("  !! than the baskets being scored -- a ladder results file holds no "
              "weather or")
        print("  !! sports legs. Build the right one with build_mece_leg_spreads.py "
              "and pass")
        print("  !! --spreads mece_leg_spreads.parquet.")
    print("  !! Do NOT describe these results as using measured per-leg spreads.")
    print("  " + "!" * 86)
    print()


def basket_family(basket) -> str:
    """Segment before the first hyphen, matching event_family() in
    mece_sum_to_one_check.py. Non-string baskets (an int hub index, when
    node_ids is absent) have no family and are never excluded."""
    return str(basket).split("-", 1)[0] if isinstance(basket, str) else ""


def apply_filters(events, args):
    """Drop events failing the opt-in robustness filters, reporting each
    filter's attribution separately.

    Attribution is computed on the ORIGINAL population, one filter at a
    time, so the printed counts say what each filter costs on its own.
    They therefore overlap and will not sum to the total removed -- that is
    the point: a single 'n dropped' figure cannot tell you whether two
    filters are removing the same baskets or different ones.
    """
    n0 = len(events)
    fams = set(args.exclude_families or [])
    mv, mt = args.min_leg_volume, args.min_ttc
    if not fams and mv <= 0.0 and mt <= 0.0:
        print(f"\nrobustness filters: none applied ({n0:,} complete baskets)")
        return events

    def thin(e):
        return mv > 0.0 and min(min(e["vol_in"]), min(e["vol_later"])) < mv

    def near_close(e):
        return mt > 0.0 and e["ttc"] < mt

    def excluded(e):
        return bool(fams) and basket_family(e["basket"]) in fams

    print(f"\nrobustness filters, attributed one at a time over {n0:,} "
          f"complete baskets:")
    if mv > 0.0:
        n = sum(1 for e in events if thin(e))
        print(f"  --min-leg-volume {mv:g}      would drop {n:,} ({n / n0:.1%}) "
              f"-- a leg below this had no tradeable last price")
    if mt > 0.0:
        n = sum(1 for e in events if near_close(e))
        print(f"  --min-ttc {mt:g}             would drop {n:,} ({n / n0:.1%}) "
              f"-- near resolution, post-determination prints possible")
    if fams:
        n = sum(1 for e in events if excluded(e))
        by = {}
        for e in events:
            if excluded(e):
                f = basket_family(e["basket"])
                by[f] = by.get(f, 0) + 1
        detail = ", ".join(f"{k}={v:,}" for k, v in sorted(by.items())) or "none matched"
        print(f"  --exclude-families          would drop {n:,} ({n / n0:.1%})"
              f"  [{detail}]")
        for f in sorted(fams):
            if f not in by:
                print(f"    WARNING: '{f}' matched NOTHING -- check the spelling "
                      f"against a basket ticker's leading segment")

    kept = [e for e in events if not (thin(e) or near_close(e) or excluded(e))]
    print(f"  combined: kept {len(kept):,} of {n0:,} ({len(kept) / n0:.1%}), "
          f"dropped {n0 - len(kept):,}")
    if kept:
        import statistics as _st
        d0 = _st.median(abs(e["dev_t"]) for e in events)
        d1 = _st.median(abs(e["dev_t"]) for e in kept)
        print(f"  median |dev_t|: {d0 * PRICE_SCALE:.2f}c before -> "
              f"{d1 * PRICE_SCALE:.2f}c after")
        if d1 > d0:
            print("    NOTE: filtering RAISED the median deviation. The filters "
                  "removed thin baskets, so this says the survivors are not the "
                  "small-deviation ones -- read it alongside the PnL, not instead.")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_mece/best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val",
                    help="DEFAULTS TO VAL. Tune here; spend test once.")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--chunk-len", type=int, default=168)
    # WAS pairwise_monotonicity_taker_side_results_corrected.parquet -- the
    # LADDER results file. That file's own docstring says it "only contains
    # crypto/financials (+ a handful of mentions/entertainment/politics)
    # pairs, since weather/economics/exotics correctly produce zero valid
    # pairs now." MECE baskets are ~58% weather and ~24% sports at a 5c
    # threshold, so NOT ONE of their legs was ever in that table: every
    # lookup missed on ticker AND series and fell through to the global
    # median, which was itself the median of crypto/financials ladder legs.
    # Every settlement PnL figure produced before this change was costed at
    # a flat 5.308c/leg borrowed from a different mechanism, while the run
    # reported "measured spreads". build_mece_leg_spreads.py exists
    # precisely to fix this; it was never passed. See the coverage guard at
    # the end of main().
    ap.add_argument("--spreads", default="mece_leg_spreads.parquet",
                    help="per-leg measured spreads for the MECE population, as "
                         "written by build_mece_leg_spreads.py (leg_a == leg_b == "
                         "ticker). 'none' to assume a flat spread instead. Do NOT "
                         "point this at a ladder results file: those contain no "
                         "weather or sports legs, so every lookup silently "
                         "degrades to the global median.")
    ap.add_argument("--spread-multiplier", type=float, default=0.5,
                    help="0.5 = cross half the spread from mid, per crossing.")
    ap.add_argument("--horizon", type=int, default=1,
                    help="steps ahead the timing decision looks. Also the delay "
                         "incurred by a 'wait' decision.")
    ap.add_argument("--entry-threshold", type=float, default=0.01,
                    help="minimum |dev(t)| in dollars to consider a basket at all.")
    ap.add_argument("--contracts", type=float, default=100.0)
    ap.add_argument("--max-participation", type=float, default=0.0,
                    help="cap size at this share of the thinnest leg's window "
                         "volume. 0 disables, which is not executable.")
    ap.add_argument("--execution", choices=["taker", "maker"], default="taker",
                    help="taker crosses the spread; maker posts and EARNS half "
                         "the spread, but only fills when flow arrives against it.")
    ap.add_argument("--fill-model", choices=["flow", "optimistic"], default="flow",
                    help="maker only. 'flow' requires net_flow to oppose the side "
                         "on EVERY leg; 'optimistic' assumes every post fills.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    # ---- robustness filters (all OFF by default) ------------------------
    #
    # Default-off deliberately. Every figure already reported in this
    # project was produced without these, and changing a default would
    # silently break comparability with the numbers in the results docs.
    # Run once without and once with, and report both.
    ap.add_argument("--min-leg-volume", type=float, default=0.0,
                    help="require EVERY leg to have at least this much volume "
                         "at both t and t+h. mece_sum_to_one_check.py found only "
                         "27.4%% of full baskets have every leg trading >=20 times "
                         "a day, and that roughly half the >5c deviations at tight "
                         "time gaps are thin-leg artefacts whose 'last price' was "
                         "never tradeable. NOTE: this is a CONTRACT VOLUME bar, not "
                         "a trade-count bar -- it is not the same quantity as that "
                         "script's >=20 trades, so do not quote the two as if they "
                         "were. 0 = off.")
    ap.add_argument("--min-ttc", type=float, default=0.0,
                    help="require time-to-close above this (same units as the "
                         "TIME_TO_CLOSE feature slot). Drops near-resolution "
                         "snapshots, where a post-determination print can land in "
                         "the price series: KXHIGHCHI-25APR02 shows two ADJACENT "
                         "temperature brackets at $0.99 and $0.72 within 1.06h, "
                         "which no partition can do. 0 = off.")
    ap.add_argument("--exclude-families", nargs="*", default=[],
                    help="basket families to drop, matched on the segment before "
                         "the first hyphen of the basket ticker. Intended for "
                         "families that are not genuine partitions: "
                         "KXCLUBWCGAME-25JUN28BENCHE shows sum=$1.89 with "
                         "min_leg_trades=369 at a 0.02h gap -- liquid, "
                         "simultaneous and 89c, therefore not a real arbitrage. A "
                         "knockout tie most likely resolves 'tie' and 'winner' on "
                         "different bases (90 minutes vs after extra time), so the "
                         "legs are not mutually exclusive. VERIFY the leg rules "
                         "before excluding; regular-season MLS/EPL, where a draw "
                         "is a terminal outcome, is a genuine partition and must "
                         "NOT be excluded.")
    args = ap.parse_args()

    # ---- load -----------------------------------------------------------
    ck_path = None
    for cand in (_THIS_DIR / args.checkpoint, _REPO_ROOT / args.checkpoint,
                 Path(args.checkpoint)):
        if cand.exists():
            ck_path = cand
            break
    if ck_path is None:
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    cfg, horizons = ck["config"], list(ck["horizons"])
    if args.horizon not in horizons:
        raise SystemExit(f"--horizon {args.horizon} not among trained horizons {horizons}")
    hi, h = horizons.index(args.horizon), args.horizon

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _REPO_ROOT / cache
    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS, cache),
                               verbose=False)
    model = STGATBackbone(padded_feature_width=store.feature_width,
                          embed_dim=cfg.embed_dim, n_heads=cfg.n_heads,
                          max_len=cfg.chunk_len)
    head = ResidualForecastHead(embed_dim=cfg.embed_dim, horizons=horizons)
    model.load_state_dict(ck["model"], strict=True)
    head.load_state_dict(ck["head"], strict=True)
    model.eval(); head.eval()

    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    # Same resolution order and same warn-and-continue behaviour as the
    # round-trip script, so a missing spreads file degrades identically in
    # both rather than one silently assuming something the other does not.
    spreads = None
    if args.spreads.lower() != "none":
        sp = Path(args.spreads)
        if not sp.is_absolute():
            for cand in (_THIS_DIR / sp, _REPO_ROOT / sp, Path.cwd() / sp):
                if cand.exists():
                    sp = cand
                    break
        if sp.exists():
            spreads = SpreadLookup.load(sp)
        else:
            print(f"  WARNING: spreads file not found ({args.spreads}); "
                  f"falling back to a FLAT {FLAT_SPREAD_CENTS}c assumption.")
    legspread = LegSpreads(spreads)

    print(f"\ncheckpoint {ck_path}  (epoch {ck.get('epoch')}, "
          f"selected on {ck.get('selected_by')})")
    print(f"split={args.split}  horizon={h}  execution={args.execution}"
          + (f" fill={args.fill_model}" if args.execution == "maker" else "")
          + f"  cap={args.max_participation or 'none'}")
    if args.split == "test":
        print("  *** scoring TEST. Do this once, with the configuration already "
              "fixed on val. ***")

    events = collect_events(store, chunks, model, head, hi, h)
    if not events:
        raise SystemExit("no complete baskets on this split")

    events = apply_filters(events, args)
    if not events:
        raise SystemExit("every basket was removed by the robustness filters -- "
                         "loosen --min-leg-volume / --min-ttc")

    # ---- strategies -----------------------------------------------------
    #
    # All five hold to settlement, so all five pay ONE-WAY costs and book
    # |dev| at whatever moment they entered. They differ only in WHEN they
    # enter and whether they enter at all.
    names = ["static_now", "cost_gated", "model_timed", "random_timed",
             "model_timed_gated"]
    books = {n: Book(n) for n in names}
    rng = torch.Generator().manual_seed(args.seed)

    n_considered = n_sized_out = n_unfilled = 0
    sizes, ttcs, waits_model, waits_random = [], [], 0, 0
    # Composition of what the cost gate selects. A gate that only ever fires
    # on 40-leg baskets nobody trades is selecting illiquidity, not
    # opportunity, and the distinction is invisible in a PnL total.
    comp_all, comp_gated = [], []

    def costs_at(prices_cents, tickers, C, flows, side):
        """One-way entry cost for a whole basket. Returns (cost, filled)."""
        sp_c = [legspread.get(tk) for tk in tickers]
        half = sum(sp_c) * args.spread_multiplier / PRICE_SCALE * C
        if args.execution == "taker":
            fee = sum(taker_fee_dollars(p, C) for p in prices_cents)
            return fee + half, True            # cross: pay the half-spread
        fee = sum(maker_fee_dollars(pc, C) for pc in prices_cents)
        if args.fill_model == "optimistic":
            filled = True
        else:
            # A resting order fills only when flow arrives against it, and a
            # basket needs EVERY leg. Requiring all of them is the honest
            # reading and is still optimistic about queue position.
            filled = all(f * side < 0 for f in flows)
        return fee - half, filled              # post: EARN the half-spread

    for e in events:
        if abs(e["dev_t"]) <= args.entry_threshold:
            continue
        n_considered += 1

        C = args.contracts
        if args.max_participation > 0.0 and e.get("vol_in"):
            cap = min(min(vi, vl) for vi, vl in zip(e["vol_in"], e["vol_later"]))
            C = min(C, args.max_participation * cap)
        if C < 1.0:
            n_sized_out += 1
            continue
        sizes.append(C)
        ttcs.append(e["ttc"])

        # sell the basket when it is over 1, buy when under
        side_now = -1 if e["dev_t"] > 0 else 1
        side_later = -1 if e["dev_h"] > 0 else 1

        cost_now, filled_now = costs_at(e["entry_cents"], e["tickers"], C,
                                        e["flow_in"], side_now)
        cost_later, filled_later = costs_at(e["later_cents"], e["tickers"], C,
                                            e["flow_in"], side_later)
        gross_now = abs(e["dev_t"]) * C
        gross_later = abs(e["dev_h"]) * C

        if not filled_now:
            n_unfilled += 1

        def book(name, gross, cost, filled):
            if not filled:
                return
            # Book cost as positive fees with zero spread when the cost is
            # negative (a maker credit); Book subtracts both, so a negative
            # fee is a credit and the arithmetic stays correct either way.
            books[name].add(gross, cost, 0.0, e["basket"])

        # 1. enter now, always. This is the existing static_arb, one-way.
        book("static_now", gross_now, cost_now, filled_now)

        # 2. enter now only if the dislocation covers its own cost. No model.
        gated = gross_now > cost_now
        rec = (len(e["tickers"]), e["ttc"] / 86400.0, min(e["vol_in"]),
               abs(e["dev_t"]) * PRICE_SCALE, C)
        comp_all.append(rec)
        if gated:
            comp_gated.append(rec)
            book("cost_gated", gross_now, cost_now, filled_now)

        # 3. WAIT if the model says the dislocation is about to widen, i.e.
        #    the predicted change points the same way as the current sign.
        widening = (e["delta_pred"] * e["dev_t"]) > 0
        if widening:
            waits_model += 1
            book("model_timed", gross_later, cost_later, filled_later)
        else:
            book("model_timed", gross_now, cost_now, filled_now)

        # 4. THE CONTROL: identical machinery, decision replaced by a coin
        #    flip. Same events, same sizes, same cost model.
        coin = torch.rand(1, generator=rng).item() > 0.5
        if coin:
            waits_random += 1
            book("random_timed", gross_later, cost_later, filled_later)
        else:
            book("random_timed", gross_now, cost_now, filled_now)

        # 5. both filters
        if widening:
            if gross_later > cost_later:
                book("model_timed_gated", gross_later, cost_later, filled_later)
        elif gated:
            book("model_timed_gated", gross_now, cost_now, filled_now)

    # ---- report ---------------------------------------------------------
    w = 92
    print("\n" + "=" * w)
    print(f"HOLD-TO-SETTLEMENT MECE ARBITRAGE -- {args.split.upper()} split, "
          f"execution={args.execution}")
    print("=" * w)
    print(f"  spread sources: " + ", ".join(f"{k}={v:,}" for k, v in
                                            sorted(legspread.sources.items()))
          + f"   [{spread_coverage(legspread)}]")
    warn_spread_coverage(legspread, args)
    print(f"  baskets over threshold : {n_considered:,}")
    print(f"  dropped below 1 contract: {n_sized_out:,}"
          f"  ({100.0 * n_sized_out / max(n_considered, 1):.1f}%)")
    if args.execution == "maker":
        print(f"  entries not filled      : {n_unfilled:,}"
              f"  ({100.0 * n_unfilled / max(len(sizes), 1):.1f}% of sized events)")
    if sizes:
        s = torch.tensor(sizes)
        print(f"  size: median {float(s.median()):.1f}  mean {float(s.mean()):.1f}  "
              f"p90 {float(s.quantile(0.9)):.1f}")
    if ttcs:
        tt = torch.tensor(ttcs) / 86400.0
        print(f"  CAPITAL LOCKUP -- time_to_close at entry, days: "
              f"median {float(tt.median()):.1f}  p90 {float(tt.quantile(0.9)):.1f}  "
              f"max {float(tt.max()):.1f}")
        print("    (a position held to settlement is committed for this long; the "
              "cost of that capital is NOT modelled below)")
    print(f"  waited: model {waits_model:,}   coin flip {waits_random:,}")

    if comp_gated:
        print("\n  WHAT THE COST GATE SELECTS  (median of each population)")
        print(f"    {'population':<22}{'n':>8}{'legs':>8}{'days to close':>15}"
              f"{'thinnest leg vol':>19}{'|dev| cents':>13}{'size':>8}")
        for label, pop in (("all sized events", comp_all), ("cost-gated subset", comp_gated)):
            t = torch.tensor(pop, dtype=torch.float64)
            med = t.median(dim=0).values
            print(f"    {label:<22}{len(pop):>8,}{float(med[0]):>8.1f}"
                  f"{float(med[1]):>15.1f}{float(med[2]):>19,.0f}"
                  f"{float(med[3]):>13.1f}{float(med[4]):>8.1f}")
        print("    If the gated row shows far fewer legs, far longer to close, or far")
        print("    thinner volume than the population, the gate is selecting")
        print("    illiquidity rather than opportunity and the PnL is not executable.")

    print("\n" + "-" * w)
    print(f"  {'strategy':<20}{'trades':>8}{'win%':>8}{'gross/trd':>12}"
          f"{'cost/trd':>11}{'net/trd':>11}{'total net':>13}")
    print("-" * w)
    for n in names:
        b = books[n]
        if b.n == 0:
            print(f"  {n:<20}{0:>8}{'--':>8}{'--':>12}{'--':>11}{'--':>11}{'--':>13}")
            continue
        g, f, sp, net = b.totals()
        print(f"  {n:<20}{b.n:>8,}{100.0 * b.wins / b.n:>7.1f}%"
              f"{g / b.n:>12.2f}{(f + sp) / b.n:>11.2f}{net / b.n:>11.2f}"
              f"{net:>13,.0f}")
    print("-" * w)

    print("\n  MODEL minus CONTROL (identical events, sizes and costs; only the")
    print("  timing decision differs, so nothing but the forecast can explain a gap)")
    pt, lo, hi_ = _gap_bootstrap(books["model_timed"], books["random_timed"],
                                 n_boot=args.n_boot, seed=args.seed)
    print("  " + format_verdict("model_timed - random_timed", lo, hi_, pt))

    for n in ("static_now", "cost_gated", "model_timed_gated"):
        if books[n].n and books["model_timed"].n:
            pt, lo, hi_ = _gap_bootstrap(books["model_timed"], books[n],
                                         n_boot=args.n_boot, seed=args.seed)
            print("  " + format_verdict(f"model_timed - {n}", lo, hi_, pt))

    # THE DECIDING COMPARISON. If any configuration here is profitable it is
    # the cost gate, which uses no model at all. So the question that settles
    # whether the model earns its place is not whether timing beats a coin
    # flip -- it is whether adding timing to the PROFITABLE rule improves it.
    if books["model_timed_gated"].n and books["cost_gated"].n:
        print("\n  DOES THE MODEL IMPROVE THE PROFITABLE RULE?")
        pt, lo, hi_ = _gap_bootstrap(books["model_timed_gated"], books["cost_gated"],
                                     n_boot=args.n_boot, seed=args.seed)
        print("  " + format_verdict("model_timed_gated - cost_gated", lo, hi_, pt))
        ga, gb = books["model_timed_gated"], books["cost_gated"]
        print(f"    per trade : {ga.totals()[3] / ga.n:+.3f} on {ga.n:,} trades   vs   "
              f"{gb.totals()[3] / gb.n:+.3f} on {gb.n:,}")
        print(f"    TOTAL net : {ga.totals()[3]:+,.0f}   vs   {gb.totals()[3]:+,.0f}"
              f"   ({100.0 * (ga.totals()[3] / gb.totals()[3] - 1):+.1f}%)")
        print("    Per-trade and total can disagree: the model version trades a")
        print("    LARGER population. Say which one you are quoting.")

    print("\n" + "=" * w)
    print("HOW TO READ THIS")
    print("=" * w)
    print("  static_now is the existing static_arb with one-way costs. If it is")
    print("  positive here and negative in the round-trip script, the entire")
    print("  difference is the exit leg -- a cost-structure result, not a")
    print("  forecasting one, and it must not be reported as the model working.")
    print()
    print("  The model's contribution is ONLY the model_timed - random_timed row.")
    print("  Everything else differs in more than the forecast.")
    print()
    print("  A basket held to settlement captures |dev| whatever the path, so the")
    print("  model is not predicting convergence -- it is predicting whether")
    print("  waiting one step buys a wider dislocation. That is the entire claim")
    print("  this script can support.")
    print()
    print("  If maker execution shows a net credit, check the unfilled count and")
    print("  the capital-lockup line before believing it. Whelan measured Kalshi")
    print("  makers at -9.64% on average, so a model that shows makers earning")
    print("  the spread with no adverse selection is modelling something easier")
    print("  than the real venue.")


if __name__ == "__main__":
    main()