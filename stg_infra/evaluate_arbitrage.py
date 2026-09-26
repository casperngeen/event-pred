"""
evaluate_arbitrage.py

Scores the trained STGAT as an ARBITRAGE STRATEGY against the naive
rule-based baseline, on the HELD-OUT TEST SPLIT, using the metric
definitions in the project's baselines.py so the result is a literal
column-vs-column comparison rather than a differently-computed number.

Both strategies are computed HERE, on the SAME test snapshots, with the
SAME cost model. That is deliberate: the stored baseline parquets were
produced over a different date range, and comparing a model evaluated on
September against a baseline measured on some other window would not be a
comparison at all. Recomputing the naive rule on the test split costs
almost nothing (its signal is already a node feature -- see below) and
removes the alignment question entirely.

WHY THE MODEL CANNOT WIN ON DEVIATION SIZE, AND WHAT IT COMPETES ON
-------------------------------------------------------------------
For a FULLY covered basket the two strategies see the identical signal,
as arithmetic rather than as an empirical finding: MeceOutputHead's
softmax forces the predicted leg prices to sum to exactly 1, so

    naive deviation  = sum(observed) - 1
                     = sum(observed) - sum(predicted)
                     = sum over legs of (observed_i - predicted_i)
                     = total of the model's own per-leg deviations.

So on full baskets the model's aggregate mispricing estimate EQUALS the
naive rule's, exactly. Any claim that it "detects more mispricing" there
would be measuring rounding error. The two real edges, both named in the
project's own kalshi-arbitrage-findings-summary.md, are:

  SELECTION -- the naive rule trades EVERY basket whose deviation clears
    a threshold, indiscriminately, which is why its realistic median PnL
    is negative even while its mean is positive: the winners carry the
    losers. The model decides WHICH flagged baskets to trade, using the
    SHAPE of the per-leg disagreement (see build_strategies for the
    concentration criterion and its logic).

    NOT leg pruning. An earlier version of this script let the model drop
    individual legs while still crediting the whole basket's deviation as
    profit. That destroys the arbitrage: the guarantee comes from holding
    EVERY leg, since exactly one resolves YES. Dropping legs preserves the
    expected value but not the certainty, turning a riskless position into
    a directional one -- while the accounting still reported a guaranteed
    payoff. It produced a monotonically rising "PnL" as legs were dropped,
    peaking at 1.27 legs out of 4.9, which is a bookkeeping artifact and
    not a strategy. Both strategies here trade whole baskets.

  COVERAGE -- the naive rule needs EVERY leg to have traded that
    snapshot, which is why 14+ leg families contribute zero opportunities.
    The model prices an unobserved leg from its neighbours, so a basket
    with some legs missing still yields an estimate.

HONESTY ABOUT COVERAGE, WHICH IS NOT RISKLESS ARBITRAGE. On a fully
covered basket the profit is structurally guaranteed: exactly one leg
settles at $1, so selling every leg of a basket trading at S > $1 locks
in S - 1 regardless of outcome. On a PARTIALLY covered basket the sum
depends on the model's estimate of the missing legs, so the "edge" is
only as good as that estimate and the position carries model risk. It is
reported SEPARATELY below and never added into the riskless totals,
because presenting it as arbitrage would overstate the result.

COST MODEL -- MUST MATCH YOUR BACKTEST SCRIPTS. The constants below
mirror mece_sum_to_one_pnl_backtest.py. Check them against your copy
before quoting any number from this script; if they differ, the
comparison silently stops being like-for-like.
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

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.inference import masked_forward  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import STGATBackbone, TrainingConfig, true_prices  # noqa: E402
from model.training_objective import MaskedLegReconstructionObjective  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]

# --- cost model: mirror mece_sum_to_one_pnl_backtest.py ---------------
TAKER_FEE_RATE = 0.07          # Kalshi taker fee coefficient
PRICE_SCALE = 100.0            # cents per dollar
CONTRACTS_PER_OPPORTUNITY = 100.0   # contracts PER LEG
PER_LEG_HAIRCUT_CENTS = 1.0    # flat execution-cost assumption (Kalshi min tick)
VIOLATION_THRESHOLD = 0.01     # |deviation| in dollars required to trade

# Basket hub feature slots (stg/nodes/kalshi.py, KalshiMeceHyperedges)
SLOT_SUM_CENTS = 0
SLOT_DEVIATION = 1
SLOT_LEGS_KNOWN = 2
SLOT_LEGS_TOTAL = 3


def taker_fee_dollars(price_cents: float, contracts: float) -> float:
    """Kalshi taker fee, rounded up to the cent-hundredth as the backtest
    scripts do. Charged per leg."""
    p = max(0.0, min(1.0, price_cents / PRICE_SCALE))
    raw = TAKER_FEE_RATE * contracts * p * (1.0 - p)
    return math.ceil(raw * 10000) / 10000


class Strategy:
    """Accumulates per-opportunity PnL, matching baselines.py's metrics."""

    def __init__(self, name: str):
        self.name = name
        self.idealized = []
        self.fees = []
        self.haircuts = []
        self.legs_traded = []

    def add(self, edge_dollars: float, leg_prices_cents, contracts: float,
            haircut_cents: float = PER_LEG_HAIRCUT_CENTS):
        n_legs = len(leg_prices_cents)
        if n_legs == 0:
            return
        gross = edge_dollars * contracts
        fee = sum(taker_fee_dollars(p, contracts) for p in leg_prices_cents)
        haircut = n_legs * (haircut_cents / PRICE_SCALE) * contracts
        self.idealized.append(gross)
        self.fees.append(fee)
        self.haircuts.append(haircut)
        self.legs_traded.append(n_legs)

    @property
    def n(self):
        return len(self.idealized)

    def pnl(self, scenario: str):
        if scenario == "idealized":
            return list(self.idealized)
        if scenario == "fees_only":
            return [g - f for g, f in zip(self.idealized, self.fees)]
        return [g - f - h for g, f, h in zip(self.idealized, self.fees, self.haircuts)]

    def metrics(self, scenario: str, universe_size: int) -> dict:
        """Same fields, same definitions as baselines.py's metrics_row()."""
        p = self.pnl(scenario)
        n = len(p)
        if n == 0:
            return {"strategy": self.name, "scenario": scenario, "n_opportunities": 0}
        mean = sum(p) / n
        var = sum((x - mean) ** 2 for x in p) / (n - 1) if n > 1 else 0.0
        std = var ** 0.5
        srt = sorted(p)
        median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        return {
            "strategy": self.name,
            "scenario": scenario,
            "n_opportunities": n,
            "universe_size": universe_size,
            "coverage_pct": 100.0 * n / universe_size if universe_size else None,
            "total_pnl_usd": sum(p),
            "mean_pnl_usd": mean,
            "median_pnl_usd": median,
            "win_rate": sum(1 for x in p if x > 0) / n,
            "cross_sectional_sharpe": (mean / std) if std > 0 else None,
            "total_fees_usd": sum(self.fees),
            "mean_legs_traded": sum(self.legs_traded) / n,
        }


def deduplicate_opportunities(opportunities, mode: str = "first"):
    """Collapses repeated sightings of the SAME basket mispricing into ONE
    tradeable opportunity per (basket, day).

    WHY THIS IS NECESSARY, NOT OPTIONAL. The graph is built at 2-hour
    resolution, so a single basket that is mispriced for an afternoon
    appears in a dozen consecutive snapshots. Treating each as an
    independent opportunity is wrong in both directions at once:

      - You cannot trade the same mispricing twelve times and collect the
        deviation on each. It is one position.
      - It DILUTES the measured edge. A mispricing builds and decays, so
        most windows show a smaller deviation than the episode's peak;
        averaging over all of them drags the mean toward the small ones
        while still charging full per-leg costs on every count.

    Measured against this project's own earlier per-DAY backtest the
    distortion is large: 5,670 "opportunities" in one month versus 267
    across that study, and a mean |deviation| of $0.0561 against the
    $0.078-$0.119 that study's reported 1-2c breakeven implies. The
    2-hourly count is the reason this evaluation put breakeven at 0.46c
    while the project's own baseline put it at 1-2c.

    MODES, AND THE LOOK-AHEAD TRAP:
      "first" (default) -- the earliest snapshot of the day whose
          deviation clears the threshold. This is what a trader monitoring
          the market actually gets: you see a qualifying deviation and you
          trade it. NO look-ahead.
      "last"  -- the final qualifying snapshot of the day, matching the
          end-of-day convention mece_sum_to_one_pnl_backtest.py uses
          (each leg's LAST trade price that day).
      "max"   -- the largest deviation of the day. This REQUIRES KNOWING
          THE FUTURE: at 9am you cannot know 3pm will be better. It is an
          upper bound only, and must never be reported as an achievable
          result.
    """
    if mode not in ("first", "last", "max", "none"):
        raise ValueError(f"unknown dedup mode {mode!r}")
    if mode == "none":
        return opportunities

    best = {}
    for o in opportunities:
        key = (o.get("hub_id"), o.get("day"))
        if key[0] is None or key[1] is None:
            # no identity available -- keep it rather than silently merging
            best[id(o)] = o
            continue
        cur = best.get(key)
        if cur is None:
            best[key] = o
        elif mode == "last":
            best[key] = o
        elif mode == "max" and o["deviation"] > cur["deviation"]:
            best[key] = o
        # "first": keep the earliest, which is what is already stored
    return list(best.values())


def build_strategies(opportunities, concentration_max: float, haircut_cents: float):
    """Rebuilds both strategies from CACHED per-opportunity detail, so a
    sweep costs no extra forward passes.

    BOTH STRATEGIES TRADE EVERY LEG OF EVERY BASKET THEY TRADE. This is
    not a style choice, it is what makes the position an arbitrage at all.

    WHY AN EARLIER VERSION OF THIS FUNCTION WAS WRONG. It let the model
    trade a SUBSET of a basket's legs while still crediting the full
    basket-level deviation as profit. That inflates PnL mechanically:
    gross is held fixed while per-leg costs shrink, so "profit" rises
    monotonically as legs are dropped, and a sweep duly reported its best
    result at 1.27 legs out of 4.9. The economics say otherwise. Selling
    YES on every leg of a basket trading at S collects S and owes exactly
    $1 at settlement, because exactly one leg resolves YES -- profit
    S - 1, with certainty. Skip leg j and the payoff becomes S - p_j when
    j wins and S - p_j - 1 when it does not: the EXPECTATION is unchanged
    at S - 1, but the variance is no longer zero. That is a directional
    bet, not arbitrage, and reporting it with a guaranteed-payoff formula
    (and a "win rate" derived from one) overstates it badly.

    WHAT THE MODEL LEGITIMATELY DECIDES INSTEAD: which flagged baskets are
    worth trading at all -- the "selection" axis the project's own
    kalshi-arbitrage-findings-summary.md actually specifies ("predict
    which of the naive rule's flagged opportunities will actually clear
    realistic costs"), as opposed to the leg-level pruning implemented by
    mistake.

    THE CRITERION, AND ITS LOGIC. Note first that the model cannot simply
    re-estimate the deviation: MeceOutputHead's softmax forces predicted
    prices to sum to 1, so sum(observed_i - predicted_i) EQUALS the naive
    deviation identically. The per-leg gaps are not independent of it.
    What the model does add is the SHAPE of that disagreement:

      concentration = max|gap| / sum|gap|

    A deviation spread across many legs looks like genuine basket-level
    mispricing. A deviation produced almost entirely by ONE leg, where the
    model's neighbour- and history-based estimate disagrees sharply with
    that leg's last print, looks instead like a stale or unrepresentative
    quote -- and a stale quote is not tradeable, because the price you
    would actually fill at is not the one being differenced. The naive
    rule cannot tell those apart; it sees only the sum. Baskets with
    concentration above ``concentration_max`` are therefore skipped.

    This is a hypothesis the numbers can reject: if the filter does not
    beat trading everything, that is a real negative result and should be
    reported as one.
    """
    naive = Strategy("naive_all_baskets")
    stgat = Strategy("stgat_selected_baskets")
    for o in opportunities:
        edge, legs = o["deviation"], o["legs"]
        all_cents = [c for _, c in legs]
        # Both strategies always trade the WHOLE basket -- the guarantee.
        naive.add(edge, all_cents, CONTRACTS_PER_OPPORTUNITY, haircut_cents)

        gaps = [g for g, _ in legs if g is not None]
        if not gaps:
            continue
        total_gap = sum(gaps)
        if total_gap <= 0:
            continue
        concentration = max(gaps) / total_gap
        if concentration <= concentration_max:
            stgat.add(edge, all_cents, CONTRACTS_PER_OPPORTUNITY, haircut_cents)
    return naive, stgat


def sweep_leg_threshold(opportunities, thresholds, haircut_cents):
    """Sweeps the BASKET-selection criterion (gap concentration). Named
    for its CLI flag, which is kept for compatibility.

    Every row trades whole baskets, so every row is a genuine arbitrage
    portfolio and the PnL figures are directly comparable to the naive
    rule's. Contrast the earlier leg-pruning sweep, whose apparent gains
    came from crediting full basket profit to partial positions.

    Read the table for the point where skipping concentrated-gap baskets
    stops helping. Because the filter DISCARDS opportunities, total PnL
    can fall simply from trading less; the columns that matter are MEAN
    and MEDIAN per opportunity, which say whether the ones kept are
    better than the ones dropped.

    TUNE THIS ON VAL, NOT TEST. Picking the value that maximises a test
    number is fitting the held-out set."""
    print()
    print("=" * 96)
    print("BASKET-SELECTION SWEEP (gap concentration) -- realistic scenario")
    print("Every row trades WHOLE baskets, so every row is a real arbitrage portfolio.")
    print("Judge on MEAN/MEDIAN per opportunity, not total (the filter trades fewer).")
    print("Choose on VAL. Reading the best value off a TEST run is test-set fitting.")
    print("=" * 96)
    base_n, base = None, None
    print(f"{'max_concn':>10} {'n_opps':>8} {'kept%':>7} {'total_pnl':>13} "
          f"{'mean':>10} {'median':>10} {'win%':>7}")
    print("-" * 96)
    for th in thresholds:
        n, s = build_strategies(opportunities, th, haircut_cents)
        if base_n is None:
            base_n = n.n
            base = n.metrics("realistic", 1)
            print(f"{'(naive)':>10} {base['n_opportunities']:>8} {100.0:>6.1f}% "
                  f"${base['total_pnl_usd']:>12,.2f} ${base['mean_pnl_usd']:>9,.2f} "
                  f"${base['median_pnl_usd']:>9,.2f} {100 * base['win_rate']:>6.1f}%")
            print("-" * 96)
        m = s.metrics("realistic", 1)
        if not m.get("n_opportunities"):
            print(f"{th:>10.2f} {0:>8} {0.0:>6.1f}% {'(filter rejects everything)':>13}")
            continue
        print(f"{th:>10.2f} {m['n_opportunities']:>8} "
              f"{100 * m['n_opportunities'] / max(base_n, 1):>6.1f}% "
              f"${m['total_pnl_usd']:>12,.2f} ${m['mean_pnl_usd']:>9,.2f} "
              f"${m['median_pnl_usd']:>9,.2f} {100 * m['win_rate']:>6.1f}%")
    print("-" * 96)
    if base:
        print(f"  Baseline to beat on MEAN: ${base['mean_pnl_usd']:,.2f} per opportunity "
              f"(median ${base['median_pnl_usd']:,.2f}).")
        print("  A filter that does not raise mean/median above these has no selection value,")
        print("  which is a legitimate finding and should be reported as one.")


def sweep_friction(opportunities, leg_threshold, haircuts):
    """Finds the BREAKEVEN execution friction for each strategy.

    Why this matters more than either PnL number on its own: the project's
    own backtest documents PER_LEG_HAIRCUT_CENTS as "a flat, non-data-driven
    assumption", not a measured cost, and MECE breakeven sits at 1-2c of
    per-leg friction -- exactly where the default sits. A single point
    estimate at an assumed cost is therefore decided by the assumption.
    Reporting the friction level at which each strategy turns profitable
    is both more honest and a sharper result: if the model breaks even at
    a higher friction than the naive rule does, it is strictly more robust
    to execution costs, and that claim does not depend on guessing the
    true cost."""
    print()
    print("=" * 96)
    print("EXECUTION-FRICTION SENSITIVITY -- total realistic PnL vs per-leg haircut")
    print("The haircut is an ASSUMPTION, not a measurement. Breakeven is the real result.")
    print("=" * 96)
    print(f"{'haircut(c)':>11} {'naive_total':>15} {'stgat_total':>15} {'naive_median':>14} "
          f"{'stgat_median':>14}")
    print("-" * 96)
    naive_be = stgat_be = None
    for hc in haircuts:
        n, s = build_strategies(opportunities, leg_threshold, hc)
        nm, sm = n.metrics("realistic", 1), s.metrics("realistic", 1)
        nt = nm.get("total_pnl_usd", 0.0) or 0.0
        st = sm.get("total_pnl_usd", 0.0) or 0.0
        if naive_be is None and nt < 0:
            naive_be = hc
        if stgat_be is None and st < 0:
            stgat_be = hc
        print(f"{hc:>11.2f} ${nt:>14,.2f} ${st:>14,.2f} "
              f"${nm.get('median_pnl_usd', 0.0):>13,.2f} ${sm.get('median_pnl_usd', 0.0):>13,.2f}")
    print("-" * 96)
    print(f"  naive rule turns negative at roughly {naive_be if naive_be else '>max tested'}c per leg")
    print(f"  STGAT      turns negative at roughly {stgat_be if stgat_be else '>max tested'}c per leg")
    if naive_be and stgat_be and stgat_be > naive_be:
        print(f"  --> the model tolerates {stgat_be - naive_be:.2f}c MORE friction per leg before")
        print("      losing money. That is a cost-robustness result independent of which")
        print("      friction figure is actually correct.")


def validate_coverage(model, objective, cfg, store, chunks, threshold, hide_frac=0.35, seed=0):
    """Tests whether the COVERAGE EXTENSION measures real mispricing or
    the model's own estimation error.

    THE PROBLEM IT CHECKS. On a partially covered basket the estimated
    sum is (observed legs) + (model's guess for the rest), so the
    estimated deviation from $1 contains the model's error as well as any
    genuine mispricing. If the model guesses a missing leg 20c wrong,
    that error appears as a 20c "deviation" and books as 20c of profit
    that cannot be captured, because the trade needs a real counterparty
    at a real price. A first test run showed exactly the signature of
    this: coverage baskets had a median estimated deviation around 20c
    against ~3c for fully-observed ones, and a strongly POSITIVE realistic
    PnL where the riskless version was negative.

    THE TEST. Take baskets that ARE fully covered, so the truth is known.
    Hide a fraction of their legs, have the model estimate those legs from
    their neighbours exactly as the coverage path does, and compare the
    ESTIMATED deviation against the TRUE one. Reports:

      bias / MAE    -- how far the estimate lands from the truth
      |est| vs |true| -- the number that matters. If estimated deviations
                      are systematically LARGER in magnitude, the coverage
                      PnL is inflated by estimation error rather than
                      finding real arbitrage.
      inflation     -- mean(|est|) / mean(|true|). At 1.0 the estimate is
                      unbiased in magnitude; well above 1.0 means the
                      extension mostly trades on its own noise.
    """
    gen = torch.Generator().manual_seed(seed)
    true_devs, est_devs = [], []
    flagged_true, flagged_est = 0, 0

    for c in chunks:
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)
        mece_adj = adj.get("mece_leg_to_basket")
        if mece_adj is None:
            del ct
            continue

        # Hide a random subset of mechanism legs; the REST stay visible,
        # which is what makes this a simulation of partial coverage rather
        # than the mask-everything protocol.
        is_ticker = features[..., -1] == 0.0
        hide = mask & is_ticker & (torch.rand(mask.shape, generator=gen) < hide_frac)
        if int(hide.sum()) == 0:
            del ct
            continue
        with torch.no_grad():
            h = masked_forward(model, features, mask, adj, hide,
                               objective.mask_token, cfg.raw_feature_width)
            mece_out = model.mece_head(h, mece_adj)

        for t in range(features.shape[0]):
            edges = mece_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            res = mece_out.get(t)
            if res is None:
                continue
            leg_g, hub_g = edges.edge_index[0], edges.edge_index[1]
            for hub in torch.unique(hub_g).tolist():
                sel = (hub_g == hub).nonzero(as_tuple=True)[0]
                legs = leg_g[sel].tolist()
                legs_total = float(features[t, hub, SLOT_LEGS_TOTAL])
                observed = [l for l in legs if bool(mask[t, l])]
                if not legs_total or len(observed) < int(round(legs_total)):
                    continue  # only fully covered baskets have a known truth
                if not any(bool(hide[t, l]) for l in legs):
                    continue  # nothing hidden -> nothing to estimate

                true_sum, est_sum, ok = 0.0, 0.0, True
                for l in legs:
                    obs = float(prices[t, l])
                    true_sum += obs
                    if bool(hide[t, l]):
                        hit = (res["leg_idx"] == l).nonzero(as_tuple=True)[0]
                        if hit.numel() == 0:
                            ok = False
                            break
                        est_sum += float(res["fair_price"][hit[0]])
                    else:
                        est_sum += obs
                if not ok:
                    continue
                td, ed = true_sum - 1.0, est_sum - 1.0
                true_devs.append(td)
                est_devs.append(ed)
                if abs(td) > threshold:
                    flagged_true += 1
                if abs(ed) > threshold:
                    flagged_est += 1
        del ct

    n = len(true_devs)
    if n == 0:
        print("\n(coverage validation: no fully-covered baskets with hidden legs to test)")
        return

    bias = sum(e - t for e, t in zip(est_devs, true_devs)) / n
    mae = sum(abs(e - t) for e, t in zip(est_devs, true_devs)) / n
    mean_abs_true = sum(abs(t) for t in true_devs) / n
    mean_abs_est = sum(abs(e) for e in est_devs) / n
    inflation = mean_abs_est / mean_abs_true if mean_abs_true else float("nan")

    print()
    print("=" * 96)
    print("COVERAGE VALIDATION -- is the extension finding mispricing, or its own error?")
    print(f"Fully covered baskets with {hide_frac:.0%} of legs hidden and model-estimated,")
    print("so the TRUE deviation is known and the estimate can be scored against it.")
    print("=" * 96)
    print(f"  baskets tested:                     {n:,}")
    print(f"  mean |true deviation|:              ${mean_abs_true:.4f}")
    print(f"  mean |estimated deviation|:         ${mean_abs_est:.4f}")
    print(f"  bias (est - true):                  ${bias:+.4f}")
    print(f"  MAE of the estimate:                ${mae:.4f}")
    print(f"  INFLATION mean|est| / mean|true|:   {inflation:.2f}x")
    print(f"  flagged as opportunities: {flagged_true:,} by truth, {flagged_est:,} by estimate "
          f"({100 * flagged_est / max(flagged_true, 1):.0f}% of truth)")
    print("-" * 96)
    if inflation > 1.5:
        print("  VERDICT: estimated deviations are substantially LARGER than the real ones, so the")
        print("  coverage extension is trading mostly on model error. Its PnL is NOT a real")
        print("  arbitrage result and must not be reported as one. The honest framing is that")
        print("  coverage extends REACH, while the edge it appears to find is largely noise.")
    elif inflation > 1.15:
        print("  VERDICT: estimates are somewhat inflated. Coverage PnL is optimistic; report it")
        print("  with this inflation factor stated alongside, never as a clean arbitrage number.")
    else:
        print("  VERDICT: estimated deviations track the real ones in magnitude. The coverage")
        print("  extension is finding genuine mispricing, not manufacturing it -- though it")
        print("  still carries model risk that fully-observed arbitrage does not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--threshold", type=float, default=VIOLATION_THRESHOLD,
                    help="|deviation| in dollars to flag an opportunity (match your backtest)")
    ap.add_argument("--leg-threshold", type=float, default=0.60,
                    help="BASKET filter: max allowed gap concentration "
                         "max|gap|/sum|gap|. A basket whose deviation is dominated by ONE "
                         "leg looks like a stale quote rather than real mispricing and is "
                         "skipped. 1.0 disables the filter (trade everything, = naive).")
    ap.add_argument("--sweep-leg-threshold", action="store_true",
                    help="try several per-leg thresholds and report realistic PnL for each. "
                         "At 0.02 the model kept ~3.9 of 4.1 legs, i.e. it was barely selecting; "
                         "the threshold is a free parameter and worth tuning on VAL, not test.")
    ap.add_argument("--dedup", choices=["first", "last", "max", "none"], default="first",
                    help="collapse repeated 2-hourly sightings of the same basket mispricing "
                         "into one tradeable opportunity per (basket, day). 'first' (default) "
                         "trades the earliest qualifying snapshot -- no look-ahead. 'last' "
                         "matches the end-of-day convention of the project's own backtest. "
                         "'max' needs knowledge of the future and is an UPPER BOUND ONLY. "
                         "'none' reproduces the old per-snapshot counting, which over-counts "
                         "opportunities ~20x and dilutes the measured edge.")
    ap.add_argument("--sweep-friction", action="store_true",
                    help="report total realistic PnL across a range of per-leg execution "
                         "haircuts, and the friction level at which each strategy turns "
                         "negative. The haircut is an assumption, not a measurement, so "
                         "breakeven is a more defensible result than any single point estimate.")
    ap.add_argument("--skip-coverage-validation", action="store_true",
                    help="skip the check that the coverage extension is finding real mispricing "
                         "rather than its own estimation error (on by default -- it should be)")
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

    print(f"Checkpoint: {ckpt_path.name} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss'):.4f})")
    print(f"Split: {args.split.upper()}  |  flag threshold: ${args.threshold:.3f}  "
          f"|  basket gap-concentration max: {args.leg_threshold:.2f}")
    print(f"Cost model: fee_rate={TAKER_FEE_RATE}, {CONTRACTS_PER_OPPORTUNITY:.0f} contracts/leg, "
          f"{PER_LEG_HAIRCUT_CENTS:.1f}c haircut/leg\n")

    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len,
                                      min_chunk_len=cfg.min_chunk_len) if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")
    print(f"{len(chunks)} {args.split} chunk(s)\n")

    opportunities = []   # cached per-opportunity leg detail, for the sweeps
    coverage = Strategy("stgat_coverage_extension")   # MODEL RISK, reported separately

    n_full_baskets = 0
    n_partial_baskets = 0
    n_flagged = 0

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask, adj = ct["features"], ct["mask"], ct["adjacency_by_type"]
        prices = true_prices(features)
        mece_adj = adj.get("mece_leg_to_basket")
        if mece_adj is None:
            del ct
            continue

        # Mask EVERY mechanism leg so each gets a context-only fair value,
        # which is what a deviation signal requires (see inference.py).
        is_ticker = features[..., -1] == 0.0
        target_mask = mask & is_ticker
        with torch.no_grad():
            h = masked_forward(model, features, mask, adj, target_mask,
                               objective.mask_token, cfg.raw_feature_width)
            mece_out = model.mece_head(h, mece_adj)

        T = features.shape[0]
        for t in range(T):
            edges = mece_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            res = mece_out.get(t)
            leg_g, hub_g = edges.edge_index[0], edges.edge_index[1]

            for hub in torch.unique(hub_g).tolist():
                sel = (hub_g == hub).nonzero(as_tuple=True)[0]
                legs = leg_g[sel].tolist()
                if not legs:
                    continue
                legs_total = float(features[t, hub, SLOT_LEGS_TOTAL])
                observed_here = [l for l in legs if bool(mask[t, l])]
                is_full = legs_total > 0 and len(observed_here) >= int(round(legs_total))

                # --- the naive rule's own signal, straight off the hub node ---
                deviation = float(features[t, hub, SLOT_DEVIATION])
                if is_full:
                    n_full_baskets += 1
                else:
                    n_partial_baskets += 1

                leg_cents = [float(prices[t, l]) * PRICE_SCALE for l in observed_here]

                if is_full and abs(deviation) > args.threshold:
                    n_flagged += 1
                    # Cache each leg's (model gap, price) ONCE. Both
                    # strategies and every sweep below are then pure
                    # post-processing of this, with no further model runs.
                    leg_detail = []
                    for l in observed_here:
                        obs = float(prices[t, l])
                        gap = None
                        if res is not None:
                            hit = (res["leg_idx"] == l).nonzero(as_tuple=True)[0]
                            if hit.numel() > 0:
                                gap = abs(obs - float(res["fair_price"][hit[0]]))
                        leg_detail.append((gap, obs * PRICE_SCALE))
                    ts = ct["timestamps"][t]
                    node_ids = ct.get("node_ids")
                    opportunities.append({
                        "deviation": abs(deviation),
                        "legs": leg_detail,
                        # identity, so repeated 2-hourly sightings of the SAME
                        # basket mispricing can be collapsed to one tradeable
                        # opportunity (see deduplicate_opportunities)
                        "hub_id": node_ids[hub] if node_ids else int(hub),
                        "day": getattr(ts, "date", lambda: ts)(),
                    })

                # --- COVERAGE: baskets the naive rule cannot touch -------
                elif not is_full and res is not None and legs_total > 0:
                    # estimate the basket sum using observed legs where we
                    # have them and the model's fair value where we don't
                    est = 0.0
                    ok = True
                    for l in legs:
                        hit = (res["leg_idx"] == l).nonzero(as_tuple=True)[0]
                        if bool(mask[t, l]):
                            est += float(prices[t, l])
                        elif hit.numel() > 0:
                            est += float(res["fair_price"][hit[0]])
                        else:
                            ok = False
                            break
                    if ok:
                        dev_est = est - 1.0
                        if abs(dev_est) > args.threshold and leg_cents:
                            coverage.add(abs(dev_est), leg_cents, CONTRACTS_PER_OPPORTUNITY)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} done", flush=True)

    raw_n = len(opportunities)
    opportunities = deduplicate_opportunities(opportunities, args.dedup)
    if args.dedup != "none":
        print(f"\nDeduplication ('{args.dedup}'): {raw_n:,} per-snapshot sightings "
              f"-> {len(opportunities):,} tradeable opportunities "
              f"({raw_n / max(len(opportunities), 1):.1f}x over-counting removed)")
        if opportunities:
            md = sum(o["deviation"] for o in opportunities) / len(opportunities)
            print(f"  mean |deviation| after dedup: ${md:.4f}")
        if args.dedup == "max":
            print("  *** 'max' uses knowledge of the future -- UPPER BOUND ONLY, not achievable.")
        print()

    naive, stgat = build_strategies(opportunities, args.leg_threshold, PER_LEG_HAIRCUT_CENTS)
    universe = n_full_baskets + n_partial_baskets
    print(f"\nbasket snapshots seen: {universe:,} "
          f"({n_full_baskets:,} fully covered, {n_partial_baskets:,} partial)")
    print(f"flagged by the naive rule (|deviation| > ${args.threshold:.3f}): {n_flagged:,}\n")

    print("=" * 96)
    print(f"RISKLESS ARBITRAGE -- fully covered baskets only, {args.split.upper()} split")
    print("Both strategies: identical opportunities, identical cost model.")
    print("=" * 96)
    hdr = (f"{'strategy':<22} {'scenario':<11} {'n':>6} {'legs':>6} {'total_pnl':>12} "
           f"{'mean':>10} {'median':>10} {'win%':>7} {'sharpe':>8}")
    print(hdr)
    print("-" * 96)
    for scen in ("idealized", "fees_only", "realistic"):
        for s in (naive, stgat):
            m = s.metrics(scen, universe)
            if not m.get("n_opportunities"):
                continue
            sh = f"{m['cross_sectional_sharpe']:.3f}" if m["cross_sectional_sharpe"] else "n/a"
            print(f"{m['strategy']:<22} {scen:<11} {m['n_opportunities']:>6} "
                  f"{m['mean_legs_traded']:>6.1f} ${m['total_pnl_usd']:>11,.2f} "
                  f"${m['mean_pnl_usd']:>9,.2f} ${m['median_pnl_usd']:>9,.2f} "
                  f"{100 * m['win_rate']:>6.1f}% {sh:>8}")
        print("-" * 96)

    if coverage.n:
        print()
        print("=" * 96)
        print("COVERAGE EXTENSION -- partially covered baskets the naive rule cannot trade.")
        print("NOT riskless: the basket sum uses MODEL ESTIMATES for unobserved legs, so this")
        print("carries model risk and is deliberately NOT added to the totals above.")
        print("=" * 96)
        print(hdr)
        print("-" * 96)
        for scen in ("idealized", "fees_only", "realistic"):
            m = coverage.metrics(scen, universe)
            sh = f"{m['cross_sectional_sharpe']:.3f}" if m["cross_sectional_sharpe"] else "n/a"
            print(f"{m['strategy']:<22} {scen:<11} {m['n_opportunities']:>6} "
                  f"{m['mean_legs_traded']:>6.1f} ${m['total_pnl_usd']:>11,.2f} "
                  f"${m['mean_pnl_usd']:>9,.2f} ${m['median_pnl_usd']:>9,.2f} "
                  f"{100 * m['win_rate']:>6.1f}% {sh:>8}")
        print("-" * 96)
        print(f"\nnaive coverage:  {100 * naive.n / universe:.2f}% of basket snapshots")
        print(f"with extension:  {100 * (naive.n + coverage.n) / universe:.2f}% "
              f"(+{coverage.n:,} opportunities the naive rule has no way to reach)")

    if args.sweep_leg_threshold:
        sweep_leg_threshold(opportunities,
                            [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00],
                            PER_LEG_HAIRCUT_CENTS)

    if args.sweep_friction:
        sweep_friction(opportunities, args.leg_threshold,
                       [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    if not args.skip_coverage_validation:
        validate_coverage(model, objective, cfg, store, chunks, args.threshold)

    print("\nCOMPARE AGAINST: baselines.py's 'realistic' rows, which are the actual bar.")
    print("Note this script recomputes the naive rule on THIS split rather than reading the")
    print("stored parquets, so both columns come from the same data and the same cost model.")


if __name__ == "__main__":
    main()