"""
diagnose_staleness.py

IS THE VIOLATION PERSISTENCE REAL, OR IS THE PRICE SIMPLY NOT UPDATING?

`diagnose_persistence.py` found that 96.8% of ladder violations survive two
hours and 95.8% survive twenty-four. Read naively that says deviations are
sticky and a resting limit order has plenty of time to fill.

There is a second reading, and this script exists to tell them apart.

The node feature is `last_yes_price` -- a LAST TRADED PRICE, not a live
quote. On a rung that has not traded since yesterday, that number is just
the last print. Two stale prints struck at different moments will show a
monotonicity violation forever, and NEITHER is a price anyone can trade
against. Under that reading the persistence is an artifact of the data not
updating, and some share of the "violations" were never opportunities.

Three pieces of evidence already point that way:
  - conditional survival barely decays (96.8% -> 95.8% over 24h)
  - the median gap sits at exactly $0.0500, suggesting tick clustering
  - the persistence MAE does NOT grow with horizon (0.78c, 1.08c, 1.21c,
    1.20c, 1.11c). A real price process diffuses and its forecast error
    grows roughly as sqrt(h). Flat error is the signature of a series that
    is not moving.

So this script measures the thing the gap can never reveal: whether the
underlying prices move at all, and whether violation survival is CONDITIONAL
on them not moving.

The decisive cross-tabulation is:

    of violations that survive to t+1, what fraction had ZERO price
    movement on BOTH legs?

If that fraction is near 100%, the persistence is staleness and the
opportunities are not tradeable. If violations survive even when both legs
reprice, the mispricing is real and survives active trading -- which is a
much stronger economic finding than persistence alone.

No model is loaded. Pure measurement over the cached graph.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "examples"))

from model.chunking import build_split_ranges, chunk_ranges  # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths  # noqa: E402
from model.train import true_prices  # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
GAP_THRESHOLD = 0.01
TICK = 0.01           # Kalshi minimum tick, in dollars


def _median(v):
    if not v:
        return float("nan")
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def _pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--chunk-len", type=int, default=84)
    ap.add_argument("--threshold", type=float, default=GAP_THRESHOLD)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    ap.add_argument("--move-eps", type=float, default=TICK / 2,
                    help="a leg counts as having REPRICED if it moved more than this "
                         "(default half a tick, so any genuine 1c move counts)")
    args = ap.parse_args()

    months = args.months or PILOT_MONTHS
    store = MonthlyBundleStore(build_month_paths(months, _REPO_ROOT / "cache"), verbose=False)
    t_index = {ts: i for i, ts in enumerate(store.timestamps)}

    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"No '{args.split}' chunks for months {months}.")

    px = defaultdict(dict)          # ticker -> {global_t: price}
    pairs = defaultdict(dict)       # (ta, tb) -> {global_t: gap}

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        mask, prices = ct["mask"], true_prices(ct["features"])
        ladder_adj = ct["adjacency_by_type"].get("ladder_monotonic")
        node_ids = ct.get("node_ids")
        if ladder_adj is None or not node_ids:
            del ct
            continue
        for t in range(prices.shape[0]):
            edges = ladder_adj[t]
            if edges.edge_index.numel() == 0:
                continue
            gt = t_index.get(ct["timestamps"][t])
            if gt is None:
                continue
            a_idx, b_idx = edges.edge_index[0], edges.edge_index[1]
            for k in range(a_idx.shape[0]):
                a, b = int(a_idx[k]), int(b_idx[k])
                if not (bool(mask[t, a]) and bool(mask[t, b])):
                    continue
                ta, tb = str(node_ids[a]), str(node_ids[b])
                pa, pb = float(prices[t, a]), float(prices[t, b])
                px[ta][gt] = pa
                px[tb][gt] = pb
                pairs[(ta, tb)][gt] = pb - pa
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned", flush=True)

    violations = [(key, gt) for key, d in pairs.items()
                  for gt, g in d.items() if g > args.threshold]
    print(f"\nladder pairs: {len(pairs):,} | violation sightings: {len(violations):,}")

    # ---- 1. do prices move at all? -----------------------------------
    print("\n" + "=" * 100)
    print("PRICE UPDATE RATE -- the question the gap cannot answer")
    print("=" * 100)
    moved = frozen = 0
    run_lengths = []
    for ticker, d in px.items():
        ts = sorted(d)
        run = 1
        for i in range(1, len(ts)):
            if ts[i] != ts[i - 1] + 1:          # gap in quoting; end the run
                run_lengths.append(run)
                run = 1
                continue
            if abs(d[ts[i]] - d[ts[i - 1]]) > args.move_eps:
                moved += 1
                run_lengths.append(run)
                run = 1
            else:
                frozen += 1
                run += 1
        run_lengths.append(run)
    tot = moved + frozen
    print(f"  consecutive-snapshot observations: {tot:,}")
    print(f"  price CHANGED:                     {moved:>9,}  {_pct(moved, tot)}")
    print(f"  price UNCHANGED:                   {frozen:>9,}  {_pct(frozen, tot)}")
    print(f"  median run of unchanged price:     {_median(run_lengths):>9.1f} snapshots "
          f"({_median(run_lengths) * 2:.0f} hours)")

    # ---- 2. is survival conditional on NOT repricing? ----------------
    print("\n" + "=" * 100)
    print("DOES THE VIOLATION SURVIVE WHEN THE PRICES ACTUALLY MOVE?")
    print("=" * 100)
    print(f"{'h':>3}  {'both legs frozen':>34}  {'>=1 leg repriced':>34}")
    print(f"{'':>3}  {'n':>8} {'survived':>10} {'rate':>8}  "
          f"{'n':>8} {'survived':>10} {'rate':>8}")
    print("-" * 100)
    for h in args.horizons:
        fz_n = fz_s = mv_n = mv_s = 0
        for (ta, tb), gt in violations:
            nxt = pairs[(ta, tb)].get(gt + h)
            if nxt is None:
                continue
            pa0, pb0 = px[ta].get(gt), px[tb].get(gt)
            pa1, pb1 = px[ta].get(gt + h), px[tb].get(gt + h)
            if None in (pa0, pb0, pa1, pb1):
                continue
            repriced = (abs(pa1 - pa0) > args.move_eps) or (abs(pb1 - pb0) > args.move_eps)
            survived = nxt > args.threshold
            if repriced:
                mv_n += 1
                mv_s += survived
            else:
                fz_n += 1
                fz_s += survived
        print(f"{h:>3}  {fz_n:>8,} {fz_s:>10,} {_pct(fz_s, fz_n):>8}  "
              f"{mv_n:>8,} {mv_s:>10,} {_pct(mv_s, mv_n):>8}")
    print("-" * 100)

    # ---- 3. the decisive number --------------------------------------
    surv_total = surv_frozen = 0
    for (ta, tb), gt in violations:
        nxt = pairs[(ta, tb)].get(gt + 1)
        if nxt is None or nxt <= args.threshold:
            continue
        pa0, pb0 = px[ta].get(gt), px[tb].get(gt)
        pa1, pb1 = px[ta].get(gt + 1), px[tb].get(gt + 1)
        if None in (pa0, pb0, pa1, pb1):
            continue
        surv_total += 1
        if abs(pa1 - pa0) <= args.move_eps and abs(pb1 - pb0) <= args.move_eps:
            surv_frozen += 1

    print(f"\nOf violations surviving one snapshot: {surv_total:,}")
    print(f"  with ZERO price movement on BOTH legs: {surv_frozen:,} "
          f"({_pct(surv_frozen, surv_total)})")
    share = surv_frozen / surv_total if surv_total else 0.0
    print("-" * 100)
    if share > 0.85:
        print("  VERDICT: the persistence is STALENESS. The violations survive because")
        print("  the prints do not update, not because a mispricing is going uncorrected.")
        print("  These are not tradeable at the quoted numbers: a resting order does not")
        print("  fill against a price nobody is posting. The violation COUNT in the")
        print("  results document should be qualified, and the tradeable subset")
        print("  restricted to pairs where both legs repriced recently.")
    elif share > 0.5:
        print("  VERDICT: MIXED. A majority of surviving violations are frozen prints,")
        print("  but a substantial minority survive active repricing. Split the")
        print("  population and re-run the PnL on the repricing subset only -- that")
        print("  subset is the real opportunity set.")
    else:
        print("  VERDICT: the persistence is REAL. Violations survive even while both")
        print("  legs actively reprice, which means the market is repeatedly quoting")
        print("  through a monotonicity breach. That is a much stronger finding than")
        print("  persistence alone, and the maker route is genuinely open.")
    print()
    print("  NOTE: whichever way this lands, the tradeable universe should be defined")
    print("  by RECENT REPRICING, not by the existence of a gap between two last")
    print("  prints. That filter belongs upstream of every PnL number in the results")
    print("  document.")


if __name__ == "__main__":
    main()