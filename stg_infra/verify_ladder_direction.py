"""
verify_ladder_direction.py

CONFIRMS THE ONE CONVENTION EVERY LADDER NUMBER DEPENDS ON.

For "value >= T" markets, crossing a HIGHER threshold implies crossing a
lower one, so no-arbitrage requires

    price(T_low)  >=  price(T_high)

and a violation is the reverse. Every gap in this project is computed as

    gap = price[edge_index[1]] - price[edge_index[0]]

which is only a violation measure if edge_index[0] is the LOWER strike.
If the convention is reversed, every violation count, every PnL figure and
every model target has the wrong sign, and nothing downstream is salvageable.

WHY THIS NEEDS CHECKING RATHER THAN ASSUMING. stg/edges/kalshi.py's
KalshiLadderChainEdges docstring states "leg_a's threshold implies leg_b's",
which read literally puts leg_a at the HIGHER strike -- the opposite of what
the code does. The measured violation rate (~18%, against ~82% if reversed)
says the code is right and the sentence is wrong, but inferring a convention
from an aggregate is weaker than reading the strikes themselves.

This script parses the numeric strike out of each ticker and reports, over
the pairs file that actually builds the graph, what fraction have
strike(leg_a) < strike(leg_b).

Read-only. Touches no model and no cache.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

# Kalshi threshold tickers end in a strike marker, e.g.
#   KXBTCD-25SEP19-T64250     (T = threshold)
#   KXHIGHNY-25SEP19-B85.5    (B = bracket/above)
# The strike is the trailing number after the final letter-prefixed field.
_STRIKE = re.compile(r"-([TB])(-?\d+(?:\.\d+)?)$")


def strike_of(ticker: str):
    m = _STRIKE.search(str(ticker))
    return (m.group(1), float(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",
                    default="pairwise_monotonicity_taker_side_results_corrected.parquet")
    args = ap.parse_args()

    p = Path(args.pairs)
    if not p.is_absolute():
        for c in (_THIS_DIR / p, _REPO_ROOT / p, Path.cwd() / p):
            if c.exists():
                p = c
                break
    if not p.exists():
        raise SystemExit(f"pairs file not found: {args.pairs}")

    df = pd.read_parquet(p, columns=["leg_a", "leg_b"])
    print(f"pairs file: {p.name}  rows={len(df):,}\n")

    a_lower = a_higher = equal = unparsed = 0
    mixed_suffix = 0
    fam = Counter()
    for a, b in zip(df["leg_a"].astype(str), df["leg_b"].astype(str)):
        sa, va = strike_of(a)
        sb, vb = strike_of(b)
        if va is None or vb is None:
            unparsed += 1
            continue
        if sa != sb:
            mixed_suffix += 1
        if va < vb:
            a_lower += 1
            fam[a.split("-", 1)[0]] += 1
        elif va > vb:
            a_higher += 1
        else:
            equal += 1

    parsed = a_lower + a_higher + equal
    print("=" * 78)
    print("STRIKE ORDERING WITHIN EACH PAIR")
    print("=" * 78)
    if not parsed:
        print("  No strikes parsed -- the ticker format differs from the assumed")
        print("  '-T<number>' / '-B<number>' suffix. Fix strike_of() before trusting")
        print("  ANY ladder number; the convention is currently unverified.")
        return
    print(f"  strike(leg_a) <  strike(leg_b):  {a_lower:>9,}  {100*a_lower/parsed:5.1f}%"
          "   <- required convention")
    print(f"  strike(leg_a) >  strike(leg_b):  {a_higher:>9,}  {100*a_higher/parsed:5.1f}%")
    print(f"  strike(leg_a) == strike(leg_b):  {equal:>9,}  {100*equal/parsed:5.1f}%")
    print(f"  unparsed tickers:                {unparsed:>9,}")
    print(f"  pairs mixing T and B suffixes:   {mixed_suffix:>9,}  "
          "(a threshold paired with a bracket is not a ladder)")

    print("\n" + "-" * 78)
    share = a_lower / parsed
    if share > 0.99:
        print("  VERDICT: convention CONFIRMED. leg_a is the lower strike, so")
        print("    gap = price[leg_b] - price[leg_a] > 0  IS a monotonicity violation,")
        print("    and every downstream sign in this project is correct.")
    elif share < 0.01:
        print("  VERDICT: convention REVERSED. leg_a is the HIGHER strike. Every gap,")
        print("    violation count, PnL figure and model target in this project has the")
        print("    wrong sign. Flip to gap = price[leg_a] - price[leg_b] and rerun")
        print("    everything -- do not patch results, recompute them.")
    else:
        print(f"  VERDICT: MIXED ({100*share:.1f}% ordered correctly). The pairs file")
        print("    does not use one consistent direction, so a single global gap formula")
        print("    is wrong for some fraction of pairs. Sort each pair by strike before")
        print("    building edges, rather than trusting the stored column order.")
    print("-" * 78)

    if fam:
        print("\n  Top families among correctly-ordered pairs:")
        for name, cnt in fam.most_common(10):
            print(f"    {name:<24} {cnt:>9,}")


if __name__ == "__main__":
    main()