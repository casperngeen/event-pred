"""
diagnose_basket_composition.py

WHAT KIND OF MARKET IS THIS MODEL ACTUALLY TRADING?

An earlier phase of this project found that MECE opportunities were
**249 of 267 (93%) weather/climate**, and that only small baskets ever
achieve full-leg coverage: the 6-leg KXHIGH* weather families and the
8-leg KXAPRPOTUS, while every 14+ leg family contributed zero full-basket
snapshots at any candidate-pool size tested. The binding constraint was
liquidity fragmentation, not candidate identification.

That number was computed on the 2025-10/11 window. The STGAT is trained
and evaluated on 2025-05..09. Nothing guarantees the composition is the
same, and three things in the write-up depend on it:

  1. GENERALISATION. If the baskets are overwhelmingly one family, the
     claim is "6-leg daily weather baskets on Kalshi", not "prediction
     markets". Better to say so than to let a reader infer breadth that
     was never tested.

  2. SEASONALITY. Weather markets have seasonal structure. Train May-Jun /
     val Jul-Aug / test Sep is entirely summer; extending to Feb-Nov
     crosses seasons. If accuracy moves, "the model does not generalise"
     and "October weather is not July weather" are different findings and
     need separating.

  3. WHAT THE COST GATE SELECTS. If the profitable subset is one family,
     the +$617 result is a statement about that family.

This script answers it from the cached bundles alone: no model, no
checkpoint, no trades file. It reports the basket population by SERIES --
the ticker prefix, which is a fact rather than a guess -- and rolls series
up into categories only if a category lookup is supplied.

Two populations are reported separately, because they answer different
questions:
  ALL        every complete basket-snapshot: what the model sees.
  OPPORTUNITY  those with |dev| over a threshold: what could be traded,
               and the population the 93% figure referred to.

USAGE
    python diagnose_basket_composition.py --split val
    python diagnose_basket_composition.py --split val --threshold 0.05
    python diagnose_basket_composition.py --self-test
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

from model.chunking import build_split_ranges, chunk_ranges            # noqa: E402
from model.month_store import MonthlyBundleStore, build_month_paths    # noqa: E402
from model.train import true_prices                                    # noqa: E402

PILOT_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
SLOT_LEGS_TOTAL = 3

# Keyword fallback, used only when no category parquet is supplied. These
# are the families this project has actually seen; anything unmatched is
# reported as "other" rather than guessed at, so an unfamiliar family shows
# up as unclassified instead of being silently folded into a known bucket.
_CATEGORY_KEYWORDS = (
    ("weather", ("HIGH", "LOW", "TEMP", "RAIN", "SNOW", "ARCTIC", "CITIESWEATHER")),
    ("crypto", ("BTC", "ETH", "SHIBA", "XRP", "DOGE", "SOL")),
    ("politics", ("POTUS", "APRPOTUS", "ELECT", "SENATE", "HOUSE")),
    ("financials", ("NASDAQ", "SPX", "SP500", "CPI", "FED", "YIELD", "TREASURY")),
    # Football leagues, motorsport and the rest were landing in "other"
    # until a real run showed KXMLSGAME alone was 26% of all baskets. The
    # keyword list is the weak point of this classifier -- prefer
    # --categories with Kalshi's own field, and treat "other" as a prompt
    # to extend this list rather than as a category.
    ("sports", ("MLB", "NBA", "NFL", "EPL", "NHL", "UFC", "MLS", "LALIGA",
                "BUNDESLIGA", "LIGUE1", "SERIEA", "UCL", "F1RACE", "NASCAR",
                "PGA", "ATP", "WTA", "NCAA", "WNBA", "MMA", "BOX")),
    ("entertainment", ("SPOTIFY", "BILLBOARD", "ROTTEN", "OSCAR")),
)


def series_of(ticker: str) -> str:
    """Kalshi tickers are SERIES-EVENT-STRIKE; the series is everything
    before the first hyphen. This is a fact about the string, not an
    inference about the market."""
    return str(ticker).split("-", 1)[0]


def category_of(series: str, lookup: dict | None) -> str:
    """Supplied lookup wins; otherwise match the FAMILY PREFIX, not a
    substring.

    Substring matching is wrong here and the self-test proves it:
    "SOMETHINGNEW" contains "ETH" and would be classified as crypto. Kalshi
    series are KX + FAMILY + qualifiers, so stripping the KX and requiring
    the keyword at the START of what remains is both stricter and closer to
    how the tickers are actually built. Anything unmatched is reported as
    'other' rather than guessed at -- an unfamiliar family should be
    visible, not quietly folded into a known bucket.
    """
    if lookup:
        got = lookup.get(series)
        if got:
            return str(got)
    s = series.upper()
    if s.startswith("KX"):
        s = s[2:]
    for cat, keys in _CATEGORY_KEYWORDS:
        if any(s.startswith(k) for k in keys):
            return cat
    return "other"


def _self_test():
    cases = [
        ("KXHIGHNY-25MAY01-B52.5", "KXHIGHNY", "weather"),
        ("KXHIGHLAX-25JUN02-T70", "KXHIGHLAX", "weather"),
        ("KXAPRPOTUS-25-45", "KXAPRPOTUS", "politics"),
        ("KXBTCD-25MAY01-T100000", "KXBTCD", "crypto"),
        ("KXNASDAQ100U-25-A", "KXNASDAQ100U", "financials"),
        # adversarial: contains "ETH" as a substring but is not crypto.
        # Substring matching classified this as crypto until the prefix
        # rule replaced it.
        ("SOMETHINGNEW-25-X", "SOMETHINGNEW", "other"),
        ("KXETHD-25MAY01-T4000", "KXETHD", "crypto"),
        ("KXARCTICICEMIN-25", "KXARCTICICEMIN", "weather"),
        ("KXCITIESWEATHER-25-NY", "KXCITIESWEATHER", "weather"),
    ]
    ok = bad = 0
    print("SELF-TEST  (series extraction and category fallback)")
    for tk, want_s, want_c in cases:
        gs, gc = series_of(tk), category_of(series_of(tk), None)
        good = (gs == want_s and gc == want_c)
        print(f"  {'PASS' if good else 'FAIL'}  {tk:<28} -> {gs:<14} {gc}")
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
    # a supplied lookup must win over the keyword guess
    got = category_of("KXHIGHNY", {"KXHIGHNY": "Climate and Weather"})
    good = got == "Climate and Weather"
    print(f"  {'PASS' if good else 'FAIL'}  supplied lookup overrides keyword -> {got}")
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
    print(f"\n{ok}/{ok + bad} passed")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--months", nargs="+", default=None)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--chunk-len", type=int, default=168)
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="|dev| in dollars above which a basket counts as an "
                         "OPPORTUNITY. The 93% figure used 0.05.")
    ap.add_argument("--categories", default=None,
                    help="optional parquet mapping a series ticker to a "
                         "category, e.g. kalshi_series_categories.parquet. "
                         "Without it, a keyword fallback is used and anything "
                         "unmatched is reported as 'other'.")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if args.self_test:
        _self_test()

    lookup = None
    if args.categories:
        import polars as pl
        cp = Path(args.categories)
        if not cp.is_absolute():
            for c in (_THIS_DIR / cp, _REPO_ROOT / cp, Path.cwd() / cp):
                if c.exists():
                    cp = c
                    break
        if not cp.exists():
            raise SystemExit(f"categories file not found: {args.categories}")
        df = pl.read_parquet(cp)
        cols = df.columns
        key = next((c for c in ("series_ticker", "series", "ticker") if c in cols), None)
        val = next((c for c in ("category", "cat") if c in cols), None)
        if not key or not val:
            raise SystemExit(f"{cp} needs a series and a category column; has {cols}")
        lookup = dict(zip(df[key].to_list(), df[val].to_list()))
        print(f"category lookup: {len(lookup):,} series from {cp.name}")

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = _REPO_ROOT / cache
    store = MonthlyBundleStore(build_month_paths(args.months or PILOT_MONTHS, cache),
                               verbose=False)
    ranges = build_split_ranges(store.timestamps)
    chunks = [c for c in chunk_ranges(ranges, chunk_len=args.chunk_len, min_chunk_len=4)
              if c.split == args.split]
    if not chunks:
        raise SystemExit(f"no '{args.split}' chunks")

    # series -> stats, for both populations
    stats = {"all": defaultdict(lambda: {"n": 0, "legs": [], "dev": []}),
             "opp": defaultdict(lambda: {"n": 0, "legs": [], "dev": []})}
    n_incomplete = 0

    for ci, c in enumerate(chunks):
        ct = store.materialize_chunk(c)
        features, mask = ct["features"], ct["mask"]
        adj = ct["adjacency_by_type"].get("mece_leg_to_basket")
        node_ids = ct.get("node_ids")
        if adj is None:
            del ct
            continue
        prices = true_prices(features)
        T = features.shape[0]
        for t in range(T):
            e = adj[t]
            if e.edge_index.numel() == 0:
                continue
            leg_g, hub_g = e.edge_index[0], e.edge_index[1]
            for hub in torch.unique(hub_g).tolist():
                legs = leg_g[(hub_g == hub).nonzero(as_tuple=True)[0]]
                if legs.numel() == 0:
                    continue
                total = float(features[t, hub, SLOT_LEGS_TOTAL])
                seen = mask[t, legs]
                # Only complete baskets: a partial sum is below 1 for a
                # trivial reason and is not a dislocation.
                if total <= 0 or int(seen.sum()) < int(round(total)):
                    n_incomplete += 1
                    continue
                L = legs[seen]
                idx = L.tolist()
                tk = node_ids[idx[0]] if node_ids else str(idx[0])
                ser = series_of(tk)
                dev = abs(float(prices[t, L].sum()) - 1.0)
                for pop in ("all",) + (("opp",) if dev > args.threshold else ()):
                    d = stats[pop][ser]
                    d["n"] += 1
                    d["legs"].append(int(L.numel()))
                    d["dev"].append(dev)
        del ct
        print(f"  chunk {ci + 1}/{len(chunks)} scanned", flush=True)

    if not stats["all"]:
        raise SystemExit("no complete MECE baskets on this split")

    def report(pop, title):
        rows = sorted(stats[pop].items(), key=lambda kv: -kv[1]["n"])
        total = sum(v["n"] for _, v in rows)
        if total == 0:
            print(f"\n  {title}: none")
            return
        print("\n" + "=" * 96)
        print(f"{title}   {total:,} basket-snapshots, {len(rows)} distinct series")
        print("=" * 96)
        print(f"  {'series':<22}{'category':<22}{'n':>10}{'share':>9}"
              f"{'med legs':>10}{'med |dev|':>11}")
        print("  " + "-" * 82)
        for ser, v in rows[:args.top]:
            legs = torch.tensor(v["legs"], dtype=torch.float)
            dev = torch.tensor(v["dev"], dtype=torch.float)
            print(f"  {ser[:21]:<22}{category_of(ser, lookup)[:21]:<22}{v['n']:>10,}"
                  f"{100.0 * v['n'] / total:>8.1f}%{float(legs.median()):>10.0f}"
                  f"{100 * float(dev.median()):>10.1f}c")
        if len(rows) > args.top:
            rest = sum(v["n"] for _, v in rows[args.top:])
            print(f"  {'... ' + str(len(rows) - args.top) + ' more series':<44}"
                  f"{rest:>10,}{100.0 * rest / total:>8.1f}%")

        # category rollup
        cat = defaultdict(int)
        for ser, v in rows:
            cat[category_of(ser, lookup)] += v["n"]
        print("\n  BY CATEGORY")
        for c, n in sorted(cat.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<26}{n:>10,}{100.0 * n / total:>8.1f}%")
        if not lookup and cat.get("other", 0) > 0.1 * total:
            unk = sorted({s_ for s_, _ in rows if category_of(s_, lookup) == "other"})
            print(f"    ^ 'other' is over 10% of this population. The keyword")
            print(f"      fallback is guessing; these series are unclassified:")
            print(f"      {', '.join(unk[:12])}" + (" ..." if len(unk) > 12 else ""))
            print(f"      Extend _CATEGORY_KEYWORDS, or pass --categories.")

        top1 = rows[0][1]["n"] / total
        top3 = sum(v["n"] for _, v in rows[:3]) / total
        print(f"\n  concentration: top series {100 * top1:.1f}%,  "
              f"top 3 {100 * top3:.1f}%,  distinct series {len(rows)}")

    report("all", f"ALL COMPLETE BASKETS -- {args.split.upper()} split")
    report("opp", f"OPPORTUNITIES (|dev| > {100 * args.threshold:.0f}c) -- "
                  f"{args.split.upper()} split")

    print(f"\n  incomplete basket-snapshots skipped: {n_incomplete:,}")
    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  If one category is most of the OPPORTUNITY population, the trading")
    print("  results are a statement about that category, and the thesis should")
    print("  scope its claim accordingly rather than say 'prediction markets'.")
    print()
    print("  Compare against the earlier 2025-10/11 finding of 249/267 (93%)")
    print("  weather. A different mix here is not an error -- it is a fact about")
    print("  market growth between the two windows, and worth reporting as one.")
    print()
    print("  If the population is concentrated in weather, seasonality becomes a")
    print("  first-order concern for the train/val/test split: a summer-only")
    print("  split cannot distinguish 'does not generalise' from 'a different")
    print("  season is a different market'.")


if __name__ == "__main__":
    main()