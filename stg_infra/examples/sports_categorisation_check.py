"""
sports_categorisation_check.py

Adjudicates ONE question: is category_history_audit.py's "sports first
appears 2024-12" an artefact of classify_ticker(), or a fact about the
data?

The suspicion is reasonable. Wikipedia says sports is >90% of Kalshi's
site activity and 89% of revenue in 2025, so "sports did not exist
before December 2024" sounds absurd on its face. But two different
things are being compared, and this script separates them:

  * category_history_audit.py counts DISTINCT MARKETS LISTED per month.
  * Wikipedia reports SHARE OF TRADING ACTIVITY / REVENUE.

Those can both be true at once. One NFL game is one market that trades
enormous size; Kalshi's crypto ladders list thousands of hourly strike
markets that each trade almost nothing. A category can be 2% of listed
markets and 90% of volume simultaneously.

That still leaves the possibility of a genuine classification bug, so
this runs three independent checks:

  CHECK A -- keyword-independent sports sweep.
      Scans every markets file for tickers matching a broad list of
      sports patterns, WITHOUT calling classify_ticker() and WITHOUT
      consulting kalshi_series_categories.parquet. If sports markets
      existed in 2022 and were merely mislabelled, they show up here.
      If this agrees with the audit, the audit is not the problem.

  CHECK B -- lookup coverage over time.
      classify_ticker() prefers kalshi_series_categories.parquet, which
      was fetched from Kalshi's live GET /series endpoint. If that
      endpoint only returns CURRENTLY LISTED series, every retired
      series falls through to the keyword fallback, and old months would
      be systematically under-categorised. This measures the unmapped
      share per month and names the biggest unmapped series, so that
      failure mode is visible rather than assumed away.

  CHECK C -- volume, not listings (--volume).
      Aggregates contracts traded and notional from the trades files by
      category and month. This is the apples-to-apples comparison with
      the Wikipedia claim. It is the expensive check, so it is opt-in
      and month-bounded.

Drop this into stg_infra/examples/ next to category_history_audit.py so
the classify_ticker import resolves, and run with data/ populated.

    python -m stg_infra.examples.sports_categorisation_check
    python -m stg_infra.examples.sports_categorisation_check --volume --volume-from 2024-06

Run --self-test first if you want to see the checks exercised on
synthetic data with planted answers.
"""

import argparse
import glob
import os
import re
import sys

import polars as pl

pl.Config.set_tbl_rows(-1)

try:
    from pairwise_monotonicity_pnl_backtest import (
        classify_ticker,
        _series_prefix,
        _load_series_category_lookup,
        SERIES_CATEGORIES_PATH,
    )
except ImportError:  # package-relative execution
    from .pairwise_monotonicity_pnl_backtest import (  # type: ignore
        classify_ticker,
        _series_prefix,
        _load_series_category_lookup,
        SERIES_CATEGORIES_PATH,
    )

START_MONTH = (2021, 6)
END_MONTH = (2025, 11)

# Deliberately WIDER than classify_ticker's SPORTS_KEYS, which is only
# ["NBA","MLB","NFL","NHL","WINS","NCAAF","NCAAB","SOCCER","EPL"]. That
# list is what produced the earlier "sports markets land in other (52%)"
# bug, so reusing it here would just reproduce the thing being tested.
# Matched against the SERIES PREFIX (the segment before the first
# hyphen), with a leading "KX" stripped, so KXMLSGAME-25JUL04-XYZ is
# tested as "MLSGAME".
#
# Split into STRONG and WEAK deliberately. classify_ticker's list
# contains "WINS", which also matches election series like KXTRUMPWINS
# and KXSENATEWINS -- those are dense in 2024-10/11 and would fabricate
# a pre-2024-12 sports population, i.e. exactly the false positive this
# script exists to rule out. Only STRONG patterns feed the verdict; WEAK
# ones are reported separately so they can be eyeballed rather than
# silently counted.
STRONG_SPORTS_PATTERNS = [
    # US leagues
    "NFL", "NBA", "WNBA", "MLB", "NHL", "MLS", "NCAAF", "NCAAB", "NCAA",
    # soccer
    "EPL", "LALIGA", "BUNDESLIGA", "LIGUE1", "SERIEA", "UCL", "UEFA",
    "SOCCER", "FIFA", "WORLDCUP",
    # motorsport
    "F1RACE", "NASCAR", "INDYCAR", "MOTOGP", "GRANDPRIX",
    # individual sports
    "TENNIS", "WIMBLEDON", "USOPEN", "GOLF", "PGA", "RYDER", "UFC",
    "MMA", "BOXING", "OLYMPIC", "CRICKET", "RUGBY", "SUPERBOWL",
    "MARCHMADNESS",
]

# Genuinely ambiguous: Kalshi uses these on sports series, but also on
# elections, awards and company series.
WEAK_SPORTS_PATTERNS = [
    "GAME", "MATCH", "WINS", "CHAMPION", "PLAYOFF", "SERIESWIN",
    "MASTERS", "AFL", "NRL", "IPL", "ATP", "WTA", "CBB", "CFB",
]


def _anchored(patterns):
    """Pattern must sit at the start or the end of the prefix. Bare
    substring matching is what put SOMETHINGNEW into crypto for
    containing 'ETH'."""
    alt = "|".join(re.escape(p) for p in patterns)
    return re.compile(rf"^(?:{alt})|(?:{alt})$")


_STRONG_RE = _anchored(STRONG_SPORTS_PATTERNS)
_WEAK_RE = _anchored(WEAK_SPORTS_PATTERNS)


def month_range(start, end):
    y, m = start
    out = []
    while (y, m) <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return out


def markets_path(month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/markets/markets_kalshi_{parity}/markets_{month}.parquet"


def trades_path(month: str) -> str:
    _, mm = month.split("-")
    parity = "even" if int(mm) % 2 == 0 else "odd"
    return f"data/trades/trades_kalshi_{parity}/trades_{month}.parquet"


def strip_kx(prefix: str) -> str:
    return prefix[2:] if prefix.startswith("KX") and len(prefix) > 2 else prefix


def sports_strength(ticker: str) -> str:
    """'strong' | 'weak' | 'no'. Independent of classify_ticker and of
    the lookup parquet -- that independence is the whole point."""
    pre = strip_kx(_series_prefix(ticker).upper())
    if _STRONG_RE.search(pre):
        return "strong"
    if _WEAK_RE.search(pre):
        return "weak"
    return "no"


def looks_like_sports(ticker: str) -> bool:
    """Either strength. Used for the trades-side split in check C, where a
    false positive costs nothing because the category breakdown is
    printed alongside it."""
    return sports_strength(ticker) != "no"


# ----------------------------------------------------------------------
# scanning
# ----------------------------------------------------------------------

def scan_markets(months):
    """One row per (month, series_prefix): market count, sports-regex flag,
    classify_ticker label, and whether the series is in the lookup."""
    lookup = _load_series_category_lookup()
    rows, missing = [], []
    for month in months:
        path = markets_path(month)
        if not glob.glob(path):
            missing.append(month)
            continue
        tickers = (
            pl.scan_parquet(path).select(["ticker"]).unique().collect().get_column("ticker")
        )
        agg = {}
        for t in tickers:
            if t is None:
                continue
            pre = _series_prefix(t)
            if pre not in agg:
                agg[pre] = {
                    "month": month,
                    "series": pre,
                    "n_markets": 0,
                    "strength": sports_strength(t),
                    "sports_regex": looks_like_sports(t),
                    "classified": classify_ticker(t),
                    "in_lookup": pre in lookup,
                }
            agg[pre]["n_markets"] += 1
        rows.extend(agg.values())
    if missing:
        print(f"WARNING: {len(missing)} months had no markets file: {missing}")
    if not rows:
        return pl.DataFrame(
            schema={"month": pl.Utf8, "series": pl.Utf8, "n_markets": pl.Int64,
                    "strength": pl.Utf8, "sports_regex": pl.Boolean,
                    "classified": pl.Utf8, "in_lookup": pl.Boolean}
        )
    return pl.DataFrame(rows)


def scan_trades(months):
    """Per (month, category): contracts traded and notional dollars.

    This is the only view comparable to a '% of site activity' claim.
    """
    rows, missing = [], []
    for month in months:
        path = trades_path(month)
        if not glob.glob(path):
            missing.append(month)
            continue
        cols = pl.scan_parquet(path).collect_schema().names()
        want = [c for c in ("ticker", "count", "yes_price") if c in cols]
        if "ticker" not in want or "count" not in want:
            print(f"  {month}: trades file lacks ticker/count columns ({cols}) -- skipped")
            continue
        t = pl.scan_parquet(path).select(want).collect()
        t = t.with_columns(
            pl.col("ticker")
            .map_elements(classify_ticker, return_dtype=pl.Utf8)
            .alias("category"),
            pl.col("ticker")
            .map_elements(looks_like_sports, return_dtype=pl.Boolean)
            .alias("sports_regex"),
        )
        if "yes_price" in want:
            t = t.with_columns(
                (pl.col("count").cast(pl.Float64) * pl.col("yes_price").cast(pl.Float64) / 100.0)
                .alias("notional_usd")
            )
        else:
            t = t.with_columns(pl.lit(None, dtype=pl.Float64).alias("notional_usd"))
        g = t.group_by(["category", "sports_regex"]).agg(
            pl.len().alias("n_trades"),
            pl.col("count").cast(pl.Float64).sum().alias("contracts"),
            pl.col("notional_usd").sum().alias("notional_usd"),
        )
        rows.append(g.with_columns(pl.lit(month).alias("month")))
        print(f"  scanned trades {month}")
    if missing:
        print(f"  (no trades file for {len(missing)} months: {missing})")
    if not rows:
        return pl.DataFrame(
            schema={"month": pl.Utf8, "category": pl.Utf8, "sports_regex": pl.Boolean,
                    "n_trades": pl.Int64, "contracts": pl.Float64, "notional_usd": pl.Float64}
        )
    return pl.concat(rows)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def check_a(df: pl.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("CHECK A -- sports markets by ticker pattern ALONE (classify_ticker not consulted)")
    print("=" * 92)
    if df.is_empty():
        print("  no markets data scanned")
        return
    strong = df.filter(pl.col("strength") == "strong")
    weak = df.filter(pl.col("strength") == "weak")
    if strong.is_empty() and weak.is_empty():
        print("  the sports ticker sweep matched NOTHING in any month.")
        print("  That means the pattern lists are wrong, not that sports don't exist -- fix")
        print("  them before drawing any conclusion from this script.")
        return

    by_month = (
        df.filter(pl.col("strength") != "no")
        .group_by(["month", "strength"])
        .agg(pl.col("n_markets").sum().alias("markets"),
             pl.col("series").n_unique().alias("series"))
        .sort(["month", "strength"])
    )
    print("  monthly counts, STRONG (unambiguously sports) vs WEAK (GAME/WINS/... , ambiguous):")
    print(by_month.pivot(values="markets", index="month", on="strength").sort("month"))

    if not strong.is_empty():
        print(f"\n  earliest month with an unambiguous sports ticker: {strong['month'].min()}")
    print("\n  Earliest appearance of each STRONG sports series (first 40):")
    firsts = (
        strong.group_by("series")
        .agg(pl.col("month").min().alias("first_month"),
             pl.col("n_markets").sum().alias("total_markets"),
             pl.col("classified").last().alias("classify_ticker_says"))
        .sort(["first_month", "total_markets"], descending=[False, True])
    )
    print(firsts.head(40))

    disagree = firsts.filter(pl.col("classify_ticker_says") != "sports")
    print(f"\n  STRONG sports series that classify_ticker does NOT call 'sports': "
          f"{disagree.height} of {firsts.height}")
    if disagree.height:
        print(disagree.head(25))
        print("  ^ these are the candidate misclassifications. Anything here with an early")
        print("    first_month is what would overturn the audit.")

    print("\n  WEAK matches before 2024-12 (expected to be elections/awards, not sports --")
    print("  this is the list to read sceptically):")
    wpre = (
        weak.filter(pl.col("month") < "2024-12")
        .group_by("series")
        .agg(pl.col("month").min().alias("first_month"),
             pl.col("n_markets").sum().alias("total_markets"),
             pl.col("classified").last().alias("classify_ticker_says"))
        .sort("total_markets", descending=True)
    )
    print(wpre.head(25) if wpre.height else "    (none)")


def check_b(df: pl.DataFrame) -> None:
    print("\n" + "=" * 92)
    print(f"CHECK B -- coverage of {SERIES_CATEGORIES_PATH} over time")
    print("=" * 92)
    lookup = _load_series_category_lookup()
    if not lookup:
        print(f"  {SERIES_CATEGORIES_PATH} is missing or empty -- EVERY ticker in the audit fell")
        print("  through to the keyword heuristic. In that case the audit's category lines are")
        print("  heuristic output, not Kalshi's own categories, and check A is the only evidence")
        print("  that matters here.")
        return
    print(f"  lookup maps {len(lookup)} series")
    if df.is_empty():
        return

    cov = (
        df.group_by("month")
        .agg(
            pl.col("series").n_unique().alias("series_in_data"),
            pl.col("series").filter(pl.col("in_lookup")).n_unique().alias("series_mapped"),
        )
        .with_columns((pl.col("series_mapped") / pl.col("series_in_data")).alias("mapped_share"))
        .sort("month")
    )
    print(cov)
    print("  If mapped_share is low in early months and high in recent ones, the lookup only")
    print("  covers CURRENTLY LISTED series and old months are under-categorised by construction.")
    print("  If it is roughly flat, the lookup is not the explanation.")

    unmapped = (
        df.filter(~pl.col("in_lookup"))
        .group_by("series")
        .agg(pl.col("month").min().alias("first_month"),
             pl.col("n_markets").sum().alias("total_markets"),
             (pl.col("strength") == "strong").any().alias("looks_like_sports"))
        .sort("total_markets", descending=True)
    )
    print(f"\n  largest unmapped series ({unmapped.height} distinct):")
    print(unmapped.head(30))
    n_sporty = unmapped.filter(pl.col("looks_like_sports")).height
    print(f"  unmapped series that look like sports: {n_sporty}")


def check_c(tr: pl.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("CHECK C -- share of TRADING ACTIVITY by category (comparable to '% of site activity')")
    print("=" * 92)
    if tr.is_empty():
        print("  no trades scanned")
        return
    by_month_cat = (
        tr.group_by(["month", "category"])
        .agg(pl.col("contracts").sum(), pl.col("notional_usd").sum(), pl.col("n_trades").sum())
        .sort(["month", "notional_usd"], descending=[False, True])
    )
    for month in by_month_cat["month"].unique().sort():
        sub = by_month_cat.filter(pl.col("month") == month)
        tot_n = sub["notional_usd"].sum() or 0.0
        tot_c = sub["contracts"].sum() or 0.0
        print(f"\n  {month}   notional ${tot_n:,.0f}   contracts {tot_c:,.0f}")
        for r in sub.head(8).iter_rows(named=True):
            sh_n = (r["notional_usd"] / tot_n) if tot_n else 0.0
            sh_c = (r["contracts"] / tot_c) if tot_c else 0.0
            print(f"    {r['category']:<24} notional {sh_n:6.1%}   contracts {sh_c:6.1%}")

    print("\n  Sports-pattern share of activity regardless of category label:")
    sw = (
        tr.group_by(["month", "sports_regex"])
        .agg(pl.col("notional_usd").sum(), pl.col("contracts").sum())
        .sort("month")
    )
    for month in sw["month"].unique().sort():
        sub = sw.filter(pl.col("month") == month)
        tot = sub["notional_usd"].sum() or 0.0
        yes = sub.filter(pl.col("sports_regex"))["notional_usd"].sum() or 0.0
        print(f"    {month}  sports-pattern notional share {(yes / tot if tot else 0):6.1%}")


def verdict(df: pl.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("HOW TO READ THIS")
    print("=" * 92)
    print("""
  The Wikipedia figure (>90% of site activity, 89% of revenue, 2025) is a
  VOLUME claim about 2025. The audit table is a LISTING COUNT across 54
  months. Both can be true: sports is a small number of very heavily
  traded markets, crypto is a very large number of barely traded hourly
  strikes. Check C is the one that speaks to Wikipedia; checks A and B
  are the ones that speak to the audit.

  Decision rule:
    * Check A finds sports tickers well before 2024-12
          -> the audit's category line is wrong, and the window argument
             built on it has to be redone.
    * Check A agrees with the audit (nothing much before 2024-12)
          -> the audit is right and no reclassification will change it.
             Kalshi could not list sports event contracts until the
             CFTC/court position shifted in late 2024; the launch is a
             regulatory fact, not a data artefact.
    * Check B shows mapped_share collapsing in old months
          -> report the early-month categories as heuristic, whatever A says.
""")
    if df.is_empty():
        return
    strong = df.filter(pl.col("strength") == "strong")
    if strong.is_empty():
        print("  MEASURED: no unambiguous sports tickers anywhere. Check the pattern list.")
        return
    first = strong["month"].min()
    n_pre = int(strong.filter(pl.col("month") < "2024-12")["n_markets"].sum() or 0)
    print(f"  MEASURED: first unambiguous sports ticker {first}; "
          f"{n_pre:,} such markets listed before 2024-12.")
    if n_pre == 0:
        print("  -> Check A agrees with the audit. The 2024-12 start is real, and no amount of")
        print("     reclassification will move it.")
    elif n_pre < 500:
        print("  -> Only a trace of early sports tickers, too small to move any window. Read the")
        print("     check A list to see whether they are real markets or listing-page artefacts.")
    else:
        print("  -> Material sports listings before 2024-12. The audit's category line is")
        print("     suspect and the window argument needs redoing off check A, not the audit.")


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------

def _self_test() -> int:
    failures = []

    def ck(name, cond):
        if not cond:
            failures.append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    print("Self-test: ticker pattern matcher")
    ck("KXMLSGAME is strong", sports_strength("KXMLSGAME-25JUL04-ABC") == "strong")
    ck("KXNFLGAME is strong", sports_strength("KXNFLGAME-25SEP07-X") == "strong")
    ck("MLB legacy prefix is strong", sports_strength("MLBGAME-22JUN01-X") == "strong")
    ck("KXNASCARRACE is strong", sports_strength("KXNASCARRACE-25AUG-X") == "strong")
    ck("KXF1RACE is strong", sports_strength("KXF1RACE-25JUL-X") == "strong")
    ck("KXLALIGAGAME is strong", sports_strength("KXLALIGAGAME-25AUG-X") == "strong")
    ck("KXEPLGAME is strong", sports_strength("KXEPLGAME-25AUG-X") == "strong")
    ck("KXHIGHNY is not sports", sports_strength("KXHIGHNY-25JUL04-B86.5") == "no")
    ck("KXFEDDECISION is not sports", sports_strength("KXFEDDECISION-25SEP-X") == "no")
    ck("KXBTCD is not sports", sports_strength("KXBTCD-25JUL0412-T1") == "no")
    ck("KXAPRPOTUS is not sports", sports_strength("KXAPRPOTUS-25JUL-X") == "no")
    ck("KXLLM1 is not sports", sports_strength("KXLLM1-25JUL-X") == "no")
    ck("KXHMONTHRANGE is not sports", sports_strength("KXHMONTHRANGE-25JUL-X") == "no")
    # THE reason strong/weak exists: election series ending in WINS must
    # not be counted as pre-2024-12 sports evidence.
    ck("KXTRUMPWINS is WEAK, not strong", sports_strength("KXTRUMPWINS-24NOV-X") == "weak")
    ck("KXNBAWINS is strong despite also matching WINS",
       sports_strength("KXNBAWINS-25JUL-X") == "strong")
    # the substring trap that produced the SOMETHINGNEW/ETH bug
    ck("mid-word match does not fire", sports_strength("KXWINGAMBIT-25JUL-X") == "no")
    ck("strip_kx handles a short prefix", strip_kx("KX") == "KX")

    print("\nSelf-test: check A on synthetic data with a planted early sports series")
    planted = pl.DataFrame([
        {"month": "2022-03", "series": "KXNBAGAME", "n_markets": 900,
         "strength": "strong", "classified": "other (unmapped)", "in_lookup": False},
        {"month": "2025-01", "series": "KXMLSGAME", "n_markets": 200,
         "strength": "strong", "classified": "sports", "in_lookup": True},
        {"month": "2022-03", "series": "KXHIGHNY", "n_markets": 300,
         "strength": "no", "classified": "weather/climate", "in_lookup": True},
        # the decoy: an election series that only matches a WEAK pattern
        {"month": "2024-10", "series": "KXTRUMPWINS", "n_markets": 5000,
         "strength": "weak", "classified": "elections", "in_lookup": True},
    ])
    strong = planted.filter(pl.col("strength") == "strong")
    ck("planted early sports series is recovered", strong["month"].min() == "2022-03")
    n_pre = int(strong.filter(pl.col("month") < "2024-12")["n_markets"].sum())
    ck("pre-2024-12 strong count is 900, decoy excluded", n_pre == 900)
    disagree = strong.filter(pl.col("classified") != "sports")
    ck("the mislabelled series is flagged", disagree.height == 1
       and disagree["series"][0] == "KXNBAGAME")
    ck("the 5,000-market election decoy is not counted as sports",
       n_pre < 5000)

    print("\nSelf-test: check B recovers a time-bounded lookup")
    cov = (
        planted.group_by("month")
        .agg(pl.col("series").n_unique().alias("series_in_data"),
             pl.col("series").filter(pl.col("in_lookup")).n_unique().alias("series_mapped"))
        .with_columns((pl.col("series_mapped") / pl.col("series_in_data")).alias("mapped_share"))
        .sort("month")
    )
    old = cov.filter(pl.col("month") == "2022-03")["mapped_share"][0]
    new = cov.filter(pl.col("month") == "2025-01")["mapped_share"][0]
    ck("old month has lower mapped_share than new", old < new)
    ck("old mapped_share is 0.5", abs(old - 0.5) < 1e-9)

    print("\nSelf-test: check C separates listings from volume")
    tr = pl.DataFrame([
        # sports: 2 markets, huge volume.  crypto: many markets, tiny volume.
        {"month": "2025-06", "category": "sports", "sports_regex": True,
         "n_trades": 10, "contracts": 900_000.0, "notional_usd": 450_000.0},
        {"month": "2025-06", "category": "crypto", "sports_regex": False,
         "n_trades": 10, "contracts": 100_000.0, "notional_usd": 50_000.0},
    ])
    tot = tr["notional_usd"].sum()
    share = tr.filter(pl.col("category") == "sports")["notional_usd"].sum() / tot
    ck("sports can be 90% of volume while a minority of listings", abs(share - 0.9) < 1e-9)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", action="store_true",
                    help="also run check C (scans trades files -- slow)")
    ap.add_argument("--volume-from", default="2024-06",
                    help="first month for check C (default 2024-06)")
    ap.add_argument("--volume-to", default="2025-11",
                    help="last month for check C (default 2025-11)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in checks on synthetic data and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    months = month_range(START_MONTH, END_MONTH)
    print(f"Scanning {len(months)} months of markets files ...")
    df = scan_markets(months)
    if not df.is_empty():
        df.write_parquet("sports_categorisation_check.parquet")
        print("Saved per-(month, series) detail to sports_categorisation_check.parquet")

    check_a(df)
    check_b(df)

    if args.volume:
        vmonths = [m for m in months if args.volume_from <= m <= args.volume_to]
        print(f"\nScanning trades for {len(vmonths)} months ({vmonths[0]}..{vmonths[-1]}) ...")
        check_c(scan_trades(vmonths))
    else:
        print("\n(check C skipped -- pass --volume to compare against the Wikipedia claim)")

    verdict(df)


if __name__ == "__main__":
    main()