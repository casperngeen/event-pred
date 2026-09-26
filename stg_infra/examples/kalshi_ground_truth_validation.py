"""
kalshi_ground_truth_validation.py

Validates this project's manually-reverse-engineered classification logic
against Kalshi's own authoritative API fields, found while researching
whether a ground-truth source exists for N-way cluster / ladder
identification:

  - Event.mutually_exclusive (bool) -- ground truth for whether an event is
    a genuine single-winner MECE structure. mece_sum_to_one_check.py
    currently infers this empirically (n_yes==1 consistently across >=3
    historical instances), which is a proxy, not a guarantee.
  - Market.strike_type / floor_strike / cap_strike / custom_strike -- ground
    truth for bracket vs. upper_tail vs. lower_tail vs. structured/compound,
    which classify_subtitle() currently infers from free-text yes_sub_title
    parsing -- a method that has already produced two real bugs in this
    project (the KXCITIESWEATHER compound-condition bug and the crypto
    duplicate-phrase false positive).

IMPORTANT CAVEATS BEFORE RUNNING:
  - The historical markets/trades parquet files do NOT contain these fields
    (confirmed: markets schema has no strike_type/floor_strike/cap_strike/
    custom_strike/mve_collection_ticker, and there is no separate events
    table with mutually_exclusive at all). This backfills them LIVE from
    Kalshi's public API for tickers already in your dataset. These are
    static, by-design properties of an event/market -- Kalshi doesn't
    redefine a market's strike type mid-life -- so fetching them today is
    still valid ground truth for historical tickers.
  - Uses ONLY public, unauthenticated GET endpoints. Confirmed via Kalshi's
    docs and third-party SDKs that market/event data endpoints are public;
    only trading/portfolio endpoints require RSA-signed auth. No API key
    needed for this script.
  - Response shapes follow documented conventions (GetEventResponse,
    GetMarketResponse) and one independently-confirmed live field list for
    the list endpoint. The exact wrapping key for the SINGLE-item endpoints
    was not independently verified against a live call -- if results come
    back empty/all-None on the first few rows, uncomment the debug print in
    _get() to see the raw response and adjust EVENT_KEY/MARKET_KEY below.
  - Rate-limited conservatively (10 req/sec, vs. the documented ~30 req/sec
    for public data) to be a good citizen. A few hundred lookups takes well
    under a minute either way.

Needs `requests` (not currently in requirements.txt):
    pip install requests --break-system-packages

Run from the same folder as the other example scripts, after both
mece_sum_to_one_check.py and pairwise_monotonicity_taker_side_check_v2.py
have already produced their results parquets.
"""

import random
import time

import polars as pl
import requests

try:
    from pairwise_monotonicity_taker_side_check_v2 import classify_subtitle
except ImportError:
    from .pairwise_monotonicity_taker_side_check_v2 import classify_subtitle

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
REQUEST_DELAY_SECONDS = 0.1  # ~10 req/sec -- conservative vs. the documented ~30 req/sec for public data

EVENT_KEY = "event"    # adjust if a live call shows a different wrapping key -- see caveats above
MARKET_KEY = "market"  # adjust if a live call shows a different wrapping key -- see caveats above

MECE_RESULTS_PATH = "mece_sum_to_one_results.parquet"
LADDER_RESULTS_PATH = "pairwise_monotonicity_taker_side_results_corrected.parquet"

SAMPLE_SIZE = 200  # per check -- fast, and plenty for a first agreement-rate estimate


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Uncomment on your first run to confirm the response shape:
    # print(data)
    time.sleep(REQUEST_DELAY_SECONDS)
    return data


def validate_mece(sample_size: int = SAMPLE_SIZE) -> None:
    print("=== Validating MECE family-consistency heuristic against Event.mutually_exclusive ===\n")
    results = pl.read_parquet(MECE_RESULTS_PATH)
    event_tickers = results["event_ticker"].unique().to_list()
    sample = random.sample(event_tickers, min(sample_size, len(event_tickers)))
    print(f"Checking {len(sample)} of {len(event_tickers)} distinct event_tickers currently "
          f"trusted as MECE candidates...\n")

    agree, disagree, errors = 0, 0, 0
    disagreements = []
    for i, evt in enumerate(sample):
        try:
            data = _get(f"/events/{evt}")
            event = data.get(EVENT_KEY, data)
            me = event.get("mutually_exclusive")
            if me is True:
                agree += 1
            elif me is False:
                disagree += 1
                disagreements.append(evt)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  WARNING: request failed for {evt}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} checked...")

    checked = agree + disagree
    if checked:
        print(f"\nAgree (mutually_exclusive=True): {agree}/{checked} ({agree/checked:.1%})")
        print(f"Disagree (mutually_exclusive=False): {disagree}/{checked} ({disagree/checked:.1%})")
    print(f"Errors/lookup failures: {errors}/{len(sample)}")
    if disagreements:
        print(f"\nEvents the heuristic trusted as MECE but Kalshi flags as NOT mutually exclusive "
              f"(false positives -- inspect these first):")
        for evt in disagreements[:20]:
            print(f"  {evt}")


def _get_market(ticker: str) -> dict:
    """Kalshi keeps only a 3-month rolling 'live' window
    (docs.kalshi.com/getting_started/historical_data) -- markets settled
    before that cutoff are ONLY available via GET /historical/markets/
    {ticker}, not the live GET /markets/{ticker}. Everything in this
    project's date range (2024-04 onward) is well past that window as of
    today, so this tries live first (in case a ticker is unexpectedly
    recent) and falls back to historical on a 404, rather than assuming
    either way."""
    resp = requests.get(f"{BASE_URL}/markets/{ticker}", timeout=10)
    if resp.status_code == 404:
        resp = requests.get(f"{BASE_URL}/historical/markets/{ticker}", timeout=10)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


def validate_ladder_subtitles(sample_size: int = SAMPLE_SIZE) -> None:
    print("\n=== Validating classify_subtitle() against Market.strike_type/floor_strike/cap_strike ===\n")
    results = pl.read_parquet(LADDER_RESULTS_PATH)
    tickers = list(set(results["leg_a"].to_list()) | set(results["leg_b"].to_list()))
    sample = random.sample(tickers, min(sample_size, len(tickers)))
    print(f"Checking {len(sample)} of {len(tickers)} distinct tickers currently classified "
          f"as valid ladder legs...\n")

    matches, mismatches, errors = 0, 0, 0
    mismatch_examples = []
    for i, ticker in enumerate(sample):
        try:
            data = _get_market(ticker)
            market = data.get(MARKET_KEY, data)
            floor_strike = market.get("floor_strike")
            cap_strike = market.get("cap_strike")
            custom_strike = market.get("custom_strike")
            strike_type = market.get("strike_type")
            sub_title = market.get("yes_sub_title", "")

            local_call = classify_subtitle(sub_title)

            # Ground-truth interpretation: both floor+cap populated = bracket;
            # only one populated = a tail threshold; custom_strike populated
            # = structured/compound, should locally be "unrecognized".
            if custom_strike:
                truth = "unrecognized"
            elif floor_strike is not None and cap_strike is not None:
                truth = "bracket"
            elif floor_strike is not None:
                truth = "upper_tail"
            elif cap_strike is not None:
                truth = "lower_tail"
            else:
                truth = "unrecognized"

            if local_call == truth:
                matches += 1
            else:
                mismatches += 1
                mismatch_examples.append((ticker, sub_title, local_call, truth, strike_type))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  WARNING: request failed for {ticker}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} checked...")

    checked = matches + mismatches
    if checked:
        print(f"\nMatch: {matches}/{checked} ({matches/checked:.1%})")
        print(f"Mismatch: {mismatches}/{checked} ({mismatches/checked:.1%})")
    print(f"Errors/lookup failures: {errors}/{len(sample)}")
    if mismatch_examples:
        print(f"\nMismatches (ticker, sub_title, local classify_subtitle() call, "
              f"Kalshi ground truth, raw strike_type):")
        for row in mismatch_examples[:20]:
            print(f"  {row}")


if __name__ == "__main__":
    validate_mece()
    validate_ladder_subtitles()