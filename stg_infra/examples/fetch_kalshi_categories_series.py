"""
fetch_kalshi_series_categories.py

Builds a series_ticker -> category lookup table using Kalshi's OWN category
metadata, instead of guessing category from the ticker string (which is what
classify_ticker() in pairwise_monotonicity_pnl_backtest.py currently does via
hardcoded keyword lists like CRYPTO_KEYS/SPORTS_KEYS/WEATHER_KEYS).

WHY THIS EXISTS: Kalshi's own API docs are explicit that ticker strings should
NOT be parsed to infer category/relationships -- "categories and subcategories
... are not part of the ticker convention" and are "discovery tools only."
The documented correct source is the `category` field returned directly on
series/event objects by the public market-data API. This script pulls that
field for every series Kalshi has, so "other" stops being a silent keyword-
match leftover bucket and becomes Kalshi's actual stated category for every
ticker that didn't match crypto/sports/weather.

ENDPOINT: https://external-api.kalshi.com/trade-api/v2/series
  - No authentication required (public read-only market data).
  - Paginated via `cursor`; this script walks every page with no category
    filter, and just reads the `category` field Kalshi puts on each series
    object directly -- simpler and more complete than filtering per-category.
  - GET /search/tags_by_categories is also called first, purely as a sanity
    print of the category taxonomy Kalshi currently uses (Politics, Economics,
    Sports, Climate and Weather, Crypto, Financials, etc.) -- not required for
    the lookup table itself.

NOTE: this sandbox's network egress does not reach external-api.kalshi.com
(confirmed: CONNECT to external-api.kalshi.com:443 returns 403 through the
proxy here), so this script could not be run/tested end-to-end from where it
was written. The endpoint paths, params, and field names below are taken
directly from Kalshi's published API docs, but you should sanity-check the
first raw response (the script prints it) against what you actually get back
before trusting the full pull -- field names occasionally drift between doc
snapshots and the live API.

OUTPUT: kalshi_series_categories.parquet with columns:
  series_ticker (str), category (str), tags (str, comma-joined)

USAGE: python fetch_kalshi_series_categories.py
"""

import sys
import time

import polars as pl
import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
OUT_PATH = "kalshi_series_categories.parquet"
PAGE_LIMIT = 200
REQUEST_PAUSE_SECONDS = 0.1  # be polite, this is someone else's public API


def get(path: str, **params) -> dict:
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def print_category_taxonomy() -> None:
    try:
        data = get("/search/tags_by_categories")
        cats = data.get("tags_by_categories", data)
        print(f"Kalshi's current category taxonomy ({len(cats)} categories):")
        for cat in sorted(cats):
            print(f"  - {cat}")
        print()
    except Exception as e:
        print(f"NOTE: could not fetch /search/tags_by_categories ({e}); "
              f"continuing without it, it's only a sanity print.\n")


def fetch_all_series() -> pl.DataFrame:
    rows = []
    cursor = None
    page = 0
    while True:
        params = {"limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        data = get("/series", **params)

        series_list = data.get("series", [])
        if page == 0 and series_list:
            print("First raw series object (sanity check the field names below "
                  "match what's used further down this script):")
            print(series_list[0])
            print()

        for s in series_list:
            ticker = s.get("ticker") or s.get("series_ticker")
            category = s.get("category")
            tags = s.get("tags") or []
            if ticker is None:
                print(f"WARNING: series object with no ticker/series_ticker field: {s}")
                continue
            rows.append({
                "series_ticker": ticker,
                "category": category,
                "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
            })

        page += 1
        cursor = data.get("cursor")
        print(f"page {page}: {len(series_list)} series (running total: {len(rows)})")
        if not cursor:
            break
        time.sleep(REQUEST_PAUSE_SECONDS)

    return pl.DataFrame(rows)


def main():
    print_category_taxonomy()

    df = fetch_all_series()
    if df.height == 0:
        print("No series returned -- check the endpoint/params above before trusting this.")
        sys.exit(1)

    print(f"\nTotal series fetched: {df.height}")
    print("Category breakdown (by series count, not by trade/pair volume):")
    print(df.group_by("category").agg(pl.len().alias("n_series")).sort("n_series", descending=True))

    df.write_parquet(OUT_PATH)
    print(f"\nWrote {OUT_PATH}. Use this as the join target in classify_ticker() "
          f"instead of (or as a check against) the CRYPTO_KEYS/SPORTS_KEYS/WEATHER_KEYS "
          f"keyword heuristic.")


if __name__ == "__main__":
    main()