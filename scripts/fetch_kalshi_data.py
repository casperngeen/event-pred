import httpx
import polars as pl
import time
import os
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/historical/trades"
HISTORICAL_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/historical/markets"
MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


def _fetch_single_market(client: httpx.Client, ticker: str) -> dict | None:
    """Try historical/markets first, fall back to markets."""
    errors = {}
    for url in [HISTORICAL_MARKETS_URL, MARKETS_URL]:
        try:
            response = client.get(f"{url}/{ticker}")

            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 5))
                print(f"\nRate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                response = client.get(f"{url}/{ticker}")

            if response.status_code == 404:
                errors[url] = 404
                continue

            response.raise_for_status()
            return response.json().get("market")

        except httpx.HTTPStatusError as e:
            errors[url] = e.response.status_code
            continue

    print(f"\nBoth endpoints failed for {ticker}: {errors}")
    return None


def fetch_markets_for_tickers(
    tickers: list[str],
    output_path: str = "markets_fetched.parquet",
    rate_limit_delay: float = 0.1
) -> pl.DataFrame:
    all_markets = []

    with httpx.Client(timeout=30) as client:
        for i, ticker in enumerate(tickers):
            try:
                market = _fetch_single_market(client, ticker)
                if market:
                    all_markets.append(market)
                else:
                    print(f"\nNot found on either endpoint: {ticker}")
            except Exception as e:
                print(f"\nError fetching {ticker}: {e}")

            print(f"[{i+1}/{len(tickers)}] {ticker}", end="\r")
            time.sleep(rate_limit_delay)

    print(f"\nFetched {len(all_markets)} markets")

    if not all_markets:
        return pl.DataFrame()

    df = normalize_markets(all_markets).with_columns(
        pl.lit(datetime.now()).cast(pl.Datetime("ns")).alias("_fetched_at")
    )
    df.write_parquet(output_path)
    print(f"Saved to {output_path}")
    return df

def fetch_trades_for_ticker(
    client: httpx.Client,
    ticker: str,
    limit: int = 1000
) -> list[dict]:
    all_trades = []
    cursor = None

    while True:
        params = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        try:
            response = client.get(BASE_URL, params=params)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 5))
                print(f"\nRate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            data = response.json()

            trades = data.get("trades", [])
            if not trades:
                break

            all_trades.extend(trades)
            cursor = data.get("cursor")

            if not cursor:
                break

        except httpx.HTTPStatusError as e:
            print(f"\nHTTP error for {ticker}: {e}")
            break
        except Exception as e:
            print(f"\nError fetching {ticker}: {e}")
            break

    return all_trades


def normalize_trades(trades: list[dict]) -> pl.DataFrame:
    """Normalize API response to match your existing trades schema."""
    if not trades:
        return pl.DataFrame()
    
    df = pl.DataFrame(trades)
    
    return df.with_columns([
        # count_fp (string) → count (int)
        pl.col('count_fp').cast(pl.Float64).cast(pl.Int64).alias('count'),
        
        # yes_price_dollars → yes_price in cents (int)
        (pl.col('yes_price_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('yes_price'),
        
        # no_price_dollars → no_price in cents (int)
        (pl.col('no_price_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('no_price'),
        
        # Parse created_time to datetime
        pl.col('created_time').str.to_datetime(
            format='%Y-%m-%dT%H:%M:%S%.fZ',
            time_unit='ns',
            time_zone='UTC'
        )
    ]).select([
        'trade_id',
        'ticker',
        'count',
        'yes_price',
        'no_price',
        'taker_side',
        'created_time'
    ])


def fetch_all_econ_trades(
    tickers: list[str],
    output_path: str = "econ_trades_new.parquet",
    rate_limit_delay: float = 0.1
) -> pl.DataFrame:
    
    checkpoint_path = output_path.replace('.parquet', '_checkpoint.parquet')
    
    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        existing = pl.read_parquet(checkpoint_path)
        fetched_tickers = set(existing['ticker'].unique().to_list())
        tickers_remaining = [t for t in tickers if t not in fetched_tickers]
        all_dfs = [existing]
        print(f"Resuming: {len(fetched_tickers)} done, "
              f"{len(tickers_remaining)} remaining")
    else:
        tickers_remaining = tickers
        all_dfs = []
        print(f"Starting fetch for {len(tickers_remaining)} tickers")

    with httpx.Client(timeout=30) as client:
        for i, ticker in enumerate(tickers_remaining):
            trades = fetch_trades_for_ticker(client, ticker)
            
            if trades:
                df = normalize_trades(trades)
                all_dfs.append(df)

            # Progress
            print(f"[{i+1}/{len(tickers_remaining)}] "
                  f"{ticker}: {len(trades)} trades")

            # Save checkpoint every 50 tickers
            if (i + 1) % 50 == 0 and all_dfs:
                checkpoint = pl.concat(all_dfs)
                checkpoint.write_parquet(checkpoint_path)
                print(f"Checkpoint saved: {len(checkpoint)} total trades")

            time.sleep(rate_limit_delay)

    # Final concat and save
    if not all_dfs:
        print("No trades fetched")
        return pl.DataFrame()
    
    final_df = pl.concat(all_dfs)
    final_df.write_parquet(output_path)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"\nDone.")
    print(f"Total trades:   {len(final_df)}")
    print(f"Unique tickers: {final_df['ticker'].n_unique()}")
    print(f"Date range:     {final_df['created_time'].min()} "
          f"to {final_df['created_time'].max()}")
    
    return final_df

def normalize_markets(markets: list[dict]) -> pl.DataFrame:
    """Normalize API response to match your existing markets schema."""
    df = pl.DataFrame(markets)
    
    return df.with_columns([
        # Convert dollar prices to cents (int)
        (pl.col('last_price_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('last_price'),
        
        (pl.col('yes_ask_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('yes_ask'),
        
        (pl.col('yes_bid_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('yes_bid'),
        
        (pl.col('no_ask_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('no_ask'),
        
        (pl.col('no_bid_dollars').cast(pl.Float64) * 100)
        .round(0).cast(pl.Int64).alias('no_bid'),
        
        # Convert fp fields to int
        pl.col('volume_fp').cast(pl.Float64)
        .round(0).cast(pl.Int64).alias('volume'),
        
        pl.col('volume_24h_fp').cast(pl.Float64)
        .round(0).cast(pl.Int64).alias('volume_24h'),
        
        pl.col('open_interest_fp').cast(pl.Float64)
        .round(0).cast(pl.Int64).alias('open_interest'),
        
        # Parse datetimes
        pl.col('close_time').str.to_datetime(
            format='%Y-%m-%dT%H:%M:%S%.fZ',
            time_unit='ns', time_zone='UTC'
        ),
        pl.col('open_time').str.to_datetime(
            format='%Y-%m-%dT%H:%M:%S%.fZ',
            time_unit='ns', time_zone='UTC'
        ),
        pl.col('created_time').str.to_datetime(
            format='%Y-%m-%dT%H:%M:%S%.fZ',
            time_unit='ns', time_zone='UTC'
        ),
    ]).select([
        'ticker',
        'event_ticker',
        'market_type',
        'title',
        'yes_sub_title',
        'no_sub_title',
        'status',
        'yes_bid',
        'yes_ask',
        'no_bid',
        'no_ask',
        'last_price',
        'volume',
        'volume_24h',
        'open_interest',
        'result',
        'created_time',
        'open_time',
        'close_time',
    ])

def fetch_markets_since(
    since: datetime,
    output_path: str = "markets_latest.parquet",
    limit: int = 1000,
    rate_limit_delay: float = 0.1
) -> pl.DataFrame:
    
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
    all_markets = []
    cursor = None
    page = 0

    with httpx.Client(timeout=30) as client:
        while True:
            params = {
                "limit": limit,
                "status": "finalized",
                "min_close_time": since.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            if cursor:
                params["cursor"] = cursor

            try:
                response = client.get(BASE_URL, params=params)

                if response.status_code == 429:
                    retry_after = int(response.headers.get('retry-after', 5))
                    print(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()

                markets = data.get("markets", [])
                if not markets:
                    break

                all_markets.extend(markets)
                page += 1
                print(f"Page {page}: {len(all_markets)} markets...", end='\r')

                cursor = data.get("cursor")
                if not cursor:
                    break

                time.sleep(rate_limit_delay)

            except Exception as e:
                print(f"Error on page {page}: {e}")
                break

    print(f"\nTotal fetched: {len(all_markets)}")

    if not all_markets:
        print("No markets fetched")
        return pl.DataFrame()

    # Save raw before normalizing
    raw_df = pl.DataFrame(all_markets)
    raw_path = output_path.replace('.parquet', '_raw.parquet')
    raw_df.write_parquet(raw_path)
    print(f"Raw markets saved to {raw_path}")

    # Normalize and save
    df = normalize_markets(all_markets)
    df.write_parquet(output_path)
    print(f"Normalized markets saved to {output_path}")

    return df
