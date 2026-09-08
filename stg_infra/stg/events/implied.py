"""Shared utilities for recovering implied distributions from Kalshi markets.

Provides ticker parsing, PDF recovery, and implied mean/std computation
for markets that follow the "Above X%" threshold structure (CPI, FED, etc.).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import polars as pl

# --------------------------------------------------------------------------
# Contract classification
#
# Kalshi macro markets come in three structurally different flavours. Feeding
# the wrong one into recover_pdf() silently produces nonsense, so classify
# first and route accordingly.
#
#   THRESHOLD   "Above 0.7%"      ticker -T0.7   cumulative P(X > t) -> recover_pdf
#   BUCKET      "$67 to 67.99"    ticker -B67.5  direct P(a <= X <= b), already a pmf
#   CATEGORICAL "Hike 25bps"      ticker -H25    not numeric at all
#
# Measured on data/markets/ (2026-08): WTI is 7818/9222 BUCKET, PROLLS mixes
# all three, FEDDECISION is entirely CATEGORICAL.
# --------------------------------------------------------------------------

THRESHOLD = "threshold"
BUCKET = "bucket"
CATEGORICAL = "categorical"
UNKNOWN = "unknown"

# Series whose contracts are categorical outcomes, not points on a numeric
# ladder. Excluded explicitly so "correctly skipped" is distinguishable from
# "parser failed" in the output.
CATEGORICAL_SERIES = frozenset({"FEDDECISION"})

# Ordered most-specific-first. Order matters: "-T-25000" must be tried before
# the bare-number pattern, which would otherwise capture "25000" and drop the
# minus sign.
# Digits may carry thousands separators in a handful of legacy tickers
# (e.g. PROLLS-22AUG-T600,000), so allow commas and strip them on parse.
_D = r"[\d,]+\.?\d*"
_TICKER_PATTERNS = [
    (re.compile(rf"-T(-{_D})$"), 1.0),     # T-0.1     -> -0.1
    (re.compile(rf"-TN({_D})$"), -1.0),    # TN0.3     -> -0.3
    (re.compile(rf"-T({_D})$"), 1.0),      # T0.4      ->  0.4
    (re.compile(rf"-N({_D})$"), -1.0),     # N100000   -> -100000  (ADP 25MAR)
    (re.compile(rf"-({_D})$"), 1.0),       # 225000    ->  225000  (claims, ISMPMI)
]

_BUCKET_TICKER_RE = re.compile(r"-B(-?\d+\.?\d*)$")
_CATEGORICAL_TICKER_RE = re.compile(r"-(?:H|C)(?:-?\d+\.?\d*)$")

_NUM = r"(-?[\d,]+\.?\d*)"

# Subtitle patterns. Anchored to the phrasing rather than grabbing the first
# number: "Hike 25bps" contains a number but is categorical, and a naive
# number-grab would silently treat 25 as a threshold.
_SUB_THRESHOLD_PATTERNS = [
    (re.compile(rf"^Above\s+{_NUM}", re.I), "exclusive"),
    (re.compile(rf"^At least\s+{_NUM}", re.I), "inclusive"),
    (re.compile(rf"^{_NUM}\s+or above", re.I), "inclusive"),
    (re.compile(rf"^{_NUM}\s+or higher", re.I), "inclusive"),
    (re.compile(rf"^\${_NUM}\s+or above", re.I), "inclusive"),
]

_SUB_BUCKET_PATTERNS = [
    re.compile(rf"^\$?{_NUM}\s+to\s+{_NUM}", re.I),
]


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def series_of(ticker: str) -> str:
    """Series prefix of a market ticker, with the KX era-prefix stripped."""
    return re.sub(r"^KX", "", ticker).split("-")[0]


def classify_contract(ticker: str, sub_title: Optional[str] = None) -> str:
    """Classify a market as THRESHOLD, BUCKET, CATEGORICAL or UNKNOWN.

    Ticker structure is authoritative where present (-B/-H/-C prefixes are
    unambiguous); the subtitle disambiguates the rest.
    """
    if series_of(ticker) in CATEGORICAL_SERIES:
        return CATEGORICAL
    if _CATEGORICAL_TICKER_RE.search(ticker):
        return CATEGORICAL
    if _BUCKET_TICKER_RE.search(ticker):
        return BUCKET
    if sub_title:
        for pat in _SUB_BUCKET_PATTERNS:
            if pat.search(sub_title.strip()):
                return BUCKET
        for pat, _conv in _SUB_THRESHOLD_PATTERNS:
            if pat.search(sub_title.strip()):
                return THRESHOLD
    for pat, _sign in _TICKER_PATTERNS:
        if pat.search(ticker):
            return THRESHOLD
    return UNKNOWN


def parse_threshold_from_ticker(ticker: str) -> Optional[float]:
    """Extract the numeric threshold encoded in a submarket ticker suffix."""
    for pat, sign in _TICKER_PATTERNS:
        m = pat.search(ticker)
        if m:
            return sign * _to_float(m.group(1))
    return None


def parse_threshold_from_subtitle(
    sub_title: Optional[str],
) -> tuple[Optional[float], Optional[str]]:
    """Extract (threshold, convention) from a yes_sub_title.

    Convention is "exclusive" for `X > t` phrasing ("Above 0.7%") and
    "inclusive" for `X >= t` phrasing ("At least 225000", "300,000 or above").
    Returns (None, None) if the subtitle is not a threshold phrasing.
    """
    if not sub_title:
        return None, None
    s = sub_title.strip()
    for pat, convention in _SUB_THRESHOLD_PATTERNS:
        m = pat.search(s)
        if m:
            return _to_float(m.group(1)), convention
    return None, None


def parse_threshold(ticker: str, sub_title: Optional[str] = None) -> Optional[float]:
    """Extract the numeric threshold for an 'Above X'-style submarket.

    Prefers ``yes_sub_title`` (99% coverage across the core macro universe)
    and falls back to the ticker suffix (90%). Returns None for BUCKET and
    CATEGORICAL contracts — those are not points on a cumulative ladder and
    must not be fed to :func:`recover_pdf`.

    ``sub_title`` is optional for backwards compatibility with callers that
    only have a ticker, but supplying it is strongly preferred: the ticker-only
    path misses JOBLESSCLAIMS and ISMPMI entirely.
    """
    kind = classify_contract(ticker, sub_title)
    if kind in (BUCKET, CATEGORICAL):
        return None

    value, _convention = parse_threshold_from_subtitle(sub_title)
    if value is not None:
        return value
    return parse_threshold_from_ticker(ticker)


def parse_bucket(
    ticker: str, sub_title: Optional[str] = None
) -> Optional[tuple[float, float]]:
    """Extract (low, high) bounds from a range/bucket contract.

    Bucket contracts ("$67 to 67.99") price P(a <= X <= b) directly — they are
    already a probability mass function and need normalisation, not the
    successive differencing :func:`recover_pdf` applies to cumulative ladders.
    """
    if sub_title:
        for pat in _SUB_BUCKET_PATTERNS:
            m = pat.search(sub_title.strip())
            if m:
                return _to_float(m.group(1)), _to_float(m.group(2))
    return None


def infer_tick(thresholds: np.ndarray) -> float:
    """Smallest reporting increment implied by a threshold ladder.

    Used to convert inclusive thresholds to exclusive ones. Integer ladders
    (payrolls, claims, ISMPMI) give 1.0; one-decimal ladders (CPI, U3) give 0.1.
    """
    finite = np.asarray([t for t in thresholds if np.isfinite(t)], dtype=float)
    if finite.size == 0:
        return 1.0
    decimals = 0
    for t in finite:
        s = f"{t:.10f}".rstrip("0").rstrip(".")
        if "." in s:
            decimals = max(decimals, len(s.split(".")[1]))
    return float(10.0 ** (-decimals))


def normalise_to_exclusive(
    thresholds: np.ndarray,
    conventions: "list[Optional[str]]",
    tick: Optional[float] = None,
) -> np.ndarray:
    """Convert a mixed inclusive/exclusive ladder to a uniformly exclusive one.

    ``recover_pdf`` assumes ``above_probs[i] = P(X > thresholds[i])``. An
    inclusive contract prices ``P(X >= t)``, which equals ``P(X > t - tick)``.
    Leaving the two mixed shifts bin edges by one tick — immaterial for
    payrolls (one job) but material for CPI, where strikes sit 0.1pp apart.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if tick is None:
        tick = infer_tick(thresholds)
    out = thresholds.copy()
    for i, conv in enumerate(conventions):
        if conv == "inclusive":
            out[i] -= tick
    return out


def infer_spacing(thresholds: np.ndarray, default: float = 0.1) -> float:
    """Median gap between adjacent strikes in a ladder.

    Spacing is a property of the individual event, not a global constant: CPI
    ladders step by 0.1 (percentage points), payrolls by ~50,000 (jobs),
    jobless claims by ~5,000. It also varies *within* a ladder — payrolls uses
    25,000 gaps mid-ladder and 100,000 in the tails — so the median is the
    appropriate summary.
    """
    t = np.sort(np.asarray([x for x in thresholds if np.isfinite(x)], dtype=float))
    if t.size < 2:
        return default
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return default
    return float(np.median(diffs))

def recover_pdf(
    thresholds: np.ndarray,
    above_probs: np.ndarray,
    tail_width: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a discrete PDF from P(X > threshold) values.

    Parameters
    ----------
    thresholds  : (N,) sorted ascending
    above_probs : (N,) P(value > threshold_i) in [0, 1]
    tail_width  : width of open tail buckets (defaults to median spacing)

    Returns
    -------
    midpoints : (N+1,) bucket midpoints
    probs     : (N+1,) probability mass per bucket, sums to 1

    Buckets:
        0      : value ≤ thresholds[0]              (left tail)
        1..N-1 : thresholds[i-1] < value ≤ thresholds[i]
        N      : value > thresholds[-1]              (right tail)
    """
    n = len(thresholds)
    spacing = float(np.median(np.diff(thresholds))) if n > 1 else 0.1
    if tail_width is None:
        tail_width = spacing

    raw = np.empty(n + 1)
    raw[0] = 1.0 - above_probs[0]
    raw[1:n] = above_probs[:-1] - above_probs[1:]
    raw[n] = above_probs[-1]

    raw = np.clip(raw, 0.0, None)
    total = raw.sum()
    probs = raw / total if total > 0 else np.ones(n + 1) / (n + 1)

    midpoints = np.empty(n + 1)
    midpoints[0] = thresholds[0] - tail_width / 2.0
    midpoints[1:n] = (thresholds[:-1] + thresholds[1:]) / 2.0
    midpoints[n] = thresholds[-1] + tail_width / 2.0

    return midpoints, probs


def pdf_implied_mean(thresholds: np.ndarray, above_probs: np.ndarray) -> float:
    midpoints, probs = recover_pdf(thresholds, above_probs)
    return float(np.dot(midpoints, probs))


def pdf_implied_std(thresholds: np.ndarray, above_probs: np.ndarray) -> float:
    midpoints, probs = recover_pdf(thresholds, above_probs)
    mean = float(np.dot(midpoints, probs))
    return float(np.sqrt(np.dot(probs, (midpoints - mean) ** 2)))


def pdf_implied_stats(
    midpoints: np.ndarray,
    probs: np.ndarray,
) -> dict:
    """Compute a full set of distributional statistics from a discrete PDF.

    Parameters
    ----------
    midpoints : bucket midpoints (already from recover_pdf)
    probs     : probability masses summing to 1 (already from recover_pdf)

    Returns
    -------
    dict with keys: mean, std, skew, kurtosis, entropy, median
        skew     — 3rd standardised moment (positive = right-skewed)
        kurtosis — excess kurtosis (0 for normal; positive = fat tails)
        entropy  — Shannon entropy in nats
        median   — midpoint where cumulative mass first reaches 0.5
    """
    mean = float(np.dot(midpoints, probs))
    variance = float(np.dot(probs, (midpoints - mean) ** 2))
    std = float(np.sqrt(variance))

    if std > 0:
        skew = float(np.dot(probs, (midpoints - mean) ** 3) / std ** 3)
        kurtosis = float(np.dot(probs, (midpoints - mean) ** 4) / std ** 4) - 3.0
    else:
        skew = 0.0
        kurtosis = 0.0

    eps = 1e-12
    entropy = float(-np.sum(probs * np.log(probs + eps)))

    cdf = np.cumsum(probs)
    median_idx = min(int(np.searchsorted(cdf, 0.5)), len(midpoints) - 1)
    median = float(midpoints[median_idx])

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "entropy": entropy,
        "median": median,
    }


def resolved_value(event_markets: pl.DataFrame) -> Optional[float]:
    """Infer the resolved value from submarket yes/no results.

    Handles three cases:
    - Both yes and no: midpoint of (highest yes threshold, lowest no threshold)
    - Only no: resolved value was below the lowest no threshold
    - Only yes: resolved value was above the highest yes threshold
    """
    has_subtitle = "yes_sub_title" in event_markets.columns
    if has_subtitle:
        threshold_expr = (
            pl.struct(["ticker", "yes_sub_title"])
            .map_elements(
                lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
                return_dtype=pl.Float64,
            )
            .alias("threshold")
        )
    else:
        threshold_expr = (
            pl.col("ticker")
            .map_elements(parse_threshold, return_dtype=pl.Float64)
            .alias("threshold")
        )

    rows = (
        event_markets
        .with_columns(threshold_expr)
        .filter(pl.col("threshold").is_not_null())
        .filter(pl.col("result").is_in(["yes", "no"]))
        .sort("threshold")
    )
    if rows.is_empty():
        return None

    # Spacing is per-event, not a global constant. Hardcoding 0.1 was correct
    # for CPI/U3/GDP and wrong by ~6 orders of magnitude for PAYROLLS, whose
    # strikes sit ~50,000 apart.
    spacing = infer_spacing(rows["threshold"].to_numpy())
    yes_rows = rows.filter(pl.col("result") == "yes")
    no_rows  = rows.filter(pl.col("result") == "no")

    if not yes_rows.is_empty() and not no_rows.is_empty():
        hy = yes_rows["threshold"].max()
        ln = no_rows["threshold"].min()
        if hy is None or ln is None:
            return None
        highest_yes, lowest_no = float(hy), float(ln) # type: ignore
        if lowest_no <= highest_yes:
            return None
        return (highest_yes + lowest_no) / 2.0

    if yes_rows.is_empty():
        ln = no_rows["threshold"].min()
        return float(ln) - spacing / 2.0 if ln is not None else None # type: ignore

    hy = yes_rows["threshold"].max()
    return float(hy) + spacing / 2.0 if hy is not None else None # type: ignore

def compute_threshold_series(
    markets: pl.DataFrame,
    trades: pl.DataFrame,
    event_pattern: str,
    series_type: str,
    date_start: "pl.Expr | None" = None,
    date_end: "pl.Expr | None" = None,
) -> pl.DataFrame:
    """Build a daily implied-mean series for any threshold-based event type.

    Filters markets/trades by ``event_pattern``, builds daily OHLCV,
    parses thresholds, and returns the PDF-derived implied mean/std.

    Parameters
    ----------
    markets, trades  : raw Kalshi parquet DataFrames
    event_pattern    : regex matched against ``event_ticker``
    series_type      : label stored in the ``series_type`` column
    date_start/end   : override the module-level DATE_START/DATE_END;
                       pass a ``pl.date(...)`` expression if needed
    """
    # import here to avoid circular dependency (KalshiOHLCV → stg)
    from stg.io.kalshi import KalshiOHLCV

    from stg.events.config import DATE_START as _DS, DATE_END as _DE
    ds = date_start if date_start is not None else _DS
    de = date_end   if date_end   is not None else _DE

    evt_markets = markets.filter(pl.col("event_ticker").str.contains(event_pattern))
    tickers     = evt_markets["ticker"].unique().to_list()
    evt_trades  = trades.filter(pl.col("ticker").is_in(tickers))

    if evt_trades.is_empty():
        return pl.DataFrame()

    # Carry yes_sub_title through so thresholds can be parsed from it rather
    # than from the ticker suffix (99% vs 90% coverage; the ticker-only path
    # misses JOBLESSCLAIMS and ISMPMI entirely).
    sub_lookup = (
        evt_markets.select(["ticker", "yes_sub_title"]).unique(subset=["ticker"])
        if "yes_sub_title" in evt_markets.columns
        else None
    )

    daily = KalshiOHLCV.build_daily(evt_trades, evt_markets).filter(
        (pl.col("date") >= ds) & (pl.col("date") <= de)
    )
    if sub_lookup is not None and "yes_sub_title" not in daily.columns:
        daily = daily.join(sub_lookup, on="ticker", how="left")

    if "yes_sub_title" in daily.columns:
        daily = daily.with_columns(
            pl.struct(["ticker", "yes_sub_title"])
            .map_elements(
                lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
                return_dtype=pl.Float64,
            )
            .alias("threshold"),
            pl.col("yes_sub_title")
            .map_elements(
                lambda s: parse_threshold_from_subtitle(s)[1], return_dtype=pl.Utf8
            )
            .alias("threshold_convention"),
        )
    else:
        daily = daily.with_columns(
            pl.col("ticker")
            .map_elements(parse_threshold, return_dtype=pl.Float64)
            .alias("threshold"),
            pl.lit(None).cast(pl.Utf8).alias("threshold_convention"),
        )

    daily = daily.filter(pl.col("threshold").is_not_null())

    return build_daily_implied_means(daily, evt_markets, series_type=series_type)


def build_daily_implied_means(
    daily_ohlcv: pl.DataFrame,
    markets: pl.DataFrame,
    series_type: str,
) -> pl.DataFrame:
    """Compute daily implied mean/std from a daily OHLCV DataFrame.

    Expects `daily_ohlcv` to already be filtered to the relevant tickers
    and date range, and to have a `threshold` column parsed from the ticker.

    Parameters
    ----------
    daily_ohlcv : output of KalshiOHLCV.build_daily with `threshold` column
    markets     : markets DataFrame for the same events (used for resolved values)
    series_type : label to store in the `series_type` column

    Returns
    -------
    DataFrame with columns:
        event_ticker, date, implied_mean, implied_std,
        implied_skew, implied_kurtosis, implied_entropy, implied_median,
        n_submarkets, series_type, resolved_value
    """
    def _group_mean(df: pl.DataFrame) -> pl.DataFrame:
        thr = df["threshold"].to_numpy()
        prb = df["close"].to_numpy() / 100.0

        # Normalise a mixed inclusive/exclusive ladder before differencing.
        # recover_pdf assumes above_probs[i] = P(X > thresholds[i]); an
        # inclusive contract prices P(X >= t) = P(X > t - tick).
        if "threshold_convention" in df.columns:
            conventions = df["threshold_convention"].to_list()
            if any(c == "inclusive" for c in conventions):
                thr = normalise_to_exclusive(thr, conventions)

        idx = np.argsort(thr)
        thr, prb = thr[idx], prb[idx]

        null_cols = [
            pl.lit(None).cast(pl.Float64).alias("implied_mean"),
            pl.lit(None).cast(pl.Float64).alias("implied_std"),
            pl.lit(None).cast(pl.Float64).alias("implied_skew"),
            pl.lit(None).cast(pl.Float64).alias("implied_kurtosis"),
            pl.lit(None).cast(pl.Float64).alias("implied_entropy"),
            pl.lit(None).cast(pl.Float64).alias("implied_median"),
            pl.lit(len(thr)).cast(pl.Int32).alias("n_submarkets"),
            pl.lit(series_type).alias("series_type"),
        ]
        if len(thr) < 2:
            return df.head(1).select(["event_ticker", "date"]).with_columns(null_cols)

        midpoints, probs = recover_pdf(thr, prb)
        s = pdf_implied_stats(midpoints, probs)
        return df.head(1).select(["event_ticker", "date"]).with_columns([
            pl.lit(s["mean"]).alias("implied_mean"),
            pl.lit(s["std"]).alias("implied_std"),
            pl.lit(s["skew"]).alias("implied_skew"),
            pl.lit(s["kurtosis"]).alias("implied_kurtosis"),
            pl.lit(s["entropy"]).alias("implied_entropy"),
            pl.lit(s["median"]).alias("implied_median"),
            pl.lit(len(thr)).cast(pl.Int32).alias("n_submarkets"),
            pl.lit(series_type).alias("series_type"),
        ])

    result = (
        daily_ohlcv
        .group_by(["event_ticker", "date"])
        .map_groups(_group_mean)
        .sort(["event_ticker", "date"])
    )

    events = result["event_ticker"].unique().to_list()
    resolved = {
        e: resolved_value(markets.filter(pl.col("event_ticker") == e))
        for e in events
    }
    return result.with_columns(
        pl.col("event_ticker")
        .map_elements(lambda e: resolved.get(e), return_dtype=pl.Float64)
        .alias("resolved_value")
    )
