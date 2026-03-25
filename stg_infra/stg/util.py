from datetime import timedelta
import polars as pl

def parse_duration_td(s: str) -> timedelta:
    """Convert a Polars-style duration string to a timedelta."""
    s = s.strip()
    units = {
        "us": "microseconds", "ms": "milliseconds", "s": "seconds",
        "m": "minutes", "h": "hours", "d": "days", "w": "weeks",
    }
    for suffix in sorted(units, key=len, reverse=True):
        if s.endswith(suffix):
            return timedelta(**{units[suffix]: int(s[: -len(suffix)])})
    raise ValueError(f"Cannot parse duration: {s!r}")


def last_taker_price(trades: pl.DataFrame) -> float:
    """Last taker price (yes_price for yes takers, no_price for no takers)."""
    row = trades.tail(1)
    side = row["taker_side"][0]
    if side == "yes":
        return float(row["yes_price"][0])
    return float(row["no_price"][0])
