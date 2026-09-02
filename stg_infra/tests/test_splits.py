import datetime as dt

import polars as pl
import pytest

from stg.splits import OOS_START, assert_no_oos, filter_is


def _frame(dates):
    return pl.DataFrame({"created_time": dates, "x": list(range(len(dates)))})


def test_assert_no_oos_passes_on_is_only():
    f = _frame([dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc)])
    assert_no_oos(f)  # no raise


def test_assert_no_oos_raises_on_planted_oos_row():
    f = _frame([
        dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
    ])
    with pytest.raises(AssertionError):
        assert_no_oos(f)


def test_filter_is_drops_at_wall():
    f = _frame([
        OOS_START - dt.timedelta(days=1),
        OOS_START,
        OOS_START + dt.timedelta(days=1),
    ])
    out = filter_is(f)
    assert out.height == 1
    assert_no_oos(out)
