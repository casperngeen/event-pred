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


def test_sports_holdout_unseal_switch(monkeypatch, tmp_path):
    from stg.splits import SPORTS_HOLDOUT_DIR, SPORTS_HOLDOUT_START, SPORTS_UNSEAL_ENV, assert_no_sports_holdout
    from stg.sports.loaders import safe_path
    monkeypatch.setenv("KALSHI_SPORTS_DIR", str(tmp_path))
    late = pl.DataFrame({"close_time": [SPORTS_HOLDOUT_START]})
    # sealed (the default, and any other value of the switch): the holdout folder is refused
    for v in (None, "1", "yes"):
        if v is None:
            monkeypatch.delenv(SPORTS_UNSEAL_ENV, raising=False)
        else:
            monkeypatch.setenv(SPORTS_UNSEAL_ENV, v)
        assert safe_path("trades") == tmp_path / "trades"
        with pytest.raises(PermissionError):
            safe_path(SPORTS_HOLDOUT_DIR, "trades")
        with pytest.raises(AssertionError):
            assert_no_sports_holdout(late)
    # unsealed: the root becomes the holdout folder and nothing outside it can be reached
    monkeypatch.setenv(SPORTS_UNSEAL_ENV, "final-test")
    assert safe_path("trades") == tmp_path / SPORTS_HOLDOUT_DIR / "trades"
    with pytest.raises(PermissionError):
        safe_path("..", "trades")
    assert_no_sports_holdout(late)  # no raise
