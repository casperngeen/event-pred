"""Stage-2 direction study: leakage guards and behaviour on planted signal.

The failure modes that matter here are silent ones — a fold that trains on its
own test rows, an edge weight fitted on the outcomes it predicts, a
"neighbour" feature that reads the future. Each gets an explicit test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from stg.direction.dataset import finalise, pair_counts
from stg.direction.evaluate import _auc, _balanced, coverage_composition, run_ladder
from stg.direction.folds import walk_forward
from stg.direction.learners import BaseRate, SignRule, design, ladder, neighbour_signal
from stg.direction.structure import fit_structure
from stg.splits import PURGE_DAYS

UTC = dt.timezone.utc


def _panel(n_per_pair: int = 60, rho: float = 0.9, seed: int = 0) -> pl.DataFrame:
    """Two pairs: one with a planted positive edge, one pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    t = dt.datetime(2022, 1, 1, tzinfo=UTC)
    for i in range(n_per_pair):
        for pair, signal in (("A->B/any", rho), ("C->D/any", 0.0)):
            trig, rest = pair.split("->")
            tgt = rest.split("/")[0]
            s = float(rng.normal())
            resp = signal * s + (1 - abs(signal)) * float(rng.normal())
            rows.append(dict(
                trigger=trig, target=tgt, side="any",
                trigger_event=f"{trig}-{i}", target_event=f"{tgt}-{i}",
                t0=t + dt.timedelta(days=3 * i), surprise=s, implied_std=1.0,
                implied_entropy=1.0, coverage=1.0, gap_days=1.0,
                days_to_close=10, p0=50.0, response=resp,
            ))
    return finalise(pl.DataFrame(rows))


def test_finalise_labels_and_drops_flat_responses():
    p = _panel(10)
    assert set(p["y"].unique().to_list()) <= {-1, 1}
    assert (p["z_surprise"] == p["surprise"]).all()   # implied_std = 1
    flat = p.head(1).with_columns(response=pl.lit(0.0))
    assert finalise(flat).height == 0


def test_pair_counts_covers_every_pair():
    p = _panel(20)
    assert set(pair_counts(p)["pair"]) == set(p["pair"].unique())


def test_folds_are_disjoint_purged_and_forward_only():
    p = _panel(80)
    t0 = p["t0"].to_numpy().astype("datetime64[ns]")
    for f in walk_forward(p, n_folds=4, min_train=20):
        assert not (f.train & f.test).any()
        assert t0[f.train].max() < t0[f.test].min()
        gap = (t0[f.test].min() - t0[f.train].max()) / np.timedelta64(1, "D")
        assert gap >= PURGE_DAYS


def test_fit_structure_recovers_the_planted_edge_only():
    st = fit_structure(_panel(120), min_n=10, q=0.10)
    assert st.rho["A->B/any"] > 0.5
    assert abs(st.rho["C->D/any"]) < 0.3
    assert "A->B/any" in st.survives
    assert "C->D/any" not in st.survives


def test_structure_gates_are_nested():
    st = fit_structure(_panel(120))
    pairs = ["A->B/any", "C->D/any"]
    assert st.covered(pairs, "all").all()
    assert st.covered(pairs, "bh").sum() <= st.covered(pairs, "p05").sum()
    with pytest.raises(ValueError):
        st.covered(pairs, "nonsense")


def test_structure_is_fitted_on_train_rows_only():
    """The same pair estimated on two disjoint halves must give two answers;
    if it did not, fit_structure would be reading the whole panel."""
    p = _panel(120)
    half = p.height // 2
    a = fit_structure(p.head(half))
    b = fit_structure(p.tail(half))
    assert a.rho["C->D/any"] != b.rho["C->D/any"]


def test_neighbour_signal_is_strictly_causal():
    p = _panel(20)
    # one target event shared by three rows at increasing t0
    shared = p.head(3).with_columns(
        target_event=pl.lit("SHARED"),
        t0=pl.Series([dt.datetime(2023, 1, d, tzinfo=UTC) for d in (1, 2, 3)]))
    st = fit_structure(p)
    nbr = neighbour_signal(shared, st)
    assert nbr[0] == 0.0                      # first mover sees no neighbours
    assert nbr[1] != 0.0 and nbr[2] != nbr[1]  # each sees only earlier rows


def test_design_columns_are_finite_and_shaped():
    p = _panel(30)
    st = fit_structure(p)
    cols = ("signal", "sign_signal", "z", "abs_z", "dtc", "p0c", "same_rel", "nbr")
    X = design(p, st, cols)
    assert X.shape == (p.height, len(cols))
    assert np.isfinite(X).all()


def test_sign_rule_beats_base_rate_on_the_planted_edge():
    p = _panel(120, rho=0.95)
    summary, folds, _ = run_ladder(p, n_folds=4, n_perm=200)
    bh = summary.filter(pl.col("subset") == "bh")
    sign = bh.filter(pl.col("rung") == "sign_rule")["acc"][0]
    base = bh.filter(pl.col("rung") == "base_rate")["acc"][0]
    assert sign > base + 0.15
    assert bh.filter(pl.col("rung") == "sign_rule")["perm_p"][0] < 0.05
    assert folds["n_test"].sum() > 0


def test_ladder_finds_nothing_in_pure_noise():
    p = _panel(120, rho=0.0, seed=7)
    summary, _, _ = run_ladder(p, n_folds=4, n_perm=200)
    allrows = summary.filter(pl.col("subset") == "all")
    assert allrows.filter(pl.col("rung") == "sign_rule")["perm_p"][0] > 0.05
    assert abs(allrows.filter(pl.col("rung") == "sign_rule")["acc"][0] - 0.5) < 0.08


def test_run_ladder_predicts_each_scored_row_exactly_once():
    p = _panel(80)
    summary, folds, _ = run_ladder(p, n_folds=4, n_perm=100)
    n_all = summary.filter((pl.col("subset") == "all")
                           & (pl.col("rung") == "base_rate"))["n"][0]
    per_rung = folds.filter(pl.col("rung") == "base_rate")["n_test"].sum()
    assert n_all == per_rung
    assert len(ladder()) == summary["rung"].n_unique()


def test_coverage_composition_accounts_for_every_covered_row():
    p = _panel(120)
    summary, _, structures = run_ladder(p, n_folds=4, n_perm=100)
    comp = coverage_composition(p, structures, "bh")
    n_bh = summary.filter((pl.col("subset") == "bh")
                          & (pl.col("rung") == "base_rate"))["n"][0]
    assert comp["n"].sum() == n_bh


def test_auc_of_a_constant_predictor_is_exactly_half():
    y = np.array([1, -1, 1, -1, 1])
    assert _auc(y, np.full(5, 0.7)) == 0.5


def test_balanced_accuracy_penalises_majority_guessing():
    y = np.array([1] * 8 + [-1] * 2)
    yh = np.ones(10, int)
    assert _balanced(y, yh) == 0.5


def test_base_rate_learner_uses_train_majority():
    p = _panel(30)
    st = fit_structure(p)
    m = BaseRate().fit(p, st)
    assert np.isclose(m.p, float((p["y"] > 0).mean()))
    assert np.allclose(m.predict_proba(p, st), m.p)


def test_sign_rule_falls_back_to_base_rate_without_an_edge():
    p = _panel(120)
    st = fit_structure(p)
    m = SignRule(survivors_only=True).fit(p, st)
    noise = p.filter(pl.col("pair") == "C->D/any")
    assert np.allclose(m.predict_proba(noise, st), m.p_base)


# ------------------------------------------------------------ tradability
def _trades(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [dict(ticker=t, created_time=dt.datetime(2024, 1, 1, tzinfo=UTC)
              + dt.timedelta(seconds=s), yes_price=float(p), taker_side=side,
              series="X")
         for t, s, p, side in rows])


def test_effective_spread_reads_the_book_from_taker_direction():
    from stg.direction.tradability import effective_spread
    # yes-taker at 52 is the ask, no-taker at 50 is the bid -> 2c spread
    sp = effective_spread(_trades([("A", 0, 52, "yes"), ("A", 10, 50, "no"),
                                   ("A", 20, 52, "yes"), ("A", 30, 50, "no")]),
                          by="series")
    assert sp["spread_median"][0] == 2.0
    assert sp["n_pairs"][0] == 3


def test_effective_spread_ignores_same_side_and_stale_pairs():
    from stg.direction.tradability import effective_spread
    same_side = _trades([("A", 0, 52, "yes"), ("A", 10, 55, "yes")])
    assert effective_spread(same_side, by="series").is_empty()
    stale = _trades([("A", 0, 52, "yes"), ("A", 9999, 50, "no")])
    assert effective_spread(stale, max_gap_s=60, by="series").is_empty()


def test_effective_spread_collapses_sweep_fills_at_one_timestamp():
    from stg.direction.tradability import effective_spread
    t = dt.datetime(2024, 1, 1, tzinfo=UTC)
    swept = pl.DataFrame([
        dict(ticker="A", created_time=t, yes_price=50.0, taker_side="no", series="X"),
        dict(ticker="A", created_time=t, yes_price=45.0, taker_side="no", series="X"),
        dict(ticker="A", created_time=t + dt.timedelta(seconds=5), yes_price=52.0,
             taker_side="yes", series="X"),
    ])
    sp = effective_spread(swept, by="series")
    assert sp["n_pairs"][0] == 1        # the 45c sweep leg is not a quote
    assert sp["spread_median"][0] == 2.0


def _ledger_panel() -> pl.DataFrame:
    """One row: p0 = 40, first print 43, exit 45, so the estimator sees +5c but
    only +2c was ever available to trade."""
    return pl.DataFrame([dict(
        pair="A->B/any", trigger="A", target="B", side="any",
        t0=dt.datetime(2024, 1, 1, tzinfo=UTC),
        t_entry=dt.datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
        t_exit=dt.datetime(2024, 1, 1, 1, 10, tzinfo=UTC),
        p0=40.0, p_entry=43.0, response=5.0)])


def test_ledger_prices_entry_at_the_first_post_resolution_print():
    from stg.direction.tradability import trade_ledger
    led = trade_ledger(_ledger_panel(), np.array([1]), np.array([True]),
                       spreads={"B": 2.0})
    assert led["gross_p0"][0] == 5.0        # what the estimator measures
    assert led["gross_entry"][0] == 2.0     # what a trader could capture
    assert led["hold_hours"][0] == 1.0
    assert led["entry_lag_min"][0] == 10.0


def test_ledger_net_subtracts_spread_and_both_legs_of_fees():
    from stg.direction.tradability import trade_ledger
    led = trade_ledger(_ledger_panel(), np.array([1]), np.array([True]),
                       spreads={"B": 2.0})
    # fees are ceil-to-the-cent on p(1-p) at entry (43c) and exit (45c)
    assert led["fees"][0] == 4.0
    assert led["net"][0] == led["gross_entry"][0] - 2.0 - 4.0
    assert led["hit"][0] == 1               # hit is judged at the entry price


def test_ledger_direction_flips_the_sign():
    from stg.direction.tradability import trade_ledger
    led = trade_ledger(_ledger_panel(), np.array([-1]), np.array([True]),
                       spreads={"B": 2.0})
    assert led["gross_entry"][0] == -2.0
    assert led["hit"][0] == 0


def test_ledger_summary_reports_the_slippage_decomposition():
    from stg.direction.tradability import ledger_summary, trade_ledger
    led = trade_ledger(_ledger_panel(), np.array([1]), np.array([True]),
                       spreads={"B": 2.0})
    s = ledger_summary(led)
    assert s["n_trades"] == 1
    assert np.isclose(s["entry_slippage"], s["gross_p0"] - s["gross_entry"])
    assert np.isclose(s["net"], s["gross_entry"] - s["spread"] - s["fees"])


def test_representative_tickers_breaks_ties_deterministically():
    """Two identical calls must agree. Ties in trade count used to resolve by
    whatever order group_by emitted, which moved a row in the panel between
    runs — see stg/panel/targets.py::representative_tickers."""
    from stg.panel.targets import representative_tickers
    mk = pl.DataFrame([
        dict(event_ticker="E1", ticker="E1-B", series_raw="CPI",
             close_time=dt.datetime(2024, 3, 1, tzinfo=UTC)),
        dict(event_ticker="E1", ticker="E1-A", series_raw="CPI",
             close_time=dt.datetime(2024, 3, 1, tzinfo=UTC)),
    ])
    trades = pl.DataFrame([
        dict(ticker=t, yes_price=50.0, created_time=dt.datetime(2024, 1, 1, tzinfo=UTC))
        for t in ("E1-A", "E1-B")            # exactly one trade each: a tie
    ]).lazy()
    picks = {representative_tickers("CPI", markets=mk, trades=trades)["ticker"][0]
             for _ in range(5)}
    assert picks == {"E1-A"}                  # lexicographic tiebreak, every time
