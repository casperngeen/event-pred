import numpy as np
import polars as pl
import pytest
import torch

from stg.models import (
    AGCRN, LinearBaseline, RUNGS, build_labels, build_tensor, count_params,
    sequence_windows,
)
from stg.models.agcrn import count_params as cp
from stg.models.capacity import agcrn_param_count
from stg.models.tensors import global_label_sd
from stg.models.train import _fold_cuts, run_linear, PURGE
from stg.splits import OOS_START

from conftest import requires_archive


# --- capacity: formula must match the real module ------------------------
@pytest.mark.parametrize("d,h,emb", [(2, 16, "learned"), (2, 16, "shared_mlp"),
                                     (10, 64, "learned"), (10, 64, "shared_mlp")])
def test_param_formula_matches_model(d, h, emb):
    formula = agcrn_param_count(11, d, h, n_horizons=1, embedding=emb, n_nodes=19)["total"]
    model = AGCRN(19, 11, hidden=h, d_emb=d, n_horizons=1, embedding=emb)
    assert formula == cp(model)


def test_default_config_is_heavily_overparameterised():
    """Bai default (d=10, h=64) against ~24k scalar labels."""
    p = agcrn_param_count(11, 10, 64, n_horizons=3, embedding="shared_mlp")["total"]
    assert p > 100_000                       # >> labels
    assert agcrn_param_count(11, 2, 16, n_horizons=3)["total"] < 10_000


# --- tensors & windows (needs the built panel) --------------------------
@pytest.fixture(scope="module")
def pt(node_panel_df):
    return build_tensor(node_panel_df)


@pytest.fixture(scope="module")
def node_panel_df():
    from pathlib import Path
    p = Path("artifacts/panels/node_panel_event.parquet")
    if not p.exists():
        pytest.skip("node panel not built")
    return pl.read_parquet(p)


@requires_archive
def test_tensor_shapes_and_mask(pt):
    assert pt.X.shape == (pt.T, pt.N, pt.F)
    assert pt.N == 19 and pt.F == 11
    assert pt.mask.shape == (pt.T, pt.N)
    assert 0.2 < pt.mask.mean() < 0.9
    assert (pt.dates < np.datetime64(OOS_START.date())).all()


@requires_archive
def test_labels_only_where_node_present_both_ends(pt):
    Y, lm = build_labels(pt, k=3, kind="belief_z")
    t, i = np.where(lm)
    assert np.isfinite(Y[lm]).all()
    assert pt.mask[t, i].all() and pt.mask[t + 3, i].all()


@requires_archive
def test_sequence_windows_bounds(pt):
    Y, lm = build_labels(pt, k=3, kind="belief_z")
    L = 12
    win = sequence_windows(pt, Y, lm, L=L)
    assert win["Xs"].shape[1:] == (L, pt.N, pt.F)
    assert win["t_idx"].min() >= L - 1
    assert win["t_idx"].max() <= pt.T - 1


@requires_archive
def test_agcrn_forward_pass(pt):
    Y, lm = build_labels(pt, k=3, kind="belief_z")
    win = sequence_windows(pt, Y, lm, L=8)
    m = AGCRN(pt.N, pt.F, hidden=16, d_emb=2, n_horizons=1, embedding="shared_mlp")
    x = torch.tensor(win["Xs"][:4])
    msk = torch.tensor(win["Ms"][:4])
    out = m(x, msk)
    assert out.shape == (4, pt.N, 1)
    A = m.learned_adjacency(x, msk)
    assert A.shape == (pt.N, pt.N)
    assert (A >= 0).all()                     # softmax(ReLU(.)) — non-negative


@requires_archive
def test_walk_forward_has_purged_train_val_gap(pt):
    Y, lm = build_labels(pt, k=3, kind="belief_z")
    win = sequence_windows(pt, Y, lm, L=12)
    dates = win["dates"]
    cuts = _fold_cuts(dates, 8)
    for i in range(8):
        tr = dates < (cuts[i] - PURGE)
        va = (dates >= cuts[i]) & (dates < cuts[i + 1])
        if tr.any() and va.any():
            assert dates[tr].max() + PURGE <= dates[va].min()


@requires_archive
def test_baseline_zero_is_reference_and_neighbours_dont_help(pt):
    """Reproduces the agcrn_complexity T5 shape: graph rungs ≤ zero OOS."""
    Y, lm = build_labels(pt, k=3, kind="belief_z")
    win = sequence_windows(pt, Y, lm, L=12)
    lsd = global_label_sd(Y, lm, "belief_z")
    z = run_linear(lambda: LinearBaseline("zero", pt.nodes), win, lsd)
    assert abs(z["r2_vs_zero"]) < 1e-9
    for rung in ("neighbour_all", "neighbour_stage1"):
        r = run_linear(lambda rung=rung: LinearBaseline(rung, pt.nodes), win, lsd)
        assert r["r2_vs_zero"] < 0.02          # no meaningful OOS gain
