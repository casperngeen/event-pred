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


# --- masking / orientation / init (synthetic, no archive) ---------------
def _synthetic(B=6, L=5, N=7, F=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, N, F, generator=g)
    m = torch.rand(B, L, N, generator=g) > 0.5
    m[:, :, 0] = True                         # at least one active node
    return x, m


@pytest.mark.parametrize("emb", ["learned", "shared_mlp", "hybrid"])
def test_per_step_masking_ignores_padded_cells(emb):
    """Whatever sits in a padded cell must not reach a node that is active."""
    x, m = _synthetic()
    torch.manual_seed(0)
    model = AGCRN(7, 11, hidden=8, d_emb=2, embedding=emb, masking="per_step").eval()
    x2 = x.clone()
    x2[~m] = torch.randn_like(x2[~m]) * 100
    with torch.no_grad():
        d = (model(x, m) - model(x2, m)).squeeze(-1)
    assert d[m[:, -1]].abs().max() < 1e-5


def test_legacy_masking_leaks_padding():
    """Documents the bug the per_step mode fixes: under full-batch training
    every node is active somewhere, so the legacy mask removes nothing."""
    x, m = _synthetic()
    torch.manual_seed(0)
    model = AGCRN(7, 11, hidden=8, d_emb=2, embedding="shared_mlp").eval()
    x2 = x.clone()
    x2[~m] = torch.randn_like(x2[~m]) * 100
    with torch.no_grad():
        d = (model(x, m) - model(x2, m)).squeeze(-1)
    assert d[m[:, -1]].abs().max() > 1e-3


def test_per_step_adjacency_excludes_padded_senders_and_topk():
    x, m = _synthetic()
    torch.manual_seed(0)
    model = AGCRN(7, 11, hidden=8, d_emb=2, embedding="shared_mlp",
                  masking="per_step", topk=2)
    act = m[:, -1]
    A = model.learned_adjacency(x, m)          # batch mean; check per sample below
    x_t = torch.cat([x[:, -1] * act.float()[..., None], act.float()[..., None]], -1)
    As = model._adjacency(model._embed(x_t), act).detach()
    assert (As * (~act)[:, None, :].float()).abs().max() == 0
    assert torch.allclose(As.sum(-1), torch.ones(As.shape[:2]))
    assert A.shape == (7, 7)


def test_stage1_prior_is_oriented_trigger_to_target():
    """[i, j] = i -> j, so the TARGET j must aggregate the trigger i."""
    S = np.zeros((4, 4))
    S[1, 3] = 0.5                              # node 1 drives node 3
    model = AGCRN(4, 3, hidden=4, d_emb=2, adjacency="stage1", stage1_adj=S)
    assert model.A_prior[3, 1] == pytest.approx(1.0)
    assert model.A_prior[1].abs().sum() == 0


def test_zero_head_starts_at_predict_zero():
    x, m = _synthetic()
    model = AGCRN(7, 11, hidden=8, d_emb=2, masking="per_step", zero_head=True)
    with torch.no_grad():
        assert model(x, m).abs().max() == 0


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
    """N is 22, not the 19 this asserted before.

    The three asset-price targets (INXU, INXD, NASDAQ100U) joined the target
    universe in ``3513010``, but the on-disk node panel predated that commit,
    so this test was passing against a stale artifact. Rebuilding the panel
    surfaced it. ``F`` is unchanged: ``n_fresh_legs`` is a diagnostic column,
    not a member of ``nodes.kalshi.FEATURE_ORDER``.
    """
    assert pt.X.shape == (pt.T, pt.N, pt.F)
    assert pt.N == 22 and pt.F == 11
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


def test_signed_adjacency_is_directed_signed_and_masks_padded_senders():
    torch.manual_seed(0)
    m = AGCRN(4, 3, hidden=4, d_emb=2, masking="per_step", embedding="learned",
              adjacency="signed")
    active = torch.tensor([[True, True, False, True]])
    A = m._adjacency(m.E, active)[0]
    assert torch.all(A[:, 2] == 0)                       # padded sender sends nothing
    assert not torch.allclose(A, A.t())                  # directed
    with torch.no_grad():
        m.E_dst.mul_(-1)
    assert (m._adjacency(m.E, active)[0] < 0).any() or (A < 0).any()   # can be negative
