"""The macro AGCRN must reproduce its recorded outputs (see golden_agcrn.py)."""
from __future__ import annotations

import pytest
import torch

from golden_agcrn import GOLDEN, record

TOL = dict(rtol=1e-6, atol=1e-9)     # CPU runs are deterministic; this only absorbs platform noise


@pytest.fixture(scope="module")
def both():
    if not GOLDEN.exists():
        pytest.skip("golden file missing: run stg_infra/tests/golden_agcrn.py")
    return torch.load(GOLDEN), record()


def test_same_configs(both):
    old, new = both
    assert set(old) == set(new)


def test_outputs_adjacency_and_params_unchanged(both):
    old, new = both
    for name in old:
        if name.startswith("_"):
            continue
        assert new[name]["params"] == old[name]["params"], name
        assert torch.allclose(new[name]["out"], old[name]["out"], **TOL), name
        assert torch.allclose(new[name]["hid"], old[name]["hid"], **TOL), name
        assert torch.allclose(new[name]["adj"], old[name]["adj"], **TOL), name


def test_capacity_formula_unchanged(both):
    old, new = both
    assert new["_capacity"] == old["_capacity"]


def test_fit_fold_history_unchanged(both):
    old, new = both
    assert torch.allclose(new["_fit_fold"], old["_fit_fold"], **TOL)
