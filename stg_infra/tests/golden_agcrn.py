"""Golden outputs of the macro AGCRN, recorded before the sports-model changes.

    venv/bin/python stg_infra/tests/golden_agcrn.py     # (re)record: only on purpose

Every valid configuration is run on fixed synthetic inputs with fixed seeds;
``test_agcrn_golden.py`` checks the current code reproduces them, so adding the
relation-typed event model (reports/sports_research_plan.md §8) cannot move a
macro result. Recorded against commit 4b38838's ``stg/models``.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from stg.models.agcrn import AGCRN, count_params
from stg.models.capacity import agcrn_param_count
from stg.models.train import fit_fold

GOLDEN = Path(__file__).resolve().parent / "golden" / "agcrn_v1.pt"
N, F, B, L = 6, 5, 4, 5


def inputs():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(B, L, N, F, generator=g)
    m = torch.rand(B, L, N, generator=g) > 0.35
    m[:, :, 0] = True
    S = torch.randn(N, N, generator=g).numpy()
    return x, m, S


def configs(S):
    out = {}
    for emb, mask, adj, w in itertools.product(("learned", "shared_mlp", "hybrid"), ("legacy", "per_step"),
                                               ("adaptive", "stage1", "signed"), ("pool", "shared")):
        if mask == "legacy" and emb == "hybrid":
            continue
        if adj == "signed" and (emb != "learned" or mask != "per_step"):
            continue
        kw = dict(embedding=emb, masking=mask, adjacency=adj, weights=w)
        if adj == "stage1":
            kw["stage1_adj"] = S
        out["-".join((emb, mask, adj, w))] = kw
    base = dict(embedding="shared_mlp", masking="per_step")
    out["prior"] = dict(base, prior_adj=S, prior_lambda=0.5)
    out["topk"] = dict(base, topk=2)
    out["zero_head"] = dict(base, zero_head=True)
    out["dropout_eval"] = dict(base, dropout=0.3)
    out["horizons2"] = dict(base, n_horizons=2)
    return out


def record() -> dict:
    x, m, S = inputs()
    rec = {}
    for name, kw in configs(S).items():
        torch.manual_seed(0)
        model = AGCRN(N, F, hidden=8, d_emb=3, **kw).eval()
        seen = {}
        hook = model.head.register_forward_hook(lambda mod, i, o: seen.setdefault("hid", i[0].detach()))
        with torch.no_grad():
            y = model(x, m)
        hook.remove()
        # the hidden state too: a zero head hides everything behind it
        rec[name] = dict(out=y, hid=seen["hid"], adj=torch.tensor(model.learned_adjacency(x, m)),
                         params=count_params(model))
    rec["_capacity"] = {f"{c}-{d}-{h}-{e}": agcrn_param_count(c, d, h, embedding=e, n_nodes=N)["total"]
                        for c, d, h, e in itertools.product((5, 11), (2, 10), (16, 64), ("learned", "shared_mlp"))}
    torch.manual_seed(0)
    y = torch.randn(B, N, generator=torch.Generator().manual_seed(7))
    ym = torch.rand(B, N, generator=torch.Generator().manual_seed(8)) > 0.3
    _, hist = fit_fold(lambda: AGCRN(N, F, hidden=8, d_emb=3, masking="per_step", embedding="shared_mlp"),
                       (x, m, y, ym), (x, m, y, ym), max_epochs=5, patience=10)
    rec["_fit_fold"] = torch.tensor(hist["train"] + hist["es"])
    return rec


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    rec = record()
    torch.save(rec, GOLDEN)
    print(f"recorded {len(rec) - 2} configs + capacity + fit_fold -> {GOLDEN}")
