#!/usr/bin/env python
"""GPU smoke test: train the library AGCRN for a few steps at the planned sizes.

    python cluster/bench_agcrn.py            # uses cuda, else mps, else cpu

Checks that torch sees the GPU and that ``stg`` imports, and times one
optimiser step at the between-game and in-game sizes of the sports plan. No
data is read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stg_infra"))
from stg.models import AGCRN  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__}  device {device}"
      + (f"  ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})" if device == "cuda" else ""))


def sync() -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench(B: int, L: int, N: int, F: int, hidden: int, reps: int = 5) -> float:
    """Seconds per optimiser step (forward + backward) on one batch."""
    torch.manual_seed(0)
    m = AGCRN(N, F, hidden=hidden, d_emb=6, masking="per_step", embedding="learned",
              weights="shared", zero_head=True).to(device)
    X = torch.randn(B, L, N, F, device=device)
    M = torch.ones(B, L, N, dtype=torch.bool, device=device)
    y = torch.randn(B, N, device=device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for i in range(reps + 1):
        if i == 1:                      # the first step includes kernel start-up
            sync()
            t = time.perf_counter()
        opt.zero_grad()
        loss = ((m(X, M).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
    sync()
    return (time.perf_counter() - t) / reps


for label, args in [("between-game, one MLB season (B=1500, L=8, N=32)", (1500, 8, 32, 10, 16)),
                    ("between-game, 3 leagues pooled (B=4000, L=8, N=32)", (4000, 8, 32, 10, 16)),
                    ("in-game batch (B=256, L=32, N=30)", (256, 32, 30, 12, 32)),
                    ("in-game batch (B=2048, L=32, N=30)", (2048, 32, 30, 12, 32))]:
    print(f"{label:52s} {bench(*args):.3f} s/step")
