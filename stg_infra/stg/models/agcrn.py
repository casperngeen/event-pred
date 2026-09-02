"""Inductive AGCRN (Bai et al. 2020) for the series-level macro graph.

Adaptations for this setting (research_summary.md §8, update_2026_08.md §2):

* **Node embedding source is a flag.** ``embedding="learned"`` is the canonical
  E ∈ R^{N×d} per fixed identity (feasible here — the union tensor gives every
  series a stable row). ``embedding="shared_mlp"`` computes E_t = MLP(x_t) each
  step, the CA report's inductive adaptation, so the graph does not depend on
  node identities persisting.
* **Frozen-prior variant** (``adjacency="stage1"``): the adaptive
  ``softmax(ReLU(EE^T))`` is replaced by the Stage-1 *signed* adjacency. The
  softmax version cannot represent a negative edge — the U3 dovish sign, the
  WTI→JOBLESSCLAIMS negative edge — so this variant is the direct test of
  whether carrying that sign in the graph (rather than the conv weights) helps.
* **Masking.** Inactive (series, snapshot) cells are zeroed out of the adaptive
  adjacency and excluded from the loss.

Kept deliberately small; N = 19, so full-batch training is milliseconds/epoch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_SUPPORT = 2   # [I, A]


class AVWGCN(nn.Module):
    """Adaptive vertex-wise graph convolution with a node-adaptive weight pool."""

    def __init__(self, c_in: int, c_out: int, d_emb: int):
        super().__init__()
        self.weight_pool = nn.Parameter(torch.empty(d_emb, K_SUPPORT, c_in, c_out))
        self.bias_pool = nn.Parameter(torch.empty(d_emb, c_out))
        nn.init.xavier_normal_(self.weight_pool)
        nn.init.zeros_(self.bias_pool)

    def forward(self, x: torch.Tensor, E: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, N, c_in)   E: (N, d)   A: (N, N) adaptive adjacency (rows sum ~1)
        N = x.size(1)
        eye = torch.eye(N, device=x.device)
        supports = torch.stack([eye, A], dim=0)                 # (K, N, N)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)        # (B, K, N, c_in)
        x_g = x_g.permute(0, 2, 1, 3)                            # (B, N, K, c_in)
        weights = torch.einsum("nd,dkio->nkio", E, self.weight_pool)   # (N, K, c_in, c_out)
        bias = E @ self.bias_pool                                # (N, c_out)
        out = torch.einsum("bnki,nkio->bno", x_g, weights) + bias
        return out


class AGCRNCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int, d_emb: int):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(in_dim + hidden, 2 * hidden, d_emb)
        self.update = AVWGCN(in_dim + hidden, hidden, d_emb)

    def forward(self, x, h, E, A):
        combined = torch.cat([x, h], dim=-1)
        zr = torch.sigmoid(self.gate(combined, E, A))
        z, r = torch.split(zr, self.hidden, dim=-1)
        cand = torch.cat([x, r * h], dim=-1)
        hc = torch.tanh(self.update(cand, E, A))
        return z * h + (1 - z) * hc


def _norm_adj(A: torch.Tensor) -> torch.Tensor:
    return A / A.sum(-1, keepdim=True).clamp_min(1e-8)


class AGCRN(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden: int = 64, d_emb: int = 10,
                 n_horizons: int = 1, embedding: str = "learned",
                 adjacency: str = "adaptive",
                 stage1_adj: np.ndarray | None = None, mlp_hidden: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.d_emb = d_emb
        self.embedding = embedding
        self.adjacency = adjacency

        if embedding == "learned":
            self.E = nn.Parameter(torch.randn(n_nodes, d_emb) * 0.05)
        elif embedding == "shared_mlp":
            self.emb_mlp = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, d_emb))
        else:
            raise ValueError(embedding)

        if adjacency == "stage1":
            if stage1_adj is None:
                raise ValueError("adjacency='stage1' needs stage1_adj")
            A = torch.tensor(stage1_adj, dtype=torch.float32)
            # signed prior: row-normalise by |rho|, keep sign
            denom = A.abs().sum(-1, keepdim=True).clamp_min(1e-8)
            self.register_buffer("A_prior", A / denom)
        elif adjacency != "adaptive":
            raise ValueError(adjacency)

        self.cell = AGCRNCell(in_dim, hidden, d_emb)
        self.head = nn.Linear(hidden, n_horizons)

    def _embed(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.embedding == "learned":
            return self.E
        return self.emb_mlp(x_t).mean(0)          # (N, d) — pool over batch

    def _adjacency(self, E: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if self.adjacency == "stage1":
            A = self.A_prior.clone()
        else:
            A = F.relu(E @ E.t())
            A = _norm_adj(F.softmax(A, dim=-1))
        # mask inactive nodes out of the graph
        m = active.float()
        A = A * m[None, :] * m[:, None]
        if self.adjacency != "stage1":
            A = _norm_adj(A + torch.eye(self.n_nodes, device=A.device) * 1e-6)
        return A

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        # seq: (B, L, N, F)   seq_mask: (B, L, N)
        B, L, N, _ = seq.shape
        h = torch.zeros(B, N, self.hidden, device=seq.device)
        active = seq_mask.any(dim=(0, 1))                       # (N,) ever active
        for t in range(L):
            x_t = seq[:, t]
            E = self._embed(x_t)
            A = self._adjacency(E, active)
            h = self.cell(x_t, h, E, A)
        return self.head(h)                                     # (B, N, n_horizons)

    @torch.no_grad()
    def learned_adjacency(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> np.ndarray:
        """Ã at the final step, averaged over the batch."""
        self.eval()
        B, L, N, _ = seq.shape
        active = seq_mask.any(dim=(0, 1))
        E = self._embed(seq[:, -1])
        return self._adjacency(E, active).cpu().numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
