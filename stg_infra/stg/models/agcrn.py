"""Inductive AGCRN (Bai et al. 2020) for the series-level macro graph.

Adaptations for this setting (research_summary.md §8, update_2026_08.md §2):

* **Node embedding source is a flag.** ``embedding="learned"`` is the canonical
  E ∈ R^{N×d} per fixed identity (feasible here — the union tensor gives every
  series a stable row). ``embedding="shared_mlp"`` computes E_t = MLP(x_t) each
  step, the CA report's inductive adaptation, so the graph does not depend on
  node identities persisting. ``embedding="hybrid"`` feeds the MLP a learned
  per-series id vector alongside x_t, so a stable series-level relation
  ("WTI moves CPI") is representable without giving up the feature path.
* **Frozen-prior variant** (``adjacency="stage1"``): the adaptive
  ``softmax(ReLU(EE^T))`` is replaced by the Stage-1 *signed* adjacency. The
  softmax version cannot represent a negative edge — the U3 dovish sign, the
  WTI→JOBLESSCLAIMS negative edge — so this variant is the direct test of
  whether carrying that sign in the graph (rather than the conv weights) helps.
* **Signed directed variant** (``adjacency="signed"``): the adjacency is
  ``E_dst E_srcᵀ`` with a second learned embedding and no softmax, so an edge
  can be one-way and negative (``reports/recovery_test.md``). Needs
  ``embedding="learned"`` and ``masking="per_step"``.
* **Soft prior** (``prior_adj`` + ``prior_lambda``): the adaptive logits become
  ``ReLU(EE^T) + λ·|P|`` — the event-study graph as a starting point the model
  can move away from, rather than a replacement.
* **Zero-initialised head** (``zero_head=True``): the model starts exactly at
  predict-zero. With the default init the untrained head emits noise of about
  the target's own spread, and early stopping on a signal-free target often
  keeps an epoch-0 state, so the default init alone costs R² vs zero.
* **Masking.** ``masking="legacy"`` reproduces the September 2026 study
  exactly: a node counts as active if it is active *anywhere in the batch and
  window*, and the shared-MLP embedding is pooled over the batch. Under
  full-batch training every node is active somewhere, so that mask is a no-op
  and padded cells — zero-filled *before* standardisation, so |z| ≈ 10 on
  ``implied_mean`` — flow into both the graph and the GRU
  (``analysis/agcrn_checklist_2026_09/``). ``masking="per_step"`` masks per
  (sample, step): padded inputs are zeroed and a presence bit appended, padded
  nodes are removed from the softmax (they send nothing), and a padded node's
  hidden state is carried forward unchanged.

Orientation: every adjacency passed in uses the Stage-1 convention
``[i, j] = influence of i on j``. Internally the aggregation is ``A_in @ x``
with ``A_in[target, source]``, so priors are transposed on the way in, and
:meth:`AGCRN.learned_adjacency` returns ``A_in`` (row = receiving node).

Kept deliberately small; N ≈ 20, so full-batch training is milliseconds/epoch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_SUPPORT = 2   # [I, A]
MASKING = ("legacy", "per_step")


def _aggregate(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A @ x for a shared (N, N) or per-sample (B, N, N) adjacency."""
    if A.dim() == 2:
        return torch.einsum("nm,bmc->bnc", A, x)
    return torch.bmm(A, x)


class AVWGCN(nn.Module):
    """Adaptive vertex-wise graph convolution.

    ``weights="pool"`` is Bai et al.'s node-adaptive weight pool (per-node
    weights ``E @ W_pool``); ``weights="shared"`` is one weight matrix for every
    node, so E only shapes the adjacency.
    """

    def __init__(self, c_in: int, c_out: int, d_emb: int, weights: str = "pool"):
        super().__init__()
        self.weights = weights
        if weights == "pool":
            self.weight_pool = nn.Parameter(torch.empty(d_emb, K_SUPPORT, c_in, c_out))
            self.bias_pool = nn.Parameter(torch.empty(d_emb, c_out))
            nn.init.xavier_normal_(self.weight_pool)
            nn.init.zeros_(self.bias_pool)
        elif weights == "shared":
            self.weight = nn.Parameter(torch.empty(K_SUPPORT, c_in, c_out))
            self.bias = nn.Parameter(torch.zeros(c_out))
            nn.init.xavier_normal_(self.weight)
        else:
            raise ValueError(weights)

    def forward(self, x: torch.Tensor, E: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, N, c_in)   E: (N, d) or (B, N, d)   A: (N, N) or (B, N, N)
        x_g = torch.stack([x, _aggregate(A, x)], dim=2)          # (B, N, K, c_in)
        if self.weights == "shared":
            return torch.einsum("bnki,kio->bno", x_g, self.weight) + self.bias
        if E.dim() == 2:
            weights = torch.einsum("nd,dkio->nkio", E, self.weight_pool)   # (N, K, c_in, c_out)
            return torch.einsum("bnki,nkio->bno", x_g, weights) + E @ self.bias_pool
        # per-sample E: contract the pool first so (B, N, K, c_in, c_out) is never built
        tmp = torch.einsum("bnki,dkio->bndo", x_g, self.weight_pool)
        return (torch.einsum("bndo,bnd->bno", tmp, E)
                + torch.einsum("bnd,do->bno", E, self.bias_pool))


class AGCRNCell(nn.Module):
    def __init__(self, in_dim: int, hidden: int, d_emb: int, weights: str = "pool"):
        super().__init__()
        self.hidden = hidden
        self.gate = AVWGCN(in_dim + hidden, 2 * hidden, d_emb, weights)
        self.update = AVWGCN(in_dim + hidden, hidden, d_emb, weights)

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
                 stage1_adj: np.ndarray | None = None, mlp_hidden: int = 64,
                 masking: str = "legacy", weights: str = "pool",
                 prior_adj: np.ndarray | None = None, prior_lambda: float = 0.0,
                 topk: int | None = None, dropout: float = 0.0, d_id: int = 4,
                 zero_head: bool = False):
        super().__init__()
        if masking not in MASKING:
            raise ValueError(masking)
        if masking == "legacy" and (topk or prior_lambda or embedding == "hybrid"):
            raise ValueError("topk / prior / hybrid embedding need masking='per_step'")
        self.n_nodes = n_nodes
        self.hidden = hidden
        self.d_emb = d_emb
        self.embedding = embedding
        self.adjacency = adjacency
        self.masking = masking
        self.topk = topk
        self.prior_lambda = prior_lambda
        # per_step appends a presence bit to every input
        cell_in = in_dim + (1 if masking == "per_step" else 0)

        if embedding == "learned":
            self.E = nn.Parameter(torch.randn(n_nodes, d_emb) * 0.05)
        elif embedding in ("shared_mlp", "hybrid"):
            mlp_in = cell_in + (d_id if embedding == "hybrid" else 0)
            if embedding == "hybrid":
                self.node_id = nn.Parameter(torch.randn(n_nodes, d_id) * 0.1)
            self.emb_mlp = nn.Sequential(
                nn.Linear(mlp_in, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, d_emb))
        else:
            raise ValueError(embedding)

        if adjacency == "stage1":
            if stage1_adj is None:
                raise ValueError("adjacency='stage1' needs stage1_adj")
            A = torch.tensor(stage1_adj, dtype=torch.float32).t()   # -> [target, source]
            # signed prior: row-normalise by |rho|, keep sign
            denom = A.abs().sum(-1, keepdim=True).clamp_min(1e-8)
            self.register_buffer("A_prior", A / denom)
        elif adjacency == "signed":
            if embedding != "learned" or masking != "per_step":
                raise ValueError("adjacency='signed' needs embedding='learned', masking='per_step'")
            self.E_dst = nn.Parameter(torch.randn(n_nodes, d_emb) * 0.05)
        elif adjacency != "adaptive":
            raise ValueError(adjacency)

        if prior_lambda:
            if prior_adj is None:
                raise ValueError("prior_lambda needs prior_adj")
            P = torch.tensor(np.abs(prior_adj), dtype=torch.float32).t()
            self.register_buffer("P", P / P.max().clamp_min(1e-8))

        self.cell = AGCRNCell(cell_in, hidden, d_emb, weights)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_horizons)
        if zero_head:
            # start at predict-zero: training has to earn every departure from it
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _embed(self, x_t: torch.Tensor) -> torch.Tensor:
        if self.embedding == "learned":
            return self.E
        if self.embedding == "hybrid":
            ids = self.node_id.expand(x_t.size(0), -1, -1)
            return self.emb_mlp(torch.cat([x_t, ids], dim=-1))  # (B, N, d)
        if self.masking == "legacy":
            return self.emb_mlp(x_t).mean(0)      # (N, d) — pooled over batch
        return self.emb_mlp(x_t)                  # (B, N, d)

    def _adjacency(self, E: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """``active``: (N,) under legacy masking, (B, N) under per_step."""
        if self.masking == "legacy":
            if self.adjacency == "stage1":
                A = self.A_prior.clone()
            else:
                A = F.softmax(F.relu(E @ E.t()), dim=-1)
            m = active.float()
            A = A * m[None, :] * m[:, None]
            if self.adjacency != "stage1":
                A = _norm_adj(A + torch.eye(self.n_nodes, device=A.device) * 1e-6)
            return A

        # per_step: padded nodes send nothing. Receivers are left alone — a
        # padded node's state is frozen in forward(), so its row is unused.
        send = active[:, None, :]                                  # (B, 1, N)
        if self.adjacency == "stage1":
            return self.A_prior[None] * send.float()
        if self.adjacency == "signed":
            return (self.E_dst @ E.t())[None] * send.float()       # [target, source]
        logits = F.relu(E @ E.transpose(-1, -2))                   # (N,N) or (B,N,N)
        if self.prior_lambda:
            logits = logits + self.prior_lambda * self.P
        logits = logits.expand(active.size(0), -1, -1).masked_fill(~send, float("-inf"))
        if self.topk:
            k = min(self.topk, self.n_nodes)
            kth = logits.topk(k, dim=-1).values[..., -1:]          # ties are kept
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        # seq: (B, L, N, F)   seq_mask: (B, L, N)
        B, L, N, _ = seq.shape
        h = torch.zeros(B, N, self.hidden, device=seq.device)
        if self.masking == "legacy":
            active = seq_mask.any(dim=(0, 1))                   # (N,) ever active
        for t in range(L):
            x_t = seq[:, t]
            if self.masking == "per_step":
                active = seq_mask[:, t]                         # (B, N)
                m = active.float()[..., None]
                x_t = torch.cat([x_t * m, m], dim=-1)
            E = self._embed(x_t)
            A = self._adjacency(E, active)
            h_new = self.cell(x_t, h, E, A)
            h = h_new if self.masking == "legacy" else m * h_new + (1 - m) * h
        return self.head(self.drop(h))                          # (B, N, n_horizons)

    @torch.no_grad()
    def learned_adjacency(self, seq: torch.Tensor, seq_mask: torch.Tensor) -> np.ndarray:
        """Ã at the final step, averaged over the batch. Row = receiving node."""
        self.eval()
        x_t = seq[:, -1]
        if self.masking == "legacy":
            active = seq_mask.any(dim=(0, 1))
        else:
            active = seq_mask[:, -1]
            m = active.float()[..., None]
            x_t = torch.cat([x_t * m, m], dim=-1)
        A = self._adjacency(self._embed(x_t), active)
        return (A.mean(0) if A.dim() == 3 else A).cpu().numpy()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
