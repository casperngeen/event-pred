"""
model/temporal_attention.py

Temporal attention layer for the STGAT: the second half of the DySAT-style
backbone. The spatial attention layer (spatial_attention.py) handles
WITHIN-snapshot relationships; this layer lets each node attend over its
OWN history ACROSS snapshots, independently of every other node.

CAUSAL BY DESIGN, not an afterthought: a node at time t may only attend to
snapshots <= t, never the future. Letting a representation at time t see
future snapshots would be a lookahead-bias leak -- exactly the failure
mode this project's shared train/val/test split (data_windows.py) was
built to prevent at the data level. Getting this wrong here would
reintroduce the same problem one layer up, silently, since a forward pass
with a causality bug still produces correctly-shaped, non-NaN output --
it just secretly cheats. This is the first thing verified in testing, not
assumed.

Per-node masking: each node has its own observed/absent timestep pattern
(a ticker may only be active in some of the T snapshots) -- attention
must only consider a node's OWN genuinely-observed past timesteps, never
padded/absent ones, and a node with no observed history yet has nothing
valid to attend to at all (handled explicitly, not left to produce NaN
from an all-masked softmax).

SCALE LIMIT ON T, BY DESIGN, NOT AN OVERSIGHT: this layer computes a dense
(N, H, T, T) attention tensor. That is fine for a bounded training chunk
(e.g. 1-2 weeks -- a few hundred snapshots at 2h resolution), but a full
multi-month range (~7200 snapshots at 2h over ~20 months) would need a
(N, H, T, T) tensor with tens of billions of elements -- infeasible
regardless of max_len. Raising max_len alone would just trade a fast,
legible IndexError for a slow OOM kill further down the line (the same
failure mode already hit once in spatial_attention.py's original dense
implementation). The real fix is temporal chunking, done by the CALLER
(see model/chunking.py + model/train.py): the training loop slices the
full timeline into bounded windows and calls this layer once per window,
so T here is always a chunk length, never the full training range.
forward() below turns a chunk larger than max_len into an explicit,
actionable error rather than a confusing crash inside pos_embedding.

SECOND SCALE LIMIT, ON N -- FOUND LATER, THE HARD WAY: chunking T alone
was NOT enough. chunking.py's slice_bundle_chunk() deliberately leaves N
as the bundle's full GLOBAL padded node universe (every node that was
EVER active anywhere across the whole build range), reasoning at the
time that "spatial_attention.py and the output heads already restrict to
each snapshot's actually-active nodes internally, so keeping N global
here costs nothing extra" -- true for THOSE layers (they loop per
snapshot and index down to that snapshot's active set), but NOT true
here: forward() below used to run its (N, H, T, T) computation over
EVERY one of the N global nodes unconditionally, including the vast
majority that are all-zero padding for the entire chunk. Confirmed
directly on real data, not assumed: a single real month of Kalshi data
produced N=14,983 (every ticker that traded at ANY point that month,
union across 310 snapshots) against only ~3.5% mask coverage -- the
`scores` tensor alone at chunk_len=168 is 14983 * 4 heads * 168 * 168 *
4 bytes ~= 6.8GB, and several more same-sized tensors (masked scores,
softmax output, the where()'d output) are live simultaneously, all on
CPU RAM with no GPU to offload to, well before autograd's saved-tensors
overhead for backward() is even counted. This is exactly what hung a
real training run for the user of this project after graph-building and
chunking had ALREADY completed successfully -- the third occurrence of
this exact "dense tensor sized to the GLOBAL node count instead of the
per-chunk ACTIVE node count" failure pattern in this project (spatial
attention's original dense implementation, then the bridge's adjacency
export, now this). "Fixed" the same way both previous instances were:
restrict to the nodes that are actually active in this chunk (here:
active in AT LEAST ONE of the chunk's T snapshots, since a node's own
attention sequence spans the WHOLE chunk, unlike spatial attention which
only ever needs one snapshot's active set at a time) before running any
(N, H, T, T)-shaped computation, then scatter the result back into a
residual base at those same node indices -- a node never active anywhere
in the chunk simply keeps its (all-zero, padding) input unchanged, the
same convention TwoChannelSpatialAttention already uses for isolated
nodes.

THIRD SCALE LIMIT, ON "ACTIVE-ANYWHERE-IN-THE-CHUNK" N -- the above fix
was verified against a synthetic scale test with ~600 truly-active nodes
out of N=14,983, which passed in ~1s. It was NOT enough: a real one-month
run's instrumented log (added to model/train.py and spatial_attention.py
while chasing this exact hang) showed graph-building, chunking, the
projection layer, and TwoChannelSpatialAttention ALL completing fast
(spatial's own per-snapshot active counts stayed in the 200-900 range,
matching the graph's real nodes_min=186/nodes_max=842 stats) -- then
nothing. No "[backbone] temporal done" line ever printed. This layer was
still the hang, just not for the reason the previous fix addressed.

The real number: "active anywhere in the chunk" is a union across ALL
168 snapshots in a ~2-week window, not a single snapshot's count. Kalshi
markets are short-lived (many expire within hours to a few days), so a
2-week window sees a large fraction of a MONTH's worth of distinct
tickers churn through it -- plausibly several thousand, not the ~600 the
synthetic test assumed. At n_active in the thousands, the exact same
(n_active, H, T, T) blowup this layer already fixed once reappears,
just under a bigger n_active than was tested. Guessing a "safe enough"
n_active and re-verifying at that number would only move the crash
threshold, not remove it -- the same mistake as raising max_len instead
of chunking T in the first place.

THE ACTUAL FIX: batch over active nodes. Every node's temporal attention
is entirely independent of every OTHER node's -- there is no cross-node
term anywhere in this layer (unlike spatial attention, where nodes
genuinely interact via edges). That independence means splitting
active_node_idx into fixed-size batches and running this layer's exact
same computation once per batch is mathematically IDENTICAL to running
it on all active nodes at once (same q/k/v projections, same causal
masking, same softmax, applied to a subset of rows -- nothing about one
node's output depends on which other nodes happen to be in its batch).
This is also not a novel trick: it's the same node-mini-batching DySAT
and later scalable-STGNN work (e.g. SGP) use to keep a structurally
identical attention computation within a fixed memory budget regardless
of how large the active node set grows -- see the batch_size=256
mini-batching DySAT's own paper describes for exactly this reason.
Peak memory becomes O(node_batch_size * H * T * T), a constant
independent of true n_active -- BUT ONLY UNDER torch.no_grad(). See the
next section, which is the part that actually mattered.

FOURTH SCALE LIMIT -- AUTOGRAD RETENTION, THE ONE THAT ACTUALLY BIT:
node-batching alone did NOT fix the real crash, and the reason is a
verification mistake worth recording so it isn't repeated. The scale
test that "proved" batching worked ran the forward pass inside
torch.no_grad(). Under no_grad, each batch's (b, H, T, T) intermediates
are freed as soon as that batch's loop iteration ends, so peak memory is
genuinely O(one batch) and the test passed at n_active up to 12,000.

Real TRAINING does not run under no_grad. It needs backward(), so every
batch's intermediates (the masked scores, the softmax output, the
where()'d attention, the q/k/v projections) are RETAINED in the autograd
graph until backward() runs. Batching therefore DEFERS the memory rather
than BOUNDING it: total retained memory is O(n_batches * batch_size * H
* T * T) = O(n_active * H * T * T), exactly what it was before batching.
Measured directly, with grad enabled, at real T=168/H=4/batch=512:
~595MB retained PER BATCH, growing linearly with batch count (no_grad in
the same test stayed flat at +0MB, confirming the two regimes differ).
The user's real run had n_active=7791 -> 16 batches -> ~9.5GB of
retained activations on top of a 4.8GB baseline; their log died partway
through batch 6 of 16, which is ~8.4GB -- consistent with a WSL2 VM
holding ~8GB.

THE FIX FOR THAT: gradient checkpointing (torch.utils.checkpoint) around
each node-batch. Checkpointing stores only each batch's INPUTS and
OUTPUTS, discards its internals, and recomputes them during backward().
Because the batches are independent, only ONE batch's internals are ever
live at a time, in both the forward and the backward pass -- so retained
memory drops from O(n_active * H * T * T) to O(n_active * T * D) for the
saved inputs/outputs plus O(batch_size * H * T * T) transient. The cost
is one extra forward pass per batch during backward (~30% more compute),
which is the standard and correct trade for this situation: the run
finishing slowly is strictly better than the run not finishing.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NEG_INF = -1e9  # matches spatial_attention.py's convention -- avoids NaN from inf - inf edge cases


class TemporalAttention(nn.Module):
    """Per-node causal multi-head self-attention across the T snapshots.

    Treats each of the N nodes as an independent sequence of length T,
    exactly like a Transformer encoder layer applied per-node along time.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, max_len: int = 512, dropout: float = 0.0,
                 node_batch_size: int = 512, use_checkpointing: bool = True):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # See module docstring's "THIRD SCALE LIMIT" section: bounds peak
        # memory to O(node_batch_size * n_heads * T * T), independent of
        # how many nodes are actually active anywhere in the chunk. 512 is
        # a starting point, not a tuned value -- at chunk_len=168, H=4,
        # this keeps the scores tensor for one batch under ~250MB
        # (512*4*168*168*4 bytes), comfortably below what crashed the
        # user's WSL machine at n_active in the thousands.
        self.node_batch_size = node_batch_size
        # See module docstring's "FOURTH SCALE LIMIT" section. Without
        # this, batching only defers the memory to backward() instead of
        # bounding it. Exposed as a flag purely so tests can compare
        # checkpointed vs non-checkpointed output for equality -- it must
        # be an exact-equality refactor, never an approximation.
        self.use_checkpointing = use_checkpointing

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned positional embedding per absolute snapshot index -- simple
        # and adequate given snapshots are evenly spaced by construction
        # (FixedWindowTemporal). Swap for a continuous time2vec-style
        # encoding using actual timestamps if a future temporal strategy
        # ever produces irregularly-spaced snapshots.
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def _attend_batch(self, h_active: torch.Tensor, mask_active: torch.Tensor,
                      pos_emb: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        """One node-batch's full causal attention. Factored out of
        forward() so it can be wrapped in torch.utils.checkpoint (see the
        module docstring's "FOURTH SCALE LIMIT" section) -- everything
        allocated in here is recomputed during backward() rather than
        retained, which is the whole point.

        h_active:    (T, b, D)
        mask_active: (T, b) bool
        pos_emb:     (1, T, D) -- precomputed once by forward(), shared by all batches
        causal:      (T, T) bool -- ditto

        Returns (T, b, D), the ATTENTION OUTPUT ONLY (the residual add
        happens in forward(), outside the checkpoint, so the cheap part
        isn't needlessly recomputed).
        """
        T, b, D = h_active.shape

        # (T, b, D) -> (b, T, D): each node becomes its own sequence,
        # time is the sequence axis -- standard attention convention.
        h_nt = h_active.permute(1, 0, 2) + pos_emb
        mask_nt = mask_active.permute(1, 0)  # (b, T)

        q = self.q_proj(h_nt).view(b, T, self.n_heads, self.head_dim).transpose(1, 2)  # (b, H, T, Dh)
        k = self.k_proj(h_nt).view(b, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_nt).view(b, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("nhqd,nhkd->nhqk", q, k) / (self.head_dim ** 0.5)  # (b, H, T, T)

        # causal[i, j] = True iff j <= i -- query at time i may only see keys up to and including i.
        key_valid = mask_nt.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, T)
        valid = causal.unsqueeze(0).unsqueeze(0) & key_valid  # -> (b, 1, T, T) via broadcasting

        scores = scores.masked_fill(~valid, NEG_INF)
        attn = F.softmax(scores, dim=-1)

        # A query position with NO valid keys at all (e.g. this node has
        # never been observed up to and including time i) softmaxes over
        # all NEG_INF, which is NaN (0/0), not 0 -- zero those explicitly.
        row_has_any_valid = valid.any(dim=-1, keepdim=True)  # (b, 1, T, 1)
        attn = torch.where(row_has_any_valid.expand_as(attn), attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        out_active = torch.einsum("nhqk,nhkd->nhqd", attn, v)
        out_active = out_active.transpose(1, 2).reshape(b, T, D)
        out_active = self.out_proj(out_active)
        return out_active.permute(1, 0, 2)  # (T, b, D)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    (T, N, D) -- N is the chunk's full GLOBAL padded node universe
        mask: (T, N) bool -- True where node was actually observed

        Returns (T, N, D).

        Restricts computation to only the nodes active in AT LEAST ONE of
        this chunk's T snapshots (a node that's global padding for the
        whole chunk contributes nothing to attend to or from), THEN
        processes those active nodes in fixed-size batches (see module
        docstring's "THIRD SCALE LIMIT" section) rather than as one single
        (n_active, H, T, T) tensor -- n_active itself can reach into the
        thousands on real, short-lived-market data even though any single
        snapshot's active count stays small, and batching is exact (not
        an approximation) because one node's attention never depends on
        any other node's.
        """
        T, N, D = h.shape
        device = h.device

        if T > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"TemporalAttention received T={T} snapshots, which exceeds "
                f"max_len={self.pos_embedding.num_embeddings}. This is not a "
                f"config bug to patch by raising max_len -- a realistic full "
                f"training range (~7200 snapshots) makes the dense (N,H,T,T) "
                f"attention tensor this layer computes infeasible regardless "
                f"of max_len. Slice the input into bounded chunks before "
                f"calling forward() (see model/chunking.py's chunk_ranges / "
                f"model/train.py, which never hands this layer more than "
                f"chunk_len snapshots at once)."
            )

        out = h.clone()  # residual base -- a node never active anywhere in
        # this chunk (pure global padding for the whole window) simply
        # keeps its input unchanged, same convention
        # TwoChannelSpatialAttention already uses for isolated nodes.

        active_node_idx = mask.any(dim=0).nonzero(as_tuple=True)[0]  # (n_active,) GLOBAL indices, sorted ascending
        n_active = active_node_idx.numel()
        if n_active == 0:
            return out  # nothing active anywhere in this chunk -- nothing to attend over

        # Computed once, shared by every batch below -- neither depends on
        # which nodes are in a given batch.
        positions = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # (1, T, D)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # (T, T)

        bs = self.node_batch_size
        n_batches = (n_active + bs - 1) // bs
        print(f"      [temporal] n_active={n_active}  node_batch_size={bs}  -> {n_batches} batch(es)  T={T}",
              flush=True)
        _t_start = time.time()

        # Checkpointing is only meaningful when autograd is actually
        # recording -- under no_grad (inference/validation) each batch's
        # internals are already freed at the end of its iteration, so
        # wrapping would add a pointless recompute.
        do_checkpoint = self.use_checkpointing and torch.is_grad_enabled()

        for bi, start in enumerate(range(0, n_active, bs)):
            batch_idx = active_node_idx[start:start + bs]  # (b,) GLOBAL indices, still sorted ascending
            b = batch_idx.numel()

            h_active = h[:, batch_idx, :]      # (T, b, D)
            mask_active = mask[:, batch_idx]   # (T, b)

            if do_checkpoint:
                out_active = checkpoint(
                    self._attend_batch, h_active, mask_active, pos_emb, causal,
                    use_reentrant=False,
                )
            else:
                out_active = self._attend_batch(h_active, mask_active, pos_emb, causal)

            out[:, batch_idx, :] = h_active + out_active  # residual, only for this batch's nodes

            if bi % 5 == 0 or bi == n_batches - 1:
                print(f"      [temporal] batch {bi + 1:3d}/{n_batches}  b={b:4d}  "
                      f"ckpt={do_checkpoint}  elapsed={time.time() - _t_start:6.1f}s", flush=True)

        return out