"""In-game graph model: a play memory (v1 a GRU, v2 a small causal transformer) feeding a
relation-typed contract graph.

Data: the panel arrays from analysis/graph_2026_10/b4_export.py (panel_v2; see its docstring). A live
feed's delay is applied when batches are built (delay_s): a play counts as known at a step only once
its time + delay_s has passed, and everything derived from the play-by-play (the plays the memory
reads, the game state, each contract's distance to its threshold, the structural baseline) is
recomputed from that. Market data are never delayed. Per 5-s step of a game, the model sees
    the play memory   a GRU over the game's plays so far (only plays known at the step); with
                      memory="none" it is replaced by zeros (the ablation)
    the game state    the step's features (time left, score, possession, winner price, activity)
    the contracts     each live-or-quoted contract's state (price range, quote age, distance to its
                      threshold, baseline price and gap)
and passes messages between contracts by kind (winner / spread / total), with a separate weight for
each (receiving kind, sending kind) pair, for N_ROUNDS rounds.

Outputs per contract: the change in value at +10 s / +60 s / +600 s (cents), and a correction to
the market's log-odds for the contract's final result.

Loss: the predicted level (value + change) is scored against the target RANGE at each horizon
(bid..ask then; a point when only a value exists): zero inside the range, Huber outside; plus
LAMBDA_Y x binary cross-entropy on the final result.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr, ndtri

N_KINDS = 7                     # winner, spread, total + four season-long futures (absent kinds are inert)


# ----------------------------------------------------------------------------- data

class Panel:
    """Memory-mapped panel_v2 arrays for one league."""

    ARRAYS = ("plays_x", "plays_t", "plays_state", "game_play_ptr", "steps_t", "steps_x", "game_step_ptr", "step_node_ptr",
              "nodes_x", "nodes_kind", "node_strike", "node_side", "value", "lo", "hi", "value_h", "y", "game_L", "game_fold")

    def __init__(self, path: str | Path, horizons: list[str] | None = None) -> None:
        """horizons: train and evaluate on these only (names from the manifest, e.g. ["30s", "60s",
        "120s"]); default all."""
        p = Path(path)
        self.manifest = json.loads((p / "manifest.json").read_text())
        for k in self.ARRAYS:
            setattr(self, k, np.load(p / f"{k}.npy", mmap_mode="r"))
        for k in ("game_play_ptr", "game_step_ptr", "step_node_ptr"):           # small: load as int64
            setattr(self, k, np.asarray(getattr(self, k)).astype(np.int64))
        for k in ("plays_t", "plays_state", "steps_t", "steps_x", "game_L", "game_fold", "plays_x"):
            setattr(self, k, np.asarray(getattr(self, k)))
        self.n_games = len(self.game_fold)
        all_h = self.manifest["horizons"]
        self.horizons = list(horizons) if horizons else list(all_h)
        missing = [h for h in self.horizons if h not in all_h]
        if missing:
            raise ValueError(f"horizons {missing} not in the panel ({all_h}); rebuild the table / export")
        self.h_idx = np.array([all_h.index(h) for h in self.horizons])
        self.n_h = len(self.horizons)
        self.market_node_features = self.manifest["node_market_features"]
        self.fp = self.plays_x.shape[1]
        self.fs = self.steps_x.shape[1] + len(STEP_DERIVED)
        self.fc = self.nodes_x.shape[1] + len(NODE_DERIVED)
        self.step_has_nodes = np.diff(self.step_node_ptr) > 0
        self.s_m, self.s_t = self.manifest["baseline_S"]
        self.scale, self.tot_scale = self.manifest["scale"], self.manifest["tot_scale"]

    def game_steps(self, g: int) -> np.ndarray:
        a, b = self.game_step_ptr[g], self.game_step_ptr[g + 1]
        s = np.arange(a, b)
        return s[self.step_has_nodes[a:b]]


STEP_DERIVED = ["tau", "margin", "total", "off_home", "off_away", "new_plays", "log_since_play"]
NODE_DERIVED = ["dist", "base", "base_gap", "has_base"]


def known_state(pn: Panel, g: int, step_t: np.ndarray, delay_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For steps of game g: plays known (count), the raw state after the last known play
    [tau, margin, total, offense], and the derived step features."""
    a, b = pn.game_play_ptr[g], pn.game_play_ptr[g + 1]
    pt = pn.plays_t[a:b]
    known = np.searchsorted(pt, step_t - delay_s, side="right")
    before = np.searchsorted(pt, step_t - 5.0 - delay_s, side="right")
    st = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (len(step_t), 1))
    has = known > 0
    st[has] = pn.plays_state[a + known[has] - 1]
    since = np.where(has, step_t - pt[np.maximum(known - 1, 0)] if len(pt) else 0.0, 600.0).clip(0, 3600)
    feats = np.stack([st[:, 0], st[:, 1] / pn.scale, st[:, 2] / pn.tot_scale, (st[:, 3] > 0), (st[:, 3] < 0),
                      (known > before), np.log1p(since) / 6], 1).astype(np.float32)
    return known, st, feats


def baseline_price(kind: np.ndarray, side: np.ndarray, strike: np.ndarray, p_home: np.ndarray, tau: np.ndarray,
                   total: np.ndarray, L: np.ndarray, s_m: float, s_t: float) -> np.ndarray:
    """The structural baseline (b1_baseline) in cents; NaN for winner contracts."""
    tau = np.clip(tau, 1e-3, 1.0)
    z = ndtri(np.clip(p_home / 100, 0.01, 0.99))
    s = s_m * np.sqrt(tau)
    spread = np.where(side > 0, ndtr(z - strike / s), ndtr(-z - strike / s))
    tot = ndtr((total + L * tau - strike) / (s_t * np.sqrt(tau)))
    return np.where(kind == 1, 100 * spread, np.where(kind == 2, 100 * tot, np.nan)).astype(np.float32)


@dataclass
class Batch:
    plays: torch.Tensor          # [Bg, Pmax, Fp]
    step_game: torch.Tensor      # [Ns] index into Bg
    step_plays: torch.Tensor     # [Ns] plays known (0 = none yet)
    steps: torch.Tensor          # [Ns, Fs]
    nodes: torch.Tensor          # [Ns, C, Fc]
    mask: torch.Tensor           # [Ns, C]
    kind: torch.Tensor           # [Ns, C]
    value: torch.Tensor          # [Ns, C]
    lo: torch.Tensor             # [Ns, C, H]
    hi: torch.Tensor
    value_h: torch.Tensor
    y: torch.Tensor              # [Ns, C]
    base: torch.Tensor           # [Ns, C]
    game_of_node: torch.Tensor   # [Ns, C] global game index
    step_of_node: torch.Tensor   # [Ns, C] global step index


def make_batch(pn: Panel, games: list[int], steps_per_game: int | None, rng: np.random.Generator | None,
               device: torch.device, stride: int = 1, delay_s: float = 0.0) -> Batch:
    play_blocks, step_ids, step_game, known_l, state_l, sfeat_l = [], [], [], [], [], []
    for bi, g in enumerate(games):
        a, b = pn.game_play_ptr[g], pn.game_play_ptr[g + 1]
        play_blocks.append(pn.plays_x[a:b])
        s = pn.game_steps(g)
        if steps_per_game is not None and len(s) > steps_per_game:
            s = np.sort(rng.choice(s, steps_per_game, replace=False))
        else:
            s = s[::stride]
        kn, st, sf = known_state(pn, g, pn.steps_t[s], delay_s)
        step_ids.append(s), step_game.append(np.full(len(s), bi)), known_l.append(kn), state_l.append(st), sfeat_l.append(sf)
    pmax = max(1, max(len(x) for x in play_blocks))
    plays = np.zeros((len(games), pmax, pn.fp), np.float32)
    for i, x in enumerate(play_blocks):
        plays[i, :len(x)] = x
    sid, sg = np.concatenate(step_ids), np.concatenate(step_game)
    known, state, sfeat = np.concatenate(known_l), np.concatenate(state_l), np.concatenate(sfeat_l)
    a, b = pn.step_node_ptr[sid], pn.step_node_ptr[sid + 1]
    cnt = b - a
    cmax, ns = int(cnt.max()), len(sid)
    rows = np.full((ns, cmax), -1, np.int64)
    for i in range(ns):
        rows[i, :cnt[i]] = np.arange(a[i], b[i])
    m = rows >= 0
    flat = rows[m]
    lo_, hi_ = int(flat.min()), int(flat.max()) + 1          # one contiguous read of the memory maps
    idx = flat - lo_

    def take(arr, fill, tail=(), dtype=np.float32, cols=None):
        out = np.full((ns, cmax) + tail, fill, dtype)
        src = np.asarray(arr[lo_:hi_])[idx]
        out[m] = src if cols is None else src[:, cols]
        return out
    kind = take(pn.nodes_kind, 0, dtype=np.int64)
    value = take(pn.value, 50.0)
    strike, side = take(pn.node_strike, 0.0), take(pn.node_side, 0, dtype=np.int64)
    # derived node features at this delay: distance to the threshold and the baseline
    st_n = np.repeat(state[:, None, :], cmax, 1)                                  # [Ns, C, 4]
    p_home = np.repeat((100 * pn.steps_x[sid, 0])[:, None], cmax, 1)
    L = np.repeat(pn.game_L[np.asarray(games)[sg]][:, None], cmax, 1)
    dist = np.where(kind == 1, strike - side * st_n[..., 1], np.where(kind == 2, strike - st_n[..., 2], 0.0))
    base = baseline_price(kind, side, strike, p_home, st_n[..., 0], st_n[..., 2], L, pn.s_m, pn.s_t)
    has_base = np.isfinite(base)
    nderived = np.stack([dist / pn.scale, np.where(has_base, base, value) / 100,
                         np.where(has_base, base - value, 0.0) / 10, has_base], -1).astype(np.float32)
    nodes = np.nan_to_num(np.concatenate([take(pn.nodes_x, 0.0, (pn.nodes_x.shape[1],)), nderived], -1))
    nodes[~m] = 0
    steps = np.concatenate([pn.steps_x[sid].astype(np.float32), sfeat], 1)
    gon = np.repeat(np.asarray(games)[sg][:, None], cmax, 1)
    son = np.repeat(sid[:, None], cmax, 1)
    t = lambda x: torch.as_tensor(x, device=device)           # noqa: E731
    return Batch(plays=t(plays), step_game=t(sg.astype(np.int64)), step_plays=t(known.astype(np.int64)), steps=t(steps),
                 nodes=t(nodes), mask=t(m), kind=t(kind), value=t(value), lo=t(take(pn.lo, np.nan, (pn.n_h,), cols=pn.h_idx)),
                 hi=t(take(pn.hi, np.nan, (pn.n_h,), cols=pn.h_idx)),
                 value_h=t(take(pn.value_h, np.nan, (pn.n_h,), cols=pn.h_idx)), y=t(take(pn.y, np.nan)),
                 base=t(np.where(has_base, base, np.nan).astype(np.float32)), game_of_node=t(gon), step_of_node=t(son))


# ----------------------------------------------------------------------------- model

class InGameGraph(nn.Module):
    def __init__(self, fp: int, fs: int, fc: int, hidden: int = 64, memory: str = "gru", n_rounds: int = 2,
                 dropout: float = 0.1, n_horizons: int = 3, tf_layers: int = 2, tf_heads: int = 4,
                 tf_window: int = 0, max_plays: int = 512) -> None:
        """memory: "gru", "transformer" (causal; tf_window > 0 lets each play see only the last
        tf_window plays, itself included) or "none"."""
        super().__init__()
        self.memory = memory
        h = hidden
        self.play_in = nn.Linear(fp, h)
        self.gru = nn.GRU(h, h, batch_first=True) if memory == "gru" else None
        self.tf = None
        if memory == "transformer":
            self.pos = nn.Embedding(max_plays, h)
            layer = nn.TransformerEncoderLayer(h, tf_heads, 2 * h, dropout, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
            self.tf_norm = nn.LayerNorm(h)
            self.tf_window = tf_window
        self.ctx = nn.Sequential(nn.Linear(fs + h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h))
        self.node_in = nn.Sequential(nn.Linear(fc, h), nn.ReLU(), nn.Linear(h, h))
        self.ctx_to_node = nn.Linear(h, h)
        # W[r][k, s]: the message a node of kind k receives from the mean of kind s
        self.rel = nn.ParameterList([nn.Parameter(torch.randn(N_KINDS, N_KINDS, h, h) / math.sqrt(h)) for _ in range(n_rounds)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout)) for _ in range(n_rounds)])
        self.delta = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, n_horizons))
        self.outcome = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def play_states(self, plays: torch.Tensor) -> torch.Tensor:
        """[Bg, Pmax+1, H]: state after 0, 1, ... plays (index 0 = before any play)."""
        bg, p, _ = plays.shape
        if self.tf is not None:
            i = torch.arange(p, device=plays.device)
            future = i[None, :] > i[:, None]                                   # [P, P] True = blocked
            if self.tf_window > 0:
                future = future | (i[None, :] <= i[:, None] - self.tf_window)
            mask = torch.zeros(p, p, device=plays.device).masked_fill(future, float("-inf"))
            x = torch.relu(self.play_in(plays)) + self.pos(i.clamp(max=self.pos.num_embeddings - 1))[None]
            out = self.tf_norm(self.tf(x, mask=mask))
            return torch.cat([torch.zeros_like(out[:, :1]), out], 1)
        if self.gru is None:
            return torch.zeros(bg, p + 1, self.play_in.out_features, device=plays.device)
        out, _ = self.gru(torch.relu(self.play_in(plays)))
        return torch.cat([torch.zeros_like(out[:, :1]), out], 1)

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.play_states(b.plays)
        hp = ps[b.step_game, b.step_plays.clamp(max=ps.shape[1] - 1)]          # [Ns, H]
        ctx = self.ctx(torch.cat([b.steps, hp], -1))                            # [Ns, H]
        x = self.node_in(b.nodes) + self.ctx_to_node(ctx)[:, None]              # [Ns, C, H]
        m = b.mask.float()
        onehot = F.one_hot(b.kind.clamp(min=0), N_KINDS).float() * m[..., None]  # [Ns, C, K]
        for W, upd in zip(self.rel, self.upd):
            cnt = onehot.sum(1).clamp(min=1)                                    # [Ns, K]
            mean_s = torch.einsum("nck,nch->nkh", onehot, x) / cnt[..., None]   # [Ns, K(send), H]
            msg_k = torch.einsum("nsh,ksho->nko", mean_s, W)                    # [Ns, K(recv), H]
            msg = torch.einsum("nck,nkh->nch", onehot, msg_k)                   # each node: its kind's message
            x = x + upd(torch.cat([x, msg], -1))
        z = torch.cat([x, ctx[:, None].expand_as(x)], -1)
        return 10.0 * self.delta(z), self.outcome(z).squeeze(-1)


PANEL_IDENTITY = ("league", "version", "targets", "futures", "horizons", "play_features", "step_market_features",
                  "node_market_features", "kinds", "baseline_S", "scale", "tot_scale")


def model_config(pn: Panel, *, hidden: int = 64, memory: str = "gru", n_rounds: int = 2, dropout: float = 0.1,
                 tf_window: int = 0) -> dict:
    """InGameGraph's arguments for this panel (stored with saved weights)."""
    return dict(fp=pn.fp, fs=pn.fs, fc=pn.fc, hidden=hidden, memory=memory, n_rounds=n_rounds, dropout=dropout,
                n_horizons=pn.n_h, tf_window=tf_window)


def save_model(path: str | Path, model: InGameGraph, pn: Panel, config: dict, info: dict) -> None:
    """Weights + model arguments + what the panel looked like (features, scales, horizons), so a later
    panel (the holdout) can be checked to match before predicting."""
    ident = {k: pn.manifest.get(k) for k in PANEL_IDENTITY} | {"train_horizons": pn.horizons}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": config,
                "panel": ident, "info": info}, path)


def load_model(path: str | Path, pn: Panel, device: torch.device) -> tuple[InGameGraph, dict]:
    """Saved weights for this panel; refuses a panel whose features, scales or horizons differ."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ident = {k: pn.manifest.get(k) for k in PANEL_IDENTITY} | {"train_horizons": pn.horizons}
    bad = {k: (ck["panel"].get(k), v) for k, v in ident.items() if k != "league" and ck["panel"].get(k) != v}
    if bad or ck["panel"]["league"] != ident["league"]:
        raise ValueError(f"{path}: panel differs from the one the model was trained on: {bad or 'league'}")
    model = InGameGraph(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck


# ----------------------------------------------------------------------------- loss and metrics

def interval_error(pred_level: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return F.relu(lo - pred_level) + F.relu(pred_level - hi)


def loss_fn(b: Batch, delta: torch.Tensor, corr: torch.Tensor, lambda_y: float, huber: float = 5.0) -> tuple[torch.Tensor, dict]:
    level = b.value[..., None] + delta
    ok = b.mask[..., None] & torch.isfinite(b.lo) & torch.isfinite(b.hi)
    e = interval_error(level, torch.nan_to_num(b.lo), torch.nan_to_num(b.hi))
    li = F.huber_loss(e, torch.zeros_like(e), delta=huber, reduction="none")[ok].mean() / 10
    p = (b.value / 100).clamp(0.005, 0.995)
    logit = torch.log(p / (1 - p)) + corr
    oky = b.mask & torch.isfinite(b.y)
    ly = F.binary_cross_entropy_with_logits(logit[oky], b.y[oky]) if oky.any() else torch.zeros((), device=delta.device)
    return li + lambda_y * ly, {"interval": float(li.detach()), "outcome": float(ly.detach())}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- training and evaluation

KIND_NAMES = ["winner", "spread", "total", "champ", "conference", "division", "playoffs"]


def fit_baseline_reversion(pn: Panel, games: list[int], device: torch.device, delay_s: float) -> np.ndarray:
    """beta[kind, h]: change(h) = beta * (baseline - value), least squares on (a sample of) the training games."""
    beta = np.zeros((N_KINDS, pn.n_h))
    acc = {k: [] for k in ("kind", "gap", "dv")}
    for i in range(0, len(games), 8):
        b = make_batch(pn, games[i:i + 8], None, None, torch.device("cpu"), stride=6, delay_s=delay_s)
        m = b.mask.numpy()
        acc["kind"].append(b.kind.numpy()[m])
        acc["gap"].append((b.base - b.value).numpy()[m])
        acc["dv"].append((b.value_h - b.value[..., None]).numpy()[m])
    k, gap, dv = (np.concatenate(acc[x]) for x in ("kind", "gap", "dv"))
    for kind in (1, 2):
        for h in range(pn.n_h):
            ok = (k == kind) & np.isfinite(gap) & np.isfinite(dv[:, h])
            if ok.sum() > 100:
                beta[kind, h] = float((gap[ok] * dv[ok, h]).sum() / (gap[ok] ** 2).sum())
    return beta


@torch.no_grad()
def predict(model: InGameGraph, pn: Panel, games: list[int], device: torch.device, stride: int = 1,
            games_per_batch: int = 4, delay_s: float = 0.0) -> dict[str, np.ndarray]:
    model.eval()
    feats = pn.market_node_features
    fi = {k: feats.index(k) for k in ("bid", "ask", "has_bid", "has_ask", "log_age", "live")}
    out = {k: [] for k in ("delta", "corr", "kind", "value", "lo", "hi", "value_h", "y", "base", "game", "step", "nodes")}
    for i in range(0, len(games), games_per_batch):
        b = make_batch(pn, games[i:i + games_per_batch], None, None, device, stride=stride, delay_s=delay_s)
        d, c = model(b)
        m = b.mask
        for k, v in (("delta", d), ("corr", c), ("kind", b.kind), ("value", b.value), ("lo", b.lo), ("hi", b.hi),
                     ("value_h", b.value_h), ("y", b.y), ("base", b.base), ("game", b.game_of_node),
                     ("step", b.step_of_node), ("nodes", b.nodes[..., [fi[x] for x in fi]])):
            out[k].append(v[m].float().cpu().numpy() if v.dtype.is_floating_point else v[m].cpu().numpy())
    model.train()
    r = {k: np.concatenate(v) for k, v in out.items()}
    nd = r.pop("nodes")                       # entry quote: bid / ask in cents, sides present, age in s, live
    r["entry_bid"], r["entry_ask"] = 100 * nd[:, 0], 100 * nd[:, 1]
    r["has_bid"], r["has_ask"] = nd[:, 2] > 0.5, nd[:, 3] > 0.5
    r["quote_age_s"] = np.expm1(8 * nd[:, 4])
    r["live"] = nd[:, 5] > 0.5
    return r


def huber_np(e: np.ndarray, delta: float = 5.0) -> np.ndarray:
    a = np.abs(e)
    return np.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def metrics(pr: dict[str, np.ndarray], beta: np.ndarray, horizons: list[str]) -> dict:
    """Per contract kind and horizon: R^2 of the value change (model / baseline reversion, against no
    change), interval loss (model / no change / baseline reversion); drift R^2; final-result log loss
    (model / market)."""
    res = {}
    for kind, name in enumerate(KIND_NAMES):
        k = pr["kind"] == kind
        r = {}
        for h, hn in enumerate(horizons):
            act = pr["value_h"][:, h] - pr["value"]
            ok = k & np.isfinite(act)
            if ok.sum() < 100:
                continue
            dm = pr["delta"][ok, h]
            db = beta[kind, h] * np.nan_to_num(pr["base"][ok] - pr["value"][ok])
            ss0 = float((act[ok] ** 2).sum())
            oki = k & np.isfinite(pr["lo"][:, h]) & np.isfinite(pr["hi"][:, h])
            lvl = lambda d: pr["value"][oki] + d                                       # noqa: E731
            ie = lambda d: float(huber_np(np.maximum(pr["lo"][oki, h] - lvl(d), 0) + np.maximum(lvl(d) - pr["hi"][oki, h], 0)).mean())
            r[hn] = dict(rows=int(ok.sum()), r2_model=1 - float(((act[ok] - dm) ** 2).sum()) / ss0,
                         r2_baseline=1 - float(((act[ok] - db) ** 2).sum()) / ss0,
                         interval_model=ie(pr["delta"][oki, h]), interval_no_change=ie(np.zeros(oki.sum())),
                         interval_baseline=ie(beta[kind, h] * np.nan_to_num(pr["base"][oki] - pr["value"][oki])))
        if "60s" in horizons and "600s" in horizons:
            i60, i600 = horizons.index("60s"), horizons.index("600s")
            dr = pr["value_h"][:, i600] - pr["value_h"][:, i60]
            ok = k & np.isfinite(dr)
            if ok.sum() > 100:
                dm = pr["delta"][ok, i600] - pr["delta"][ok, i60]
                r["drift_60_600"] = dict(rows=int(ok.sum()), r2_model=1 - float(((dr[ok] - dm) ** 2).sum() / (dr[ok] ** 2).sum()),
                                         corr=float(np.corrcoef(dr[ok], dm)[0, 1]))
        oky = k & np.isfinite(pr["y"])
        if oky.sum() > 100:
            p = np.clip(pr["value"][oky] / 100, 0.005, 0.995)
            lg = np.log(p / (1 - p)) + pr["corr"][oky]
            q = 1 / (1 + np.exp(-lg))
            y = pr["y"][oky]
            ll = lambda pp: float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())   # noqa: E731
            r["outcome"] = dict(rows=int(oky.sum()), games=int(len(np.unique(pr["game"][oky]))),
                                logloss_market=ll(p), logloss_model=ll(np.clip(q, 1e-6, 1 - 1e-6)))
        res[name] = r
    return res


def train(pn: Panel, test_fold: int | None, *, memory: str = "gru", hidden: int = 64, n_rounds: int = 2, dropout: float = 0.1,
          lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 40, patience: int = 5, games_per_batch: int = 8,
          steps_per_game: int = 128, lambda_y: float = 0.1, val_share: float = 0.15, eval_stride: int = 1,
          seed: int = 0, device: str = "auto", log=print, return_preds: bool = False, delay_s: float = 0.0,
          tf_window: int = 0, huber: float = 5.0) -> dict:
    """Walk-forward: train on folds < test_fold (the last val_share of those games, by date, for early
    stopping), test on fold == test_fold. test_fold=None: the final model, trained the same way on every
    game in the panel (no test set, no metrics). huber: the loss's switch from squared to linear error, in
    cents (a large value is close to squared error)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    fold = np.asarray(pn.game_fold)
    train_all = [g for g in range(pn.n_games) if test_fold is None or fold[g] < test_fold]   # games are in date order
    n_val = max(1, int(len(train_all) * val_share))
    tr, va = train_all[:-n_val], train_all[-n_val:]
    te = [g for g in range(pn.n_games) if test_fold is not None and fold[g] == test_fold]
    model = InGameGraph(**model_config(pn, hidden=hidden, memory=memory, n_rounds=n_rounds, dropout=dropout,
                                       tf_window=tf_window)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    log(f"device {dev}; train {len(tr)} games, val {len(va)}, test {len(te)} "
        f"({'final model, no test fold' if test_fold is None else f'fold {test_fold}'}); memory={memory}; "
        f"feed delay {delay_s:g} s; "
        f"params {sum(p.numel() for p in model.parameters()):,}")
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        order = rng.permutation(tr)
        tot, n = 0.0, 0
        for i in range(0, len(order), games_per_batch):
            b = make_batch(pn, list(order[i:i + games_per_batch]), steps_per_game, rng, dev, delay_s=delay_s)
            d, c = model(b)
            loss, info = loss_fn(b, d, c, lambda_y, huber=huber)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + float(loss.detach()), n + 1
        pv = predict(model, pn, va, dev, stride=6, delay_s=delay_s)
        okv = np.isfinite(pv["lo"]) & np.isfinite(pv["hi"])
        lvl = pv["value"][:, None] + pv["delta"]
        e = np.maximum(pv["lo"] - lvl, 0) + np.maximum(lvl - pv["hi"], 0)
        val = float(huber_np(e[okv], huber).mean())
        log(f"epoch {ep:2d}  train loss {tot / max(n, 1):.4f}  val interval {val:.4f}")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if test_fold is None:
        res = dict(league=pn.manifest["league"], test_fold=None, memory=memory, tf_window=tf_window, seed=seed, hidden=hidden,
                   delay_s=delay_s, huber=huber, lambda_y=lambda_y, params=sum(p.numel() for p in model.parameters()),
                   n_rounds=n_rounds, epochs_run=ep + 1, best_val_interval=best, train_games=len(tr), val_games=len(va),
                   test_games=0, horizons=pn.horizons, metrics={})
        return (res, model, None) if return_preds else (res, model)
    beta = fit_baseline_reversion(pn, tr, dev, delay_s)
    pt = predict(model, pn, te, dev, stride=eval_stride, delay_s=delay_s)
    if return_preds:
        res_preds = pt
    res = dict(league=pn.manifest["league"], test_fold=test_fold, memory=memory, tf_window=tf_window, seed=seed,
               hidden=hidden, delay_s=delay_s, huber=huber, lambda_y=lambda_y, params=sum(p.numel() for p in model.parameters()),
               n_rounds=n_rounds, epochs_run=ep + 1, best_val_interval=best, train_games=len(tr), val_games=len(va),
               test_games=len(te), horizons=pn.horizons, metrics=metrics(pt, beta, pn.horizons))
    return (res, model, res_preds) if return_preds else (res, model)
