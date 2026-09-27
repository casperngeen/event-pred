#!/usr/bin/env python
"""Do extra (non-Kalshi) releases help *learn* the macro graph?

    venv/bin/python -W ignore analysis/spillover_2026_09/augmented.py \
        > analysis/spillover_2026_09/out/augmented.txt     # needs releases.py and event_time build_panel.py

Instants: the 154 Kalshi release instants (event-time panel) and the 1,036
calendar release instants (``releases.py``: 34 releases with a consensus and no
Kalshi market, > 1 h from any Kalshi release). Sources: the 17 Kalshi series
(z = their panel surprise) and the 34 calendar releases (z = standardised
actual − consensus). Targets: the 17 Kalshi series.

Label, the same for both kinds of instant: the target's lead contract from the
release to +4 h (``_tape.response``; a print after the release is required),
in per-target sd units. At a Kalshi instant, targets that released, or share a
release with a series that did, are masked (as in the event-time panel).

Edges: every Kalshi → Kalshi candidate of the panel (cross-release, 142) and
every calendar release → Kalshi target (578): 720. Theory sign = HAWKISH
product. Channel = source type → target type, with calendar families mapped
onto the same types (inflation → inflation, activity and sentiment → growth,
labour → labour), so pooling can share strength across the two kinds of source.

Models, walk-forward over all instants (8 folds, training labels must end
21 days before the fold):
  theory rule        Σ sign·z (no fitting)
  one slope          ridge on the theory-signed signal (1 parameter)
  free linear graph  ridge on all 720 edges: learned, no sign prior
  low-rank graph     W = (E_s E_dᵀ) ⊙ candidates, rank 4, Adam + early
                     stopping: the STG-shaped learner, no sign prior
  Bayes, soft sign   channel-pooled, each edge's sign free (``bayes.py``)
each fitted twice: on the Kalshi-release training cells only, and on Kalshi +
calendar. Scored on the same test cells: the Kalshi-release cells (does the
extra data help learn the Kalshi graph?) and all test cells.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "stg_infra")
from stg.models.baselines import _ridge
from stg.models.train import PURGE, _fold_cuts
from stg.panel.registry import is_same_release

sys.path.insert(0, "analysis/spillover_2026_09")
sys.path.insert(0, "analysis/event_time_2026_09")
from _tape import H, HAWKISH, LEGS, TYPE, lead_contract, response  # noqa: E402
import bayes  # noqa: E402

OUT = "analysis/spillover_2026_09/out"
HORIZON = "+4h"
N_BOOT = 2000
torch.set_num_threads(4)
rng = np.random.default_rng(0)
CAL_TYPE = {"inflation": "inflation", "activity": "growth", "sentiment": "growth", "labour": "labour"}
from calendar_triggers import TRIGGERS as CAL_TRIG  # noqa: E402

KSRC = sorted(LEGS)
TARGETS = sorted(LEGS)
CSRC = sorted(CAL_TRIG)
SOURCES = [("k", s) for s in KSRC] + [("c", c) for c in CSRC]
SI = {src: i for i, src in enumerate(SOURCES)}
TI = {t: i for i, t in enumerate(TARGETS)}
ttype = lambda s: "policy" if s == "FED" else TYPE[s]            # noqa: E731

# ------------------------------------------------------------------ instants and surprises
ev = (pl.read_parquet("analysis/event_time_2026_09/out/event_nodes.parquet")
      .filter(pl.col("released") & pl.col("z").is_not_null())
      .select(pl.col("instant").dt.replace_time_zone(None).cast(pl.Datetime("us")).alias("t"),
              "series", "z"))
kal = {t: dict(zip(g["series"], g["z"])) for (t,), g in ev.group_by("t")}
cal_rows = (pl.read_parquet(f"{OUT}/releases.parquet").select("t", "ev", "z").unique(["t", "ev"]))
cal = {t: dict(zip(g["ev"], g["z"])) for (t,), g in cal_rows.group_by("t")}
print(f"instants: Kalshi {len(kal)}, calendar {len(cal)}")

# ------------------------------------------------------------------ cells
rows = []
for kind, book in (("kalshi", kal), ("calendar", cal)):
    for t, zs in book.items():
        tn = np.datetime64(t, "us")
        x = np.zeros(len(SOURCES))
        for name, z in zs.items():
            x[SI[("k" if kind == "kalshi" else "c", name)]] = z
        for b in TARGETS:
            if kind == "kalshi" and (b in zs or any(is_same_release(b, s) for s in zs)):
                continue
            tb = lead_contract(b, tn)
            if tb is None:
                continue
            _, after = response(tb, tn, tn)
            y = after[HORIZON]
            if np.isfinite(y):
                rows.append((kind, tn, TI[b], x, y))
kind = np.array([r[0] for r in rows])
tc = np.array([r[1] for r in rows])
tgt = np.array([r[2] for r in rows])
Xs = np.stack([r[3] for r in rows])
y_c = np.array([r[4] for r in rows])
sd = np.array([y_c[tgt == b].std() if (tgt == b).sum() > 1 else 1.0 for b in range(len(TARGETS))])
sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
y = y_c / sd[tgt]
print(f"labelled cells: {len(y)} (Kalshi instants {int((kind == 'kalshi').sum())}, calendar "
      f"{int((kind == 'calendar').sum())})")

# ------------------------------------------------------------------ edges
EDGES, SIGN, CH = [], [], []
for si, (sk, s) in enumerate(SOURCES):
    for b in TARGETS:
        if sk == "k":
            if s == b or is_same_release(s, b):
                continue
            sign, stype = HAWKISH[s] * HAWKISH[b], ttype(s)
        else:
            fam, hk = CAL_TRIG[s]
            sign, stype = hk * HAWKISH[b], CAL_TYPE[fam]
        EDGES.append((si, TI[b])); SIGN.append(sign); CH.append(f"{stype}→{ttype(b)}")
SIGN = np.array(SIGN, float)
GROUPS = sorted(set(CH))
GIDX = np.array([GROUPS.index(c) for c in CH])
E = len(EDGES)
XE = np.zeros((len(y), E))
for e, (si, b) in enumerate(EDGES):
    m = tgt == b
    XE[m, e] = Xs[m, si]
CAND = np.zeros((len(SOURCES), len(TARGETS)))
for si, b in EDGES:
    CAND[si, b] = 1
print(f"edges: {E} ({len(KSRC)} Kalshi sources, {len(CSRC)} calendar sources), {len(GROUPS)} channels")


# ------------------------------------------------------------------ models
class LowRank(nn.Module):
    def __init__(self, r=4):
        super().__init__()
        self.Es = nn.Parameter(torch.randn(len(SOURCES), r) * 0.1)
        self.Ed = nn.Parameter(torch.randn(len(TARGETS), r) * 0.1)
        self.c = nn.Parameter(torch.zeros(len(TARGETS)))
        self.register_buffer("mask", torch.tensor(CAND, dtype=torch.float32))

    def forward(self, x, b):
        W = (self.Es @ self.Ed.t()) * self.mask                 # [source, target]
        return (x @ W)[torch.arange(len(b)), b] + self.c[b]


def fit_lowrank(tr, seed=0):
    torch.manual_seed(seed)
    order = np.argsort(tc[tr])
    idx = np.nonzero(tr)[0][order]
    cut = int(len(idx) * 0.85)
    fit, es = idx[:cut], idx[cut:]
    X = torch.tensor(Xs, dtype=torch.float32)
    Y = torch.tensor(np.clip(y, -4, 4), dtype=torch.float32)
    B = torch.tensor(tgt)
    m = LowRank()
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    best, state, bad = np.inf, None, 0
    for ep in range(1000):
        opt.zero_grad()
        loss = ((m(X[fit], B[fit]) - Y[fit]) ** 2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            ev_ = ((m(X[es], B[es]) - Y[es]) ** 2).mean().item()
        if ev_ < best - 1e-6:
            best, bad, state = ev_, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 50:
                break
    m.load_state_dict(state)
    with torch.no_grad():
        return m(X, B).numpy()


def predict_all(tr):
    """Predictions for every cell from models trained on cells ``tr``."""
    out = {"theory rule": XE @ SIGN}
    s = XE @ SIGN
    c = _ridge(s[tr][:, None], np.clip(y[tr], -4, 4), 10.0)
    out["one slope"] = c[0] + c[1] * s
    c = _ridge(XE[tr], np.clip(y[tr], -4, 4), 10.0)
    out["free linear graph"] = c[0] + XE @ c[1:]
    out["low-rank graph (rank 4)"] = np.mean([fit_lowrank(tr, s_) for s_ in (0, 1, 2)], 0)
    d = bayes.fit(XE[tr], y[tr], "soft", sign=SIGN, gidx=GIDX, burn=500, draws=1000)
    out["Bayes, soft sign"] = d["alpha"].mean() + XE @ d["beta"].mean(0)
    return out, d["rhat_max"]


def scores(yy, p):
    s = (yy != 0) & (p != 0)
    up, pu = yy[s] > 0, p[s] > 0
    bal = np.mean([(pu & up).sum() / max(up.sum(), 1), (~pu & ~up).sum() / max((~up).sum(), 1)])
    nz = yy != 0
    auc = roc_auc_score(yy[nz] > 0, p[nz]) if len(np.unique(yy[nz] > 0)) == 2 else np.nan
    return 1 - ((yy - p) ** 2).sum() / (yy ** 2).sum(), bal, auc


def boot_delta(mask, pa, pb):
    """AUC(pb) − AUC(pa) on cells ``mask``, bootstrap over instants."""
    yy, a, b, t = y[mask], pa[mask], pb[mask], tc[mask]
    nz = yy != 0
    yy, a, b, t = yy[nz], a[nz], b[nz], t[nz]
    u, inv = np.unique(t, return_inverse=True)
    groups = [np.nonzero(inv == k)[0] for k in range(len(u))]
    est = roc_auc_score(yy > 0, b) - roc_auc_score(yy > 0, a)
    bs = []
    for _ in range(N_BOOT):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]).astype(int)
        if len(np.unique(yy[ix] > 0)) == 2:
            bs.append(roc_auc_score(yy[ix] > 0, b[ix]) - roc_auc_score(yy[ix] > 0, a[ix]))
    return est, *np.percentile(bs, [2.5, 97.5])




def main():
    uniq = np.unique(tc)
    cuts = _fold_cuts(uniq, 8)
    end = tc + 4 * H
    P = {arm: {} for arm in ("Kalshi only", "Kalshi + calendar")}
    test_mask = np.zeros(len(y), bool)
    rh = []
    for i in range(8):
        te = (tc >= cuts[i]) & (tc < cuts[i + 1])
        base_tr = end < (cuts[i] - PURGE)
        if base_tr.sum() < 50 or te.sum() == 0:
            continue
        test_mask |= te
        for arm, tr in (("Kalshi only", base_tr & (kind == "kalshi")), ("Kalshi + calendar", base_tr)):
            if tr.sum() < 30:
                continue
            preds, r_ = predict_all(tr)
            rh.append(r_)
            for name, p in preds.items():
                P[arm].setdefault(name, np.full(len(y), np.nan))[te] = p[te]
        print(f"fold {i}: train cells Kalshi {int((base_tr & (kind == 'kalshi')).sum())}, "
              f"all {int(base_tr.sum())}; test cells {int(te.sum())} "
              f"(Kalshi {int((te & (kind == 'kalshi')).sum())})", flush=True)


    print(f"\nmax split-R̂ of the Bayes fits: {np.nanmax(rh):.3f}")
    for lab, m in (("Kalshi-release test cells", test_mask & (kind == "kalshi")),
                   ("calendar-release test cells", test_mask & (kind == "calendar")),
                   ("all test cells", test_mask)):
        print(f"\n{'=' * 104}\n{lab}: {int(m.sum())} cells, horizon {HORIZON}\n{'=' * 104}")
        print(f"{'model':26} | {'trained on Kalshi only':>26} | {'trained on Kalshi + calendar':>30} | "
              f"{'ΔAUC (+calendar) [95% CI]':>26}")
        print(f"{'':26} | {'R²':>8} {'bal':>6} {'AUC':>6}   | {'R²':>8} {'bal':>6} {'AUC':>6}       |")
        for name in P["Kalshi + calendar"]:
            pk, pa = P["Kalshi only"].get(name), P["Kalshi + calendar"][name]
            ra = scores(y[m], pa[m])
            rk = scores(y[m], pk[m]) if pk is not None else (np.nan,) * 3
            r2k = "–" if name == "theory rule" else f"{rk[0]:+.4f}"
            r2a = "–" if name == "theory rule" else f"{ra[0]:+.4f}"
            dl = ""
            if pk is not None and name != "theory rule":
                d, lo, hi = boot_delta(m, pk, pa)
                dl = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(f"{name:26} | {r2k:>8} {rk[1]:>6.3f} {rk[2]:>6.3f}   | {r2a:>8} {ra[1]:>6.3f} {ra[2]:>6.3f}"
                  f"       | {dl:>26}")



if __name__ == "__main__":
    main()
