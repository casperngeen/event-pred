#!/usr/bin/env python
"""Learn the graph at the level of detail the data can support: channels.

    venv/bin/python -W ignore analysis/spillover_2026_09/channel.py \
        > analysis/spillover_2026_09/out/channel.txt      # needs releases.py; imports augmented.py's data

``augmented.py`` shows that 776 free edge coefficients cannot be learned even
with the calendar releases. This keeps the learning but cuts the parameters:

  channel (15)            one free coefficient per source type → target type,
                          on the theory-signed channel signal Σ_edges sign·z.
                          The coefficient's sign is learned: positive = the
                          channel moves as theory says.
  source type × target    one free coefficient per (source type, target
  (68)                    series), on the sources' surprises in hawkish units
                          (each release's own direction applied; the target's
                          sign and size are learned, no edge sign used)

against the theory rule, the one-slope rung and the per-edge free graph, each
trained on Kalshi-release cells only and on Kalshi + calendar, on the same
walk-forward folds and test cells as ``augmented.py``. Also the full-sample
channel coefficients (Kalshi + calendar) with OLS standard errors clustered by
release instant.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "stg_infra")
from stg.models.baselines import _ridge
from stg.models.train import PURGE, _fold_cuts

sys.path.insert(0, "analysis/spillover_2026_09")
import augmented as A  # noqa: E402   (builds the cells: y, XE, tc, kind, …)

y, XE, SIGN, tc, kind, tgt = A.y, A.XE, A.SIGN, A.tc, A.kind, A.tgt
GROUPS, GIDX, EDGES, CH = A.GROUPS, A.GIDX, A.EDGES, A.CH
NT = len(A.TARGETS)

# channel signals: n × 15, theory-signed within the channel
CHS = np.column_stack([XE[:, GIDX == g] @ SIGN[GIDX == g] for g in range(len(GROUPS))])

# hawkish-unit source signal per (source type, target): n × (4·17)
src_hk = []
for si, _ in EDGES:
    sk, s = A.SOURCES[si]
    src_hk.append(A.HAWKISH[s] if sk == "k" else A.CAL_TRIG[s][1])
src_hk = np.array(src_hk, float)
STYPES = sorted({c.split("→")[0] for c in CH})
stype = np.array([STYPES.index(c.split("→")[0]) for c in CH])
etgt = np.array([b for _, b in EDGES])
TS = np.zeros((len(y), len(STYPES) * NT))
for k in range(len(STYPES)):
    for b in range(NT):
        m = (stype == k) & (etgt == b)
        if m.any():
            TS[:, k * NT + b] = XE[:, m] @ src_hk[m]
TS = TS[:, np.abs(TS).sum(0) > 0]
print(f"cells {len(y)}; channels {len(GROUPS)}; source-type × target columns {TS.shape[1]}")


def ridge_pred(F, tr, lam):
    c = _ridge(F[tr], np.clip(y[tr], -4, 4), lam)
    return c[0] + F @ c[1:]


def predict_all(tr):
    s = XE @ SIGN
    return {"theory rule": s,
            "one slope (1)": ridge_pred(s[:, None], tr, 10.0),
            "channel, learned sign (15)": ridge_pred(CHS, tr, 1.0),
            "source type × target (68)": ridge_pred(TS, tr, 1.0),
            "free edge graph (776)": ridge_pred(XE, tr, 10.0)}


uniq = np.unique(tc)
cuts = _fold_cuts(uniq, 8)
end = tc + 4 * A.H
P = {arm: {} for arm in ("Kalshi only", "Kalshi + calendar")}
test_mask = np.zeros(len(y), bool)
for i in range(8):
    te = (tc >= cuts[i]) & (tc < cuts[i + 1])
    base = end < (cuts[i] - PURGE)
    if base.sum() < 50 or te.sum() == 0:
        continue
    test_mask |= te
    for arm, tr in (("Kalshi only", base & (kind == "kalshi")), ("Kalshi + calendar", base)):
        for name, p in predict_all(tr).items():
            P[arm].setdefault(name, np.full(len(y), np.nan))[te] = p[te]

for lab, m in (("Kalshi-release test cells", test_mask & (kind == "kalshi")),
               ("calendar-release test cells", test_mask & (kind == "calendar")),
               ("all test cells", test_mask)):
    print(f"\n{'=' * 100}\n{lab}: {int(m.sum())} cells, horizon {A.HORIZON}\n{'=' * 100}")
    print(f"{'model':28} | {'Kalshi only: bal   AUC':>22} | {'+ calendar: bal   AUC':>22} | "
          f"{'ΔAUC (+calendar) [95% CI]':>26} | {'ΔAUC vs theory rule, + cal.':>28}")
    for name in P["Kalshi + calendar"]:
        rk = A.scores(y[m], P["Kalshi only"][name][m])
        ra = A.scores(y[m], P["Kalshi + calendar"][name][m])
        dl = dt = ""
        if name != "theory rule":
            d, lo, hi = A.boot_delta(m, P["Kalshi only"][name], P["Kalshi + calendar"][name])
            dl = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            d, lo, hi = A.boot_delta(m, P["Kalshi + calendar"]["theory rule"], P["Kalshi + calendar"][name])
            dt = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"{name:28} | {rk[1]:>14.3f} {rk[2]:>6.3f} | {ra[1]:>14.3f} {ra[2]:>6.3f} | {dl:>26} | {dt:>28}")

# full-sample channel coefficients, Kalshi + calendar, SE clustered by instant
Xc = np.column_stack([np.ones(len(y)), CHS])
yc = np.clip(y, -4, 4)
inv = np.linalg.pinv(Xc.T @ Xc)
beta = inv @ Xc.T @ yc
e = yc - Xc @ beta
u, gi = np.unique(tc, return_inverse=True)
S = np.zeros((Xc.shape[1],) * 2)
for k in range(len(u)):
    mk = gi == k
    v = Xc[mk].T @ e[mk]
    S += np.outer(v, v)
V = inv @ S @ inv * len(u) / (len(u) - 1)
se = np.sqrt(np.diag(V))
nk = [(np.abs(CHS[kind == "kalshi", g]) > 0).sum() for g in range(len(GROUPS))]
nc = [(np.abs(CHS[kind == "calendar", g]) > 0).sum() for g in range(len(GROUPS))]
print(f"\n{'=' * 100}\nfull-sample channel coefficients (Kalshi + calendar), OLS, SE clustered by instant\n"
      f"+ = the channel moves in the theory direction\n{'=' * 100}")
print(f"{'channel':24} {'cells: Kalshi':>13} {'calendar':>9} {'coef':>8} {'t':>6}")
for g, ch in enumerate(GROUPS):
    print(f"{ch:24} {nk[g]:>13} {nc[g]:>9} {beta[g + 1]:>+8.4f} {beta[g + 1] / se[g + 1]:>+6.2f}")
pos = sum(beta[g + 1] > 0 for g in range(len(GROUPS)))
sig = sum(beta[g + 1] / se[g + 1] > 2 for g in range(len(GROUPS)))
print(f"{pos} of {len(GROUPS)} channels positive (theory direction); {sig} with t > 2; "
      f"{sum(beta[g + 1] / se[g + 1] < -2 for g in range(len(GROUPS)))} with t < −2")
