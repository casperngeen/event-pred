"""Why did AGCRN (and every rung) fail to beat predict-zero?

Six diagnostics, all in-sample. Run from event-pred/:
    venv/bin/python ../analysis/agcrn_diagnostics_2026_09/postmortem.py

  1  Target predictability ceiling  — can an oracle beat zero at all?
  2  Overlapping-window autocorrelation — is the "temporal structure" an artifact?
  3  Signal concentration           — what share of labelled cells is trigger-adjacent?
  4  Conditional predictability     — restrict eval to trigger-adjacent cells
  5  Sign vs magnitude              — the Stage-1 signal is sign-only; the target is not
  6  Snapshot cadence               — how stale is the sequence a node's GRU sees?
"""
from __future__ import annotations
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.models import build_tensor, build_labels, sequence_windows
from stg.models.tensors import global_label_sd
from stg.models.baselines import _stage1_in_edges
from stg.panel.registry import same_release_pairs

NP = "artifacts/panels/node_panel_event.parquet"
SP = "artifacts/panels/surprise_panel.parquet"


def rule(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


pt = build_tensor(pl.read_parquet(NP))
surprise = pl.read_parquet(SP)
A_s1 = _stage1_in_edges(pt.nodes, survivors_only=False)      # (N,N) signed rho, all searched
A_bh = _stage1_in_edges(pt.nodes, survivors_only=True)
ni = {n: i for i, n in enumerate(pt.nodes)}

# =====================================================================
rule("1  TARGET PREDICTABILITY CEILING")
print("R2 vs zero for oracles that CANNOT be beaten by any causal model.\n")
print(f"{'k':>3} {'n':>6} {'|z| med':>8} {'oracle: per-node mean':>22} "
      f"{'oracle: expanding mean':>24} {'oracle: AR(1) fit':>18}")
for k in (1, 2, 3, 5):
    Y, lm = build_labels(pt, k=k, kind="belief_z")
    sd = global_label_sd(Y, lm, "belief_z")
    z = Y / sd[None]
    t_idx, n_idx = np.where(lm)
    yv = z[lm]
    sse0 = (yv ** 2).sum()

    # (a) per-node global mean (leaks future — hard upper bound)
    gm = np.array([z[lm[:, i], i].mean() if lm[:, i].any() else 0.0 for i in range(pt.N)])
    r2_gm = 1 - ((yv - gm[n_idx]) ** 2).sum() / sse0

    # (b) causal expanding per-node mean (no leak)
    pred_b = np.zeros_like(yv)
    for i in range(pt.N):
        s = z[lm[:, i], i]
        csum = np.cumsum(s)
        run = np.concatenate([[0.0], csum[:-1] / np.maximum(np.arange(1, len(s)), 1)])
        pred_b[n_idx == i] = run
    r2_b = 1 - ((yv - pred_b) ** 2).sum() / sse0

    # (c) causal AR(1) on the per-node label series, expanding fit
    r2_c_num = 0.0
    for i in range(pt.N):
        s = z[lm[:, i], i]
        if len(s) < 20:
            r2_c_num += (s ** 2).sum()
            continue
        pr = np.zeros(len(s))
        for j in range(10, len(s)):
            x, y = s[:j - 1], s[1:j]
            b = np.polyfit(x, y, 1)
            pr[j] = np.polyval(b, s[j - 1])
        r2_c_num += ((s - pr) ** 2).sum()
    r2_c = 1 - r2_c_num / sse0

    print(f"{k:>3} {lm.sum():>6} {np.median(np.abs(yv)):>8.2f} "
          f"{r2_gm:>+22.4f} {r2_b:>+24.4f} {r2_c:>+18.4f}")
print("\nIf even the leaky per-node-mean oracle is ~0, the target has no "
      "predictable central tendency and no model can score positive R2.")

# =====================================================================
rule("2  OVERLAPPING-WINDOW AUTOCORRELATION")
print("k-step-ahead Δ at consecutive snapshots shares k-1 steps, so it is\n"
      "mechanically autocorrelated. That looks like 'temporal structure' but\n"
      "carries no predictive information — the same trap as the CA report's\n"
      "consecutive-day rule (research_summary.md §3).\n")
for k in (1, 2, 3, 5):
    Y, lm = build_labels(pt, k=k, kind="belief_z")
    sd = global_label_sd(Y, lm, "belief_z")
    z = Y / sd[None]
    acs = []
    for i in range(pt.N):
        s = z[lm[:, i], i]
        if len(s) > 15:
            acs.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    # non-overlapping subsample (every k-th window)
    acs_no = []
    for i in range(pt.N):
        s = z[lm[:, i], i][::k]
        if len(s) > 15:
            acs_no.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    print(f"k={k}: lag-1 autocorr  overlapping={np.nanmedian(acs):+.3f}   "
          f"non-overlapping (stride k)={np.nanmedian(acs_no):+.3f}")

# =====================================================================
rule("3  SIGNAL CONCENTRATION — how much of the panel is trigger-adjacent?")
sp_by_series = {s: surprise.filter(pl.col("series") == s).sort("close_time")
                for s in surprise["series"].unique()}
res_dates = {s: df["close_time"].dt.date().to_numpy().astype("datetime64[D]")
             for s, df in sp_by_series.items()}

Y3, lm3 = build_labels(pt, k=3, kind="belief_z")
sd3 = global_label_sd(Y3, lm3, "belief_z")
z3 = Y3 / sd3[None]

for W in (5, 10, 21):
    adj = np.zeros_like(lm3)
    for j, nid in enumerate(pt.nodes):
        # in-neighbours with a searched (non-zero) Stage-1 edge into this node
        ins = [pt.nodes[i] for i in range(pt.N) if A_s1[i, j] != 0]
        for i, d in enumerate(pt.dates):
            if not lm3[i, j]:
                continue
            for src in ins:
                rd = res_dates.get(src)
                if rd is not None and ((d - rd >= np.timedelta64(0, "D")) &
                                       (d - rd <= np.timedelta64(W, "D"))).any():
                    adj[i, j] = True
                    break
    on = adj[lm3]
    q = z3[lm3][on]
    quiet = z3[lm3][~on]
    print(f"W={W:2d}d: trigger-adjacent cells = {on.sum():5d} / {lm3.sum()} "
          f"({100*on.mean():.1f}%)   |z| adjacent={np.median(np.abs(q)):.2f}  "
          f"quiet={np.median(np.abs(quiet)):.2f}   var ratio={q.var()/quiet.var():.2f}")

# keep W=10 adjacency for section 4/5
W = 10
adj10 = np.zeros_like(lm3)
for j, nid in enumerate(pt.nodes):
    ins = [pt.nodes[i] for i in range(pt.N) if A_s1[i, j] != 0]
    for i, d in enumerate(pt.dates):
        if not lm3[i, j]:
            continue
        for src in ins:
            rd = res_dates.get(src)
            if rd is not None and ((d - rd >= np.timedelta64(0, "D")) &
                                   (d - rd <= np.timedelta64(W, "D"))).any():
                adj10[i, j] = True
                break

# =====================================================================
rule("4  CONDITIONAL PREDICTABILITY — restrict to trigger-adjacent cells")
print("A surprise-weighted linear predictor (Stage-1 rho x recent trigger\n"
      "surprise), walk-forward, scored ONLY on trigger-adjacent cells vs all.\n")

# build per-cell 'incoming surprise signal' = sum_i rho[i,j] * (last surprise of i within W days, aligned)
sig = np.zeros_like(z3)
for j, nid in enumerate(pt.nodes):
    for i_src in range(pt.N):
        rho = A_bh[i_src, j]
        if rho == 0:
            continue
        src = pt.nodes[i_src]
        df = sp_by_series.get(src)
        if df is None:
            continue
        rd = df["close_time"].dt.date().to_numpy().astype("datetime64[D]")
        sv = df["surprise"].to_numpy()
        ssd = np.std(sv) or 1.0
        for i_t, d in enumerate(pt.dates):
            if not lm3[i_t, j]:
                continue
            m = (d - rd >= np.timedelta64(0, "D")) & (d - rd <= np.timedelta64(W, "D"))
            if m.any():
                sig[i_t, j] += rho * np.sign(sv[m][-1])   # sign-only, aligned by rho sign

from stg.models.train import _fold_cuts, PURGE
dates = pt.dates
cuts = _fold_cuts(dates, 8)
pred = np.full_like(z3, np.nan)
for f in range(8):
    tr = (dates < (cuts[f] - PURGE))[:, None] & lm3
    te = ((dates >= cuts[f]) & (dates < cuts[f + 1]))[:, None] & lm3
    if tr.sum() < 50 or te.sum() == 0:
        continue
    x, y = sig[tr], z3[tr]
    if x.std() < 1e-9:
        b = [0, 0]
    else:
        b = np.polyfit(x, y, 1)
    pred[te] = np.polyval(b, sig[te])

for label, sub in [("all labelled cells", lm3 & np.isfinite(pred)),
                   ("trigger-adjacent only", adj10 & np.isfinite(pred)),
                   ("quiet cells only", lm3 & ~adj10 & np.isfinite(pred))]:
    yv, pv = z3[sub], pred[sub]
    r2 = 1 - ((yv - pv) ** 2).sum() / (yv ** 2).sum()
    da = np.mean(np.sign(pv[pv != 0]) == np.sign(yv[pv != 0])) if (pv != 0).any() else np.nan
    print(f"{label:24}  n={sub.sum():5d}  R2 vs zero={r2:+.4f}  dir acc={da:.3f}")

# =====================================================================
rule("5  SIGN vs MAGNITUDE on trigger-adjacent cells")
print("Stage-1's signal is in the DIRECTION of the surprise, not its size\n"
      "(research_log.md §2). A magnitude regression dilutes a sign-only signal\n"
      "toward zero — so the model can be 'right' on sign and still score R2<0.\n")
sub = adj10 & (sig != 0)
yv = z3[sub]
sv = sig[sub]
print(f"trigger-adjacent, non-zero incoming signal: n={sub.sum()}")
print(f"  sign agreement (sign of incoming signal vs sign of Δ) : "
      f"{np.mean(np.sign(sv) == np.sign(yv)):.3f}")
print(f"  corr(|signal|, |Δz|)                                  : "
      f"{np.corrcoef(np.abs(sv), np.abs(yv))[0,1]:+.3f}")
print(f"  corr(signal, Δz)  (signed)                            : "
      f"{np.corrcoef(sv, yv)[0,1]:+.3f}")
print(f"  median |Δz| when signal!=0 vs overall                 : "
      f"{np.median(np.abs(yv)):.2f}  vs  {np.median(np.abs(z3[lm3])):.2f}")

# =====================================================================
rule("6  SNAPSHOT CADENCE — how stale is a node's own sequence?")
print("Event-driven snapshots fire on ANY macro resolution. For a given node\n"
      "most steps carry no news about it; the GRU sees a mostly-flat sequence.\n")
gaps_all = []
for j, nid in enumerate(pt.nodes):
    ts = np.where(pt.mask[:, j])[0]
    if len(ts) < 5:
        continue
    g = np.diff(pt.dates[ts]).astype("timedelta64[D]").astype(int)
    own_res = res_dates.get(nid)
    frac_news = 0.0
    if own_res is not None:
        frac_news = np.mean([np.any(np.abs(pt.dates[t] - own_res) <= np.timedelta64(3, "D"))
                             for t in ts])
    print(f"  {nid:14} snapshots={len(ts):3d}  median gap={np.median(g):4.0f}d  "
          f"p90 gap={np.percentile(g,90):4.0f}d  frac near own release={frac_news:.2f}")
    gaps_all.extend(g)
print(f"\n  overall: median inter-snapshot gap {np.median(gaps_all):.0f}d, "
      f"an L=12 window spans ~{12*np.median(gaps_all):.0f}d of very uneven time")

# =====================================================================
rule("7  HORIZON — where the Stage-1 signal actually lives")
print("The SAME trigger->target pairs, scored at the dormant horizon (next 3\n"
      "trades, ~0.5h) vs the snapshot horizon the model predicts (~weeks).\n")
from stg.panel.targets import response_panel
edges = pl.read_parquet("artifacts/adjacency_is.parquet").filter(pl.col("survives"))
rows = []
for r in edges.iter_rows(named=True):
    sp_t = surprise.filter(pl.col("series") == r["trigger"])
    for hz in ("dormant", "liquid"):
        rp = response_panel(sp_t, r["target"], r["side"], horizon=hz)
        if rp.height >= 8:
            s = np.sign(rp["surprise"].to_numpy()) * np.sign(r["rho"])
            resp = rp["response"].to_numpy()
            m = (s != 0) & (resp != 0)
            rows.append(dict(edge=f'{r["trigger"]}->{r["target"]}/{r["side"]}',
                             horizon=hz, n=int(m.sum()),
                             sign_agree=float((s[m] * np.sign(resp[m]) > 0).mean()),
                             corr_mag=float(np.corrcoef(np.abs(rp["surprise"].to_numpy()),
                                                        np.abs(resp))[0, 1])))
d = pl.DataFrame(rows)
for hz in ("dormant", "liquid"):
    sub = d.filter(pl.col("horizon") == hz)
    w = np.array(sub["n"]); sa = np.array(sub["sign_agree"])
    print(f"  {hz:8}: pooled sign agreement {100*np.average(sa, weights=w):.1f}%  "
          f"(n={w.sum()})   mean corr(|surp|,|resp|) {sub['corr_mag'].mean():+.2f}")
print("\n  vs section 5: at the snapshot horizon the model predicts, sign\n"
      "  agreement on the same mechanism is ~52% (coin flip) and the magnitude\n"
      "  correlation is ~0. The signal has decayed out of the window.")