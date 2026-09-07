"""Stress-test the pooled sign result for two ways my own design could inflate it.

(1) The permutation shuffled surprises WITHIN each trigger independently, which
    destroys the real cross-trigger correlation (CPI/CPIYOY come from one BLS
    release, r=0.686; PAYROLLS/U3 come from one jobs report). A null with less
    cross-cell co-movement than reality has too little variance => p too small.
    Fix: BLOCK-permute by release date, so all triggers move together.

(2) For one FOMC event, P(hike) and P(cut) are near-complementary, and I gave
    them opposite expected signs -- so those two cells may be near-duplicates,
    doubling apparent n without adding information.
    Fix: report a deduplicated variant with one contract per (trigger, event).
"""
import sys, datetime as dt, numpy as np, polars as pl
sys.path.insert(0, "/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
sys.path.insert(0, "stg_infra")
from sign_diagnostics import build, spearman, pooled_stats, SPECS, N_PERM

RNG = np.random.default_rng(1)

# ---------- first: how complementary ARE hike and cut? ----------
print("=== how redundant are the hike/cut cells? ===")
for horizon in ["dormant", "liquid"]:
    h = build("CPI", "FEDDECISION", "hike", +1, horizon)
    c = build("CPI", "FEDDECISION", "cut", -1, horizon)
    if h and c:
        eh, Sh, Rh = h; ec, Sc, Rc = c
        common = sorted(set(eh) & set(ec))
        rh = {e: r for e, r in zip(eh, Rh)}; rc = {e: r for e, r in zip(ec, Rc)}
        a = np.array([rh[e] for e in common]); b = np.array([rc[e] for e in common])
        if len(a) > 3:
            corr = float(np.corrcoef(a, b)[0, 1])
            print(f"  {horizon}: corr(R_hike, R_cut) = {corr:+.3f} over n={len(a)} shared events"
                  f"   ({'near-duplicate' if corr < -0.7 else 'largely independent'})")

# ---------- build all cells once, per horizon ----------
def assemble(horizon, dedup):
    cells = []
    for trig, tgt, side, sign in SPECS:
        if dedup and side == "cut":
            continue          # keep one FEDDECISION side only
        out = build(trig, tgt, side, sign, horizon)
        if out is None or len(out[0]) < 8:
            continue
        cells.append((trig, tgt, side, out))
    return cells

def date_of(ev):
    """Release-period key shared by triggers from the same report."""
    parts = ev.replace("KX", "").split("-")
    return parts[1] if len(parts) > 1 else ev

def run(horizon, dedup):
    cells = assemble(horizon, dedup)
    datasets = [(o[1], o[2]) for _, _, _, o in cells]
    obs_sign, obs_rho, n_tot = pooled_stats(datasets)

    # block permutation: permute release-period labels GLOBALLY, so every
    # trigger's surprise for a given report moves together.
    periods = sorted({date_of(e) for _, _, _, o in cells for e in o[0]})
    null_sign, null_rho = [], []
    for _ in range(N_PERM):
        shuffled = RNG.permutation(periods)
        pmap = dict(zip(periods, shuffled))
        # within each trigger, remap event -> surprise of the permuted period
        pdatasets = []
        for trig, tgt, side, (ev, S, R) in cells:
            by_period = {}
            for e, s in zip(ev, S):
                by_period.setdefault(date_of(e), s)
            Sp = np.array([by_period.get(pmap[date_of(e)], np.nan) for e in ev], float)
            ok = ~np.isnan(Sp)
            if ok.sum() >= 3:
                pdatasets.append((Sp[ok], R[ok]))
        s_, r_, _ = pooled_stats(pdatasets)
        if not np.isnan(s_):
            null_sign.append(s_); null_rho.append(r_)
    null_sign = np.array(null_sign); null_rho = np.array(null_rho)
    p_sign = float((null_sign >= obs_sign).mean())
    p_rho = float((null_rho >= obs_rho).mean())
    tag = "DEDUP (hike/FED only)" if dedup else "ALL CELLS"
    print(f"\n  [{horizon} / {tag}]  cells={len(cells)}  n={n_tot}")
    print(f"    observed : sign={100*obs_sign:.1f}%  Spearman={obs_rho:.3f}")
    print(f"    block-perm null: sign={100*null_sign.mean():.1f}% (sd {100*null_sign.std():.1f}%)"
          f"   rho={null_rho.mean():+.3f} (sd {null_rho.std():.3f})")
    print(f"    ONE-SIDED p: sign={p_sign:.4f}   Spearman={p_rho:.4f}")

print("\n=== block-permutation (preserves cross-trigger correlation) ===")
for horizon in ["dormant", "liquid"]:
    for dedup in [False, True]:
        run(horizon, dedup)
