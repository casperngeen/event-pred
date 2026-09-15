#!/usr/bin/env python
"""Hold from trigger resolution to target close: what does the signal earn?

    venv/bin/python analysis/leadlag_2026_09/economics.py

``signal_model.py`` leaves a tension that only this can resolve. The model-free
test says the signal separates outcomes (+3.80pp top tercile minus bottom,
block-permutation p = 0.0005), but tilting the market's quote by it does not
improve Brier or log loss. Both can be true: a proper scoring rule is dominated
by the thousands of near-certain legs, where the signal is silent and a tilt
only adds noise, while the separation lives in a few hundred mid-priced legs.

P&L asks the question the other way round -- you do not have to price every leg,
only the ones you choose to trade -- and it is the object the research question
is actually about.

The rule
--------
Fixed in advance, per price bucket:

* ``signal`` in the **top** tercile -> buy YES at ``p_entry``
* ``signal`` in the **bottom** tercile -> sell YES (buy NO at ``100 - p_entry``)
* middle tercile -> no position

Hold to settlement. One crossing, no exit fee, because settlement is not a
trade. Terciles are cut **within the price bucket**, so the rule never uses the
price level to decide direction.

Two versions, because in-sample terciles are a free parameter:

* **in-sample** terciles, reported to show the shape;
* **walk-forward**, terciles cut on prior years only -- the honest number.

Return on capital
-----------------
Kalshi collateralises at full notional, so capital is ``p_entry`` for a YES and
``100 - p_entry`` for a NO. A 1c edge on a 5c contract is a 20% return; the same
1c on a 95c contract is 1%. Reported as ``ret_cap``, and as ``ret_cap_yr``,
annualised over ``gap_days`` -- the plan's edge-per-capital-day, which is the
metric that stops a book filling with correct-but-slow positions.

Everything clusters on ``target_event``.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/leadlag_2026_09/out")
N_BOOT = 10000
FEE_RATE, CONTRACTS = 0.07, 100

BUCKETS = [(1, 5), (5, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99)]


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(price_cents) -> np.ndarray:
    """Measured round-trip effective spread by moneyness (research_log §11.2)."""
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cluster_boot(vals: np.ndarray, groups: np.ndarray, seed: int = 0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s = np.bincount(inv, weights=vals, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


def price_legs(px, win, side):
    """Net cents and capital for taking `side` (+1 YES, -1 NO) at `px`."""
    entry = np.where(side > 0, px, 100.0 - px)
    payoff = np.where(side > 0, 100.0 * win, 100.0 * (1 - win))
    gross = payoff - entry
    cost = fee_cents(entry) + spread_cents(entry) / 2.0
    return gross - cost, entry, gross


def run(d: pl.DataFrame, sig: np.ndarray, walk_forward: bool, label: str):
    px = d["p_entry"].to_numpy()
    win = d["win"].to_numpy().astype(float)
    ev = d["target_event"].to_numpy()
    yr = d["yr"].to_numpy()
    gap = np.maximum(d["gap_days"].to_numpy(), 0.5)
    years = sorted(np.unique(yr))

    side = np.zeros(len(px))
    for lo, hi in BUCKETS:
        m = (px >= lo) & (px < hi)
        if m.sum() < 60:
            continue
        if not walk_forward:
            t = np.percentile(sig[m], [33.3, 66.7])
            side[m] = np.where(sig[m] > t[1], 1.0,
                               np.where(sig[m] <= t[0], -1.0, 0.0))
        else:
            idx = np.where(m)[0]
            for Y in years[1:]:
                tr = idx[yr[idx] < Y]
                te = idx[yr[idx] == Y]
                if len(tr) < 60 or len(te) == 0:
                    continue
                t = np.percentile(sig[tr], [33.3, 66.7])
                side[te] = np.where(sig[te] > t[1], 1.0,
                                    np.where(sig[te] <= t[0], -1.0, 0.0))

    taken = side != 0
    net, cap, gross = price_legs(px[taken], win[taken], side[taken])
    sub_px, sub_ev, sub_gap = px[taken], ev[taken], gap[taken]
    sub_side = side[taken]

    print(f"\n=== {label} ===")
    print(f"positions {taken.sum()}   target events {len(np.unique(sub_ev))}\n")
    rows = []
    for lo, hi in BUCKETS:
        m = (sub_px >= lo) & (sub_px < hi)
        if m.sum() < 40:
            continue
        obs, clo, chi, pneg = cluster_boot(net[m], sub_ev[m])
        rc = net[m] / cap[m]
        rows.append(dict(band=f"{lo}-{hi}c", n=int(m.sum()),
                         n_ev=int(len(np.unique(sub_ev[m]))),
                         n_long=int((sub_side[m] > 0).sum()),
                         mean_cap=float(cap[m].mean()),
                         gross=float(gross[m].mean()), net=obs,
                         ci_lo=clo, ci_hi=chi, p_le0=pneg,
                         ret_cap=100.0 * float(rc.mean()),
                         ret_cap_yr=100.0 * float((rc * 365.0 / sub_gap[m]).mean())))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=240):
        print(pl.DataFrame(rows))

    obs, clo, chi, pneg = cluster_boot(net, sub_ev)
    print(f"ALL: net {obs:+.2f}c  CI [{clo:+.2f}, {chi:+.2f}]  P(<=0) = {pneg:.3f}"
          f"   mean capital {cap.mean():.1f}c   ret_cap {100*np.mean(net/cap):+.2f}%")

    # long vs short
    print("\nby side:")
    rows = []
    for s, nm in [(1.0, "long YES"), (-1.0, "short YES")]:
        m = sub_side == s
        if m.sum() < 40:
            continue
        obs, clo, chi, pneg = cluster_boot(net[m], sub_ev[m])
        rows.append(dict(side=nm, n=int(m.sum()),
                         n_ev=int(len(np.unique(sub_ev[m]))),
                         mean_entry=float(cap[m].mean()), net=obs,
                         ci_lo=clo, ci_hi=chi, p_le0=pneg,
                         ret_cap=100.0 * float(np.mean(net[m] / cap[m]))))
    with pl.Config(float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    return pl.DataFrame(dict(
        net=net, cap=cap, gross=gross, px=sub_px, ev=sub_ev,
        gap=sub_gap, side=sub_side, yr=yr[taken]))


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    z = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z), 99))
    sig = d["direction"].to_numpy() * np.clip(z, -lim, lim)

    run(d, sig, walk_forward=False, label="in-sample terciles (shape only)")
    pos = run(d, sig, walk_forward=True,
              label="walk-forward terciles (the honest number)")
    net = pos["net"].to_numpy()

    # ------------------------------------------------------------- by year
    print("\n=== walk-forward, by year ===")
    rows = []
    for Y in sorted(pos["yr"].unique().to_list()):
        s = pos.filter(pl.col("yr") == Y)
        if s.height < 40:
            continue
        obs, clo, chi, pneg = cluster_boot(s["net"].to_numpy(), s["ev"].to_numpy())
        rows.append(dict(year=int(Y), n=s.height,
                         n_ev=s["ev"].n_unique(), net=obs,
                         ci_lo=clo, ci_hi=chi, p_le0=pneg,
                         ret_cap=100.0 * float((s["net"] / s["cap"]).mean())))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    obs, clo, chi, pneg = cluster_boot(net, pos["yr"].to_numpy(), seed=2)
    print(f"year-clustered: {obs:+.2f}c  CI [{clo:+.2f}, {chi:+.2f}]  "
          f"P(<=0) = {pneg:.3f}")

    # ------------------------------------------------- the metric conflict
    print("\n=== the two metrics disagree, and both are right ===")
    print("Net cents weights every position equally. Return on capital weights")
    print("by how much money each one ties up. Cheap legs lose here and tie up")
    print("almost nothing, so they barely dent the cent total and dominate the")
    print("percentage. Which number is 'the' answer depends on whether your")
    print("constraint is positions or capital.\n")
    cap_a = pos["cap"].to_numpy()
    rows = [
        dict(metric="mean net cents / position", value=float(net.mean())),
        dict(metric="mean return on capital %",
             value=100.0 * float((net / cap_a).mean())),
        dict(metric="capital-weighted return % (total net / total capital)",
             value=100.0 * float(net.sum() / cap_a.sum())),
        dict(metric="mean net cents, legs >= 10c only",
             value=float(net[pos["px"].to_numpy() >= 10].mean())),
        dict(metric="mean return on capital %, legs >= 10c only",
             value=100.0 * float((net[pos["px"].to_numpy() >= 10]
                                  / cap_a[pos["px"].to_numpy() >= 10]).mean())),
    ]
    with pl.Config(float_precision=3, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    # ------------------------------------------------- permutation on the P&L
    print("=== block-permutation null on the walk-forward P&L ===")
    print("Shuffle z_surprise among trigger events within each trigger series,")
    print("rebuild the signal, re-run the whole walk-forward rule. This is the")
    print("same null signal_model.py uses, applied to cents instead of points.\n")
    trig_ev = d["trigger_event"].to_numpy()
    trig_sr = d["trigger"].to_numpy()
    direction = d["direction"].to_numpy().astype(float)
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    te_z = np.zeros(len(uniq_te))
    te_series = np.empty(len(uniq_te), dtype=object)
    for i, te in enumerate(uniq_te):
        j = int(np.argmax(trig_ev == te))
        te_z[i] = np.clip(z[j], -lim, lim)
        te_series[i] = trig_sr[j]
    groups = [np.where(te_series == s)[0] for s in np.unique(te_series)]

    rng = np.random.default_rng(0)
    obs_mean = float(net.mean())
    null = np.empty(400)
    import io
    import contextlib
    for b in range(400):
        zp = te_z.copy()
        for g in groups:
            zp[g] = te_z[rng.permutation(g)]
        sp = direction * zp[row_te]
        with contextlib.redirect_stdout(io.StringIO()):
            pb = run(d, sp, walk_forward=True, label="perm")
        null[b] = float(pb["net"].mean())
    print(f"observed {obs_mean:+.3f}c   null {null.mean():+.3f} +/- {null.std():.3f}"
          f"   p(one-sided) = {(null >= obs_mean).mean():.4f}")


if __name__ == "__main__":
    main()
