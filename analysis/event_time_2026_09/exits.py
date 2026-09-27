#!/usr/bin/env python
"""Buy, then sell once the surprise is captured — taker vs maker, both legs.

    venv/bin/python analysis/event_time_2026_09/exits.py \
        > analysis/event_time_2026_09/out/exits.txt      # needs build_panel.py, models.py

``backtest.py`` holds to settlement. That bets the target is *mispriced*; this
bets it *reprices*. Same candidate trades, same sides.

Exits (position value ``v`` = YES price for a long, 100 − price for a short)
---------------------------------------------------------------------------
capture  the print at or after the target's 3rd post-release print (≤24h) — the
         window the ``imm`` label measures, where the repricing lives — and
         strictly after entry
24h      the last print in (entry, release + 24h]; the next one if none
settle   hold: v becomes the 0/100 payoff, no exit cost
If no print follows entry before the contract closes, the position settles.

Execution
---------
taker entry  cross at the first post-release print: fee + half spread
maker entry  rest a limit at the last pre-release price from the release; it
             fills if a print trades at or through it within 24h (fill at the
             limit; "strict" counts only prints through it). Unfilled = no
             trade. Adverse selection is built in: when
             the target gaps your way you are not filled (``maker_fill.py``)
taker exit   cross at the exit print: fee + half spread
maker exit   rest at the exit print's price; fills if a later print within 24h
             trades at or through it, else cross as taker at the last print in
             that window (or settle if the market goes dark)
Taker fee ``ceil(0.07·C·P(1−P))``, maker fee ``0.0175·C·P(1−P)`` (rounded the
same way), 100-contract tickets; half the measured spread per taker crossing
(``leadlag_2026_09/economics.py``). Queue position is ignored — optimistic for
the maker, which is reported with fee bounds of 0 and the full taker fee.

Strategies: the labour→policy rule and the a-priori all-edge rule over all
releases; the walk-forward linear econ model, the intercept-only control and
always-NO on the test folds (see ``backtest.py`` for why those controls). Net is
cents per contract with a 95% CI from a bootstrap over release instants.
In-sample only.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.panel._io import scan_trades
from stg.panel.registry import is_same_release

OUT = "analysis/event_time_2026_09/out"
TAKER_RATE, CONTRACTS, N_BOOT = 0.07, 100, 10000
MAKER_RATES = {"maker fee 0.0175": 0.0175, "maker fee 0": 0.0, "maker fee = taker": 0.07}
K_CAPTURE, CAP_NS = 3, 24 * 3_600_000_000_000
W_NS = 24 * 3_600_000_000_000                         # resting window, both legs

HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
TYPE = {**{s: "inflation" for s in ("CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "CPIGAS",
                                    "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL",
                                    "PCECORE")},
        **{s: "labour" for s in ("PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP")},
        "GDP": "growth", "ISMPMI": "growth", "FED": "policy"}


def fee(v, rate):
    p = np.asarray(v, dtype=float) / 100.0
    return np.ceil(rate * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def half_spread(v):
    """Half the measured round-trip effective spread (2c, 1c beyond 10/90c)."""
    return 0.5 if abs(float(v) - 50.0) > 40.0 else 1.0


def cboot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(u), size=(N_BOOT, len(u)))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return np.percentile(b, 2.5), np.percentile(b, 97.5), float((b <= 0).mean())


# ------------------------------------------------------------------ data
panel = pl.read_parquet(f"{OUT}/event_nodes.parquet").with_columns(
    pl.col("instant").dt.replace_time_zone(None))
rel = {r["instant"]: dict(zip(r["series"], r["z"])) for r in
       panel.filter(pl.col("released")).group_by("instant")
       .agg(pl.col("series"), pl.col("z")).iter_rows(named=True)}
cand = panel.filter(pl.col("p_entry").is_not_null())


def signal(inst, tgt, channels=None):
    return sum(HAWKISH[a] * HAWKISH[tgt] * z for a, z in rel[inst].items()
               if a != tgt and not is_same_release(a, tgt)
               and (channels is None or (TYPE[a], TYPE[tgt]) in channels))


raw = (scan_trades(is_only=True)
       .filter(pl.col("ticker").is_in(cand["lead_ticker"].unique().to_list()))
       .select("trade_id", "ticker", "yes_price", "count", "created_time").collect()
       .unique(subset=["trade_id", "ticker", "created_time", "yes_price", "count"]))
col = (raw.group_by("ticker", "created_time")
       .agg(((pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum()).alias("px"))
       .with_columns(pl.col("created_time").dt.epoch("ns").alias("ns"))
       .sort("ticker", "ns").group_by("ticker", maintain_order=True)
       .agg(pl.col("ns"), pl.col("px")))
TAPE = {t: (np.asarray(a, np.int64), np.asarray(b, float))
        for t, a, b in zip(col["ticker"], col["ns"], col["px"])}

# ------------------------------------------------------------------ one trade
def simulate(r: dict, side: int, entry: str, exit_: str, horizon: str, maker_rate: float,
             strict: bool = False):
    """(net, hours held, gross) in cents per contract; None if the maker entry
    never fills. gross = exit value − entry value, before any fee or spread."""
    ns, px = TAPE[r["lead_ticker"]]
    t = int(np.datetime64(r["instant"], "ns").astype(np.int64))
    close = int(r["settle_end"])
    val = (lambda p: p) if side > 0 else (lambda p: 100.0 - p)
    j0 = int(np.searchsorted(ns, t, side="right"))
    if j0 >= len(ns) or ns[j0] >= close:
        return None, None, None
    live = int(np.searchsorted(ns, close, side="left"))     # prints before close: [0, live)

    def upto(t_end):                                          # prints in [.., t_end]
        return min(int(np.searchsorted(ns, t_end, side="right")), live)

    if entry == "taker":
        e, ve = j0, val(px[j0])
        cost = fee(ve, TAKER_RATE) + half_spread(ve)
    else:
        L, hi = r["p_lead"], upto(t + W_NS)
        seg = px[j0:hi]
        # strict: only a print *through* the limit counts (you may be behind the queue at L)
        ok = np.nonzero((seg < L if strict else seg <= L) if side > 0
                        else (seg > L if strict else seg >= L))[0]
        if not ok.size:
            return None, None, None
        e, ve = j0 + int(ok[0]), val(L)
        cost = fee(ve, maker_rate)
    payoff = 100.0 * r["win"] if side > 0 else 100.0 * (1 - r["win"])
    settle = (payoff - ve - cost, (close - ns[e]) / 3.6e12, payoff - ve)

    if horizon == "settle":
        return settle
    if horizon == "capture":
        last = min(j0 + K_CAPTURE, upto(t + CAP_NS)) - 1      # K-th post-release print, ≤24h
        tgt_ns = ns[last] if last >= j0 else ns[e]
        x = max(e + 1, int(np.searchsorted(ns, tgt_ns, side="left")))
    else:                                                     # "24h"
        hi = upto(t + CAP_NS)
        x = hi - 1 if hi - 1 > e else e + 1
    if x >= live:                                             # nothing prints again: settle
        return settle
    vx = val(px[x])
    if exit_ == "taker":
        return (vx - ve - cost - fee(vx, TAKER_RATE) - half_spread(vx), (ns[x] - ns[e]) / 3.6e12,
                vx - ve)
    # maker exit: rest at vx; fills if a later print trades at or through it
    hi = upto(ns[x] + W_NS)
    nxt = np.arange(x + 1, hi)
    if nxt.size:
        vals = px[nxt] if side > 0 else 100.0 - px[nxt]
        f = np.nonzero(vals >= vx)[0]
        if f.size:
            return (vx - ve - cost - fee(vx, maker_rate), (ns[nxt[f[0]]] - ns[e]) / 3.6e12,
                    vx - ve)
        vl = float(vals[-1])                                  # give up: cross at the last print
        return (vl - ve - cost - fee(vl, TAKER_RATE) - half_spread(vl), (ns[nxt[-1]] - ns[e]) / 3.6e12,
                vl - ve)
    return settle


# ------------------------------------------------------------------ strategies
oof = pl.read_parquet(f"{OUT}/oof_settle.parquet").with_columns(
    pl.col("instant").cast(pl.Datetime("us")))
test = cand.join(oof.select("instant", "series").unique(), on=["instant", "series"])
y_all = (100.0 * cand["win"].to_numpy() - cand["p_entry"].to_numpy())
end_all = cand["settle_end"].to_numpy().astype("datetime64[ns]")


def past_mean_side(d):
    out = []
    for tk in d["instant"].to_numpy().astype("datetime64[ns]"):
        m = end_all < tk - np.timedelta64(21, "D")
        out.append(np.sign(y_all[m].mean()) if m.sum() >= 30 else 0.0)
    return np.array(out)


econ = test.join(oof.filter(pl.col("model") == "linear econ signal only (all)")
                 .select("instant", "series", "pred_c"), on=["instant", "series"])
STRATS = [
    ("rule labour→policy (all releases)", cand,
     np.array([np.sign(signal(r["instant"], r["series"], {("labour", "policy")}))
               for r in cand.iter_rows(named=True)])),
    ("rule all edges, a priori (all releases)", cand,
     np.array([np.sign(signal(r["instant"], r["series"])) for r in cand.iter_rows(named=True)])),
    ("linear econ model (walk-forward test folds)", econ, np.sign(econ["pred_c"].to_numpy())),
    ("control: intercept only (test folds)", test, past_mean_side(test)),
    ("control: always NO (test folds)", test, -np.ones(test.height)),
]
PLANS = [("taker → hold to settle", "taker", None, "settle"),
         ("maker → hold to settle", "maker", None, "settle"),
         ("maker(strict) → hold", "maker!", None, "settle"),
         ("taker → taker @capture", "taker", "taker", "capture"),
         ("taker → maker @capture", "taker", "maker", "capture"),
         ("maker → taker @capture", "maker", "taker", "capture"),
         ("maker → maker @capture", "maker", "maker", "capture"),
         ("taker → taker @24h", "taker", "taker", "24h"),
         ("maker → maker @24h", "maker", "maker", "24h")]

print(f"lead contracts with a tape: {len(TAPE)}; maker fee {MAKER_RATES['maker fee 0.0175']} "
      f"central, bounds shown for maker→maker @capture")
hdr = (f"{'plan':26} {'trades':>6} {'fill':>5} {'hold h':>7} {'hit':>5} "
       f"{'gross':>7} {'costs':>6} {'net ¢':>7} {'95% CI (clustered)':>18} {'P(≤0)':>6} {'$ @100':>8}")
for sname, d, side in STRATS:
    m = side != 0
    rows = d.filter(pl.Series(m)).to_dicts()
    sd = side[m].astype(int)
    print(f"\n=== {sname}: {len(rows)} signals over {len({r['instant'] for r in rows})} releases")
    print(hdr); print("-" * len(hdr))
    for pname, en, ex, hz in PLANS:
        res = [simulate(r, s, en.rstrip("!"), ex, hz, MAKER_RATES["maker fee 0.0175"],
                        strict=en.endswith("!")) for r, s in zip(rows, sd)]
        ok = [i for i, (v, _, _) in enumerate(res) if v is not None]
        if not ok:
            print(f"{pname:26} no fills"); continue
        net = np.array([res[i][0] for i in ok])
        hold = np.array([res[i][1] for i in ok])
        gross = np.array([res[i][2] for i in ok])
        g = np.array([rows[i]["instant"] for i in ok])
        lo, hi, p0 = cboot(net, g)
        print(f"{pname:26} {len(ok):>6} {len(ok) / len(rows):>5.0%} {np.median(hold):>7.1f} "
              f"{np.mean(net > 0):>5.0%} {gross.mean():>+7.2f} {(gross - net).mean():>6.2f} "
              f"{net.mean():>+7.2f} [{lo:>+6.2f}, {hi:>+6.2f}] "
              f"{p0:>6.3f} {net.sum():>+8.0f}", flush=True)
    for fname, rate in MAKER_RATES.items():
        if rate == MAKER_RATES["maker fee 0.0175"]:
            continue
        res = [simulate(r, s, "maker", "maker", "capture", rate) for r, s in zip(rows, sd)]
        net = np.array([v for v, _, _ in res if v is not None])
        print(f"{'  maker→maker, ' + fname:26} {net.size:>6} {'':>5} {'':>7} {'':>5} "
              f"{'':>7} {'':>6} {net.mean():>+7.2f}")

print("\nnet: cents per contract after all fees and spreads; hit = share of trades")
print("with net > 0; hold h = median hours from entry fill to exit; $ @100 = total")
print("net dollars at 100 contracts per trade (Σ net ¢ × 100 / 100).")


# ================================================================== per channel
# Every channel of the a-priori graph traded on its own signal alone, all
# releases. A target cell can fall in several channels (one per releasing
# series' type). Permutation p for the taker-hold trade: shuffle surprise
# vectors across release instants (vectorised); BH across channels.
N_PERM = 5000
nodes = sorted(HAWKISH)
ni = {n: i for i, n in enumerate(nodes)}
insts = sorted(rel)
ii = {t: k for k, t in enumerate(insts)}
Zmat = np.zeros((len(insts), len(nodes)))
for t, zm in rel.items():
    for a, z in zm.items():
        Zmat[ii[t], ni[a]] = z
c_i = np.array([ii[t] for t in cand["instant"].to_list()])
c_j = np.array([ni[s] for s in cand["series"].to_list()])
pe, wn = cand["p_entry"].to_numpy(), cand["win"].to_numpy().astype(float)


def hold_net(side):
    entry = np.where(side > 0, pe, 100.0 - pe)
    payoff = np.where(side > 0, 100.0 * wn, 100.0 * (1 - wn))
    ent = np.asarray(entry)
    return payoff - ent - fee(ent, TAKER_RATE) - np.where(np.abs(ent - 50) > 40, 0.5, 1.0)


NET_YES, NET_NO = hold_net(np.ones(len(pe))), hold_net(-np.ones(len(pe)))


def channel_graph(ch):
    G = np.zeros((len(nodes), len(nodes)))
    for a in nodes:
        for b in nodes:
            if a != b and not is_same_release(a, b) and (TYPE[a], TYPE[b]) == ch:
                G[ni[a], ni[b]] = HAWKISH[a] * HAWKISH[b]
    return G


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[o] = np.minimum(q, 1)
    return out


chans = sorted({(TYPE[a], TYPE[b]) for a in nodes for b in nodes
                if a != b and not is_same_release(a, b) and Zmat[:, ni[a]].any()})
rng = np.random.default_rng(7)
perms = [rng.permutation(len(insts)) for _ in range(N_PERM)]
summary = []
cols = [p[0] for p in PLANS]
print("\n\n" + "=" * 100 + "\nPER CHANNEL — net ¢/contract by plan, all releases "
      "(maker fee 0.0175)\n" + "=" * 100)
short = {"taker → hold to settle": "T hold", "maker → hold to settle": "M hold",
         "maker(strict) → hold": "Ms hold", "taker → taker @capture": "TT cap",
         "taker → maker @capture": "TM cap", "maker → taker @capture": "MT cap",
         "maker → maker @capture": "MM cap", "taker → taker @24h": "TT 24h",
         "maker → maker @24h": "MM 24h"}
print(f"{'channel':22} {'n':>4} {'rel':>4} {'fill':>5} " + " ".join(f"{short[c]:>8}" for c in cols))
rows_all = cand.to_dicts()
for ch in chans:
    G = channel_graph(ch)
    sig = (Zmat @ G)[c_i, c_j]
    m = sig != 0
    if m.sum() < 10:
        continue
    side = np.sign(sig)
    rows = [rows_all[k] for k in np.nonzero(m)[0]]
    sd = side[m].astype(int)
    means, fills, extra, pos_cap = [], None, {}, []
    for pname, en, ex, hz in PLANS:
        res = [simulate(r, s, en.rstrip("!"), ex, hz, MAKER_RATES["maker fee 0.0175"],
                        strict=en.endswith("!")) for r, s in zip(rows, sd)]
        net = np.array([v for v, _, _ in res if v is not None])
        g = np.array([r["instant"] for r, (v, _, _) in zip(rows, res) if v is not None])
        means.append(net.mean() if net.size else np.nan)
        if pname == "maker → hold to settle":
            fills = net.size / len(rows)
        if pname in ("taker → hold to settle", "maker → hold to settle") and net.size >= 5:
            extra[pname] = cboot(net, g)
        if hz != "settle" and net.size >= 5 and net.mean() > 0:
            lo, hi, p0 = cboot(net, g)
            pos_cap.append(f"{short[pname]} {net.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] n={net.size}")
    # permutation null for taker hold (vectorised)
    obs = np.where(side[m] > 0, NET_YES[m], NET_NO[m]).mean()
    null = np.empty(N_PERM)
    for b, pm in enumerate(perms):
        s2 = np.sign((Zmat[pm] @ G)[c_i, c_j])
        k = s2 != 0
        null[b] = np.where(s2[k] > 0, NET_YES[k], NET_NO[k]).mean() if k.any() else 0.0
    pval = (1 + np.sum(null >= obs)) / (1 + N_PERM)
    plow = (1 + np.sum(null <= obs)) / (1 + N_PERM)
    summary.append(dict(ch=ch, n=int(m.sum()), rel=len({r["instant"] for r in rows}),
                        T=means[0], M=means[1], p=pval, plow=plow, pos_cap=pos_cap,
                        ciT=extra.get(cols[0]),
                        ciM=extra.get(cols[1]), best_cap=np.nanmax(means[3:])))
    print(f"{ch[0] + '→' + ch[1]:22} {int(m.sum()):>4} {summary[-1]['rel']:>4} {fills:>5.0%} "
          + " ".join(f"{v:>+8.2f}" for v in means), flush=True)

q = bh([r["p"] for r in summary])
print(f"\nhold-to-settlement by channel, with 95% clustered CIs; perm p = share of {N_PERM}")
print("surprise-shuffled draws with taker-hold net ≥ observed; q = BH across channels.")
print(f"{'channel':22} {'n':>4} {'taker hold [95% CI]':>28} {'maker hold [95% CI]':>28} "
      f"{'perm p':>7} {'BH q':>6} {'p(≤)':>7} {'best capture':>12}")
for r, qq in sorted(zip(summary, q), key=lambda x: x[0]["p"]):
    ct = r["ciT"]; cm = r["ciM"]
    ts = f"{r['T']:+.2f} [{ct[0]:+.1f}, {ct[1]:+.1f}]" if ct else f"{r['T']:+.2f}"
    ms = f"{r['M']:+.2f} [{cm[0]:+.1f}, {cm[1]:+.1f}]" if cm else f"{r['M']:+.2f}"
    print(f"{r['ch'][0] + '→' + r['ch'][1]:22} {r['n']:>4} {ts:>28} {ms:>28} "
          f"{r['p']:>7.4f} {qq:>6.3f} {r['plow']:>7.4f} {r['best_cap']:>+12.2f}")
print("\nT/M hold = taker/maker entry held to settlement; Ms = maker, strict-through fill;")
print("TT/TM/MT/MM = entry→exit execution; cap = exit at the capture print, 24h = at 24h.")
print("best capture = the best of the six capture/24h plans (selected, so optimistic).")
print("p(≤) = lower tail: the economic sign doing significantly WORSE than shuffled surprises.")
print("\npositive capture/24h cells, with 95% clustered CIs:")
for r in summary:
    for c in r["pos_cap"]:
        print(f"  {r['ch'][0] + '→' + r['ch'][1]:22} {c}")
