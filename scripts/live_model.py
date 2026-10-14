"""Live in-game forecaster (holdout_spec section 2): the model's inputs for one game at one 5-s step,
rebuilt from raw data as it arrives, and the 5 seeds' average forecast. MLB (every pitch a play) and NFL
(every ESPN play).

The inputs follow analysis/graph_2026_10/b3_table.py (plays_mlb_pitch / plays_nfl, quotes, print_columns,
p_home) and b4_export.py (play_arrays, node_frame; mode "prints" = panel_v3) rule for rule, so the model
sees what it was trained on; live_parity.py checks this against the training panel on a past game.

Raw inputs (what scripts/live_record.py stores):
    feed        MLB: the statsapi /feed/live document + winProbability (fetch_pbp.parse_mlb's input)
                NFL: the ESPN /summary document (fetch_pbp.parse_espn's input, plus the drive in progress)
    trades      Kalshi trades: created_time, yes_price (cents), count, taker_side, trade_id
    candles     Kalshi 1-min candles: end_period_ts, yes_bid / yes_ask close (cents)

NFL differs from training in two inputs that only exist after the game (stg.sports.loaders._nfl_play_ends):
a play's end time (nflverse; live: the time our feed first delivered the play) and nflverse's win
probability (live: ESPN's, which the training data used only where nflverse had none).

Feed delay: the model was trained with plays reaching it 10 s after they happen. Live, a play is known
when the feed delivers it, so delay_s (added on top) defaults to 0 here; the real delay is logged by the
runner (live_trade.py), and the model is retrained if it differs much from 10 s.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

# small models, one forecast every 5 s: one thread. Torch's default is one per core of the machine (32 in a Lumid
# sandbox), far above the sandbox's CPU quota, and the threads only queue behind each other
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_pbp import parse_espn, parse_mlb  # noqa: E402

# stg.models.ingame, loaded from its file (the stg.models package imports the whole training stack)
_spec = importlib.util.spec_from_file_location("stg_ingame", ROOT / "stg_infra" / "stg" / "models" / "ingame.py")
ingame = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingame)

# b3_table / b4_export / b1_response constants
STEP, ACT_S, FLOW_S, PRINT_MAX_AGE_S = 5, 300, 60, 120
GAP, LIVE_LO, LIVE_HI, LIVE_AGE_S, VALUE_AGE_S, MID_AGE_S = 10.0, 3.0, 97.0, 300, 600, 600
HR_DECIDED_S = 7
MLB_GROUPS = [("home_run", "home_run"), ("extra_base_hit", "double|triple"), ("single", "single"),
              ("walk_hbp", "walk|hit_by_pitch|intent"), ("strikeout", "strikeout"),
              ("double_play", "double_play|triple_play"), ("error", "error"), ("out_in_play", "out|sac|fielders_choice")]
PITCH_GROUPS = [("p_ball", "^(Ball|Ball In Dirt|Pitchout)$"), ("p_called_strike", "^Called Strike$"),
                ("p_swinging_strike", "^(Swinging Strike|Swinging Strike \\(Blocked\\)|Missed Bunt|Foul Tip)$"),
                ("p_foul", "^(Foul|Foul Bunt)$")]
# b1_prelim.NFL_GROUPS on ESPN's event type / text: (group, test)
NFL_GROUPS = [
    ("review", lambda e, d: bool(re.search(r"(?i)challenged|reversed|replay", d))),
    ("injury", lambda e, d: bool(re.search(r"(?i)injured", d))),
    ("touchdown", lambda e, d: "Touchdown" in e),
    ("turnover", lambda e, d: e in ("Pass Interception Return", "Fumble Recovery (Opponent)", "Sack Opp Fumble Recovery",
                                    "Muffed Punt Recovery (Opponent)", "Blocked Punt", "Blocked Field Goal")),
    ("field_goal", lambda e, d: e in ("Field Goal Good", "Field Goal Missed", "Safety")),
    ("penalty", lambda e, d: e == "Penalty"),
    ("stoppage", lambda e, d: e in NFL_STOPPAGES),
    ("special", lambda e, d: e in ("Kickoff", "Punt")),
    ("scrimmage", lambda e, d: True)]
NFL_STOPPAGES = ("Timeout", "Official Timeout", "Two-minute warning", "End Period", "End of Half", "End of Regulation",
                 "End of Game")
PLAY_TAIL = ["dwp", "dmargin", "dpoints", "off_home", "off_away", "tau", "margin", "total", "wp", "s1", "s2", "s3",
             "stoppage", "scoring", "log_gap"]
CFG = {
    "MLB": dict(S_M=4.0, S_T=4.25, SCALE=4.0, TOT_SCALE=9.0, PLAY_SCALE=1.0, S_SCALES=(3.0, 4.0, 3.0), PLAY_UNIT="pitch",
                L_FALLBACK=8.26,      # median pregame total of the training panel's games (b4 fills a missing L so)
                GROUPS=[g for g, _ in MLB_GROUPS] + ["other"] + [g for g, _ in PITCH_GROUPS] + ["p_other"],
                NAMES="gru_d10_h30-60-120-1800_tp_pitch",
                SERIES={"KXMLBGAME": "winner", "KXMLBSPREAD": "spread", "KXMLBTOTAL": "total"},
                ALIAS={"A": "ATH", "ARI": "AZ", "FLA": "MIA", "KAN": "KC", "WAS": "WSH"}),    # stg.sports.contracts.ALIAS
    "NFL": dict(S_M=13.0, S_T=14.5, SCALE=14.0, TOT_SCALE=45.0, PLAY_SCALE=7.0, S_SCALES=(4.0, 10.0, 100.0), PLAY_UNIT="pa",
                L_FALLBACK=44.68, GROUPS=[g for g, _ in NFL_GROUPS], NAMES="gru_d10_h30-60-120-1800_tp",
                SERIES={"KXNFLGAME": "winner", "KXNFLSPREAD": "spread", "KXNFLTOTAL": "total"},
                ALIAS={"JAX": "JAC", "LAR": "LA"}),
}
for _c in CFG.values():
    _c["PLAY_FEATS"] = [f"g_{g}" for g in _c["GROUPS"]] + PLAY_TAIL
STEP_FEATS = ["p_home", "has_p_home", "log_activity"]
NODE_FEATS = ["k_winner", "k_spread", "k_total", "side_home", "side_away", "live", "has_bid", "has_ask", "bid", "ask",
              "value", "log_age", "has_print", "print_px", "log_print_age", "flow", "log_vol"]
KIND_CODE = {"winner": 0, "spread": 1, "total": 2}
SEEDS = [0, 1, 2, 3, 4]


def epoch(t: dt.datetime | None) -> float:
    return t.timestamp() if t is not None else math.nan


# ----------------------------------------------------------------------------- contracts

def contract_meta(league: str, markets: list[dict], home: str, away: str) -> list[dict]:
    """Kind, side and strike of the game's winner / spread / total markets (Kalshi /markets rows), as
    stg.sports.contracts reads them: the team from the leg suffix, the strike on one scale (YES iff the
    quantity is > strike). home / away: Kalshi's codes for the two teams (as in the event ticker)."""
    alias = CFG[league]["ALIAS"]
    home, away = alias.get(home, home), alias.get(away, away)
    out = []
    for m in markets:
        kind = CFG[league]["SERIES"].get(m.get("series_ticker") or m["event_ticker"].split("-")[0])
        if kind is None:
            continue
        side = None
        if kind != "total":
            t = re.match(r"^([A-Z]+?)(\d+)?$", m["ticker"].split("-")[-1])
            team = alias.get(t.group(1), t.group(1)) if t else None
            side = "home" if team == home else "away" if team == away else None
            if side is None:
                continue                      # b3_table keeps winner / spread legs of the two teams only
        strike = None
        for txt, pat, off in ((m.get("yes_sub_title"), r"(?i)over ([\d.]+)", 0), (m.get("title"), r"(?i)over ([\d.]+)", 0),
                              (m.get("title"), r"(?i)more than ([\d.]+)", 0), (m.get("yes_sub_title"), r"(\d+)\+", -0.5),
                              (m.get("title"), r"(?i)at least ([\d.]+)", -0.5)):
            r = re.search(pat, txt or "")
            if r:
                strike = float(r.group(1)) + off
                break
        if kind != "winner" and strike is None:
            continue
        out.append({"ticker": m["ticker"], "kind": kind, "side": side, "strike": strike})
    return out


# ----------------------------------------------------------------------------- plays

ARCHIVE_PITCH_S = 3.9      # a pitch's endTime - startTime in the archived feed training reads (median, not in play)


def mlb_plays(doc: dict, start_utc: dt.datetime, live_clock: bool = False) -> pl.DataFrame:
    """The game's pitches as b3_table.plays_mlb_pitch + _finish_plays builds them, sorted as b4_export
    reads them (t_event, seq). Live only: a plate appearance still in progress has no result yet, so its
    latest pitch is a pitch, not the plate appearance's end (in a finished game's feed this changes nothing).
    live_clock: time the pitches as the archived feed does, which the models learn from. The live feed first
    gives a pitch's end ~0.7 s after its start, the archive ~3.9 s (a later version moves it), so each pitch
    not in play ends no sooner than start + ARCHIVE_PITCH_S; and a ball in play first shows without its result,
    ending at contact, while the archive ends it with the play, so it is left out until its result arrives."""
    rows = parse_mlb("live", doc)
    open_ab = {p["about"]["atBatIndex"] for p in doc["feed"]["liveData"]["plays"]["allPlays"]
               if not p["about"].get("isComplete", True)}
    if live_clock:
        rows = [r for r in rows if not (r["is_pitch"] and r["is_in_play"] and r["at_bat_index"] in open_ab)]
        for r in rows:
            if r["is_pitch"] and not r["is_in_play"] and r["t"] is not None and r["t_end"] is not None:
                r["t_end"] = max(r["t_end"], r["t"] + dt.timedelta(seconds=ARCHIVE_PITCH_S))
    for r in rows:
        if r["at_bat_index"] in open_ab:
            r["pa_last_event"], r["pa_event"], r["home_wp_post"] = False, None, None
    schema = {"seq": pl.Int64, "t": pl.Datetime("us", "UTC"), "t_end": pl.Datetime("us", "UTC"), "period": pl.Int64,
              "half": pl.String, "home_score": pl.Int64, "away_score": pl.Int64, "is_pitch": pl.Boolean,
              "description": pl.String, "pa_last_event": pl.Boolean, "pa_event": pl.String, "balls": pl.Int64,
              "strikes": pl.Int64, "outs": pl.Int64, "home_wp_post": pl.Float64}
    e = pl.DataFrame([{k: r[k] for k in schema} for r in rows], schema=schema, orient="row")
    h = (pl.col("t") - start_utc).dt.total_seconds() / 3600
    e = e.filter(h.is_between(-1, 8).fill_null(False), pl.col("is_pitch")).sort("seq")
    # loaders.load_pbp: a pitch is known at its end; t_event = its start
    e = e.with_columns(pl.col("t_end").alias("known_at"), pl.col("t").alias("t_event"))
    last = pl.col("pa_last_event").fill_null(False)
    pa = pl.lit("other")
    for name, pat in reversed(MLB_GROUPS):
        pa = pl.when(pl.col("pa_event").str.contains(pat)).then(pl.lit(name)).otherwise(pa)
    pitch = pl.lit("p_other")
    for name, pat in reversed(PITCH_GROUPS):
        pitch = pl.when(pl.col("description").str.contains(pat)).then(pl.lit(name)).otherwise(pitch)
    e = e.with_columns(pl.when(last & (pl.col("pa_event") == "home_run"))
                       .then(pl.min_horizontal(pl.col("t_event") + pl.duration(seconds=HR_DECIDED_S), "known_at"))
                       .otherwise(pl.max_horizontal("t_event", "known_at")).alias("t_event"))
    made = (pl.col("period") - 1) * 6 + pl.when(pl.col("half") == "bottom").then(3).otherwise(0) + pl.col("outs")
    e = e.with_columns(pl.when(last).then(pa).otherwise(pitch).alias("play_group"),
                       pl.when(last).then(pl.col("home_wp_post")).alias("wp_post"),
                       pl.when(pl.col("half") == "top").then(pl.lit("away")).otherwise(pl.lit("home")).alias("offense"),
                       ((54 - made) / 54).clip(1 / 54, 1.0).alias("tau"),
                       pl.col("outs").alias("s1"), pl.col("balls").alias("s2"), pl.col("strikes").alias("s3"))
    prev = lambda c: pl.col(c).shift(1)      # noqa: E731
    e = e.sort("seq").with_columns(pl.col("wp_post").forward_fill())
    e = e.with_columns((pl.col("wp_post") - prev("wp_post")).alias("dwp"),
                       ((pl.col("home_score") - pl.col("away_score")) - (prev("home_score") - prev("away_score"))).fill_null(0).alias("dmargin"),
                       ((pl.col("home_score") + pl.col("away_score")) - (prev("home_score") + prev("away_score"))).fill_null(0).alias("dpoints"),
                       (pl.col("home_score") - pl.col("away_score")).alias("margin"),
                       (pl.col("home_score") + pl.col("away_score")).alias("total"),
                       pl.lit(False).alias("is_stoppage"))
    return e.with_columns((pl.col("dpoints") > 0).alias("is_scoring")).sort("t_event", "seq")


def espn_rows(doc: dict) -> list[dict]:
    """fetch_pbp.parse_espn's rows of an NFL /summary document, plus the drive in progress (a finished
    game's document has none; parse_espn reads the finished drives only)."""
    cur = (doc.get("drives") or {}).get("current")
    if cur and cur.get("plays"):
        doc = dict(doc, drives=dict(doc["drives"], previous=list(doc["drives"].get("previous", [])) + [cur]))
        prev = doc["drives"]["previous"]
        if len(prev) > 1 and prev[-2].get("id") == cur.get("id"):      # the current drive listed twice
            doc["drives"]["previous"] = prev[:-1]
    return parse_espn("NFL", "live", doc)


def nfl_clock_tau(period: int | None, clock: str | None) -> float | None:
    """b1_baseline.tau_left (NFL): the share of regulation left; overtime as its clock / 60 min. None
    where that is null (no period or an unreadable clock)."""
    if period is None or clock is None:
        return None
    try:
        mm, ss = (float(clock.split(":")[0]), float(clock.split(":")[1])) if ":" in clock else (0.0, float(clock))
    except (ValueError, IndexError):
        return None
    m = mm + ss / 60
    t = ((4 - period) * 15 + m) / 60 if period <= 4 else m / 60
    return min(max(t, 1e-3), 1.0)


def nfl_plays(ev: pl.DataFrame, home_id: str | None) -> pl.DataFrame:
    """The game's plays as b3_table.plays_nfl + _finish_plays builds them, sorted as b4_export reads them.
    ev: one row per ESPN play in game order with seq, t, t_event (when the play is known), period, clock,
    scores, event_type, description, team_id, down, distance, yard_line and wp (the home win probability
    after the play: training nflverse's, else ESPN's; live ESPN's). home_id: the home team's ESPN id."""
    grp = [next(g for g, f in NFL_GROUPS if f(e or "", d or "")) for e, d in ev.select("event_type", "description").iter_rows()]
    tau = [nfl_clock_tau(p, c) for p, c in ev.select("period", "clock").iter_rows()]
    e = ev.with_columns(pl.Series("play_group", grp, pl.String), pl.Series("tau", tau, pl.Float64),
                        pl.col("wp").alias("wp_post"),
                        pl.when(pl.col("team_id").is_null()).then(None).when(pl.col("team_id") == home_id).then(pl.lit("home"))
                        .otherwise(pl.lit("away")).alias("offense"),
                        pl.col("event_type").is_in(NFL_STOPPAGES).fill_null(False).alias("is_stoppage"),
                        pl.col("down").alias("s1"), pl.col("distance").alias("s2"), pl.col("yard_line").alias("s3"))
    prev = lambda c: pl.col(c).shift(1)      # noqa: E731
    e = e.sort("seq").with_columns(pl.col("wp_post").forward_fill())
    e = e.with_columns((pl.col("wp_post") - prev("wp_post")).alias("dwp"),
                       ((pl.col("home_score") - pl.col("away_score")) - (prev("home_score") - prev("away_score"))).fill_null(0).alias("dmargin"),
                       ((pl.col("home_score") + pl.col("away_score")) - (prev("home_score") + prev("away_score"))).fill_null(0).alias("dpoints"),
                       (pl.col("home_score") - pl.col("away_score")).alias("margin"),
                       (pl.col("home_score") + pl.col("away_score")).alias("total"))
    return e.with_columns(pl.col("is_scoring").fill_null(False)).sort("t_event", "seq")


def nfl_live_plays(doc: dict, start_utc: dt.datetime, first_seen: dict[str, float]) -> tuple[pl.DataFrame, str | None]:
    """nfl_plays from a live ESPN document. A play is known when our feed first delivered it (first_seen:
    ESPN play id -> epoch s, filled in by the caller); training used its end time (nflverse), which is
    published only after the game. Returns the plays and the home team's ESPN id."""
    comp = ((doc.get("header") or {}).get("competitions") or [{}])[0]
    home_id = next((c["team"]["id"] for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
    rows = espn_rows(doc)
    cols = {"seq": pl.Int64, "t": pl.Datetime("us", "UTC"), "period": pl.Int64, "clock": pl.String, "home_score": pl.Int64,
            "away_score": pl.Int64, "event_type": pl.String, "description": pl.String, "is_scoring": pl.Boolean,
            "team_id": pl.String, "down": pl.Int64, "distance": pl.Int64, "yard_line": pl.Int64, "wp": pl.Float64,
            "espn_play_id": pl.String}
    ev = pl.DataFrame([{**{k: r.get(k) for k in cols if k != "wp"}, "wp": r.get("home_wp_post")} for r in rows],
                      schema=cols, orient="row")
    seen = pl.Series("t_event", [first_seen.get(i) for i in ev["espn_play_id"]], pl.Float64)
    h = (pl.col("t") - start_utc).dt.total_seconds() / 3600
    ev = (ev.with_columns(pl.from_epoch(seen * 1e6, time_unit="us").dt.replace_time_zone("UTC").alias("t_event"))
          .filter(h.is_between(-1, 8).fill_null(False), pl.col("t_event").is_not_null()))
    return nfl_plays(ev, home_id), home_id


def play_arrays(p: pl.DataFrame, league: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """b4_export.play_arrays for one game: features [P, Fp], times [P] (epoch s), raw state [P, 4]."""
    C = CFG[league]
    gap = p.select((pl.col("t_event") - pl.col("t_event").shift(1)).dt.total_seconds().fill_null(0).clip(0, 3600))
    feats = p.select(
        *[(pl.col("play_group") == g).alias(f"g_{g}") for g in C["GROUPS"]],
        (pl.col("dwp").fill_null(0) * 10).alias("dwp"), (pl.col("dmargin") / C["PLAY_SCALE"]).alias("dmargin"),
        (pl.col("dpoints") / C["PLAY_SCALE"]).alias("dpoints"), (pl.col("offense") == "home").alias("off_home"),
        (pl.col("offense") == "away").alias("off_away"), pl.col("tau").fill_null(0).alias("tau"),
        (pl.col("margin") / C["SCALE"]).alias("margin"), (pl.col("total") / C["TOT_SCALE"]).alias("total"),
        pl.col("wp_post").fill_null(0.5).alias("wp"),
        *[(pl.col(c).cast(pl.Float64).clip(0, 4 * sc) / sc).alias(c) for c, sc in zip(("s1", "s2", "s3"), C["S_SCALES"])],
        pl.col("is_stoppage").alias("stoppage"), pl.col("is_scoring").alias("scoring"),
        pl.Series("log_gap", np.log1p(gap.to_series().to_numpy()) / 5))
    assert feats.columns == C["PLAY_FEATS"], feats.columns
    f32 = lambda df: df.select(pl.all().cast(pl.Float32).fill_null(0)).to_numpy().astype(np.float32)   # noqa: E731
    state = p.select(pl.col("tau").fill_null(1.0), pl.col("margin").cast(pl.Float64), pl.col("total").cast(pl.Float64),
                     pl.when(pl.col("offense") == "home").then(1.0).when(pl.col("offense") == "away").then(-1.0).otherwise(0.0))
    return f32(feats), p["t_event"].dt.epoch("us").cast(pl.Float64).to_numpy() / 1e6, f32(state)


# ----------------------------------------------------------------------------- market data

class Market:
    """Per contract: the quote series (stg.sports.prices: trades + 1-min candle closes, newest side wins) and
    the trade prints (in b3_table's sweep order), as numpy arrays. Kept up to date incrementally: new rows that
    sort after a contract's last row are appended; anything else (a late row, a changed candle) re-sorts that
    contract only. The arrays are the same as a full rebuild's (scripts/test_live_market.py checks this)."""

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = tickers
        self.trades: dict[str, dict] = {}            # trade_id -> row
        self.candles: dict[tuple, dict] = {}         # (ticker, ts) -> row
        self._qrows = {t: [] for t in tickers}       # sorted (key, bid, ask)
        self._prows = {t: [] for t in tickers}       # sorted (t, yes, signed price, trade_id, price, count)
        self._new_q = {t: [] for t in tickers}
        self._new_p = {t: [] for t in tickers}
        self._full: set[str] = set()                 # contracts to re-sort from scratch
        self.q = {t: self._q_arrays([], np.nan, np.nan, np.nan, np.nan) for t in tickers}
        self.p = {t: self._p_arrays([]) for t in tickers}
        self.dirty = False

    @staticmethod
    def _qrow_trade(r: dict) -> tuple:
        yes = r["taker_side"] == "yes"
        return ((r["t"], 1, yes, r["price"] if yes else -r["price"], r["trade_id"]),
                math.nan if yes else r["price"], r["price"] if yes else math.nan)

    @staticmethod
    def _prow(r: dict) -> tuple:
        yes = r["taker_side"] == "yes"
        return (r["t"], yes, r["price"] if yes else -r["price"], r["trade_id"], r["price"], r["count"])

    @staticmethod
    def _qrow_candle(r: dict) -> tuple:
        # a candle (src 0) before a trade (src 1) at the same time; a missing candle side counts as 0 / 100
        return ((r["ts"], 0), 0.0 if r["bid"] is None else r["bid"], 100.0 if r["ask"] is None else r["ask"])

    def add_trades(self, rows: list[dict]) -> None:
        for r in rows:
            t = r["ticker"]
            if t not in self._qrows or r["trade_id"] in self.trades:
                continue
            self.trades[r["trade_id"]] = r
            self._new_q[t].append(self._qrow_trade(r))
            self._new_p[t].append(self._prow(r))
            self.dirty = True

    def add_candles(self, rows: list[dict]) -> None:
        for r in rows:
            t = r["ticker"]
            if t not in self._qrows:
                continue
            k = (t, r["ts"])
            old = self.candles.get(k)
            if old is not None:
                if (old["bid"], old["ask"]) != (r["bid"], r["ask"]):
                    self.candles[k] = r
                    self._full.add(t)
                    self.dirty = True
                continue
            self.candles[k] = r
            self._new_q[t].append(self._qrow_candle(r))
            self.dirty = True

    @staticmethod
    def _ffill(vals: np.ndarray, tt: np.ndarray, carry: float, carry_at: float) -> tuple[np.ndarray, np.ndarray]:
        """Each row's last finite value up to it (and that value's time), starting from carry."""
        idx = np.maximum.accumulate(np.where(np.isfinite(vals), np.arange(len(vals)), -1)) if len(vals) else np.array([], int)
        v = np.where(idx >= 0, vals[np.maximum(idx, 0)] if len(vals) else vals, carry)
        at = np.where(idx >= 0, tt[np.maximum(idx, 0)] if len(vals) else tt, carry_at)
        return v.astype(float), at.astype(float)

    def _q_arrays(self, rows: list, cb: float, cb_at: float, ca: float, ca_at: float) -> dict:
        tt = np.array([x[0][0] for x in rows], float)
        bid, bid_at = self._ffill(np.array([x[1] for x in rows], float), tt, cb, cb_at)
        ask, ask_at = self._ffill(np.array([x[2] for x in rows], float), tt, ca, ca_at)
        return dict(t=tt, bid=bid, bid_at=bid_at, ask=ask, ask_at=ask_at)

    @staticmethod
    def _p_arrays(rows: list) -> dict:
        return dict(t=np.array([x[0] for x in rows], float), yes=np.array([x[1] for x in rows], bool),
                    px=np.array([x[4] for x in rows], float), n=np.array([x[5] for x in rows], float))

    def _rebuild(self, t: str) -> None:
        q = [self._qrow_candle(r) for (tk, _), r in self.candles.items() if tk == t]
        q += [self._qrow_trade(r) for r in self.trades.values() if r["ticker"] == t]
        self._qrows[t] = sorted(q, key=lambda x: x[0])
        self._prows[t] = sorted((self._prow(r) for r in self.trades.values() if r["ticker"] == t), key=lambda x: x[:4])
        self.q[t] = self._q_arrays(self._qrows[t], np.nan, np.nan, np.nan, np.nan)
        self.p[t] = self._p_arrays(self._prows[t])

    def build(self) -> None:
        if not self.dirty:
            return
        for t in self.tickers:
            nq, np_ = self._new_q[t], self._new_p[t]
            if t in self._full:
                self._new_q[t], self._new_p[t] = [], []
                self._rebuild(t)
                continue
            if nq:
                nq.sort(key=lambda x: x[0])
                rows, old = self._qrows[t], self.q[t]
                if rows and nq[0][0] < rows[-1][0]:                # a late row: re-sort this contract's quotes
                    self._qrows[t] = sorted(rows + nq, key=lambda x: x[0])
                    self.q[t] = self._q_arrays(self._qrows[t], np.nan, np.nan, np.nan, np.nan)
                else:
                    last = (old["bid"][-1], old["bid_at"][-1], old["ask"][-1], old["ask_at"][-1]) if rows else (np.nan,) * 4
                    add = self._q_arrays(nq, *last)
                    rows.extend(nq)
                    self.q[t] = {k: np.concatenate([old[k], add[k]]) for k in old}
                self._new_q[t] = []
            if np_:
                np_.sort(key=lambda x: x[:4])
                rows, old = self._prows[t], self.p[t]
                if rows and np_[0][:4] < rows[-1][:4]:
                    self._prows[t] = sorted(rows + np_, key=lambda x: x[:4])
                    self.p[t] = self._p_arrays(self._prows[t])
                else:
                    add = self._p_arrays(np_)
                    rows.extend(np_)
                    self.p[t] = {k: np.concatenate([old[k], add[k]]) for k in old}
                self._new_p[t] = []
        self._full.clear()
        self.dirty = False

    def quote(self, ticker: str, t: float) -> tuple[float, float, float, float] | None:
        """(bid, ask, bid_at, ask_at) as of t (NaN = that side never seen), None if nothing yet."""
        self.build()
        q = self.q[ticker]
        i = int(np.searchsorted(q["t"], t, side="right")) - 1
        return None if i < 0 else (q["bid"][i], q["ask"][i], q["bid_at"][i], q["ask_at"][i])


def quote_age(bid_at: float, ask_at: float, t: float) -> float:
    """t - the older side's time (pl.min_horizontal ignores a missing side)."""
    s = [x for x in (bid_at, ask_at) if np.isfinite(x)]
    return t - min(s) if s else math.nan


def quote_value(bid: float, ask: float, age: float) -> float:
    """b1_response.value: a two-sided mid, or an extreme one-sided quote valued halfway to 0 / 100."""
    if not age <= VALUE_AGE_S:
        return math.nan
    b, a = (0.0 if not np.isfinite(bid) else bid), (100.0 if not np.isfinite(ask) else ask)
    if b > 0 and a < 100 and a >= b and a - b <= GAP:
        return (b + a) / 2
    if b >= 90 and a >= 99:
        return (b + 100) / 2
    if a <= 10 and b <= 1:
        return a / 2
    return math.nan


def is_live(bid: float, ask: float, age: float) -> bool:
    return bool(np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask < 100 and ask >= bid and ask - bid <= GAP
                and LIVE_LO <= (bid + ask) / 2 <= LIVE_HI and math.floor(age) <= LIVE_AGE_S)


def pregame_total(contracts: list[dict], mk: Market, t: float) -> float:
    """b1_baseline.pregame_total: the strike where the total ladder prices the over at 50c, at t (1 min
    before the first pitch); two-sided mids within GAP, sides at most 10 min old."""
    ks, ps = [], []
    for c in contracts:
        if c["kind"] != "total" or (q := mk.quote(c["ticker"], t)) is None:
            continue
        bid, ask, ba, aa = q
        age = t - min(ba, aa) if np.isfinite(ba) and np.isfinite(aa) else math.nan
        if np.isfinite(bid) and np.isfinite(ask) and 0 < bid <= ask < 100 and ask - bid <= GAP and age <= MID_AGE_S:
            ks.append(c["strike"]), ps.append((bid + ask) / 200)
    k, p = np.array(ks), np.array(ps)
    if len(k) < 2 or not (p.max() >= 0.5 >= p.min()):
        return math.nan
    o = np.argsort(-p, kind="stable")
    return float(np.interp(-0.5, -p[o], k[o]))


# ----------------------------------------------------------------------------- one step

def step_inputs(t: float, first_play: float, contracts: list[dict], mk: Market, p_home_prev: float | None) -> dict:
    """The market side of the model's inputs at step time t (on the grid first_play + 5k): the node rows
    (b3_table contract_steps -> b4_export game_rows / node_frame, mode prints) and the step's market
    features. p_home_prev: the last step's home winner price (b4 carries it forward)."""
    mk.build()
    k = math.floor((t - first_play) / STEP + 1e-9)
    rows, ph = [], []
    for c in contracts:
        tk = c["ticker"]
        q = mk.quote(tk, t)
        if q is None:
            continue
        bid, ask, ba, aa = q
        if not (np.isfinite(bid) or np.isfinite(ask)):
            continue
        age = quote_age(ba, aa, t)
        value, live = quote_value(bid, ask, age), is_live(bid, ask, age)
        p = mk.p[tk]
        n = int(np.searchsorted(p["t"], t, side="right"))                 # prints known at t
        pr_px, pr_age = (p["px"][n - 1], t - p["t"][n - 1]) if n else (math.nan, math.nan)
        bins = np.maximum(0, np.ceil((p["t"][:n] - first_play) / STEP - 1e-9))
        w = bins >= k - (FLOW_S // STEP - 1)
        flow = float((p["n"][:n][w] * np.where(p["yes"][:n][w], 1, -1)).sum())
        vol = float(p["n"][:n][w].sum())
        if c["kind"] == "winner" and live:
            mid = (bid + ask) / 2
            ph.append(mid if c["side"] == "home" else 100 - mid)
        # b3_table stores quote and print ages as whole seconds (polars dt.total_seconds truncates); the
        # value rule (b1_response.value) compares the exact age
        age_s, pr_age = math.floor(age) if np.isfinite(age) else age, math.floor(pr_age) if np.isfinite(pr_age) else pr_age
        anchor = pr_px if pr_age <= PRINT_MAX_AGE_S else value
        if not np.isfinite(anchor) or not (live or 2 <= anchor <= 98):
            continue
        has_print = pr_age <= PRINT_MAX_AGE_S
        x = [c["kind"] == "winner", c["kind"] == "spread", c["kind"] == "total", c["side"] == "home", c["side"] == "away",
             live, np.isfinite(bid) and bid > 0, np.isfinite(ask) and ask < 100,
             (bid if np.isfinite(bid) else 0) / 100, (ask if np.isfinite(ask) else 100) / 100,
             (value if np.isfinite(value) else anchor) / 100,
             math.log1p(min(max(age_s if np.isfinite(age_s) else 3600, 0), 3600)) / 8,
             has_print, (pr_px if np.isfinite(pr_px) else anchor) / 100,
             math.log1p(min(max(pr_age if np.isfinite(pr_age) else 3600, 0), 3600)) / 8,
             math.copysign(math.log1p(abs(flow)), flow) / 8 if flow else 0.0, math.log1p(vol) / 8]
        rows.append(dict(c, x=x, anchor=anchor, bid=bid, ask=ask, live=live, pr_px=pr_px, pr_age=pr_age))
    p_now = float(np.mean(ph)) if ph else None
    p_home = p_now if p_now is not None else p_home_prev
    wt = np.concatenate([mk.p[c["ticker"]]["t"] for c in contracts if c["kind"] == "winner"] or [np.array([])])
    act = int(((wt > t - ACT_S) & (wt <= t)).sum())
    sx = np.array([(p_home if p_home is not None else 50.0) / 100, p_now is not None, math.log1p(act) / 7], np.float32)
    return dict(t=t, k=k, rows=rows, steps_x=sx, p_home=p_home)


# ----------------------------------------------------------------------------- model

class _Game:
    """The fields of stg.models.ingame.Panel that known_state reads, for one game."""

    def __init__(self, pt: np.ndarray, pstate: np.ndarray, C: dict) -> None:
        self.game_play_ptr = np.array([0, len(pt)])
        self.plays_t, self.plays_state = pt, pstate
        self.scale, self.tot_scale = C["SCALE"], C["TOT_SCALE"]


class Forecaster:
    """The 5 seeds of one league's model; forecast() = their average."""

    def __init__(self, league: str, weights_dir: str | Path, device: str = "cpu") -> None:
        self.league, self.C = league, CFG[league]
        C = self.C
        self.models, self.dev = [], torch.device(device)
        for s in SEEDS:
            ck = torch.load(Path(weights_dir) / f"{league}_final_{C['NAMES']}_seed{s}.pt", map_location="cpu", weights_only=False)
            p = ck["panel"]
            # the inputs built here must be the ones the weights were trained on
            assert p["league"] == league and (p["play_features"], p["step_market_features"], p["node_market_features"]) == \
                (C["PLAY_FEATS"], STEP_FEATS, NODE_FEATS), s
            assert ((p.get("play_unit") or "pa"), tuple(p["baseline_S"]), p["scale"], p["tot_scale"], p["targets"]) == \
                (C["PLAY_UNIT"], (C["S_M"], C["S_T"]), C["SCALE"], C["TOT_SCALE"], "trade prints"), s
            m = ingame.InGameGraph(**ck["config"])
            m.load_state_dict(ck["state_dict"])
            self.models.append(m.to(self.dev).eval())
            self.horizons, self.info = p["train_horizons"], ck["info"]

    @torch.no_grad()
    def forecast(self, steps: list[dict], px: np.ndarray, pt: np.ndarray, pstate: np.ndarray, L: float,
                 delay_s: float = 0.0) -> list[dict]:
        """Forecasts for one game's steps (step_inputs outputs; steps without rows are skipped): per node
        row, delta [H] (cents, average of the seeds) and the level at each horizon (anchor + delta).
        Plays: all received so far (play_arrays); a play counts at step t once t_event + delay_s <= t."""
        C = self.C
        steps = [s for s in steps if s["rows"]]
        if not steps:
            return []
        ts = np.array([s["t"] for s in steps])
        known, state, sfeat = ingame.known_state(_Game(pt, pstate, C), 0, ts, delay_s)
        cmax, ns = max(len(s["rows"]) for s in steps), len(steps)
        nx = np.zeros((ns, cmax, len(NODE_FEATS)), np.float32)
        kind, side, strike, value = (np.zeros((ns, cmax), np.int64), np.zeros((ns, cmax), np.int64),
                                     np.zeros((ns, cmax), np.float32), np.full((ns, cmax), 50.0, np.float32))
        m = np.zeros((ns, cmax), bool)
        for i, s in enumerate(steps):
            r = s["rows"]
            n = len(r)
            # b4_export stores node features as float16: the model was trained on the rounded values
            nx[i, :n] = np.array([x["x"] for x in r], np.float32).astype(np.float16).astype(np.float32)
            kind[i, :n] = [KIND_CODE[x["kind"]] for x in r]
            side[i, :n] = [{"home": 1, "away": -1}.get(x["side"], 0) for x in r]
            strike[i, :n] = [x["strike"] or 0.0 for x in r]
            value[i, :n] = [x["anchor"] for x in r]
            m[i, :n] = True
        sx = np.stack([s["steps_x"] for s in steps])
        # make_batch's derived node features at this delay: distance to the threshold and the baseline
        st_n = np.repeat(state[:, None, :], cmax, 1)
        p_home = np.repeat((100 * sx[:, 0])[:, None], cmax, 1)
        Lm = np.full((ns, cmax), np.float32(L if np.isfinite(L) else C["L_FALLBACK"]))
        dist = np.where(kind == 1, strike - side * st_n[..., 1], np.where(kind == 2, strike - st_n[..., 2], 0.0))
        base = ingame.baseline_price(kind, side, strike, p_home, st_n[..., 0], st_n[..., 2], Lm, C["S_M"], C["S_T"])
        has_base = np.isfinite(base)
        nder = np.stack([dist / C["SCALE"], np.where(has_base, base, value) / 100, np.where(has_base, base - value, 0.0) / 10,
                         has_base], -1).astype(np.float32)
        nodes = np.nan_to_num(np.concatenate([nx, nder], -1))
        nodes[~m] = 0
        plays = np.zeros((1, max(1, len(px)), px.shape[1] if len(px) else len(C["PLAY_FEATS"])), np.float32)
        plays[0, :len(px)] = px
        t = lambda x: torch.as_tensor(x, device=self.dev)      # noqa: E731
        z = torch.zeros((ns, cmax), device=self.dev)
        b = ingame.Batch(plays=t(plays), step_game=t(np.zeros(ns, np.int64)), step_plays=t(known.astype(np.int64)),
                         steps=t(np.concatenate([sx, sfeat], 1).astype(np.float32)), nodes=t(nodes), mask=t(m),
                         kind=t(kind), value=t(value), lo=z, hi=z, value_h=z, y=z, base=z, game_of_node=z, step_of_node=z)
        d = torch.stack([mdl(b)[0] for mdl in self.models]).mean(0).cpu().numpy()       # [Ns, C, H]
        out = []
        for i, s in enumerate(steps):
            for j, r in enumerate(s["rows"]):
                out.append(dict(r, t=s["t"], k=s["k"], plays_known=int(known[i]), delta=d[i, j],
                                level={h: float(r["anchor"] + d[i, j, hi]) for hi, h in enumerate(self.horizons)}))
        return out
