"""Pull play-by-play for the Kalshi MLB / NBA / NFL game markets.

Sources, one per league, each checked against the Kalshi tape on one game
(2026-09-29; the market moves within seconds of these timestamps, not before):

    MLB  statsapi.mlb.com  /api/v1.1/game/{pk}/feed/live, every pitch with a UTC
         start time (ms), plus /api/v1/game/{pk}/winProbability per plate
         appearance. Home runs: half the price move 3-5 s after the pitch's
         startTime; the plate appearance's endTime is 20-30 s later, so align on
         the pitch, never on the plate appearance.
    NBA  ESPN site API  /summary?event={id}, every play with a wallclock (1 s)
         and ESPN's home win probability. Scoring plays: half the move 2-7 s later.
    NFL  ESPN, as for the NBA (wallclock = the snap; half the move ~9 s later,
         about one play's length), plus the nflverse season files, which carry
         the snap to the ms (time_of_day), EPA and win probability before and
         after each play. NBA's own feeds (cdn.nba.com, stats.nba.com) refuse us.

Stages, run in order; pull resumes where a previous run stopped:

    venv/bin/python scripts/fetch_pbp.py schedule  # source schedules, matched to Kalshi game events
    venv/bin/python scripts/fetch_pbp.py estimate  # requests / disk / time, from a few sample games
    venv/bin/python scripts/fetch_pbp.py pull      # one raw file per game, + nflverse season files
    venv/bin/python scripts/fetch_pbp.py build     # one tidy row per pitch / play

Games come from the Kalshi listing that ``scripts/fetch_sports.py list`` wrote
(KXMLBGAME, KXMLBSTGAME, KXNBAGAME, KXNFLGAME events). Other game-level series
(spreads, totals, props) share the event key, e.g. KXNBASPREAD-25NOV17CHIDEN
and KXNBAGAME-25NOV17CHIDEN, so join them to ``games.parquet`` on ``key``.

Holdout: a game whose Kalshi game event closes on or after HOLDOUT_START
(fetch_sports.py's) is written under data/sports/_HOLDOUT_DO_NOT_USE/pbp/ and
never into the working tree; only counts are printed for it. The schedule
tables keep no scores.

Layout (data/ is git-ignored):
    data/sports/pbp/games.parquet                  Kalshi event -> source game
    data/sports/pbp/raw/<league>/<game_id>.json.gz raw responses (MLB: feed + trimmed win prob)
    data/sports/pbp/raw/NFL/nflverse_pbp_<season>.parquet
    data/sports/pbp/events_<league>.parquet        built
    data/sports/_HOLDOUT_DO_NOT_USE/pbp/...        same, holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sports import HOLD as SPORTS_HOLD, HOLDOUT_START, OUT as SPORTS  # noqa: E402

OUT = SPORTS / "pbp"
HOLD = SPORTS_HOLD / "pbp"
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

MLB_API = "https://statsapi.mlb.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_PATH = {"NBA": "basketball/nba", "NFL": "football/nfl"}
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"

GAME_SERIES = {"KXMLBGAME": "MLB", "KXMLBSTGAME": "MLB", "KXNBAGAME": "NBA", "KXNFLGAME": "NFL"}
# Kalshi leg code -> source abbreviation (statsapi for MLB, ESPN for NBA/NFL).
# Kalshi used both LA and LAR for the Rams.
TO_SRC = {
    "MLB": {"ARI": "AZ"},
    "NBA": {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"},
    "NFL": {"JAC": "JAX", "LA": "LAR", "WAS": "WSH"},
}
ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
NOT_TEAMS = {"TIE", "AL", "NL", "ALHS", "ALLS", "NLHS", "NLLS"}       # tie legs, All-Star games
MLB_GAME_TYPES = "S,R,F,D,L,W"                                       # spring, regular, postseason rounds
FINAL = {"MLB": {"Final", "Game Over", "Completed Early"}, "NBA": {"STATUS_FINAL"}, "NFL": {"STATUS_FINAL"}}


# --------------------------------------------------------------------------- http

class Client:
    """One client per host, each with its own request-rate cap; retry on 429/5xx."""

    def __init__(self, rates: dict[str, float]) -> None:
        self.c = httpx.Client(timeout=60, follow_redirects=True, limits=httpx.Limits(max_connections=32))
        self.gap = {h: 1.0 / r for h, r in rates.items()}
        self.next_t = {h: 0.0 for h in rates}
        self.lock = threading.Lock()
        self.n = 0

    def _wait(self, host: str) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t.get(host, 0.0))
            self.next_t[host] = t + self.gap.get(host, 0.5)
            self.n += 1
        if t > now:
            time.sleep(t - now)

    def get(self, url: str, params: dict | None = None) -> httpx.Response | None:
        host = httpx.URL(url).host
        for attempt in range(8):
            self._wait(host)
            try:
                r = self.c.get(url, params=params)
            except httpx.TransportError:
                time.sleep(2 ** min(attempt, 5))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("retry-after", 2 ** min(attempt, 5))))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        raise RuntimeError(f"gave up on {url} {params}")

    def json(self, url: str, params: dict | None = None):
        r = self.get(url, params)
        return r.json() if r is not None else None


def ts(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


# --------------------------------------------------------------------------- schedule

def kalshi_games() -> pl.DataFrame:
    """One row per Kalshi game event: date and time from the ticker, the two
    team codes from its legs (in source abbreviations), and the latest close."""
    cols = ["ticker", "event_ticker", "series_ticker", "close_time"]
    mk = pl.concat([pl.read_parquet(p, columns=cols) for p in (SPORTS / "markets.parquet",
                                                                 SPORTS_HOLD / "markets.parquet") if p.exists()])
    mk = mk.filter(pl.col("series_ticker").is_in(list(GAME_SERIES)))
    rows = []
    for (ev, series), g in mk.group_by("event_ticker", "series_ticker"):
        lg = GAME_SERIES[series]
        key = ev.split("-", 1)[1]
        m = re.match(r"^(\d{2}[A-Z]{3}\d{2})(\d{4})?", key)
        codes = sorted({TO_SRC[lg].get(c, c) for c in (t.rsplit("-", 1)[1] for t in g["ticker"])} - NOT_TEAMS)
        rows.append({
            "league": lg, "series_ticker": series, "event_ticker": ev, "key": key,
            "date_local": dt.datetime.strptime(m.group(1), "%y%b%d").date() if m else None,
            "hhmm": m.group(2) if m else None,
            "game_no": int(key[-1]) if re.search(r"[A-Z]G[12]$", key) else None,   # older doubleheader tickers
            "teams": "-".join(codes) if len(codes) == 2 else None,
            "close_time": g["close_time"].max(),
        })
    return (pl.DataFrame(rows)
              .with_columns((pl.col("close_time") >= HOLDOUT_START).alias("holdout"))
              .sort("league", "date_local", "event_ticker"))


def schedule_mlb(cl: Client, first: dt.date, last: dt.date) -> list[dict]:
    rows = []
    for y in range(first.year, last.year + 1):
        a, b = max(first, dt.date(y, 1, 1)), min(last, dt.date(y, 12, 31))
        d = cl.json(f"{MLB_API}/api/v1/schedule", {"sportId": 1, "startDate": a.isoformat(),
                    "endDate": b.isoformat(), "gameType": MLB_GAME_TYPES, "hydrate": "team"}) or {}
        for day in d.get("dates", []):
            for g in day["games"]:
                t = g["teams"]
                # the date bucket, not officialDate: a postponed game's officialDate
                # moves to its makeup, which would hide it on the original date
                rows.append({"game_id": str(g["gamePk"]), "date_local": dt.date.fromisoformat(day["date"]),
                             "start_utc": ts(g["gameDate"]), "status": g["status"]["detailedState"],
                             "game_type": g["gameType"], "game_number": g.get("gameNumber"),
                             "rescheduled": g.get("rescheduledFrom") is not None,
                             "home": t["home"]["team"].get("abbreviation"),
                             "away": t["away"]["team"].get("abbreviation")})
    return rows


def schedule_espn(cl: Client, lg: str, dates: list[dt.date], workers: int) -> list[dict]:
    """ESPN has no date-range scoreboard, so one request per date. Dates are
    the Kalshi dates +- 1 day; an event's local date is its start in US Eastern."""
    def one(d: dt.date) -> list[dict]:
        j = cl.json(f"{ESPN}/{ESPN_PATH[lg]}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 1000}) or {}
        out = []
        for e in j.get("events", []):
            side = {c["homeAway"]: c["team"]["abbreviation"] for c in e["competitions"][0]["competitors"]}
            start = ts(e["date"])
            out.append({"game_id": e["id"], "date_local": start.astimezone(ET).date(), "start_utc": start,
                        "status": e["status"]["type"]["name"], "game_type": str(e["season"]["type"]),
                        "game_number": None, "rescheduled": False,
                        "home": side.get("home"), "away": side.get("away")})
        return out

    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, d) for d in dates]):
            rows += f.result()
    return rows


def match(kal: pl.DataFrame, src: pl.DataFrame, league: str) -> pl.DataFrame:
    """Kalshi event -> source game on (local date, team pair).

    On the ticker's date: one game is it; doubleheaders go by the ticker's
    G1/G2 suffix (older tickers), else the day's own game over a makeup when
    Kalshi listed one event for the date, else the ticker's HH:MM (ET), else
    the latest start before the event closed.
    If that game was postponed, the event may have been carried to the makeup:
    the same game id on a later date (statsapi keeps it), else any final game
    between the pair that started within 12 h before the event closed. Nothing
    on the date: a game a day either side, only if the event closed within
    12 h of its start (a series plays the same pair on consecutive days), else
    a makeup as above."""
    src = src.with_columns(pl.concat_list("home", "away").list.sort().list.join("-").alias("teams"))
    kal = kal.with_columns(pl.len().over("series_ticker", "date_local", "teams").alias("n_same"))
    pairs = {k[0]: g.sort("start_utc") for k, g in src.group_by("teams")}
    window = dt.timedelta(hours=12)
    out = []
    for r in kal.iter_rows(named=True):
        hit, how = None, "none"
        c = pairs.get(r["teams"]) if r["teams"] and r["date_local"] else None
        if c is not None:
            close = r["close_time"]
            ok = lambda row: row["start_utc"] <= close <= row["start_utc"] + window  # noqa: E731
            final = lambda row: row["status"] in FINAL[league]                       # noqa: E731
            day = c.filter(pl.col("date_local") == r["date_local"])
            if day.height:
                if day.height == 1:
                    hit, how = day.row(0, named=True), "date"
                elif r["game_no"] and r["game_no"] in day["game_number"].to_list():
                    hit, how = day.filter(pl.col("game_number") == r["game_no"]).row(0, named=True), "date,G#"
                elif r["n_same"] == 1 and day.filter(~pl.col("rescheduled")).height == 1:
                    # a doubleheader of the day's own game and a makeup, and Kalshi
                    # listed one event for the date: it is the day's own game (the
                    # makeup's placeholder start, 5 min after game 1, would mislead)
                    hit, how = day.filter(~pl.col("rescheduled")).row(0, named=True), "date,original"
                elif r["hhmm"]:
                    want = dt.datetime.combine(r["date_local"], dt.time(int(r["hhmm"][:2]), int(r["hhmm"][2:])), ET)
                    i = min(range(day.height), key=lambda i: abs((day["start_utc"][i] - want).total_seconds()))
                    hit, how = day.row(i, named=True), "date,hhmm"
                else:
                    before = day.filter(pl.col("start_utc") <= close)
                    hit, how = (before if before.height else day).row(-1, named=True), "date,close"
                if not final(hit):
                    later = [x for x in c.iter_rows(named=True)
                             if x["date_local"] > r["date_local"] and final(x) and ok(x)]
                    same = [x for x in later if x["game_id"] == hit["game_id"]]
                    if same or later:
                        hit, how = (same or later)[-1], "makeup"
            else:
                near = [x for x in c.iter_rows(named=True)
                        if abs((x["date_local"] - r["date_local"]).days) == 1 and ok(x)]
                later = [x for x in c.iter_rows(named=True)
                         if x["date_local"] > r["date_local"] and final(x) and ok(x)]
                if near:
                    hit, how = near[-1], "date+-1"
                elif later:     # the schedule lists a postponed game only on its makeup date
                    hit, how = later[-1], "makeup"
        out.append({**r, "match": how,
                    **{k: (hit or {}).get(k) for k in ("game_id", "start_utc", "status", "game_type",
                                                          "game_number", "rescheduled", "home", "away")}})
    return pl.DataFrame(out, infer_schema_length=None)


def stage_schedule(cl: Client, workers: int) -> None:
    kal = kalshi_games()
    frames = []
    for lg in ("MLB", "NBA", "NFL"):
        k = kal.filter(pl.col("league") == lg)
        ds = k["date_local"].drop_nulls()
        if lg == "MLB":
            src = schedule_mlb(cl, ds.min() - dt.timedelta(days=1), ds.max() + dt.timedelta(days=1))
        else:
            dates = sorted({d + dt.timedelta(days=o) for d in ds for o in (-1, 0, 1)})
            src = schedule_espn(cl, lg, dates, workers)
        # a postponed MLB game appears twice under one id (original date, makeup): keep both
        src = pl.DataFrame(src, infer_schema_length=None).unique(["game_id", "date_local"], keep="last")
        frames.append(match(k, src, lg).with_columns(pl.lit("statsapi" if lg == "MLB" else "espn").alias("source")))
    games = pl.concat(frames, how="diagonal_relaxed")

    # ESPN season type 1 = preseason: listed, never pulled or built (MLB spring
    # training, game_type S, is kept)
    games = games.with_columns(((pl.col("source") == "espn") & (pl.col("game_type") == "1")).alias("preseason"))
    # the nflverse game id, for joining the season files
    games = games.with_columns(
        pl.when(pl.col("league") == "NFL").then(pl.format(
            "{}_{}_{}", pl.col("start_utc").dt.convert_time_zone("America/New_York").dt.date(),
            pl.col("away").replace(ESPN_TO_NFLVERSE), pl.col("home").replace(ESPN_TO_NFLVERSE)))
        .alias("nflverse_key"))

    for hold, root in ((False, OUT), (True, HOLD)):
        root.mkdir(parents=True, exist_ok=True)
        games.filter(pl.col("holdout") == hold).write_parquet(root / "games.parquet")
    print("Kalshi game events -> source games (holdout: counts only)")
    print(games.group_by("league", "holdout", "match").agg(pl.len().alias("events"),
                                                            pl.col("game_id").n_unique().alias("games"))
               .sort("league", "holdout", "match"))
    un = games.filter(pl.col("game_id").is_null(), ~pl.col("holdout"))
    if un.height:
        print(f"\nunmatched, working block ({un.height}):")
        print(un.select("event_ticker", "teams", "close_time").head(30))
    st = games.filter(pl.col("game_id").is_not_null()).unique(["league", "game_id"])
    print("\nstatus of matched games:")
    print(st.group_by("league", "holdout", "status").len().sort("league", "holdout", "len", descending=[False, False, True]))


# --------------------------------------------------------------------------- pull

def load_games(final_only: bool = True) -> pl.DataFrame:
    g = pl.concat([pl.read_parquet(r / "games.parquet") for r in (OUT, HOLD)], how="diagonal_relaxed")
    g = g.filter(pl.col("game_id").is_not_null(), ~pl.col("preseason"))
    is_final = pl.struct("league", "status").map_elements(lambda s: s["status"] in FINAL[s["league"]],
                                                          return_dtype=pl.Boolean)
    if final_only:
        g = g.filter(is_final)
    # a game any of whose Kalshi events is holdout is holdout
    return (g.sort("start_utc")
             .group_by("league", "game_id")
             .agg(pl.col("holdout").any(), pl.col("status").last(), pl.col("start_utc").last(),
                  pl.col("nflverse_key").last(), pl.col("event_ticker"))
             .sort("league", "start_utc"))


def raw_path(league: str, game_id: str, hold: bool) -> Path:
    return (HOLD if hold else OUT) / "raw" / league / f"{game_id}.json.gz"


def fetch_game(cl: Client, league: str, game_id: str) -> bytes:
    if league == "MLB":
        feed = cl.json(f"{MLB_API}/api/v1.1/game/{game_id}/feed/live")
        wp = cl.json(f"{MLB_API}/api/v1/game/{game_id}/winProbability") or []
        # the win-prob response repeats every play; keep only what the feed lacks
        wp = [{k: w.get(k) for k in ("atBatIndex", "homeTeamWinProbability", "awayTeamWinProbability",
                                     "homeTeamWinProbabilityAdded")} for w in wp]
        doc = {"feed": feed, "win_probability": wp}
    else:
        doc = cl.json(f"{ESPN}/{ESPN_PATH[league]}/summary", {"event": game_id})
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode())


def write_atomic(p: Path, b: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(b)
    tmp.replace(p)


def nfl_seasons(games: pl.DataFrame) -> dict[int, bool]:
    """season -> holdout, for the nflverse files (a season is holdout if any game is)."""
    nfl = games.filter(pl.col("league") == "NFL").with_columns(
        pl.when(pl.col("start_utc").dt.month() >= 3).then(pl.col("start_utc").dt.year())
          .otherwise(pl.col("start_utc").dt.year() - 1).alias("season"))
    return dict(nfl.group_by("season").agg(pl.col("holdout").any()).iter_rows())


def stage_pull(cl: Client, workers: int, leagues: list[str]) -> None:
    games = load_games().filter(pl.col("league").is_in(leagues))
    todo = [r for r in games.iter_rows(named=True) if not raw_path(r["league"], r["game_id"], r["holdout"]).exists()]
    print(f"{games.height:,} final games, {games.height - len(todo):,} already on disk, {len(todo):,} to pull")

    if "NFL" in leagues:
        for season, hold in sorted(nfl_seasons(games).items()):
            p = (HOLD if hold else OUT) / "raw" / "NFL" / f"nflverse_pbp_{season}.parquet"
            r = cl.get(NFLVERSE.format(season))
            if r is None:
                print(f"nflverse {season}: not published")
                continue
            write_atomic(p, r.content)
            print(f"nflverse {season}: {len(r.content) / 2**20:.1f} MB" + ("  [holdout]" if hold else ""))

    t0, done, size, lock = time.time(), 0, 0, threading.Lock()

    def one(r: dict) -> None:
        nonlocal done, size
        b = fetch_game(cl, r["league"], r["game_id"])
        write_atomic(raw_path(r["league"], r["game_id"], r["holdout"]), b)
        with lock:
            done += 1
            size += len(b)
            if done % 250 == 0 or done == len(todo):
                print(f"[{done:,}/{len(todo):,}] games  {size / 2**20:,.0f} MB  "
                      f"{(time.time() - t0) / 60:.0f} min  ({cl.n:,} requests)", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            f.result()


# --------------------------------------------------------------------------- build

def parse_mlb(game_id: str, doc: dict) -> list[dict]:
    """One row per play event (pitch, pickoff, action such as a stolen base or
    wild-pitch run), timed at its startTime. Score is the score after the
    event: actions carry it, the last event of a plate appearance gets the
    plate appearance's result, other events carry the running score."""
    feed = doc["feed"]
    wp = {w["atBatIndex"]: w["homeTeamWinProbability"] for w in doc.get("win_probability") or []}
    rows, home, away, seq = [], 0, 0, 0
    for p in feed["liveData"]["plays"]["allPlays"]:
        a, res, evs = p["about"], p["result"], p["playEvents"]
        for i, e in enumerate(evs):
            d = e.get("details", {})
            last = i == len(evs) - 1
            if "homeScore" in d:
                home, away = d["homeScore"], d["awayScore"]
            if last:
                home, away = res.get("homeScore", home), res.get("awayScore", away)
            c = e.get("count", {})
            rows.append({
                "game_id": game_id, "seq": seq, "t": ts(e.get("startTime")), "t_end": ts(e.get("endTime")),
                "period": a["inning"], "half": a["halfInning"], "clock": None,
                "home_score": home, "away_score": away,
                "event_type": e.get("type"), "is_pitch": e.get("isPitch", False),
                "description": d.get("description"),
                "is_scoring": bool(d.get("isScoringPlay")) or (last and a.get("isScoringPlay", False)),
                "at_bat_index": a["atBatIndex"], "pa_last_event": last,
                "pa_event": res.get("eventType") if last else None,
                "balls": c.get("balls"), "strikes": c.get("strikes"), "outs": c.get("outs"),
                "pitch_type": (d.get("type") or {}).get("code") if e.get("isPitch") else None,
                "is_in_play": d.get("isInPlay"),
                # MLB's win prob after the plate appearance, on its last event
                "home_wp_post": wp[a["atBatIndex"]] / 100 if last and a["atBatIndex"] in wp else None,
            })
            seq += 1
    return rows


def parse_espn(league: str, game_id: str, doc: dict) -> list[dict]:
    """One row per ESPN play, timed at its wallclock (NFL: the snap). Scores
    are after the play; home_wp_post is ESPN's home win prob after it."""
    wp = {w.get("playId"): w.get("homeWinPercentage") for w in doc.get("winprobability") or []}
    if league == "NFL":
        plays = [p for d in (doc.get("drives") or {}).get("previous", []) for p in d.get("plays", [])]
    else:
        plays = doc.get("plays") or []
    rows = []
    # the list is in game order; sequenceNumber is not (corrections get late numbers)
    for i, p in enumerate(plays):
        start = p.get("start") or {}
        rows.append({
            "game_id": game_id, "seq": i, "espn_seq": int(p.get("sequenceNumber") or 0),
            "t": ts(p.get("wallclock")),
            "t_end": None, "period": (p.get("period") or {}).get("number"), "half": None,
            "clock": (p.get("clock") or {}).get("displayValue"),
            "home_score": p.get("homeScore"), "away_score": p.get("awayScore"),
            "event_type": (p.get("type") or {}).get("text"), "description": p.get("text"),
            "is_scoring": bool(p.get("scoringPlay")), "score_value": p.get("scoreValue"),
            "team_id": ((p.get("team") or {}).get("id") or ((start.get("team") or {}).get("id"))),
            "down": start.get("down"), "distance": start.get("distance"), "yard_line": start.get("yardLine"),
            "home_wp_post": wp.get(p.get("id")),
            "espn_play_id": p.get("id"),
        })
    return rows


def stage_build(workers: int) -> None:
    games = load_games(final_only=False)
    for hold, root in ((False, OUT), (True, HOLD)):
        for lg in ("MLB", "NBA", "NFL"):
            g = games.filter(pl.col("league") == lg, pl.col("holdout") == hold)
            paths = [(r["game_id"], raw_path(lg, r["game_id"], hold)) for r in g.iter_rows(named=True)]
            paths = [(gid, p) for gid, p in paths if p.exists()]
            if not paths:
                continue

            def one(item):
                gid, p = item
                doc = json.loads(gzip.decompress(p.read_bytes()))
                return parse_mlb(gid, doc) if lg == "MLB" else parse_espn(lg, gid, doc)

            rows = []
            with ThreadPoolExecutor(workers) as ex:
                for r in ex.map(one, paths):
                    rows += r
            ev = (pl.DataFrame(rows, infer_schema_length=None)
                    .with_columns(pl.lit(lg).alias("league"))
                    .sort("game_id", "seq"))
            # ESPN sometimes stamps part of a game a whole day off (time of day
            # right): move a time outside [start - 1 h, start + 8 h] back into it
            # when a whole-day shift does. MLB is left alone: its out-of-window
            # rows are pre-game notes and suspended games resumed another day.
            # t_outside_window flags what is still outside, after that.
            ev = ev.join(g.select("game_id", "start_utc"), on="game_id", how="left")
            h = (pl.col("t") - pl.col("start_utc")).dt.total_seconds() / 3600
            inside = lambda x: x.is_between(-1, 8)  # noqa: E731
            if lg != "MLB":
                days = (h / 24).round()
                fix = ~inside(h) & (days != 0) & inside(h - days * 24)
                ev = ev.with_columns(pl.when(fix).then(days).otherwise(0).cast(pl.Int8).alias("t_day_shift"))
                ev = ev.with_columns(pl.col("t") - pl.duration(days=pl.col("t_day_shift")))
            ev = ev.with_columns((~inside(h)).fill_null(True).alias("t_outside_window")).drop("start_utc")
            ev.write_parquet(root / f"events_{lg}.parquet")
            per = ev.group_by("game_id").agg(pl.len().alias("n"), pl.col("t").is_null().mean().alias("null_t"),
                                             pl.col("t_outside_window").mean().alias("outside"))
            if hold:
                print(f"{lg} [holdout]: {len(paths):,} games, {ev.height:,} rows")
            else:
                print(f"{lg}: {len(paths):,} games, {ev.height:,} rows "
                      f"({(root / f'events_{lg}.parquet').stat().st_size / 2**20:.1f} MB); "
                      f"games with no timed rows {per.filter(pl.col('null_t') == 1).height}, "
                      f"median share untimed {per['null_t'].median():.3f}, "
                      f"rows outside the game window {ev['t_outside_window'].mean():.4f}"
                      + (f", day-shifted {(ev['t_day_shift'] != 0).sum():,} rows in "
                         f"{ev.filter(pl.col('t_day_shift') != 0)['game_id'].n_unique()} games" if lg != "MLB" else ""))


# --------------------------------------------------------------------------- estimate

def stage_estimate(cl: Client, workers: int, rates: dict[str, float], per_league: int = 3) -> None:
    """Pull a few working-block games per league into memory, measure raw
    (gzipped) and built (parquet) bytes and rows, and scale to every game."""
    games = load_games()
    have = {(lg, gid) for lg in ("MLB", "NBA", "NFL") for gid in
            (p.name.removesuffix(".json.gz") for root in (OUT, HOLD) for p in (root / "raw" / lg).glob("*.json.gz"))}
    print("final games to pull (all blocks):")
    print(games.group_by("league", "holdout").len().sort("league", "holdout"))
    total_raw = total_built = total_req = 0
    reqs = {"MLB": 2, "NBA": 1, "NFL": 1}
    secs = {}                                            # host -> seconds of pulling
    for lg in ("MLB", "NBA", "NFL"):
        g = games.filter(pl.col("league") == lg)
        if not g.height:
            continue
        sample = g.filter(~pl.col("holdout")).sample(min(per_league, g.height), seed=0)
        raw = built = n = 0
        t0 = time.monotonic()
        for r in sample.iter_rows(named=True):
            b = fetch_game(cl, lg, r["game_id"])
            doc = json.loads(gzip.decompress(b))
            rows = parse_mlb(r["game_id"], doc) if lg == "MLB" else parse_espn(lg, r["game_id"], doc)
            tmp = OUT / f".estimate_{lg}.parquet"
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(tmp)
            raw, built, n = raw + len(b), built + tmp.stat().st_size, n + len(rows)
            tmp.unlink()
        k = sample.height
        latency = (time.monotonic() - t0) / (k * reqs[lg])   # s per request, serial (includes the rate gap)
        left = g.filter(~pl.struct("league", "game_id").map_elements(
            lambda s: (s["league"], s["game_id"]) in have, return_dtype=pl.Boolean)).height
        print(f"{lg}: {g.height:,} games ({left:,} not on disk); per game {raw / k / 1024:.0f} KB raw gz, "
              f"{built / k / 1024:.0f} KB built, {n / k:.0f} rows -> "
              f"raw {raw / k * g.height / 2**20:,.0f} MB, built {built / k * g.height / 2**20:,.0f} MB "
              f"(built size is an upper bound: one file per league compresses better), "
              f"{reqs[lg] * left:,} requests")
        host = "statsapi.mlb.com" if lg == "MLB" else "site.api.espn.com"
        per_s = min(rates[host], workers / latency)          # the rate cap, unless latency binds first
        secs[host] = secs.get(host, 0) + reqs[lg] * left / per_s
        print(f"     ~{per_s:.1f} req/s ({latency:.2f} s per request serially) -> {reqs[lg] * left / per_s / 60:,.0f} min")
        total_raw += raw / k * g.height
        total_built += built / k * g.height
        total_req += reqs[lg] * left
    nfl_mb = 0.0
    for season in nfl_seasons(games):
        r = cl.c.head(NFLVERSE.format(season), follow_redirects=True)
        nfl_mb += int(r.headers.get("content-length", 0)) / 2**20
    print(f"nflverse season files: {nfl_mb:.0f} MB")
    print(f"\ntotal: ~{(total_raw + total_built) / 2**30 + nfl_mb / 1024:.2f} GB on disk "
          f"(raw {total_raw / 2**30:.2f} GB + built {total_built / 2**30:.2f} GB + nflverse {nfl_mb / 1024:.2f} GB), "
          f"{total_req:,} requests")
    # the two hosts are rate-limited separately and pulled at once, so the slower one sets the time
    print("time: " + ", ".join(f"{h} {s / 60:,.0f} min" for h, s in secs.items())
          + f" -> ~{max(secs.values(), default=0) / 60:,.0f} min wall clock with --workers {workers}")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["schedule", "estimate", "pull", "build"])
    ap.add_argument("--leagues", nargs="+", default=["MLB", "NBA", "NFL"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mlb-rate", type=float, default=8.0, help="requests/s to statsapi.mlb.com")
    ap.add_argument("--espn-rate", type=float, default=4.0, help="requests/s to site.api.espn.com (unofficial API)")
    a = ap.parse_args()
    rates = {"statsapi.mlb.com": a.mlb_rate, "site.api.espn.com": a.espn_rate}
    cl = Client(rates)
    if a.stage == "schedule":
        stage_schedule(cl, a.workers)
    elif a.stage == "estimate":
        stage_estimate(cl, a.workers, rates)
    elif a.stage == "pull":
        stage_pull(cl, a.workers, a.leagues)
    else:
        stage_build(a.workers)
    print(f"requests made: {cl.n:,}")


if __name__ == "__main__":
    sys.exit(main())
