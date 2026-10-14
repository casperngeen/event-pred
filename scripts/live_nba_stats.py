"""Record the NBA's own play-by-play (stats.nba.com playbyplayv3) as it arrives, to compare its delay with ESPN's.
The NBA's live CDN feed (cdn.nba.com liveData) refuses us (HTTP 403 from the laptop and the sandbox); stats.nba.com
answers when sent the browser headers nba_api uses. Its actions carry the game clock but no wall-clock time, so
the delay is measured against ESPN: the same play (period, clock, score) in ESPN's feed has ESPN's wallclock, and
both feeds' first-arrival times are logged.

Per game (scoreboardv3 for the US Eastern date), from 10 min before the start until the game is final + 10 min,
playbyplayv3 with a cache-busting parameter (the response says max-age=120), every POLL_S while the ball can be in
play and more slowly when it can't, to keep the load on stats.nba.com down:

    before the first action     PREGAME_S
    between periods             BREAK_S   (from a period's "end" action until the next period's "start")
    after a timeout             BREAK_S   for TIMEOUT_S, unless a period starts or ends
    after the final             FINAL_S   (to catch late corrections)
    otherwise (in play)         POLL_S

    <out>/<date>/NBA_stats_<gameId>/game.json        the game (ids, teams, start)
    <out>/<date>/NBA_stats_<gameId>/actions.jsonl    each action the first time we saw it, + received_at
    <out>/<date>/NBA_stats_<gameId>/polls.jsonl      each poll: received_at, status, seconds taken, actions, mode

    python scripts/live_nba_stats.py [--date 2026-10-13] [--out /home/casperngeen/sports_live] [--games 0012600045 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_record as R  # noqa: E402

STATS = "https://stats.nba.com/stats"
HEADERS = {   # as nba_api sends them; without these stats.nba.com lets the request hang
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive", "Referer": "https://www.nba.com/", "Pragma": "no-cache", "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"', "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest": "empty"}
POLL_S, BREAK_S, PREGAME_S, FINAL_S, STATUS_S = 2.0, 10.0, 15.0, 30.0, 60.0
TIMEOUT_S = 60.0                                     # a timeout lasts 75-120 s; play can't resume much sooner
BEFORE, AFTER, MAX_GAME = dt.timedelta(minutes=10), dt.timedelta(minutes=10), dt.timedelta(hours=4)


def get(hc: httpx.Client, path: str, params: dict) -> httpx.Response:
    return hc.get(f"{STATS}/{path}", params=params | {"_": str(random.getrandbits(48))})


def games(hc: httpx.Client, day: dt.date) -> list[dict]:
    d = get(hc, "scoreboardv3", {"GameDate": day.isoformat(), "LeagueID": "00"}).json()
    return [dict(game_id=g["gameId"], away=g["awayTeam"]["teamTricode"], home=g["homeTeam"]["teamTricode"],
                 start=g["gameTimeUTC"], status=g["gameStatus"]) for g in d["scoreboard"]["games"]]


_board: dict = {"at": -1e9, "status": {}}           # scoreboardv3, shared by the games' threads
_board_lock = threading.Lock()


def status(hc: httpx.Client, day: dt.date, gid: str) -> int | None:
    with _board_lock:
        if time.monotonic() - _board["at"] > STATUS_S:
            try:
                _board["status"] = {g["game_id"]: g["status"] for g in games(hc, day)}
                _board["at"] = time.monotonic()
            except Exception:
                pass
        return _board["status"].get(gid)


def interval(acts: list[dict], timeout_seen: float | None, final: bool) -> tuple[float, str]:
    """How long to wait before the next poll, and why."""
    if final:
        return FINAL_S, "final"
    if not acts:
        return PREGAME_S, "pregame"
    last = next((a for a in reversed(acts) if a["actionType"] == "period"), None)
    if last is not None and last["subType"] == "end":
        return BREAK_S, "break"
    if timeout_seen is not None and time.monotonic() - timeout_seen < TIMEOUT_S:
        return BREAK_S, "timeout"
    return POLL_S, "play"


def record(g: dict, day: dt.date, out: Path) -> None:
    root = out / day.isoformat() / f"NBA_stats_{g['game_id']}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "game.json").write_text(json.dumps(g, indent=1))
    start = dt.datetime.fromisoformat(g["start"].replace("Z", "+00:00"))
    wait = (start - BEFORE - R.now()).total_seconds()
    if wait > 0:
        time.sleep(wait)
    print(f"{g['game_id']} {g['away']}@{g['home']}: recording from {R.iso(R.now())}", flush=True)
    hc, lock = httpx.Client(headers=HEADERS, timeout=20), threading.Lock()
    seen: dict[int, str] = {}                         # actionId -> the action as last seen
    final_at, last_status, acts, timeout_seen = None, 0.0, [], None
    while R.now() < start + MAX_GAME and not (final_at and R.now() > final_at + AFTER):
        t0 = time.monotonic()
        got, rec = R.now(), {"received_at": R.iso(R.now())}
        try:
            r = get(hc, "playbyplayv3", {"GameID": g["game_id"], "StartPeriod": 0, "EndPeriod": 0})
            rec |= {"status": r.status_code, "seconds": round(time.monotonic() - t0, 3)}
            if r.status_code == 200:
                acts = (r.json().get("game") or {}).get("actions") or acts
                new = []
                for a in acts:
                    k = json.dumps(a, sort_keys=True)
                    if seen.get(a["actionId"]) != k:              # new, or changed since we last saw it (actionId is
                        new.append({**a, "received_at": R.iso(got), "changed": a["actionId"] in seen})   # unique;
                        seen[a["actionId"]] = k                   # a block / steal shares its shot's actionNumber)
                        if a["actionType"] == "Timeout" and not new[-1]["changed"]:
                            timeout_seen = time.monotonic()
                        elif a["actionType"] == "period":
                            timeout_seen = None
                rec |= {"actions": len(acts), "new": len(new)}
                R.append(root / "actions.jsonl", new, lock)
        except Exception as e:
            rec |= {"error": repr(e)[:200]}
        if time.monotonic() - last_status > STATUS_S:
            last_status = time.monotonic()
            if final_at is None and status(hc, day, g["game_id"]) == 3:
                final_at = R.now()
                print(f"{g['game_id']}: final at {R.iso(final_at)}", flush=True)
        wait, rec["mode"] = interval(acts, timeout_seen, final_at is not None)
        R.append(root / "polls.jsonl", [rec], lock)
        time.sleep(max(0.0, wait - (time.monotonic() - t0)))
    print(f"{g['game_id']}: done, {len(seen)} actions", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=dt.date.fromisoformat, default=R.now().astimezone(R.ET).date())
    ap.add_argument("--out", type=Path, default=R.OUT)
    ap.add_argument("--games", nargs="*", help="NBA game ids (default: all that day)")
    a = ap.parse_args()
    with httpx.Client(headers=HEADERS, timeout=20) as hc:
        gs = [g for g in games(hc, a.date) if not a.games or g["game_id"] in a.games]
    for g in gs:
        print(f"NBA {g['game_id']} {g['away']}@{g['home']} {g['start']}", flush=True)
    th = [threading.Thread(target=record, args=(g, a.date, a.out)) for g in gs]
    for t in th:
        t.start()
    for t in th:
        t.join()


if __name__ == "__main__":
    main()
