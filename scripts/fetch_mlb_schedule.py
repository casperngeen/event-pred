"""MLB regular-season schedule and final scores from statsapi, for standings and the season simulator.

Kalshi listed MLB game markets only from 2025-04-16, so the as-of layer's game results miss
2025-03-27 .. 04-15 (about 18 games per team): standings built from them alone undercount all
season. This fills that in from the source. Team codes match Kalshi's (checked 2026-10-03).

    venv/bin/python scripts/fetch_mlb_schedule.py              # 2025 and 2026 regular seasons

Writes data/sports/pbp/mlb_schedule.parquet (git-ignored), one row per game (gamePk):
    league, season, game_id, game_type, sched_start, home, away, status,
    home_score, away_score, first_pitch, known_at

``known_at`` is when the result was known: first pitch + game duration + delays (statsapi's
gameInfo), else the scheduled start + 5 h. A suspended game listed on two dates keeps its
resumption. Holdout: scores of games whose result is known on or after HOLDOUT_START are not
stored (the schedule itself, published before the season, is kept).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import httpx
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sports import HOLDOUT_START, OUT as SPORTS  # noqa: E402

MLB_API = "https://statsapi.mlb.com/api/v1/schedule"
FINAL = {"Final", "Game Over", "Completed Early"}
FALLBACK = dt.timedelta(hours=5)


def ts(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def fetch(season: int) -> list[dict]:
    r = httpx.get(MLB_API, params={"sportId": 1, "season": season, "gameType": "R",
                                   "hydrate": "team,linescore,gameInfo"}, timeout=120)
    r.raise_for_status()
    rows = []
    for day in r.json().get("dates", []):
        for g in day["games"]:
            t, info = g["teams"], g.get("gameInfo") or {}
            start = ts(g["gameDate"])
            first = ts(info.get("firstPitch"))
            dur = info.get("gameDurationMinutes")
            final = g["status"]["detailedState"] in FINAL
            end = (first + dt.timedelta(minutes=dur + (info.get("delayDurationMinutes") or 0))
                   if first and dur else start + FALLBACK)
            keep_score = final and end < HOLDOUT_START      # the result, not the start, must precede the wall
            rows.append({
                "league": "MLB", "season": season, "game_id": str(g["gamePk"]), "game_type": g["gameType"],
                "sched_start": start, "home": t["home"]["team"]["abbreviation"],
                "away": t["away"]["team"]["abbreviation"], "status": g["status"]["detailedState"],
                "home_score": t["home"].get("score") if keep_score else None,
                "away_score": t["away"].get("score") if keep_score else None,
                "first_pitch": first if keep_score else None, "known_at": end if keep_score else None,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2025, 2026])
    a = ap.parse_args()
    df = pl.DataFrame([r for s in a.seasons for r in fetch(s)], schema_overrides={
        "home_score": pl.Int64, "away_score": pl.Int64, "first_pitch": pl.Datetime("us", "UTC"),
        "known_at": pl.Datetime("us", "UTC"), "sched_start": pl.Datetime("us", "UTC")})
    # one row per game: a suspended game appears on its original and resumption dates; keep the
    # last (final) entry, and a played game over a postponed placeholder
    df = (df.with_columns(pl.col("status").is_in(list(FINAL)).alias("_final"))
          .sort("game_id", "_final", "sched_start").group_by("game_id", maintain_order=True).last().drop("_final"))
    out = SPORTS / "pbp" / "mlb_schedule.parquet"
    df.sort("sched_start").write_parquet(out)
    fin = df.filter(pl.col("home_score").is_not_null())
    print(f"{out}: {df.height} games, {fin.height} with scores "
          f"(per season: {dict(fin.group_by('season').len().sort('season').iter_rows())})")


if __name__ == "__main__":
    main()
