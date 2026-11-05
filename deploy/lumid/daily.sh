#!/bin/bash
# On the laptop, once a day (launchd, deploy/lumid/com.fyp.lumid-ingame.plist, 12:30 UTC): replace yesterday's sandbox
# with a fresh 24-h one and start that day's runners in it (start_day.sh, kept in the sandbox's persistent /home),
# LEAD_MIN before the day's first MLB / NFL / NBA game (the runners' recorder and order-book stream start 30 min
# before each game), but no later than LATEST_UTC (00:00 Singapore): the laptop creates the sandbox and is usually
# asleep in the Singapore early morning, when most US games start. The start time goes to logs/start_at_<date> (epoch s), so the watchdog leaves the day alone
# until then. No games: no sandbox. Schedule lookup failed: start now.
# Yesterday's games end by ~06:00 UTC, so deleting yesterday's sandbox at the new start loses nothing.
set -u
. "$(dirname "$0")/sandbox_lib.sh"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOGS=$ROOT/data/sports_live/logs
LEAD_MIN=45
LATEST_UTC=16:00
D=$(TZ=America/New_York date +%F)
NAME=ingame-$(TZ=America/New_York date +%m%d)
echo "== $(date -u +%FT%TZ) $NAME"
# the backup server's recordings so far (deploy/vm), if it is set up
grep -q "^Host fyp-vm" ~/.ssh/config 2>/dev/null && "$ROOT/deploy/vm/pull.sh"
first=$(cd "$ROOT" && venv/bin/python -W ignore -c "
import sys, datetime as dt; sys.path.insert(0, 'scripts')
import live_record as R
d = dt.date.fromisoformat('$D')
s = [dt.datetime.fromisoformat(g['start'].replace('Z', '+00:00')) for L in ('MLB', 'NFL', 'NBA') for g in R.schedule(L, d)]
print(int(min(s).timestamp()) if s else 'none')" 2>/dev/null | tail -1)
case "$first" in
  none) echo "no games today: no sandbox"; echo 9999999999 > "$LOGS/start_at_$D"; exit 0 ;;
  ''|*[!0-9]*) echo "schedule lookup failed: starting now"; date +%s > "$LOGS/start_at_$D" ;;
  *) wake=$((first - LEAD_MIN * 60))
     latest=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%F) $LATEST_UTC" +%s)
     [ $wake -gt $latest ] && wake=$latest
     echo $wake > "$LOGS/start_at_$D"
     echo "first game $(date -u -r $first +%H:%MZ): starting at $(date -u -r $wake +%H:%MZ)"
     # a loop, not one long sleep: sleep does not count time the laptop spends asleep
     while [ "$(date +%s)" -lt $wake ]; do sleep 60; done ;;
esac
lock
recreate $NAME $D
