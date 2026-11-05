#!/bin/bash
# On the laptop, every hour (launchd, deploy/lumid/com.fyp.lumid-watchdog.plist): if any of the day's runners is
# neither running nor finished normally (start_day.sh's .done markers), or the sandbox is gone or unreachable,
# recreate the sandbox and restart the whole day (sandbox_lib.sh recreate). Killing single runners is not possible:
# the sandbox refuses kill across SSH sessions.
# The "day" is the one daily.sh started last: the ET date of now - 12:30 h (daily.sh runs at 12:30 UTC, then waits
# until shortly before the first game: logs/start_at_<date>).
# At most 3 restarts a day, so a runner that always fails does not recreate the sandbox every hour.
set -u
. "$(dirname "$0")/sandbox_lib.sh"
MAX=3
STATE=$(dirname "$0")/../../data/sports_live/logs/watchdog_restarts
hm=$(date -u +%H%M)
[ "$hm" -ge 1215 ] && [ "$hm" -lt 1315 ] && exit 0          # daily.sh's slot
D=${WATCHDOG_DATE:-}                                            # (testing: a fixed day)
[ -n "$D" ] || D=$(python3 -c 'import datetime as d; print((d.datetime.now(d.timezone.utc) - d.timedelta(hours=12, minutes=30)).date())')
NAME=ingame-${D:5:2}${D:8:2}
# daily.sh starts the day only shortly before its first game (or not at all without games): nothing to watch before
AT=$(cat "$(dirname "$0")/../../data/sports_live/logs/start_at_$D" 2>/dev/null)
[ -n "$AT" ] && [ "$(date +%s)" -lt "$AT" ] && exit 0
lock

check() {   # prints DOWN <runner> for each dead runner, then OK; nothing if SSH fails
  $SSH "L=/home/casperngeen/sports_live/logs; D=$D
names=\$(cat \$L/\$D.runners 2>/dev/null); day=\" .*--date \$D\"
[ -n \"\$names\" ] || { names='MLB NFL NBA NBA_stats books_ws'; day=; }    # started before start_day.sh took a date
ps=\$(for p in /proc/[0-9]*; do tr '\0' ' ' < \$p/cmdline 2>/dev/null; echo; done)
for n in \$names; do
  [ -e \$L/\${D}_\$n.done ] && continue
  case \$n in MLB) pat='live_trade.py --league MLB';; NFL) pat='live_trade.py --league NFL';;
    NBA) pat='live_trade.py --league NBA';; NBA_stats) pat=live_nba_stats.py;; books_ws) pat=live_books_ws.py;; *) pat=\$n;; esac
  echo \"\$ps\" | grep -av 'echo DOWN' | grep -aq -- \"\$pat\$day\" || echo DOWN \$n
done
echo OK" 2>/dev/null
}

# the gateway sometimes answers 502 (SSH and the API, 19 Oct): a failed listing is no evidence the sandbox is gone,
# and one failed SSH is no evidence it is unreachable; recreating would kill healthy runners mid-game
list=$(sandboxes 2>/dev/null) || { echo "$(date -u +%FT%TZ) sandbox API not answering, skipping this hour"; exit 0; }
if ! echo "$list" | grep -q "^$NAME Running"; then
  why="sandbox $NAME not running ($(echo "$list" | tr '\n' ' '))"
else
  out=; for i in 1 2 3 4 5 6 7 8; do out=$(check); [ -n "$out" ] && break; sleep 20; done
  if [ -z "$out" ]; then why="sandbox unreachable over SSH"
  elif echo "$out" | grep -q DOWN; then why=$(echo "$out" | grep DOWN | tr '\n' ' ')
  else exit 0                                                   # all running or finished: silent
  fi
fi
n=$(grep -c "^$D " "$STATE" 2>/dev/null); n=${n:-0}
echo "== $(date -u +%FT%TZ) $NAME: $why (restarts so far today: $n)"
[ "$n" -ge $MAX ] && { echo "restart limit reached, leaving it"; exit 0; }
[ "${WATCHDOG_DRY:-}" = 1 ] && { echo "dry run: would recreate"; exit 0; }
echo "$D $(date -u +%FT%TZ)" >> "$STATE"
recreate $NAME $D
