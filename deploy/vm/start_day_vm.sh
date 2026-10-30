#!/bin/bash
# On the backup server: the day's runners, as the Lumid sandbox runs them (deploy/lumid/start_day.sh) but all dry runs
# (only the sandbox publishes), each restarted up to 5 times if it fails. Started each day by systemd
# (fyp-day.timer, 12:00 UTC); the runners wait for their games themselves. Argument: ET date (default today's).
# The Kalshi key is shared with the sandbox: each copy uses about half of its 20 requests/s.
F=$HOME/fyp
D=${1:-$(TZ=America/New_York date +%F)}
LOG=$F/sports_live/logs
mkdir -p $LOG
cd $F/app
# order books an earlier day's stream left raw (it gzips them only when it ends normally): ~20x smaller gzipped
find $F/sports_live -name books_ws.jsonl -not -path "*/$D/*" -exec gzip -f {} \; 2>> $LOG/${D}_start.log &
if [ -s $F/.kalshi_key.pem ] && [ -s $F/.kalshi_key_id ]; then
  export KALSHI_KEY_ID=$(cat $F/.kalshi_key_id) KALSHI_KEY_FILE=$F/.kalshi_key.pem
else
  echo "no Kalshi key in $F: unsigned requests (~4/s) and no WebSocket streams" >> $LOG/${D}_start.log
fi
start() {   # name, command: up to 5 tries; logs/<date>_<name>.done once it finishes normally
  N=$1; shift
  if pgrep -f -- "--date $D .*# $N\$" > /dev/null; then echo "$N already running for $D" >> $LOG/${D}_start.log; return; fi
  setsid nohup bash -c "n=0; until $*; do n=\$((n+1)); [ \$n -ge 5 ] && exit 1; echo 'restart in 60 s'; sleep 60; done; \
    touch $LOG/${D}_$N.done # $N" >> $LOG/${D}_$N.log 2>&1 < /dev/null &
}
P="$F/venv/bin/python -u -W ignore scripts"
O="--date $D --out $F/sports_live --no-books"
M60=gru_d60_h30-60-120-1800_tp
start MLB $P/live_trade.py --league MLB $O --delay 10 --rate 3
start NFL $P/live_trade.py --league NFL $O --model $M60 --delay 60 --rate 3
start NBA $P/live_trade.py --league NBA $O --model $M60 --delay 60 --rate 4 --late-block
[ -n "${KALSHI_KEY_ID:-}" ] && start books_ws $P/live_books_ws.py --leagues MLB NFL NBA --date $D --out $F/sports_live
sleep 30
tail -n 3 $LOG/${D}_*.log
