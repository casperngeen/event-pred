#!/bin/sh
# In a sandbox: start today's runners (the US Eastern date). Lives in /home/casperngeen, which every sandbox on the
# home site mounts. MLB publishes its totals signals to Lumid (the PAT in .lumid_pat); NFL is a dry run; NBA is
# recorded only.
# A runner that fails at start (e.g. Kalshi rate-limits the contract lookup) is retried up to 5 times.
H=/home/casperngeen
D=$(TZ=America/New_York date +%F)
mkdir -p $H/sports_live/logs
cd $H/app
# the Kalshi key: signed requests (20 req/s instead of ~4) and the runners' WebSocket trade stream
if [ -s $H/.kalshi_key.pem ] && [ -s $H/.kalshi_key_id ]; then
  export KALSHI_KEY_ID=$(cat $H/.kalshi_key_id) KALSHI_KEY_FILE=$H/.kalshi_key.pem
fi
export PYTHONPATH=$H/pylib                      # websockets
run() {   # league, extra args
  L=$1; shift
  LUMID_PAT=$(cat $H/.lumid_pat) setsid nohup sh -c "n=0; until python3 -u -W ignore scripts/live_trade.py --league $L \
    --out $H/sports_live $*; do n=\$((n+1)); [ \$n -ge 5 ] && break; echo 'restart in 60 s'; sleep 60; done" \
    >> $H/sports_live/logs/${D}_$L.log 2>&1 < /dev/null &
}
# --delay 10: decide on the timing the models and the skip rule were validated on (a pitch counts at its archive-style
# end + 10 s, or when it arrives if later) until the 5-s models are tested; the faster feed then only removes late arrivals
run MLB --publish --publish-kinds total --no-books --delay 10
run NFL --no-books
# NBA: recorded only (feed, trades, quotes), to measure ESPN's NBA feed delay; there is no NBA trading runner yet
setsid nohup sh -c "n=0; until python3 -u -W ignore scripts/live_record.py --league NBA --no-books --rate 4 \
  --out $H/sports_live; do n=\$((n+1)); [ \$n -ge 5 ] && break; echo 'restart in 60 s'; sleep 60; done" \
  >> $H/sports_live/logs/${D}_NBA.log 2>&1 < /dev/null &
# the NBA's own play-by-play (stats.nba.com), to compare its delay with ESPN's
setsid nohup python3 -u -W ignore scripts/live_nba_stats.py --out $H/sports_live \
  >> $H/sports_live/logs/${D}_NBA_stats.log 2>&1 < /dev/null &
# every order-book change and trade over Kalshi's WebSocket (analysis only), if the Kalshi key has been copied in
if [ -s $H/.kalshi_key.pem ] && [ -s $H/.kalshi_key_id ]; then
  setsid nohup python3 -u -W ignore scripts/live_books_ws.py --leagues MLB NFL NBA --out $H/sports_live \
    >> $H/sports_live/logs/${D}_books_ws.log 2>&1 < /dev/null &
fi
sleep 45
tail -n 4 $H/sports_live/logs/${D}_*.log
