#!/bin/sh
# In a sandbox: start the runners for one US Eastern date (argument YYYY-MM-DD; default today's). Lives in
# /home/casperngeen, which every sandbox on the home site mounts. MLB publishes its totals signals to Lumid (the PAT in
# .lumid_pat); NFL and NBA are dry runs.
# A runner that fails at start (e.g. Kalshi rate-limits the contract lookup) is retried up to 5 times. A runner that
# finishes normally leaves logs/<date>_<name>.done; logs/<date>.runners lists the runners started, so the laptop's
# watchdog (deploy/lumid/watchdog.sh) can tell a finished runner from a dead one.
H=/home/casperngeen
D=${1:-$(TZ=America/New_York date +%F)}
LOG=$H/sports_live/logs
mkdir -p $LOG
rm -f $LOG/${D}_*.done
cd $H/app
# the Kalshi key: signed requests (20 req/s instead of ~4) and the runners' WebSocket trade stream
if [ -s $H/.kalshi_key.pem ] && [ -s $H/.kalshi_key_id ]; then
  export KALSHI_KEY_ID=$(cat $H/.kalshi_key_id) KALSHI_KEY_FILE=$H/.kalshi_key.pem
fi
export PYTHONPATH=$H/pylib                      # websockets
start() {   # name, command: run it with up to 5 tries; mark it done only if it finishes normally
  N=$1; shift
  echo $N >> $LOG/${D}.runners
  LUMID_PAT=$(cat $H/.lumid_pat) setsid nohup sh -c "n=0; until $*; do n=\$((n+1)); [ \$n -ge 5 ] && exit 1; \
    echo 'restart in 60 s'; sleep 60; done; touch $LOG/${D}_$N.done" >> $LOG/${D}_$N.log 2>&1 < /dev/null &
}
: > $LOG/${D}.runners
P="python3 -u -W ignore scripts"
# --delay 10: decide on the timing the models and the skip rule were validated on (a pitch counts at its archive-style
# end + 10 s, or when it arrives if later) until the 5-s models are tested; the faster feed then only removes late arrivals
# Kalshi requests/s per runner (--rate): together at most 10, as the backup server (deploy/vm) shares the 20-a-second key
start MLB $P/live_trade.py --league MLB --date $D --out $H/sports_live --publish --publish-kinds total --no-books --delay 10 --rate 3
# NFL / NBA: ESPN delivers plays ~45 s late, so models trained at a 60-s feed delay, run at --delay 60 (journal 2026-10-23);
# NBA also skips decisions during reviews and while placeholder plays show; both skip while the feed is > 2 min behind;
# --late-block (NBA, 27 Oct): no new TOTAL entries in Q4 / overtime inside live_trade.LATE_RULE's window (spreads only tagged 'late')
M60=gru_d60_h30-60-120-1800_tp
# the 60-s weights live in the home folder ($H/models), which survives a sandbox recreate; $H/app/results is a symlink
# into the image (/app/results), so anything copied there is lost when the watchdog recreates the sandbox
W60=$H/models
for L in NFL NBA; do
  n=$(ls $W60/${L}_final_${M60}_seed*.pt 2>/dev/null | wc -l)
  [ "$n" -ge 5 ] || echo "WARNING: only $n $L $M60 weights in $W60"
done
start NFL $P/live_trade.py --league NFL --date $D --out $H/sports_live --no-books --weights $W60 --model $M60 --delay 60 --rate 3
start NBA $P/live_trade.py --league NBA --date $D --out $H/sports_live --no-books --weights $W60 --model $M60 --delay 60 --rate 4 --late-block
# every order-book change and trade over Kalshi's WebSocket (analysis only), if the Kalshi key has been copied in
if [ -s $H/.kalshi_key.pem ] && [ -s $H/.kalshi_key_id ]; then
  start books_ws $P/live_books_ws.py --leagues MLB NFL NBA --date $D --out $H/sports_live
fi
sleep 45
tail -n 4 $LOG/${D}_*.log
