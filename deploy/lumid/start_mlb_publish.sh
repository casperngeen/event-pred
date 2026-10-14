#!/bin/sh
# In the sandbox: start an MLB runner that publishes MLB totals signals to Lumid, next to the dry run (a sandbox
# session cannot signal processes started from another session, so the dry run is left as is).
# Needs the PAT in /home/casperngeen/.lumid_pat (one line, mode 600), written by the user. Logs: sports_live_pub/.
H=/home/casperngeen
[ -s $H/.lumid_pat ] || { echo "no $H/.lumid_pat"; exit 1; }
mkdir -p $H/sports_live_pub/logs
cd $H/app
LUMID_PAT=$(cat $H/.lumid_pat) setsid nohup python3 -u -W ignore scripts/live_trade.py --league MLB --out $H/sports_live_pub \
  --publish --publish-kinds total >> $H/sports_live_pub/logs/MLB.log 2>&1 < /dev/null &
sleep 30; tail -n 6 $H/sports_live_pub/logs/MLB.log
