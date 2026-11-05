# Sourced by daily.sh and watchdog.sh (laptop side). Tokens come from ~/.zshrc and are never printed:
# LUMID_SANDBOX (sandbox:sandboxes:write; LUMID_SNADBOX accepted).
API=https://lum.id/sbx/api
IMAGE=harbor.lum.id/casper-fyp/fyp-ingame:live1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 31223 gw@lum.id"
TOKEN=$(grep -E '^export LUMID_(SANDBOX|SNADBOX)=' ~/.zshrc | tail -1 | sed -E 's/^export [A-Z_]+=//; s/["'\'']//g')
[ -n "$TOKEN" ] || { echo "no LUMID_SANDBOX in ~/.zshrc"; exit 1; }
H="Authorization: Bearer $TOKEN"
LOCK=/tmp/fyp-lumid-sandbox.lock

lock() {   # one sandbox job at a time (daily start vs watchdog); a lock older than 30 min is stale
  for i in $(seq 1 90); do
    mkdir $LOCK 2>/dev/null && { trap 'rmdir $LOCK' EXIT; return 0; }
    [ -n "$(find $LOCK -maxdepth 0 -mmin +30 2>/dev/null)" ] && rmdir $LOCK && continue
    sleep 10
  done
  echo "lock busy, giving up"; exit 1
}

sandboxes() {   # "name phase" per line
  curl -s -H "$H" $API/sandboxes | python3 -c "import json,sys
for s in json.load(sys.stdin)['sandboxes']: print(s['name'], s.get('phase'))"
}

recreate() {   # name, ET date: delete every ingame sandbox (and so every runner in it), create NAME, start DATE's runners once
  local name=$1 d=$2 s
  for old in $(sandboxes | awk '$1 ~ /^ingame/ {print $1}'); do
    echo "delete $old: $(curl -s -X DELETE -H "$H" $API/sandboxes/$old)"
  done
  for i in $(seq 1 60); do
    s=$(sandboxes | awk '$1 ~ /^ingame/' | tr '\n' ' ')
    [ -z "$s" ] && break
    echo "$(date -u +%T) waiting for: $s"; sleep 10
  done
  local body=$(python3 -c 'import json,sys; print(json.dumps(dict(name=sys.argv[1], cpu=2, memory_gi=8, image=sys.argv[2], ttl_hours=24)))' "$name" "$IMAGE")
  echo "create: $(curl -s -X POST -H "$H" -H 'Content-Type: application/json' --data "$body" $API/sandboxes)"
  for i in $(seq 1 60); do
    $SSH 'test -x /home/casperngeen/start_day.sh' 2>/dev/null && break
    sleep 15
  done
  echo "$(date -u +%T) starting $d"
  $SSH "sh /home/casperngeen/start_day.sh $d"
}
