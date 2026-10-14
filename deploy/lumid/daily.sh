#!/bin/bash
# On the laptop, once a day (launchd, deploy/lumid/com.fyp.lumid-ingame.plist): replace yesterday's sandbox with a
# fresh 24-h one and start that day's runners in it (start_day.sh, kept in the sandbox's persistent /home).
# Tokens come from ~/.zshrc and are never printed: LUMID_SANDBOX (sandbox:sandboxes:write; LUMID_SNADBOX accepted).
set -u
API=https://lum.id/sbx/api
IMAGE=harbor.lum.id/casper-fyp/fyp-ingame:live1
NAME=ingame-$(TZ=America/New_York date +%m%d)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -p 31223 gw@lum.id"
TOKEN=$(grep -E '^export LUMID_(SANDBOX|SNADBOX)=' ~/.zshrc | tail -1 | sed -E 's/^export [A-Z_]+=//; s/["'\'']//g')
[ -n "$TOKEN" ] || { echo "no LUMID_SANDBOX in ~/.zshrc"; exit 1; }
H="Authorization: Bearer $TOKEN"
echo "== $(date -u +%FT%TZ) $NAME"

# yesterday's games end by ~06:00 UTC: delete older ingame sandboxes (2 per site at most; ssh lands in the newest)
for old in $(curl -s -H "$H" $API/sandboxes | python3 -c "import json,sys
for s in json.load(sys.stdin)['sandboxes']:
    if s['name'].startswith('ingame') and s['name'] != '$NAME': print(s['name'])"); do
  echo "delete $old: $(curl -s -X DELETE -H "$H" $API/sandboxes/$old)"
done
BODY=$(python3 -c 'import json,sys; print(json.dumps(dict(name=sys.argv[1], cpu=2, memory_gi=8, image=sys.argv[2], ttl_hours=24)))' "$NAME" "$IMAGE")
echo "create: $(curl -s -X POST -H "$H" -H 'Content-Type: application/json' --data "$BODY" $API/sandboxes)"
for i in $(seq 1 60); do
  $SSH 'test -x /home/casperngeen/start_day.sh' 2>/dev/null && break
  sleep 15
done
$SSH 'sh /home/casperngeen/start_day.sh'
