#!/bin/bash
# Daily pull of the backup VM's live results (small files only) into data/sports_live_pull/vm/, then the live-result tables.
#   ./live_pull.sh          pull, rebuild out/b73_live_vm (b73_live_adapter.py), print live vs replay on the common games
# Pulls the exchange simulator's log (execution.jsonl), orders, decisions, summary and the feed; never the books, trades, candles, forecasts or
# published files (several GB a night; the recorder's copies of those stay on the VM). Safe to rerun.
cd "$(dirname "$0")"
rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  --include='*/' --include='execution.jsonl' --include='orders.jsonl' --include='decisions.jsonl' --include='summary.json' \
  --include='game.json' --include='calls.jsonl' --include='feed_arrivals.jsonl' --include='feed/***' --exclude='*' \
  ${VM:-fyp-vm}:fyp/sports_live/ data/sports_live_pull/vm/ && echo "$(date -u +%FT%TZ) pulled live results from the backup VM" || { echo "pull FAILED"; exit 1; }
/usr/local/bin/python3 analysis/graph_2026_10/b73_live_adapter.py build data/sports_live_pull/vm analysis/graph_2026_10/out/b73_live_vm
/usr/local/bin/python3 analysis/graph_2026_10/b73_live_adapter.py compare analysis/graph_2026_10/out/b73_live_vm
