#!/bin/bash
# Backup server -> laptop: everything the server's runners recorded, into data/sports_live/vm/ (the sandbox's copy
# is data/sports_live/sandbox/). Order books (books_ws.jsonl[.gz], GBs a night) stay on the server: fetch a night with rsync when needed. Safe to rerun: rsync copies only what changed. Run by daily.sh each day.
cd "$(dirname "$0")/../.."
rsync -a --partial --exclude "books_ws.jsonl*" -e "ssh -o BatchMode=yes -o ConnectTimeout=20" ${VM:-fyp-vm}:fyp/sports_live/ data/sports_live/vm/ \
  && echo "$(date -u +%FT%TZ) pulled from the backup server" || echo "$(date -u +%FT%TZ) backup server pull FAILED"
