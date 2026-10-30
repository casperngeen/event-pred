#!/bin/bash
# Laptop -> backup server: the runners' code, the deployed model weights, the start script and its systemd timer
# (user units: they run without anyone logged in once lingering is on). Rerun after changing any of them.
#     deploy/vm/push.sh            (server = the ssh alias fyp-vm; VM=<alias> for another)
set -eu
cd "$(dirname "$0")/../.."
VM=${VM:-fyp-vm}
rsync -av scripts/fetch_sports.py scripts/fetch_pbp.py scripts/live_record.py scripts/live_model.py scripts/live_trade.py \
  scripts/live_books_ws.py $VM:fyp/app/scripts/
rsync -av stg_infra/stg/models/ingame.py $VM:fyp/app/stg_infra/stg/models/
rsync -av results/ingame/deploy/MLB_final_gru_d10_h30-60-120-1800_tp_pitch_seed?.pt \
  results/ingame/deploy/{NFL,NBA}_final_gru_d60_h30-60-120-1800_tp_seed?.pt $VM:fyp/app/results/ingame/deploy/
rsync -av deploy/vm/start_day_vm.sh $VM:fyp/app/
ssh $VM 'mkdir -p ~/.config/systemd/user'
rsync -av deploy/vm/fyp-day.service deploy/vm/fyp-day.timer $VM:.config/systemd/user/
ssh $VM 'sudo loginctl enable-linger $USER && systemctl --user daemon-reload && systemctl --user enable --now fyp-day.timer \
  && systemctl --user list-timers fyp-day.timer --no-pager && cd ~/fyp/app && ../venv/bin/python -W ignore -c "
import sys; sys.path.insert(0, \"scripts\"); import live_model as LM
for L, n in ((\"MLB\", None), (\"NFL\", \"gru_d60_h30-60-120-1800_tp\"), (\"NBA\", \"gru_d60_h30-60-120-1800_tp\")):
    print(L, len(LM.Forecaster(L, \"results/ingame/deploy\", name=n).models), \"seeds load\")"'
