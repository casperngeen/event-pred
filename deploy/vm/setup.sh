#!/bin/bash
# On the backup server (Ubuntu 24.04, run once as the login user): Python, the runners' packages and the folders.
# The code and model weights come from the laptop (deploy/vm/push.sh); the Kalshi key is copied in by hand.
#     ssh fyp-vm 'bash -s' < deploy/vm/setup.sh
set -eu
sudo apt-get update -q
sudo apt-get install -y -q python3-venv python3-pip tzdata rsync
F=$HOME/fyp
mkdir -p $F/app/scripts $F/app/stg_infra/stg/models $F/app/results/ingame/deploy $F/sports_live/logs
[ -x $F/venv/bin/python ] || python3 -m venv $F/venv
$F/venv/bin/pip install -q --upgrade pip
$F/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
$F/venv/bin/pip install -q numpy scipy polars httpx psutil cryptography tzdata websockets
$F/venv/bin/python -c "import torch, polars, httpx, websockets, cryptography; print('packages ok, torch', torch.__version__)"
