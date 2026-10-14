# Image for paper trading on Lumid's fleet (FlowMesh job): the recorder, the forecaster (5 seeds per league,
# CPU) and the runner. Build from the repo root, amd64 because the fleet is amd64:
#   docker build --platform=linux/amd64 -f deploy/lumid/Dockerfile -t harbor.lum.id/casper-fyp/fyp-ingame:<tag> .
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy scipy polars httpx psutil cryptography tzdata
WORKDIR /app
COPY stg_infra/stg/models/ingame.py /app/stg_infra/stg/models/ingame.py
COPY results/ingame/deploy/MLB_final_gru_d10_h30-60-120-1800_tp_pitch_seed*.pt \
     results/ingame/deploy/NFL_final_gru_d10_h30-60-120-1800_tp_seed*.pt /app/results/ingame/deploy/
COPY scripts/fetch_sports.py scripts/fetch_pbp.py scripts/live_record.py scripts/live_model.py scripts/live_trade.py /app/scripts/
