# Running on the SoC GPU cluster (Slurm)

Partitions (from `sinfo`): `gpu` / `gpu-long` (3 h / 3 days) for GPU jobs,
`normal` / `long` (3 h / 3 days) for CPU jobs. Ask for a GPU by name:
`--gres=gpu:a100-40:1` (default), `gpu:h100-47:1` (an H100 slice),
`gpu:a100-80:1`. Avoid the generic `gpu:nv`, which gives an older card.

The models are small, so what matters is queue time, not GPU size. Between-game
models run fine on a laptop CPU; the cluster is for the in-game model and large
grids.

## Setup (once)

The login node is reached through the stujump jump host. Add to `~/.ssh/config`
on your laptop, so `ssh`, `rsync` and `scp` all go through it:

    Host xlogin
        HostName xlogin.comp.nus.edu.sg
        User casperng
        ProxyJump casperng@stujump.comp.nus.edu.sg

From your laptop, copy the code, never `data/` (holdout data must not leave the
laptop; jobs get prepared, working-block tensors only):

    rsync -av --exclude data/ --exclude venv/ --exclude .venv-cluster/ \
          --exclude '__pycache__/' ./ xlogin:~/event-pred/

On the cluster, from `~/event-pred`:

    bash cluster/setup_env.sh          # .venv-cluster with CUDA torch + stg
    sbatch cluster/smoke_gpu.sbatch    # ~1 min once it starts

CUDA torch is ~3 GB. If home is too small (`quota -s`), put the venv on a
bigger disk and pass the same path to jobs:

    VENV=/bigger/disk/stg-venv bash cluster/setup_env.sh
    VENV=/bigger/disk/stg-venv sbatch cluster/smoke_gpu.sbatch
    cat cluster/logs/stg-smoke-*.out   # GPU name and s/step at the plan's sizes

## Later

Training jobs will be Slurm job arrays, one task per (fold, seed, variant),
each writing its own result file, plus a collect step:
- in-game model: `gpu` partition, one GPU (or an H100 slice) per task;
- between-game grid: `normal` partition, one CPU per task.

These come with the trainer (sports plan, Phase 3b).
