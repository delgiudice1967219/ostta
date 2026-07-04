#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tsne-dump
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

env -u VIRTUAL_ENV uv run --no-sync python src/run.py \
  method=nova corruption=gaussian_noise seed=0 dump_features=true device=cuda \
  out_dir=multirun/tsne/nova
env -u VIRTUAL_ENV uv run --no-sync python src/run.py \
  method=tent corruption=gaussian_noise seed=0 dump_features=true device=cuda \
  method.marginal_lambda=0.0 method.warmup_K=10 out_dir=multirun/tsne/tent
env -u VIRTUAL_ENV uv run --no-sync python src/run.py \
  method=unient corruption=gaussian_noise seed=0 dump_features=true device=cuda \
  method.reliability_gate=false out_dir=multirun/tsne/unient
