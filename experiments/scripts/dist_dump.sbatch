#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dist-dump
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova corruption=gaussian_noise,snow seed=0 dump_per_sample=true device=cuda \
  hydra.sweep.dir=multirun/dist 'hydra.sweep.subdir=${method.name}_${corruption}'

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=tent corruption=gaussian_noise,snow seed=0 dump_per_sample=true device=cuda \
  method.marginal_lambda=0.0 method.warmup_K=10 \
  hydra.sweep.dir=multirun/dist 'hydra.sweep.subdir=${method.name}_${corruption}'
