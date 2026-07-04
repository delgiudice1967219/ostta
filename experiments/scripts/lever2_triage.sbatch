#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=lever2-triage
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

RUN="env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m"
COMMON="corruption=gaussian_noise,fog,snow seed=0,1,2 device=cuda hydra.sweep.dir=multirun/lever2"

CUDA_VISIBLE_DEVICES=0 $RUN method=nova_l2 method.lam=0.01,0.03 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}' \
  > results/lever2-lane0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN method=nova_l2 method.lam=0.1,0.3 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}' \
  > results/lever2-lane1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 $RUN method=nova_combo method.mu=0.1,0.2,0.5 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_m${method.mu}_s${seed}' \
  > results/lever2-lane2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 $RUN method=nova_l2_combo method.mu=0.1,0.2,0.5 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_m${method.mu}_s${seed}' \
  > results/lever2-lane3.log 2>&1 &

wait
echo "all lanes done"
