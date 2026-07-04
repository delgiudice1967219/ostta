#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --job-name=nova-lambda
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

ALLC=gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression
RUN="env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m"
COMMON="method=nova corruption=$ALLC seed=0,1,2 device=cuda hydra.sweep.dir=multirun/nova_lambda"
SUB='hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}'

CUDA_VISIBLE_DEVICES=0 $RUN $COMMON method.lam=0.05  "$SUB" > results/novalam-0.05.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN $COMMON method.lam=0.075 "$SUB" > results/novalam-0.075.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN $COMMON method.lam=0.1   "$SUB" > results/novalam-0.1.log   2>&1 &
wait
echo "all lanes done"
