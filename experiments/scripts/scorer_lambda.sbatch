#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=scorer-lambda
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=cell_scorer_adapting corruption=gaussian_noise,fog,snow,pixelate seed=0,1,2 \
  device=cuda hydra.sweep.dir=multirun/scorer \
  'hydra.sweep.subdir=${method.name}_${corruption}_s${seed}'

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova corruption=gaussian_noise,fog,snow method.lam=0.01,0.1,0.3 seed=0,1,2 \
  device=cuda hydra.sweep.dir=multirun/lambda \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}'
