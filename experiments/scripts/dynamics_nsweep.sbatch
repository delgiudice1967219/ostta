#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dyn-nsweep
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai
mkdir -p results

env -u VIRTUAL_ENV uv run --no-sync python src/eval/posterior_track.py \
  corruption=gaussian_noise device=cuda 'N=20'  out_dir=results/dyn_N20
env -u VIRTUAL_ENV uv run --no-sync python src/eval/posterior_track.py \
  corruption=gaussian_noise device=cuda 'N=200' out_dir=results/dyn_N200
