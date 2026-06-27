#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=open-set-tta
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

env -u VIRTUAL_ENV uv run python src/run.py -m method=bnadapt,tent,nova corruption=gaussian_noise device=cuda
