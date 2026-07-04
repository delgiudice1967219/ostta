#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sweep
#SBATCH --array=0-14
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

CORRUPTIONS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
             motion_blur zoom_blur snow frost fog brightness contrast \
             elastic_transform pixelate jpeg_compression)

C=${CORRUPTIONS[$SLURM_ARRAY_TASK_ID]}
echo "array task ${SLURM_ARRAY_TASK_ID} -> corruption ${C}"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=bnadapt,tent,nova,unient,unient_plus corruption="${C}" device=cuda \
  hydra.sweep.dir=multirun/sweep/"${C}"
