#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=continual-seeds
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-2
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

S=$SLURM_ARRAY_TASK_ID
env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m \
  method=bnadapt,tent,unient,unient_plus,nova device=cuda seed=$S \
  hydra.sweep.dir=multirun/continual_seeds/seed$S
