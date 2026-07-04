#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=openness
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-3%4
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

SUBSET=(gaussian_noise fog snow pixelate)
C=${SUBSET[$SLURM_ARRAY_TASK_ID]}
SUB='${method.name}_${corruption}_a${alpha}_s${seed}'

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova corruption=$C alpha=0.25,0.5,0.75 seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/openness "hydra.sweep.subdir=$SUB"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=unient corruption=$C alpha=0.25,0.5,0.75 seed=0,1,2 device=cuda \
  '++method.reliability_gate=false' \
  hydra.sweep.dir=multirun/openness "hydra.sweep.subdir=$SUB"
