#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bench-unient-fix
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-14%6
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# Rerun of the bench UniEnt/UniEnt+ part that failed under the plain-override bug
# (multirun validates 'method.reliability_gate' against the default method tent,
# which lacks the key -> use '++'). bnadapt/nova/tent already completed in
# bench_seeds (48084208); SVHN-C caches for seeds 0,1,2 are warm. Per-corruption
# protocol => gateless UniEnt. Writes into the same multirun/bench tree.
CORRUPTIONS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform \
  pixelate jpeg_compression)
C=${CORRUPTIONS[$SLURM_ARRAY_TASK_ID]}
SUB='${method.name}_${corruption}_s${seed}'
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=unient,unient_plus corruption=$C seed=0,1,2 device=cuda \
  '++method.reliability_gate=false' \
  hydra.sweep.dir=multirun/bench "hydra.sweep.subdir=$SUB"
