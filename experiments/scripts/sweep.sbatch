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

# 15-corruption benchmark sweep × {BN-adapt, Tent, NOVA, UniEnt, UniEnt+}, T=80,
# seeded SVHN-C (seed=0, Task 3.0), full SVHN csOOD (n_ood_examples=null).
# One SLURM array task per corruption; the 5 methods run as a sequential multirun
# within the task (the first method generates that corruption's seeded SVHN-C
# cache, the rest reuse it). Different array tasks touch different corruption
# cache files, so there is no cross-task cache race.
#
# --no-sync is REQUIRED on compute nodes (no network; avoids racing the shared .venv).

CORRUPTIONS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
             motion_blur zoom_blur snow frost fog brightness contrast \
             elastic_transform pixelate jpeg_compression)

C=${CORRUPTIONS[$SLURM_ARRAY_TASK_ID]}
echo "array task ${SLURM_ARRAY_TASK_ID} -> corruption ${C}"

# Dedicated sweep output tree (one subdir per corruption) so aggregation over
# multirun/sweep/ picks up exactly the 75 sweep runs, not the grid / old runs.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=bnadapt,tent,nova,unient,unient_plus corruption="${C}" device=cuda \
  hydra.sweep.dir=multirun/sweep/"${C}"
