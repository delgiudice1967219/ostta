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

# E5: frozen vs adapting scorer. NOVA (frozen, scorer_frozen=true) is run in
# A1/A2; here the adapting-scorer variant on a 4-corruption subset, 3 seeds.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=cell_scorer_adapting corruption=gaussian_noise,fog,snow,pixelate seed=0,1,2 \
  device=cuda hydra.sweep.dir=multirun/scorer \
  'hydra.sweep.subdir=${method.name}_${corruption}_s${seed}'

# E6: lambda sensitivity for NOVA (0.03 is run in A1/A2 as nova) on 3 corruptions, 3 seeds.
# 'lam' is nested under method (no top-level key), so override 'method.lam' (it
# exists in the default method tent too, so a plain multirun override is fine).
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova corruption=gaussian_noise,fog,snow method.lam=0.01,0.1,0.3 seed=0,1,2 \
  device=cuda hydra.sweep.dir=multirun/lambda \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}'
