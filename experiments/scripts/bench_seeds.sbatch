#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bench-seeds
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-14%6
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# E1+E4: per-corruption benchmark, 3 seeds, with FPR95/OSCR (A0 patch).
# One array task per corruption (~15 runs + first-touch cache-gen for seeds 1,2).
# This task is also the cache-warmer for all downstream per-corruption jobs.
#
# PROTOCOL FIDELITY: the per-corruption protocol uses PLAIN Tent (Wang 2021:
# no marginal term, K=10 warm-up) and GATELESS UniEnt/UniEnt+ -- the marginal
# term and the OSTTA reliability gate belong to the CONTINUAL protocol (A5).
# The canonical YAMLs currently hold the continual-faithful settings, so we
# override them back here. BN-adapt and NOVA need no override.
CORRUPTIONS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
  motion_blur zoom_blur snow frost fog brightness contrast elastic_transform \
  pixelate jpeg_compression)
C=${CORRUPTIONS[$SLURM_ARRAY_TASK_ID]}
SUB='${method.name}_${corruption}_s${seed}'

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=bnadapt,nova corruption=$C seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/bench "hydra.sweep.subdir=$SUB"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=tent corruption=$C seed=0,1,2 device=cuda \
  method.marginal_lambda=0.0 method.warmup_K=10 \
  hydra.sweep.dir=multirun/bench "hydra.sweep.subdir=$SUB"

# NOTE: '++' force-override (add-or-set). Under multirun Hydra validates a plain
# value override against the DEFAULT method (tent), which lacks reliability_gate,
# so plain 'method.reliability_gate=false' errors "not in struct"; '++' bypasses it.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=unient,unient_plus corruption=$C seed=0,1,2 device=cuda \
  '++method.reliability_gate=false' \
  hydra.sweep.dir=multirun/bench "hydra.sweep.subdir=$SUB"
