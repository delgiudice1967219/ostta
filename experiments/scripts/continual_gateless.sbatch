#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=continual-gateless
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-2
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# Gate-OFF continual UniEnt/UniEnt+ at 3 seeds, for a fair (both-gateless) continual
# comparison vs NOVA (whose 3-seed continual already exists in continual_seeds).
# '++' force-overrides reliability_gate against the default method schema.
S=$SLURM_ARRAY_TASK_ID
env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m \
  method=unient,unient_plus '++method.reliability_gate=false' device=cuda seed=$S \
  hydra.sweep.dir=multirun/continual_gateless/seed$S
