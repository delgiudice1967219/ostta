#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=continual-gate-seeds
#SBATCH -o results/%x-%A_%a.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
#SBATCH --array=0-2
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# Gated continual ablation, 3 seeds: NOVA+gate (cont_l1_gate) and
# entmax+gate (cont_entmax_gate), matching the 3-seed standard of Table 1's
# continual column. Seed 0 duplicates multirun/continual_ablation/ as a
# reproducibility check.
S=$SLURM_ARRAY_TASK_ID
env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m \
  method=cont_l1_gate,cont_entmax_gate device=cuda seed=$S \
  hydra.sweep.dir=multirun/continual_gate_seeds/seed$S
