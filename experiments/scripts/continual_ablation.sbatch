#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=continual-ablation
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai
mkdir -p results

# Continual gate × lever ablation, NOVA's detector (maxcos/pooled/soft) held fixed.
# 3 new cells; the 4th cell (L1, no-gate) = NOVA, already in multirun/continual_faithful/nova.
# seed=0 reuses the SVHN-C caches.
env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m \
  method=cont_l1_gate,cont_entmax,cont_entmax_gate device=cuda seed=0 \
  hydra.sweep.dir=multirun/continual_ablation
