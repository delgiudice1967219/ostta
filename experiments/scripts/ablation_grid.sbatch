#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ablation-grid
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

# Detector x lever ablation on gaussian_noise / SVHN-C. Anchors (nova, unient,
# unient_plus, tent, bnadapt) plus the 6 ablation cells, run as one multirun so
# every cell shares the seeded SVHN-C (Task 3.0) and the same eval (T=80, full
# SVHN csOOD, n_ood_examples=null). The controller submits this and aggregates.
#
# Lever ablation triplet  = {nova, cell_novadet_entmax, cell_det_only}
#   (identical maxcos/pooled/soft detector base; only ood_op and its weight lam differ).
# Detector ablation       = {nova, cell_energy_l1, cell_entropy_l1, cell_maxcos_perbatch_l1}
#   (NOVA's l1_norm lever fixed; one detector axis differs in each).
#
# NOTE: --no-sync is REQUIRED on compute nodes. Plain `uv run` attempts a
# dependency sync at launch; compute nodes have no network (the robustbench
# git+ dep can't be fetched) and a sync can race the shared .venv, transiently
# breaking imports. The .venv is pre-built on the login node, so the job must
# use it as-is.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova,unient,unient_plus,tent,bnadapt,cell_det_only,cell_novadet_entmax,cell_energy_l1,cell_entropy_l1,cell_maxcos_perbatch_l1,cell_unient_energy \
  corruption=gaussian_noise device=cuda
