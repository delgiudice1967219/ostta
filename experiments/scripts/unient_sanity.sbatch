#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=unient-sanity
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

# UniEnt + UniEnt+ sanity on gaussian_noise / SVHN-C, T=80, full SVHN csOOD
# (n_ood_examples=null) so AUROC/acc are directly comparable to the Phase-1
# anchors (bnadapt 0.785, tent 0.676, nova 0.930).
#
# NOTE: --no-sync is REQUIRED on compute nodes. Plain `uv run` attempts a
# dependency sync at launch; compute nodes have no network (the robustbench
# git+ dep can't be fetched) and a sync can race the shared .venv, transiently
# breaking imports. The .venv is pre-built on the login node, so the job must
# use it as-is.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=unient,unient_plus corruption=gaussian_noise device=cuda
