#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dynamics
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

# Posterior-quality-over-time: pooled vs per-batch GMM detector quality across the
# adaptation stream, frozen-theta_0 maxcos scorer (NO adaptation). Writes
# results/dynamics/posterior_quality.npz + summary.json under the Hydra run dir.
#
# NOTE: --no-sync is REQUIRED on compute nodes. Plain `uv run` attempts a
# dependency sync at launch; compute nodes have no network (the robustbench
# git+ dep can't be fetched) and a sync can race the shared .venv, transiently
# breaking imports. The .venv is pre-built on the login node, so the job must
# use it as-is.
env -u VIRTUAL_ENV uv run --no-sync python src/eval/posterior_track.py \
  corruption=gaussian_noise device=cuda
