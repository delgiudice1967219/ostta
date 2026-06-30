#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=smoke-patch
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# GPU smoke of the A0 patch on a tiny subset: confirms FPR/OSCR land in summary.json,
# dump.npz is written, and the alpha override runs -- all on CUDA (device boundaries
# only surface on a real GPU run). ~1-2 min.
env -u VIRTUAL_ENV uv run --no-sync python src/run.py \
  method=nova corruption=gaussian_noise device=cuda \
  T=2 n_examples=300 n_diag=8 batch_size=8 alpha=0.25 dump_per_sample=true \
  out_dir=results/smoke_patch
