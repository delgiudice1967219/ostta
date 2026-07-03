#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=lever2-full
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# All-15-corruption confirmation of the lever2 triage: the L1 lever vs the
# radial (squared-L2) lever vs the L1+entropy-max combo, compared at matched
# accuracy (lam=0.1 pairs; lam=0.03 for the reference point). One lane per GPU,
# 45 runs each; baselines (nova lam=0.03, novadet_entmax, det_only) already
# exist in multirun/{bench,lever}.

ALLC=gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression
RUN="env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m"
COMMON="corruption=$ALLC seed=0,1,2 device=cuda hydra.sweep.dir=multirun/lever2_full"

CUDA_VISIBLE_DEVICES=0 $RUN method=nova_l2 method.lam=0.03 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}' \
  > results/lever2full-lane0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN method=nova_l2 method.lam=0.1 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}' \
  > results/lever2full-lane1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN method=nova method.lam=0.1 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_l${method.lam}_s${seed}' \
  > results/lever2full-lane2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $RUN method=nova_combo method.mu=0.2 $COMMON \
  'hydra.sweep.subdir=${method.name}_${corruption}_m${method.mu}_s${seed}' \
  > results/lever2full-lane3.log 2>&1 &

wait
echo "all lanes done"
