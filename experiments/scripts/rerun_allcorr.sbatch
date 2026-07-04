#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rerun-allcorr
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# Make two under-sampled ablation claims reproducible on ALL 15 corruptions x 3
# seeds (they previously ran on a 4-corruption / single-seed subset):
#   - E5 frozen-vs-adapting scorer  -> cell_scorer_adapting (vs frozen NOVA=headline)
#   - "detection rule is secondary" -> energy_l1 / entropy_l1 / maxcos_perbatch_l1
#     (NOVA's L1 lever held fixed; only the score/GMM axis differs)
# One lane per GPU, 45 runs each. Frozen NOVA (89.7 headline) already in multirun/bench.

ALLC=gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression
RUN="env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m"
SUBDIR='hydra.sweep.subdir=${method.name}_${corruption}_s${seed}'

CUDA_VISIBLE_DEVICES=0 $RUN method=cell_scorer_adapting     corruption=$ALLC seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/scorer   "$SUBDIR" > results/rerun-scorer.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN method=cell_energy_l1           corruption=$ALLC seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/rule_swap "$SUBDIR" > results/rerun-energy_l1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $RUN method=cell_entropy_l1          corruption=$ALLC seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/rule_swap "$SUBDIR" > results/rerun-entropy_l1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $RUN method=cell_maxcos_perbatch_l1  corruption=$ALLC seed=0,1,2 device=cuda \
  hydra.sweep.dir=multirun/rule_swap "$SUBDIR" > results/rerun-maxcos_perbatch_l1.log 2>&1 &

wait
echo "all lanes done"
