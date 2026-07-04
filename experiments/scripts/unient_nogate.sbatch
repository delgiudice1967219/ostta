#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=unient-nogate
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# FAITHFUL UniEnt: the reliability gate is OSTTA's (wisdom-of-crowds confidence
# filter), NOT part of UniEnt (paper Eq. 8/9 have no gate). Our canonical
# unient.yaml/unient_plus.yaml had reliability_gate=true == OSTTA+UniEnt. Here we
# re-run the ACTUAL UniEnt method (gate OFF) on all 15 corruptions x 3 seeds, in
# both protocols, to report UniEnt faithfully and check vs the paper
# (TENT+UniEnt+ continual CIFAR-10-C AUROC = 93.2).

ALLC=gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression
RUN="env -u VIRTUAL_ENV uv run --no-sync python"
SUBDIR='hydra.sweep.subdir=${method.name}_${corruption}_s${seed}'

# per-corruption (T=80, reset), one method per lane
CUDA_VISIBLE_DEVICES=0 $RUN src/run.py -m method=unient      method.reliability_gate=false \
  corruption=$ALLC seed=0,1,2 device=cuda hydra.sweep.dir=multirun/unient_nogate "$SUBDIR" \
  > results/nogate-unient-percorr.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $RUN src/run.py -m method=unient_plus method.reliability_gate=false \
  corruption=$ALLC seed=0,1,2 device=cuda hydra.sweep.dir=multirun/unient_nogate "$SUBDIR" \
  > results/nogate-unient_plus-percorr.log 2>&1 &

# continual (single online pass over all 15), both methods, per seed
CUDA_VISIBLE_DEVICES=2 $RUN src/run_continual.py -m method=unient,unient_plus method.reliability_gate=false \
  seed=0,1,2 device=cuda hydra.sweep.dir=multirun/continual_nogate \
  > results/nogate-continual.log 2>&1 &

wait
echo "all lanes done"
