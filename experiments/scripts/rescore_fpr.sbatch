#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rescore-fpr
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

# Fail fast on a poisoned node (leaked processes from a dead job can hold the
# GPUs; job 48646752 hit one). Require >50 GiB free on every allocated GPU.
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '$1 > 10000 {exit 1}'; then :; else
  echo "FATAL: allocated GPU already holds >10 GiB (leaked process on $(hostname)); aborting" >&2
  exit 1
fi

# Rescore pass for the standard-convention FPR@TPR95 (csID positive, as in
# UniEnt): reproduce every Table-1 run (same seeds, same overrides as
# bench_seeds.sbatch / continual_gateless) and dump raw per-sample energies so
# any metric convention is recomputable offline. 4 GPU lanes, ~40 min each.

ALLC=gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic_transform,pixelate,jpeg_compression
RUN="env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m"
CRUN="env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m"
SUB='hydra.sweep.subdir=${method.name}_${corruption}_s${seed}'
BDIR=hydra.sweep.dir=multirun/rescore/bench

# lane 0: bench bnadapt + nova (90 runs)
CUDA_VISIBLE_DEVICES=0 $RUN method=bnadapt,nova corruption=$ALLC seed=0,1,2 \
  device=cuda '++dump_per_sample=true' $BDIR "$SUB" \
  > results/rescore-lane0.log 2>&1 &

# lane 1: bench tent + unient (90 runs), original bench overrides
CUDA_VISIBLE_DEVICES=1 bash -c "
  $RUN method=tent corruption=$ALLC seed=0,1,2 device=cuda \
    method.marginal_lambda=0.0 method.warmup_K=10 \
    '++dump_per_sample=true' $BDIR '$SUB' &&
  $RUN method=unient corruption=$ALLC seed=0,1,2 device=cuda \
    '++method.reliability_gate=false' \
    '++dump_per_sample=true' $BDIR '$SUB'
" > results/rescore-lane1.log 2>&1 &

# lane 2: bench unient_plus (45 runs), then continual seed 2 (5 runs)
CUDA_VISIBLE_DEVICES=2 bash -c "
  $RUN method=unient_plus corruption=$ALLC seed=0,1,2 device=cuda \
    '++method.reliability_gate=false' \
    '++dump_per_sample=true' $BDIR '$SUB' &&
  $CRUN method=bnadapt,tent,nova device=cuda seed=2 \
    hydra.sweep.dir=multirun/rescore/continual/seed2 &&
  $CRUN method=unient,unient_plus '++method.reliability_gate=False' \
    device=cuda seed=2 hydra.sweep.dir=multirun/rescore/continual/seed2
" > results/rescore-lane2.log 2>&1 &

# lane 3: continual seeds 0 and 1 (10 runs), gateless UniEnt as in Table 1
CUDA_VISIBLE_DEVICES=3 bash -c "
  for S in 0 1; do
    $CRUN method=bnadapt,tent,nova device=cuda seed=\$S \
      hydra.sweep.dir=multirun/rescore/continual/seed\$S &&
    $CRUN method=unient,unient_plus '++method.reliability_gate=False' \
      device=cuda seed=\$S hydra.sweep.dir=multirun/rescore/continual/seed\$S
  done
" > results/rescore-lane3.log 2>&1 &

wait
echo "all rescore lanes done"
