#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=absorb-diag
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai
set -euo pipefail
cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

SUB='${method.name}_${corruption}_s${seed}'

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=bnadapt corruption=gaussian_noise seed=0 device=cuda \
  hydra.sweep.dir=multirun/absorb "hydra.sweep.subdir=$SUB"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=nova corruption=gaussian_noise seed=0 device=cuda \
  dump_per_sample=true \
  hydra.sweep.dir=multirun/absorb "hydra.sweep.subdir=$SUB"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=tent corruption=gaussian_noise seed=0 device=cuda \
  method.marginal_lambda=0.0 method.warmup_K=10 \
  hydra.sweep.dir=multirun/absorb "hydra.sweep.subdir=$SUB"

env -u VIRTUAL_ENV uv run --no-sync python src/run.py -m \
  method=unient,unient_plus corruption=gaussian_noise seed=0 device=cuda \
  '++method.reliability_gate=false' \
  hydra.sweep.dir=multirun/absorb "hydra.sweep.subdir=$SUB"
