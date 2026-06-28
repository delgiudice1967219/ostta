#!/bin/bash
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=continual-bench
#SBATCH -o results/%x-%j.out
#SBATCH --chdir=/leonardo_work/IscrC_AIM/open-set-tta-dlai

set -euo pipefail

cd /leonardo_work/IscrC_AIM/open-set-tta-dlai

mkdir -p results

# Continual/online open-set TTA benchmark over the 15 corruptions (severity 5,
# seeded SVHN-C csOOD) for {BN-adapt, Tent, UniEnt, UniEnt+, NOVA}. Each method
# runs as one continual stream (adapter built once, NO reset across corruptions;
# only the OOD detector re-pools per corruption). Methods run as a sequential
# multirun; the first method to touch each corruption generates that corruption's
# seeded SVHN-C cache, the rest reuse it.
#
# The baselines use their faithful configs (warmup_K=0 for unient/unient_plus;
# the controller applies the faithful warm-up override for tent/bnadapt at run
# time -- this script does NOT bake those overrides in).
#
# --no-sync is REQUIRED on compute nodes: plain `uv run` attempts a dependency
# sync at launch, but compute nodes have no network (the robustbench git+ dep
# can't be fetched) and a sync can race the shared .venv. The .venv is pre-built
# on the login node, so the job uses it as-is.
env -u VIRTUAL_ENV uv run --no-sync python src/run_continual.py -m \
  method=bnadapt,tent,unient,unient_plus,nova device=cuda
