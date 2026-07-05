# Open-Set Test-Time Adaptation

Online test-time adaptation (TTA) that keeps a classifier accurate under input
corruption **while still rejecting semantically out-of-distribution (OOD) inputs**.

A source-trained classifier degrades when the test stream is corrupted (noise,
blur, weather). TTA recovers accuracy online and label-free by updating only the
BatchNorm affine parameters. But realistic streams are *open-set*: alongside
corrupted known-class inputs they contain novel-class inputs that must be
rejected. Plain entropy minimization (Tent) recovers accuracy but erodes OOD
detection over the stream. This repo studies that trade-off geometrically and
implements a norm-suppressing variant (**NOVA**) that preserves detection at no
accuracy cost.

Every method is expressed through a single **factorized adapter** parameterized
by `(score, gmm_fit, ood_op, label)`, so Tent, NOVA, and variants are configs
over one shared pipeline. Setup: CIFAR-10-C as the shifted in-distribution, SVHN
as the OOD source, a WideResNet-40-2 backbone (RobustBench / AugMix).

## Layout

```
src/
  data/      CIFAR-10-C + SVHN-C loaders, disjoint adapt/diagnostic pools, open-set stream
  models/    WideResNet-40-2 backbone wrapper (features / logits / BN-affine)
  scoring/   energy, predictive entropy, max-cosine alignment, 2-component GMM posterior
  methods/   factorized TTA adapter (Tent, NOVA, …)
  eval/      AUROC / accuracy / geometry metrics + time-tracking pipeline
  run.py     Hydra entry point
experiments/
  configs/   Hydra config tree (top-level + a `method` group)
  scripts/   SLURM batch scripts
tests/
```

## Setup

```bash
uv sync --extra dev      # create the venv + install deps (PyTorch cu118)
```

## Usage

```bash
# one method on one corruption
uv run python src/run.py method=nova corruption=gaussian_noise device=cuda

# sweep methods (Hydra multirun)
uv run python src/run.py -m method=bnadapt,tent,nova corruption=gaussian_noise device=cuda

# on SLURM
sbatch experiments/scripts/repro.sbatch
```

Each run writes `timetrack.npz` (the full metric trajectory for `t = 0..T`) and a
compact `summary.json` (start/end of each metric) to the run directory.
