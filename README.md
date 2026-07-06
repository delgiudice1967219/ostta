# The Geometry of Open-Set Test-Time Adaptation

Code for the DLAI project on open-set test-time adaptation (TTA): adapting a
classifier online to input corruption while still detecting semantically novel
inputs. The benchmark follows UniEnt: CIFAR-10-C as the corrupted known
classes, SVHN under the same corruptions as the novel source, a frozen
WideResNet-40-2 (RobustBench / AugMix) adapting only its BatchNorm affine
parameters. The repo implements Tent, BN-adapt, UniEnt/UniEnt+ and **NOVA**
(frozen max-cosine scoring, pooled GMM novelty posterior, entropy minimization
on likely-known inputs and an L1 feature-norm penalty on likely-novel ones),
plus the geometry diagnostics used in the report. Method, analysis, and
results are in the accompanying report.

## How the code is organized

Every method is one configuration of a single **factorized adapter**
(`score`, `gmm_fit`, `ood_op`, `label`, `reliability_gate`), so Tent, NOVA,
UniEnt/UniEnt+ and every ablation cell are configs over one shared pipeline.

```
src/
  data/            CIFAR-10-C + SVHN-C loaders, disjoint adapt/diagnostic pools, open-set stream
  models/          WideResNet-40-2 backbone wrapper (features / logits / BN-affine)
  scoring/         energy, predictive entropy, max-cosine alignment, 2-component GMM posterior
  methods/         factorized TTA adapter (Tent, NOVA, UniEnt(+), ablation cells)
  eval/            metrics (AUROC, FPR95 in both conventions, OSCR) + per-step time-tracking
  run.py           per-corruption protocol (adapt T steps, reset; trajectory diagnostics)
  run_continual.py continual protocol (one online pass over all 15 corruptions, no reset)
  figures.py       renders every report figure from saved artifacts (no GPU needed)
experiments/
  configs/         Hydra config tree (top-level + a `method` group; one yaml per method/cell)
  scripts/         SLURM batch scripts (each maps to one experiment batch in the report)
```

## Setup

```bash
uv sync --extra dev      # create the venv + install deps (PyTorch cu118)
```

## Usage

```bash
# per-corruption protocol: one method, one corruption
uv run python src/run.py method=nova corruption=gaussian_noise device=cuda

# sweep methods (Hydra multirun)
uv run python src/run.py -m method=bnadapt,tent,unient,unient_plus,nova \
  corruption=gaussian_noise seed=0,1,2 device=cuda

# continual protocol: one online pass over all 15 corruptions
uv run python src/run_continual.py method=nova device=cuda

# regenerate all report figures
uv run python src/figures.py
```

Each per-corruption run writes `timetrack.npz` (full metric trajectory for
`t = 0..T`, evaluated on a held-out diagnostic set) and `summary.json`;
`++dump_per_sample=true` additionally saves raw per-sample energies/scores.
Continual runs write per-corruption metrics plus the raw stream energies, so
any metric convention can be recomputed offline.
