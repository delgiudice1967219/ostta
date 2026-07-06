# The Geometry of Open-Set Test-Time Adaptation

Online test-time adaptation (TTA) that keeps a classifier accurate under input
corruption **while still detecting semantically novel inputs** — and a
last-layer-geometry account of why the standard recipe fails at exactly that.

A source-trained classifier degrades when the test stream is corrupted (noise,
blur, weather). TTA recovers accuracy online and label-free by updating only the
BatchNorm affine parameters; the standard rule, entropy minimization (Tent),
lowers uncertainty on *every* input. Real streams are open-set: they also carry
novel-class inputs the model should refuse to classify. Tent makes the model
confident on those too — accuracy rises while novelty detection collapses.

**The diagnosis.** With a frozen linear head, a class score can only grow two
ways: a larger feature norm or a better alignment to the class prototype.
Tracking both factors over adaptation shows Tent breaks detection through the
*angle* — it rotates novel features onto the known class directions (alignment
0.40→0.55, novel confidence 0.69→0.87) — while the norm gap the energy detector
reads stays intact.

**The method.** **NOVA** (Norm-Oriented Vector Alignment) scores each input by
its max-cosine alignment to the class prototypes read from the *frozen* source
model (a signal adaptation cannot corrupt), turns pooled scores into a novelty
posterior via a two-component GMM, and applies entropy minimization on
likely-known inputs and an L1 feature-norm penalty on likely-novel ones.

**Headline results** (CIFAR-10-C known vs SVHN-C novel, WideResNet-40-2,
mean over 15 corruptions ± std over 3 seeds, energy-score detection AUROC):

| protocol | Tent | UniEnt | NOVA |
|---|---|---|---|
| per-corruption (T=80, reset) | 66.1 ±1.5 | 86.9 ±0.6 | **89.7 ±0.4** |
| continual (single pass) | 65.4 ±1.5 | **86.1 ±0.1** | 84.6 ±0.9 |

The ranking is protocol-dependent and the paper reports it honestly: NOVA's
norm penalty needs several gradient steps, so UniEnt's single-step entropy
maximization edges it under a strict single pass.

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
  figures.py       renders every paper figure from saved artifacts (no GPU needed)
experiments/
  configs/         Hydra config tree (top-level + a `method` group; one yaml per method/cell)
  scripts/         SLURM batch scripts (each maps to one experiment batch in the paper)
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

# regenerate all paper figures
uv run python src/figures.py
```

Each per-corruption run writes `timetrack.npz` (full metric trajectory for
`t = 0..T`, evaluated on a held-out diagnostic set) and `summary.json`;
`++dump_per_sample=true` additionally saves raw per-sample energies/scores.
Continual runs write per-corruption metrics plus the raw stream energies, so
any metric convention can be recomputed offline.
