"""Continual / online open-set TTA benchmark runner.

Mirrors the faithful UniEnt continual protocol: the adapter (backbone + BN-affine
optimizer + GMM) is built ONCE and adapted across the 15 corruptions **without
reset**; only the OOD detector is re-pooled per corruption (``reset_detector()``
at each boundary, since the frozen score is only stationary within a corruption).

Per corruption the runner streams an online sequence of 1:1 mixed mini-batches
(``batch_csid`` csID + ``batch_csid`` csOOD, csID first). For each batch it
**predicts with the current theta** (a no-grad forward) and only THEN takes one
adaptation step on the same batch -- so the metrics see the pre-step model and
the model carries the cumulative adaptation of every earlier batch. Predictions
and energy scores are accumulated over all batches of a corruption, scored once
**per corruption**, and finally averaged over the corruptions.

Sign conventions (a flip turns AUROC into ``1 - AUROC``):

* ``energy_score(logits) = -logsumexp`` -> HIGHER = more OOD. This is passed to
  :func:`fpr_at_tpr95` (csOOD positive, higher = OOD).
* ``-energy_score`` (= ``logsumexp``) -> HIGHER = more ID. Split into its csID /
  csOOD halves it is passed to :func:`auroc` (higher = ID) and :func:`oscr`
  (higher = ID).
* Accuracy is csID-only: ``argmax`` of the first ``n_id`` logits vs ``y_csid``.

The core ``run_continual(cfg, out_dir)`` is Hydra-independent (like ``run.py``'s
``run_pipeline``) so it is callable directly from tests; the ``@hydra.main``
wrapper just resolves the output dir and delegates.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.cifar10c import CORRUPTIONS, load_cifar10c_data
from data.svhnc import load_svhn_c
from eval.metrics import auroc, fpr_at_tpr95, fpr_at_tpr95_std, oscr
from methods.factorized import FactorizedAdapter
from models.backbone import load_backbone
from scoring.energy import energy_score

log = logging.getLogger(__name__)

# csOOD source name -> loader. Mirrors run.py so the runner stays open to sources
# beyond svhn_c.
_OOD_LOADERS = {
    "svhn_c": load_svhn_c,
}

# Per-corruption metric keys (also the keys of the means block). ``fpr`` fixes
# csOOD recall at 95% (this repo's original convention); ``fpr_std`` is the
# standard OOD-literature operating point (csID recall at 95%, as in UniEnt).
_METRIC_KEYS = ("acc", "auroc", "fpr", "fpr_std", "oscr")


def _build_adapter(backbone, method_cfg: DictConfig) -> FactorizedAdapter:
    """Construct the factorized adapter from a ``method`` config group.

    Every key except the human-readable ``name`` is forwarded as a keyword
    argument to :class:`FactorizedAdapter`, so a new method is added purely by
    dropping a yaml in ``experiments/configs/method/`` (identical to ``run.py``).
    """
    kwargs = OmegaConf.to_container(method_cfg, resolve=True)
    kwargs.pop("name", None)
    return FactorizedAdapter(backbone, **kwargs)


def _corruption_metrics(
    e_ood_all: torch.Tensor,
    e_id_all: torch.Tensor,
    is_ood_all: torch.Tensor,
    preds_id: torch.Tensor,
    ys_id: torch.Tensor,
) -> dict[str, float]:
    """The four open-set metrics for one corruption from its accumulated arrays.

    All tensors are on the CPU. ``e_ood_all`` is the energy (higher = OOD) over
    the full csID+csOOD stream; ``e_id_all = -e_ood_all`` (higher = ID); the
    boolean ``is_ood_all`` marks the csOOD samples; ``preds_id`` / ``ys_id`` are
    the csID predictions / labels. The sign handed to each metric matches that
    metric's documented convention (see the module docstring).
    """
    id_mask = ~is_ood_all

    acc = float((preds_id == ys_id).float().mean().item())
    # AUROC / OSCR consume the higher = more ID score, split by membership.
    auc = auroc(e_id_all[id_mask], e_id_all[is_ood_all])
    # FPR@TPR95 consumes the higher = more OOD score with csOOD as positive;
    # fpr_std is the standard (csID-positive) operating point.
    fpr = fpr_at_tpr95(e_ood_all, is_ood_all)
    fpr_std = fpr_at_tpr95_std(e_ood_all, is_ood_all)
    osc = oscr(
        e_id_all[id_mask].numpy(),
        e_id_all[is_ood_all].numpy(),
        preds_id.numpy(),
        ys_id.numpy(),
    )
    return {"acc": acc, "auroc": auc, "fpr": fpr, "fpr_std": fpr_std, "oscr": osc}


def _save_outputs(
    out_dir: Path,
    per_corruption: dict[str, dict[str, float]],
    summary: dict,
    raw_scores: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Write ``summary.json`` (means + per-corruption) and ``continual.npz``.

    ``continual.npz`` carries the per-corruption metric arrays as parallel 1-D
    arrays (one entry per corruption, in run order) plus the corruption names, so
    the full per-corruption table can be reconstructed downstream. When
    ``raw_scores`` is given (``name -> (energy, is_ood)`` over that corruption's
    stream, energy HIGHER = more OOD) the raw per-sample arrays are saved too, as
    ``energy__{name}`` / ``is_ood__{name}``, so any metric convention can be
    recomputed offline without re-running the stream.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    names = list(per_corruption.keys())
    arrays = {"corruptions": np.array(names)}
    for key in _METRIC_KEYS:
        arrays[key] = np.array(
            [per_corruption[name][key] for name in names], dtype=np.float64
        )
    if raw_scores is not None:
        for name, (energy, is_ood) in raw_scores.items():
            arrays[f"energy__{name}"] = np.asarray(energy, dtype=np.float32)
            arrays[f"is_ood__{name}"] = np.asarray(is_ood, dtype=bool)
    np.savez(out_dir / "continual.npz", **arrays)


def run_continual(cfg: DictConfig, out_dir: Path) -> tuple[dict, FactorizedAdapter]:
    """Run the continual/online benchmark; return ``(summary, adapter)``.

    Pure (Hydra-independent) core so it can be driven from a composed config in
    tests as well as from the ``@hydra.main`` wrapper. Steps:

    1. Build the backbone + adapter ONCE (no reset across corruptions).
    2. For each corruption (``cfg.corruptions`` or all 15 in order):
       a. ``adapter.reset_detector()`` (re-pool the OOD detector per corruption).
       b. Load csID (CIFAR-10-C) and seeded csOOD (e.g. SVHN-C), both unshuffled.
       c. Stream 1:1 mixed batches: predict-with-current-theta (no grad), then
          ``adapter.step`` on the same batch; accumulate energies / preds.
       d. Compute the four metrics for the corruption.
    3. Mean each metric over the corruptions; save ``summary.json`` +
       ``continual.npz`` to ``out_dir``.

    The returned ``adapter`` lets callers inspect post-run state (e.g. the pooled
    GMM history) without re-running.
    """
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.ood_source not in _OOD_LOADERS:
        raise ValueError(
            f"ood_source must be one of {tuple(_OOD_LOADERS)}, got {cfg.ood_source!r}"
        )
    ood_loader = _OOD_LOADERS[cfg.ood_source]

    device = cfg.device
    torch.manual_seed(cfg.seed)

    corruptions = list(cfg.corruptions) if cfg.corruptions else list(CORRUPTIONS)

    # Model + adapter: built once; not reset across corruptions
    backbone = load_backbone(device)
    adapter = _build_adapter(backbone, cfg.method)
    log.info(
        "Method=%s (frozen=%s); continual over %d corruptions",
        cfg.method.name,
        adapter.frozen,
        len(corruptions),
    )

    per_corruption: dict[str, dict[str, float]] = {}
    raw_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in corruptions:
        # Re-pool the OOD detector per corruption (no-op for none/perbatch fits).
        adapter.reset_detector()

        # csID (CIFAR-10-C) + seeded csOOD (e.g. SVHN-C), both unshuffled. csOOD
        # count matches csID via indices=range(n_examples) so the 1:1 batch
        # slicing lines up.
        x_id, y_id = load_cifar10c_data(name, cfg.severity, n_examples=cfg.n_examples)
        x_ood, _ = ood_loader(
            name,
            cfg.severity,
            indices=list(range(cfg.n_examples)),
            seed=cfg.seed,
            cache=cfg.svhnc_cache,
        )

        e_ood_all: list[torch.Tensor] = []
        e_id_all: list[torch.Tensor] = []
        is_ood_all: list[torch.Tensor] = []
        preds_id: list[torch.Tensor] = []
        ys_id: list[torch.Tensor] = []

        nb = math.ceil(cfg.n_examples / cfg.batch_csid)
        for c in range(nb):
            sl = slice(c * cfg.batch_csid, (c + 1) * cfg.batch_csid)
            xi = x_id[sl].to(device)
            yi = y_id[sl].to(device)
            xo = x_ood[sl].to(device)
            xb = torch.cat([xi, xo], dim=0)  # [n_id + n_ood], csID first
            n_id = xi.shape[0]

            # predict with the current theta (pre-step), no grad
            with torch.no_grad():
                _, logits = adapter.backbone._forward(xb)
            e_ood = energy_score(logits)  # -logsumexp; HIGHER = more OOD
            is_ood = torch.cat([torch.zeros(n_id), torch.ones(xo.shape[0])]).bool()
            preds = logits[:n_id].argmax(dim=1)

            e_ood_all.append(e_ood.cpu())
            e_id_all.append((-e_ood).cpu())  # logsumexp; HIGHER = more ID (oscr/auroc)
            is_ood_all.append(is_ood)
            preds_id.append(preds.cpu())
            ys_id.append(yi.cpu())

            # then adapt one step on the SAME batch
            adapter.step(xb)

        e_ood_cat, is_ood_cat = torch.cat(e_ood_all), torch.cat(is_ood_all)
        metrics = _corruption_metrics(
            e_ood_cat,
            torch.cat(e_id_all),
            is_ood_cat,
            torch.cat(preds_id),
            torch.cat(ys_id),
        )
        per_corruption[name] = metrics
        raw_scores[name] = (e_ood_cat.numpy(), is_ood_cat.numpy())
        log.info("[%s] %s", name, json.dumps(metrics))

    # Mean each metric over the corruptions (not one global pool)
    mean = {
        key: float(np.mean([per_corruption[name][key] for name in corruptions]))
        for key in _METRIC_KEYS
    }
    summary = {
        "mean": mean,
        "per_corruption": per_corruption,
        "n_corruptions": len(corruptions),
    }

    _save_outputs(out_dir, per_corruption, summary, raw_scores=raw_scores)
    log.info("Saved summary.json + continual.npz to %s", out_dir)
    log.info("Summary mean:\n%s", json.dumps(mean, indent=2))
    return summary, adapter


@hydra.main(
    version_base=None,
    config_path="../experiments/configs",
    config_name="continual",
)
def main(cfg: DictConfig) -> None:
    """Run the continual/online benchmark for one method and save its metrics.

    Resolves the output dir (``out_dir`` override, else the Hydra run dir --
    looked up via ``HydraConfig`` so it is correct whether or not
    ``hydra.job.chdir`` moved the cwd) and delegates to :func:`run_continual`.
    """
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
    else:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    run_continual(cfg, out_dir)


if __name__ == "__main__":
    main()
