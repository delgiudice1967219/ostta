from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.cifar10c import load_cifar10c_data
from data.pools import DataPools
from data.stream import build_stream
from data.svhnc import load_svhn_c
from eval.timetrack import run_timetrack
from methods.factorized import FactorizedAdapter
from models.backbone import load_backbone

log = logging.getLogger(__name__)

# csOOD source name -> loader. The dispatch keeps the runner open to additional
# sources beyond svhn_c.
_OOD_LOADERS = {
    "svhn_c": load_svhn_c,
}

# The metric keys produced per step by run_timetrack (besides the index ``t``).
_METRIC_KEYS = (
    "auroc",
    "acc",
    "norm_gap_l2",
    "norm_gap_l1",
    "maxcos_id",
    "maxcos_ood",
    "conf_ood",
)


@dataclass
class DiagSet:
    """Held-out diagnostic set ``D`` consumed by :func:`run_timetrack`.

    Carries the ``(x_csid, y_csid, x_csood)`` slices; the time-tracker evaluates
    the post-update model on these at every step.
    """

    x_csid: torch.Tensor
    y_csid: torch.Tensor
    x_csood: torch.Tensor


def _load_sources(cfg: DictConfig) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Load csID (CIFAR-10-C) and csOOD (e.g. SVHN-C) for ``corruption@severity``.

    Returns ``(x_csid, y_csid, x_csood)``. ``cfg.n_examples`` subsets both
    sources (full sets at the default 10000; a few hundred for the CPU smoke).
    """
    if cfg.ood_source not in _OOD_LOADERS:
        raise ValueError(
            f"ood_source must be one of {tuple(_OOD_LOADERS)}, got {cfg.ood_source!r}"
        )

    x_csid, y_csid = load_cifar10c_data(
        cfg.corruption, cfg.severity, n_examples=cfg.n_examples
    )

    ood_loader = _OOD_LOADERS[cfg.ood_source]
    # csOOD count is INDEPENDENT of csID. With `n_ood_examples=null` the FULL SVHN
    # test set (~26k) is used as csOOD; a small int subsets it (keeps the CPU/GPU
    # smoke fast). svhn_c is generated per-image, so indices control how many get
    # corrupted.
    n_ood = cfg.get("n_ood_examples", None)
    ood_indices = list(range(n_ood)) if n_ood is not None else None
    x_csood, _ = ood_loader(cfg.corruption, cfg.severity, indices=ood_indices)

    return x_csid, y_csid, x_csood


def _build_adapter(backbone, method_cfg: DictConfig) -> FactorizedAdapter:
    """Construct the factorized adapter from a ``method`` config group.

    Every key except the human-readable ``name`` is forwarded as a keyword
    argument to :class:`FactorizedAdapter`, so a new method is added purely by
    dropping a yaml in ``experiments/configs/method/``.
    """
    kwargs = OmegaConf.to_container(method_cfg, resolve=True)
    kwargs.pop("name", None)
    return FactorizedAdapter(backbone, **kwargs)


def _summarize(trajectory: list[dict]) -> dict:
    """Compact ``{metric: {t0, tT}}`` summary from the full ``m(t)`` trajectory.

    Records the first (``t=0``) and last (``t=T``) value of every metric -- the
    start/end pair used to summarise a run (e.g. AUROC 0.79 -> ~0.65 for Tent).
    Also echoes the trajectory length and the actual ``t`` endpoints.
    """
    first, last = trajectory[0], trajectory[-1]
    summary = {
        "n_points": len(trajectory),
        "t0": int(first["t"]),
        "tT": int(last["t"]),
    }
    for key in _METRIC_KEYS:
        summary[key] = {"t0": float(first[key]), "tT": float(last[key])}
    return summary


def _save_outputs(out_dir: Path, trajectory: list[dict]) -> dict:
    """Write ``timetrack.npz`` (full arrays) + ``summary.json`` (start/end).

    Returns the summary dict (also logged by the caller).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full trajectory as parallel 1-D arrays, one per metric (+ the t index).
    arrays = {"t": np.array([m["t"] for m in trajectory], dtype=np.int64)}
    for key in _METRIC_KEYS:
        arrays[key] = np.array([m[key] for m in trajectory], dtype=np.float64)
    np.savez(out_dir / "timetrack.npz", **arrays)

    summary = _summarize(trajectory)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_pipeline(cfg: DictConfig, out_dir: Path) -> dict:
    """Run the full adapt + time-track + save pipeline; return the summary.

    Pure (Hydra-independent) core so it can be driven from a composed config in
    tests as well as from the ``@hydra.main`` wrapper. Steps:

    1. Load csID (CIFAR-10-C) + csOOD (e.g. SVHN-C) for ``corruption@severity``.
    2. Split into disjoint adapt / diagnostic pools (``n_diag`` held out each).
    3. Build the open-set adaptation stream over the adapt pools.
    4. Assemble the diagnostic set ``D`` from the diag pools (on ``device``).
    5. Load the backbone, build the method's adapter, trace ``m(t)`` for
       ``t = 0..T``, and write ``timetrack.npz`` + ``summary.json`` to ``out_dir``.
    """
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.device
    torch.manual_seed(cfg.seed)

    # ── Data: csID + csOOD, then the disjoint adapt / diagnostic pools ──────────
    x_csid, y_csid, x_csood = _load_sources(cfg)
    log.info("Loaded csID=%d, csOOD=%d", len(x_csid), len(x_csood))

    pools = DataPools(
        n_csid=len(x_csid),
        n_csood=len(x_csood),
        n_diag_csid=cfg.n_diag,
        n_diag_csood=cfg.n_diag,
        seed=cfg.seed,
    )

    # Frozen adaptation stream over the *adapt* pools (open-set: N//2 ID + N//2 OOD).
    stream = build_stream(
        x_csid=x_csid,
        y_csid=y_csid,
        x_csood=x_csood,
        adapt_csid_indices=pools.csid_adapt,
        adapt_csood_indices=pools.csood_adapt,
        N=cfg.N,
        T=cfg.T,
        open_set=True,
        seed=cfg.seed,
    )

    # Held-out diagnostic set D = the *diag* pools (moved to the eval device so the
    # accuracy comparison against the on-device logits lines up).
    D = DiagSet(
        x_csid=x_csid[pools.csid_diag].to(device),
        y_csid=y_csid[pools.csid_diag].to(device),
        x_csood=x_csood[pools.csood_diag].to(device),
    )

    # ── Model + adapter ─────────────────────────────────────────────────────────
    backbone = load_backbone(device)
    adapter = _build_adapter(backbone, cfg.method)
    log.info("Method=%s (frozen=%s)", cfg.method.name, adapter.frozen)

    # ── Trace m(t) for t = 0..T and persist ────────────────────────────────────
    trajectory = run_timetrack(
        backbone, adapter, stream, D, cfg.T, batch_size=cfg.batch_size
    )

    summary = _save_outputs(out_dir, trajectory)
    log.info("Saved timetrack.npz + summary.json to %s", out_dir)
    log.info("Summary:\n%s", json.dumps(summary, indent=2))
    return summary


@hydra.main(
    version_base=None,
    config_path="../experiments/configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Adapt one method on one stream and trace + save its metric trajectory.

    Resolves the output dir (``out_dir`` override, else the Hydra run dir --
    looked up via ``HydraConfig`` so it is correct whether or not
    ``hydra.job.chdir`` moved the cwd) and delegates to :func:`run_pipeline`.
    """
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
    else:
        out_dir = Path(HydraConfig.get().runtime.output_dir)
    run_pipeline(cfg, out_dir)


if __name__ == "__main__":
    main()
