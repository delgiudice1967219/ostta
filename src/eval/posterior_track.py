from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if "scoring" not in sys.modules:
    _SRC_ROOT = str(Path(__file__).resolve().parent.parent)
    if _SRC_ROOT not in sys.path:
        try:
            import scoring
        except ModuleNotFoundError:
            sys.path.insert(0, _SRC_ROOT)

import numpy as np
import torch

from eval.metrics import auroc
from scoring.gmm import OODPosterior

log = logging.getLogger(__name__)


def track_posterior_quality(
    stream_scores: list,
    eval_scores,
    eval_is_ood,
    mode: str,
) -> list[float]:
    """Trace the GMM posterior's OOD-detection AUROC over the adaptation stream.

    For each stream batch ``t = 1..T``, refit one :class:`OODPosterior` (``pooled``
    accumulates the batch into its history and refits on the whole pool; the same
    instance is reused so the pool grows; ``perbatch`` refits on batch ``t``
    alone), then score the fixed held-out set ``D`` with ``posterior()`` (``P(OOD)``,
    higher = more OOD) and record the AUROC of that posterior as an OOD detector.

    The metric ``auroc(scores_id, scores_ood)`` expects higher = more *ID*, so we
    feed it ``-P(OOD)`` split by the ground-truth mask: ``-P(OOD)`` on the csID
    samples (high when the posterior is confidently ID) and ``-P(OOD)`` on the
    csOOD samples. A perfect detector returns ``1.0``.

    :param stream_scores: length-``T`` list; each a 1-D array of frozen-scorer
        scores for the mixed csID+csOOD adapt batch ``t`` (unsupervised -- no
        labels needed to fit the GMM).
    :param eval_scores: 1-D array of frozen-scorer scores on the held-out set ``D``.
    :param eval_is_ood: bool / ``{0,1}`` array aligned with ``eval_scores`` (``D``'s
        ground-truth csOOD mask).
    :param mode: ``"pooled"`` or ``"perbatch"``.
    :returns: ``[auroc_t for t in 1..T]`` -- the posterior's OOD-AUROC per step.
    :rtype: list[float]
    """
    post = OODPosterior(mode=mode)
    eval_scores = np.asarray(eval_scores, dtype=np.float64).ravel()
    is_ood = np.asarray(eval_is_ood).astype(bool).ravel()
    is_id = ~is_ood

    out: list[float] = []
    for s in stream_scores:  # t = 1..T
        post.update(np.asarray(s))  # pooled: accumulate+refit; perbatch: refit on s
        p_ood = post.posterior(eval_scores)  # P(OOD) on the fixed held-out D
        # auroc wants higher = more ID, so rank by -P(OOD) (csID should rank above
        # csOOD). auroc concatenates with torch.cat, so feed it torch tensors.
        score_id = torch.from_numpy(-p_ood[is_id])
        score_ood = torch.from_numpy(-p_ood[is_ood])
        out.append(auroc(score_id, score_ood))
    return out


def track_fit_quality(
    stream_scores: list,
    eval_scores,
    eval_is_ood,
    mode: str,
) -> dict[str, list[float]]:
    """Trace GMM **fit quality** -- not just its ranking -- over the stream.

    Drives ONE :class:`OODPosterior` (``pooled`` accumulates the batch into its
    history and refits on the whole pool; ``perbatch`` refits on batch ``t``
    alone) across ``t = 1..T`` and, at every step, evaluates the fitted mixture on
    the FIXED held-out set ``D``. Returns three length-``T`` curves:

    * ``nll`` -- held-out negative log-likelihood, ``-post.gmm.score(D)`` (sklearn
      ``GaussianMixture.score`` is the *mean* log-likelihood). **Lower = better
      fit.** This is the metric that exposes pooling: the pooled fit DECREASES and
      stabilises as batches accumulate, while a per-batch refit on one starved
      batch stays higher and NOISY. Unlike AUROC it is sensitive to the actual fit.
    * ``split_acc`` -- balanced accuracy of the hard split ``P(OOD) >= 0.5`` vs the
      ground-truth ``eval_is_ood``: ``mean(TPR, TNR)`` with csOOD positive. The
      0.5 threshold makes it fit-sensitive (the densities, not just their ranking).
    * ``auroc`` -- the existing rank-AUROC of ``P(OOD)`` (see
      :func:`track_posterior_quality`), kept as a labelled **fit-invariant
      baseline**: a 2-component 1-D GMM posterior is monotonic in the frozen score,
      so its AUROC tracks the score ranking and stays ~flat across ``t`` and modes.

    :param stream_scores: length-``T`` list; each a 1-D array of frozen-scorer
        scores for the mixed csID+csOOD adapt batch ``t`` (unsupervised).
    :param eval_scores: 1-D array of frozen-scorer scores on the held-out set ``D``.
    :param eval_is_ood: bool / ``{0,1}`` array aligned with ``eval_scores`` (``D``'s
        ground-truth csOOD mask).
    :param mode: ``"pooled"`` or ``"perbatch"``.
    :returns: ``{"auroc": [...], "nll": [...], "split_acc": [...]}`` (each length
        ``T``).
    :rtype: dict[str, list[float]]
    """
    post = OODPosterior(mode=mode)
    eval_scores = np.asarray(eval_scores, dtype=np.float64).ravel()
    eval_col = eval_scores.reshape(-1, 1)
    is_ood = np.asarray(eval_is_ood).astype(bool).ravel()
    is_id = ~is_ood

    auroc_curve: list[float] = []
    nll_curve: list[float] = []
    split_acc_curve: list[float] = []

    for s in stream_scores:  # t = 1..T
        post.update(np.asarray(s))  # pooled: accumulate+refit; perbatch: refit on s
        p_ood = post.posterior(eval_scores)  # P(OOD) on the fixed held-out D

        # AUROC (fit-invariant baseline): rank by -P(OOD), same as track_posterior_quality.
        score_id = torch.from_numpy(-p_ood[is_id])
        score_ood = torch.from_numpy(-p_ood[is_ood])
        auroc_curve.append(auroc(score_id, score_ood))

        # Held-out NLL of the fitted GMM (mean log-likelihood -> negate). Lower=better.
        nll_curve.append(float(-post.gmm.score(eval_col)))

        # Balanced accuracy of the hard split P(OOD) >= 0.5 (csOOD positive).
        pred_ood = p_ood >= 0.5
        split_acc_curve.append(_balanced_accuracy(is_ood, pred_ood))

    return {"auroc": auroc_curve, "nll": nll_curve, "split_acc": split_acc_curve}


def _balanced_accuracy(is_ood: np.ndarray, pred_ood: np.ndarray) -> float:
    """``mean(TPR, TNR)`` with csOOD as the positive class.

    TPR = recall on the csOOD population, TNR = recall on the csID population. A
    population with no members contributes its perfect-recall convention (``1.0``)
    so an absent class never drags the balanced mean down.
    """
    is_id = ~is_ood
    n_ood = int(is_ood.sum())
    n_id = int(is_id.sum())
    tpr = float((pred_ood & is_ood).sum()) / n_ood if n_ood else 1.0
    tnr = float((~pred_ood & is_id).sum()) / n_id if n_id else 1.0
    return 0.5 * (tpr + tnr)


def _stream_batch_scores(backbone, stream, device: str) -> list[np.ndarray]:
    """Frozen-``theta_0`` maxcos scores for every stream batch -> list of 1-D arrays.

    Reuses the same maxcos orientation the GMM expects (higher = more ID): for each
    mixed batch ``x`` the penultimate features are scored with
    ``max_cosine(features, backbone.W)`` under ``no_grad`` -- no parameter is
    touched, the scorer is the unchanging ``theta_0`` backbone.
    """
    import torch
    from scoring.alignment import max_cosine

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for _t, x_batch in stream:
            features, _logits = backbone._forward(x_batch.to(device))
            s = max_cosine(features, backbone.W)  # [N], higher = more ID
            scores.append(s.detach().cpu().numpy().astype(np.float64))
    return scores


def _diag_scores(backbone, x_csid, x_csood, batch_size: int, device: str):
    """Frozen maxcos scores on the held-out D -> ``(eval_scores, eval_is_ood)``.

    Scores the csID and csOOD diagnostic tensors with the frozen scorer in batches
    (csID first, csOOD second) and returns the concatenated score array plus the
    aligned ``is_ood`` mask (csID=0, csOOD=1). Computed ONCE -- only the GMM fit
    changes over ``t``.
    """
    import torch
    from scoring.alignment import max_cosine

    def _score_all(x) -> np.ndarray:
        """Frozen maxcos score for every image in ``x``, batched, no grad."""
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = x[i : i + batch_size].to(device)
                features, _logits = backbone._forward(xb)
                parts.append(max_cosine(features, backbone.W).detach().cpu().numpy())
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts).astype(np.float64)

    s_id = _score_all(x_csid)
    s_ood = _score_all(x_csood)
    eval_scores = np.concatenate([s_id, s_ood])
    eval_is_ood = np.concatenate(
        [
            np.zeros(len(s_id), dtype=bool),
            np.ones(len(s_ood), dtype=bool),
        ]
    )
    return eval_scores, eval_is_ood


def run_dynamics(cfg, out_dir: Path) -> dict:
    """Compute + save the pooled-vs-per-batch posterior-quality curves.

    Mirrors ``run.py``'s data setup but performs NO adaptation -- the ``theta_0``
    backbone is used directly as the frozen maxcos scorer. Steps:

    1. Load csID (CIFAR-10-C) + csOOD (e.g. SVHN-C), split into disjoint adapt /
       diagnostic pools (``n_diag`` held out each), build the open-set stream over
       the adapt pools, and form the held-out set ``D`` from the diag pools.
    2. Load the frozen backbone; compute maxcos scores (no_grad) for each stream
       batch (``stream_scores[t]``) and for ``D`` (``eval_scores`` + ``eval_is_ood``,
       csID=0 / csOOD=1).
    3. Trace ``track_fit_quality`` for both modes over ``t = 1..T``, yielding the
       held-out ``nll`` + ``split_acc`` fit curves and the ``auroc`` baseline.
    4. Save ``posterior_quality.npz`` (``t`` plus ``{pooled,perbatch}_{nll,
       split_acc,auroc}``) + ``summary.json`` (per mode x metric: ``{t0, tT, mean,
       std}`` -- ``std`` over ``t`` captures per-batch JITTER: perbatch std >> pooled
       std is the reliability signal).

    :param cfg: composed Hydra config (data + dynamics knobs).
    :type cfg: DictConfig
    :param out_dir: run directory the artifacts are written to.
    :type out_dir: Path
    :returns: the summary dict (also logged).
    :rtype: dict
    """
    import torch
    from omegaconf import OmegaConf

    from data.pools import DataPools
    from data.stream import build_stream
    from models.backbone import load_backbone
    from run import _load_sources

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    device = cfg.device
    torch.manual_seed(cfg.seed)

    # Data: csID + csOOD, then disjoint adapt / diagnostic pools
    x_csid, y_csid, x_csood = _load_sources(cfg)
    log.info("Loaded csID=%d, csOOD=%d", len(x_csid), len(x_csood))

    pools = DataPools(
        n_csid=len(x_csid),
        n_csood=len(x_csood),
        n_diag_csid=cfg.n_diag,
        n_diag_csood=cfg.n_diag,
        seed=cfg.seed,
    )

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

    x_diag_csid = x_csid[pools.csid_diag]
    x_diag_csood = x_csood[pools.csood_diag]

    # Frozen theta_0 scorer (no adaptation)
    backbone = load_backbone(device)

    # Stream batch scores drive the GMM accumulation; D scores are computed once.
    stream_scores = _stream_batch_scores(backbone, stream, device)
    eval_scores, eval_is_ood = _diag_scores(
        backbone, x_diag_csid, x_diag_csood, cfg.batch_size, device
    )
    log.info(
        "Scored stream T=%d, D csID=%d csOOD=%d",
        len(stream_scores),
        int((~eval_is_ood).sum()),
        int(eval_is_ood.sum()),
    )

    # Trace pooled vs per-batch FIT QUALITY (nll + split_acc + auroc) over t
    curves = {
        mode: track_fit_quality(stream_scores, eval_scores, eval_is_ood, mode)
        for mode in ("pooled", "perbatch")
    }

    # Persist
    out_dir = Path(out_dir)
    results_dir = out_dir / "results" / "dynamics"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = ("nll", "split_acc", "auroc")
    t_axis = np.arange(1, len(stream_scores) + 1, dtype=np.int64)

    # posterior_quality.npz: t + one array per mode x metric (e.g. pooled_nll).
    arrays = {"t": t_axis}
    for mode in ("pooled", "perbatch"):
        for m in metrics:
            arrays[f"{mode}_{m}"] = np.asarray(curves[mode][m], dtype=np.float64)
    np.savez(results_dir / "posterior_quality.npz", **arrays)

    # summary.json: per mode x metric, {t0, tT, mean, std} (std over t = jitter).
    summary: dict[str, dict[str, float]] = {}
    for mode in ("pooled", "perbatch"):
        for m in metrics:
            v = np.asarray(curves[mode][m], dtype=np.float64)
            summary[f"{mode}_{m}"] = {
                "t0": float(v[0]),
                "tT": float(v[-1]),
                "mean": float(v.mean()),
                "std": float(v.std()),
            }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info("Saved posterior_quality.npz + summary.json to %s", results_dir)
    log.info("Summary:\n%s", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """``@hydra.main`` wrapper: resolve the output dir and run the dynamics sweep."""
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path="../../experiments/configs",
        config_name="dynamics",
    )
    def _run(cfg: DictConfig) -> None:
        """Hydra wrapper: resolve the output dir and delegate to run_dynamics."""
        if cfg.out_dir is not None:
            out_dir = Path(cfg.out_dir)
        else:
            out_dir = Path(HydraConfig.get().runtime.output_dir)
        run_dynamics(cfg, out_dir)

    _run()


if __name__ == "__main__":
    main()
