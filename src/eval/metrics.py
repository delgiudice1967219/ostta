from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score

from scoring.alignment import max_cosine


def auroc(scores_id: torch.Tensor, scores_ood: torch.Tensor) -> float:
    """AUROC under the convention: higher score = more ID-like.

    Labels are ``1`` for the ID population and ``0`` for the OOD population, and
    ``roc_auc_score`` measures how well ``scores`` rank the ID samples above the
    OOD samples. Perfect separation with every csID score above every csOOD score
    returns ``1.0``; the perfectly inverted ranking returns ``0.0``.

    :param scores_id: ``[N_id]`` per-sample ID scores (higher = more ID-like).
    :type scores_id: torch.Tensor
    :param scores_ood: ``[N_ood]`` per-sample OOD scores (same scale/sign).
    :type scores_ood: torch.Tensor
    :returns: AUROC as a plain Python float.
    :rtype: float
    """
    labels = torch.cat([torch.ones(len(scores_id)), torch.zeros(len(scores_ood))])
    scores = torch.cat([scores_id, scores_ood])
    # scores may live on the GPU (logits are computed on-device)
    # move to CPU before converting.
    return float(roc_auc_score(labels.cpu().numpy(), scores.detach().cpu().numpy()))


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Top-1 classification accuracy of ``logits`` against integer labels ``y``.

    :param logits: ``[N, K]`` classifier logits.
    :type logits: torch.Tensor
    :param y: ``[N]`` integer ground-truth class indices.
    :type y: torch.Tensor
    :returns: fraction correct as a plain Python float.
    :rtype: float
    """
    preds = logits.argmax(dim=-1)
    return float((preds == y).float().mean().item())


def geometry_diagnostics(
    features_id: torch.Tensor,
    features_ood: torch.Tensor,
    logits_id: torch.Tensor,
    logits_ood: torch.Tensor,
    W: torch.Tensor,
) -> dict:
    """Per-step geometry diagnostics on the *adapted* diagnostic-set tensors.

    All inputs are assumed already computed under ``torch.no_grad()`` from the
    adapted model (this function never touches the model). Reductions are plain
    means, returned as Python floats. (A distance-to-frozen-centroid diagnostic
    is not computed here -- it would need clean-CIFAR centroids this function
    does not build.)

    :param features_id: ``[N_id, d]`` adapted csID penultimate features.
    :type features_id: torch.Tensor
    :param features_ood: ``[N_ood, d]`` adapted csOOD penultimate features.
    :type features_ood: torch.Tensor
    :param logits_id: ``[N_id, K]`` adapted csID logits (accepted for symmetry;
        the confidence diagnostic is reported on the csOOD side).
    :type logits_id: torch.Tensor
    :param logits_ood: ``[N_ood, K]`` adapted csOOD logits.
    :type logits_ood: torch.Tensor
    :param W: ``[K, d]`` (frozen) classifier weight matrix, for max-cosine.
    :type W: torch.Tensor
    :returns: ``dict`` with keys:

        * ``norm_gap_l2`` -- ``mean||g||2_id - mean||g||2_ood``.
        * ``norm_gap_l1`` -- ``mean||g||1_id - mean||g||1_ood``.
        * ``maxcos_id``   -- mean max-cosine(features_id, W).
        * ``maxcos_ood``  -- mean max-cosine(features_ood, W).
        * ``conf_ood``    -- mean max-softmax over ``logits_ood``.
    :rtype: dict
    """
    norm_l2_id = features_id.norm(dim=-1)
    norm_l2_ood = features_ood.norm(dim=-1)
    norm_l1_id = features_id.abs().sum(dim=-1)
    norm_l1_ood = features_ood.abs().sum(dim=-1)

    maxcos_id = max_cosine(features_id, W)
    maxcos_ood = max_cosine(features_ood, W)

    conf_ood = logits_ood.softmax(dim=-1).max(dim=-1).values

    return {
        "norm_gap_l2": float((norm_l2_id.mean() - norm_l2_ood.mean()).item()),
        "norm_gap_l1": float((norm_l1_id.mean() - norm_l1_ood.mean()).item()),
        "maxcos_id": float(maxcos_id.mean().item()),
        "maxcos_ood": float(maxcos_ood.mean().item()),
        "conf_ood": float(conf_ood.mean().item()),
    }
