from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

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


def _to_numpy(x) -> np.ndarray:
    """Coerce a torch tensor / array-like to a 1-D ``float64`` numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def fpr_at_tpr95(scores, is_ood) -> float:
    """False-positive rate at 95% true-positive rate, csOOD as the positive class.

    Convention (sign-consistent with the runner's OOD score): a HIGHER ``score``
    means more OOD. ``is_ood`` is a boolean / ``{0,1}`` mask marking the csOOD
    samples. We trace the ROC with csOOD positive and return the FPR (csID
    wrongly flagged OOD) at the lowest threshold whose TPR (csOOD recall) reaches
    ``0.95``, linearly interpolating between ROC vertices. Lower is better.

    Perfect separation (every csOOD score above every csID score) returns
    ``0.0``; complete overlap where csOOD sits below csID returns ``1.0``.

    :param scores: ``[N]`` per-sample OOD scores (torch tensor or array-like).
    :param is_ood: ``[N]`` boolean / ``{0,1}`` mask, ``True`` for csOOD.
    :returns: FPR at TPR = 0.95 as a plain Python float.
    :rtype: float
    """
    s = _to_numpy(scores)
    y = _to_numpy(is_ood).astype(bool).astype(np.int64)  # csOOD = positive (1)
    fpr, tpr, _ = roc_curve(y, s)
    # roc_curve returns vertices sorted by increasing threshold => fpr and tpr are
    # both non-decreasing, so np.interp gives the FPR at the first TPR >= 0.95.
    return float(np.interp(0.95, tpr, fpr))


def oscr(score_id, score_ood, pred, y_id) -> float:
    """Open-Set Classification Rate: area under the CCR-vs-FPR curve.

    Convention: a HIGHER ``score`` means more ID-like, so "above threshold" is
    the region admitted as ID (sign-consistent with the open-set scorer; the
    runner negates energy so higher = more ID). Sweeping a threshold ``t`` over
    all scores:

    * **CCR(t)** -- *correct classification rate* -- the fraction of csID samples
      that are both scored above ``t`` (admitted as ID) AND classified correctly
      (``pred == y_id``), normalised by the csID count.
    * **FPR(t)** -- the fraction of csOOD samples scored above ``t`` (wrongly
      admitted as ID).

    OSCR integrates CCR over FPR (trapezoidal). Perfectly separated csID scored
    above csOOD with every csID classified correctly -> ``1.0``; every csID
    misclassified -> ``0.0``.

    :param score_id: ``[N_id]`` csID scores (higher = more ID).
    :param score_ood: ``[N_ood]`` csOOD scores (same orientation/scale).
    :param pred: ``[N_id]`` predicted class indices for the csID samples.
    :param y_id: ``[N_id]`` ground-truth class indices for the csID samples.
    :returns: OSCR (area under the CCR-FPR curve) as a plain Python float.
    :rtype: float
    """
    s_id = _to_numpy(score_id)
    s_ood = _to_numpy(score_ood)
    pred_a = _to_numpy(pred)
    y_a = _to_numpy(y_id)
    n_id = s_id.shape[0]
    n_ood = s_ood.shape[0]
    if n_id == 0 or n_ood == 0:
        return 0.0

    correct = pred_a == y_a  # [N_id] bool: csID samples classified correctly

    # Sweep every score high -> low, tracing the curve from (FPR=0, CCR=0). A
    # terminal (FPR=1, CCR=1) vertex closes the area so a perfectly separated,
    # perfectly classified csID population integrates to 1.0. (csID uses strict
    # ``>``, csOOD uses ``>=`` -- an immaterial asymmetry on continuous scores.)
    score = np.concatenate([s_id, s_ood])
    ccr = [0.0]
    fpr = [0.0]
    for t in -np.sort(-score):
        ccr.append(float(np.sum((s_id > t) & correct) / n_id))
        fpr.append(float(np.sum(s_ood >= t) / n_ood))
    fpr.append(1.0)
    ccr.append(1.0)

    fpr_arr = np.asarray(fpr)
    ccr_arr = np.asarray(ccr)
    order = np.argsort(fpr_arr, kind="stable")  # integrate CCR over increasing FPR
    return float(np.trapezoid(ccr_arr[order], fpr_arr[order]))


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
