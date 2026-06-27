"""Frozen feature/classifier-weight alignment score (max cosine similarity)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def max_cosine(features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Maximum cosine similarity between each feature and any class weight row.

    Both the features and the weight rows are L2-normalised, so the result is a
    pure direction comparison (norm-free). A feature pointing exactly along one
    of the class-weight directions scores 1; orthogonal scores 0. Higher means
    better aligned with the source feature geometry (more ID-like).

    :param features: ``[N, d]`` penultimate features.
    :type features: torch.Tensor
    :param W: ``[K, d]`` final-linear classifier weights (one row per class).
    :type W: torch.Tensor
    :returns: ``[N]`` max cosine similarity per sample (over the ``K`` classes).
    :rtype: torch.Tensor
    """
    f_norm = F.normalize(features, dim=-1)   # [N, d]
    w_norm = F.normalize(W, dim=-1)          # [K, d]
    sims = f_norm @ w_norm.T                  # [N, K]
    return sims.max(dim=-1).values            # [N]
