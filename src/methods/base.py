from __future__ import annotations

import torch


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample Shannon entropy of ``softmax(logits)``, shape ``[N]``.

    Computed from ``softmax`` and ``log_softmax`` for numerical stability rather
    than ``p * log p`` directly. Ranges from 0 (one-hot / confident) up to
    ``log K`` (uniform). This is the ID ("confident prediction") objective: a
    model that minimises it sharpens its predictions.
    """
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)


def softmax_mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the batch-mean softmax, a single scalar (shape ``[]``).

    Averages ``softmax(logits)`` over the batch to a marginal class distribution
    ``f-bar`` of shape ``[C]``, then returns ``-sum_c f-bar_c log f-bar_c``.
    Maximising this scalar pushes the batch-marginal towards uniform, countering
    the degenerate collapse onto a single class. A small ``eps`` floors the log
    argument so an empty class (``f-bar_c -> 0``) stays finite.

    :param logits: a ``[N, C]`` batch of logits (grad may flow through).
    :type logits: torch.Tensor
    :returns: the scalar marginal entropy ``H(f-bar)``.
    :rtype: torch.Tensor
    """
    mean_p = logits.softmax(dim=-1).mean(dim=0)  # [C], the batch marginal
    return -(mean_p * (mean_p + 1e-12).log()).sum()


def feature_l1(features: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 norm of the penultimate features, shape ``[N]``.

    Minimising this suppresses feature-norm inflation, the NOVA OOD penalty.
    On the non-negative (post-ReLU, pooled) features its per-sample pull
    ``d||g||_1/dg = sign(g)`` is a uniform subtraction over the active support
    --- soft-thresholding --- which kills small coordinates first and thereby
    ROTATES the surviving direction toward the dominant (class-aligned)
    coordinates while it shrinks. :func:`feature_sq_l2` is the radial
    alternative without that angular side effect.
    """
    return features.abs().sum(dim=-1)


def feature_sq_l2(features: torch.Tensor) -> torch.Tensor:
    """Per-sample squared L2 norm of the penultimate features, shape ``[N]``.

    Minimizing this shrinks the feature radially: the per-sample pull
    ``d||g||_2^2/dg = 2g`` is parallel to ``g`` itself, so the penalty
    suppresses the norm the energy detector reads without moving the feature's
    direction (no soft-thresholding, hence no rotation toward the class
    weights).
    """
    return features.pow(2).sum(dim=-1)


def warmup_factor(t: int, K: int) -> float:
    """Linear LR ramp in ``[0, 1]`` reaching 1 at step ``t >= K``.

    ``t`` is 1-indexed (the first adaptation step is ``t = 1``), so the factor is
    ``1/K`` on the first step and climbs linearly to ``1`` by step ``K``. A
    non-positive ``K`` disables the ramp (always returns ``1``).

    :param t: current (1-indexed) adaptation step.
    :type t: int
    :param K: number of steps over which to ramp the LR to its full value.
    :type K: int
    :returns: scalar ramp factor in ``[0, 1]``.
    :rtype: float
    """
    if K <= 0:
        return 1.0
    return min(t / K, 1.0)
