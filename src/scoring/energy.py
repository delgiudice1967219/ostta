from __future__ import annotations

import torch


def energy_score(logits: torch.Tensor) -> torch.Tensor:
    """Energy-based OOD score, ``-logsumexp(logits)`` over the class axis: ``E(x) = -log sum_k exp(z_k)``.
    Lower energy means a sharper / more confident logit vector (more ID-like);
    higher energy means a flatter logit vector (more OOD-like).

    :param logits: ``[N, K]`` classifier logits.
    :type logits: torch.Tensor
    :returns: ``[N]`` energy per sample.
    :rtype: torch.Tensor
    """
    return -torch.logsumexp(logits, dim=-1)


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the softmax distribution, ``-sum_k p_k log p_k``.
    
    Ranges from 0 (one-hot / confident)
    up to ``log(K)`` (uniform / maximally uncertain).

    :param logits: ``[N, K]`` classifier logits.
    :type logits: torch.Tensor
    :returns: ``[N]`` predictive entropy per sample.
    :rtype: torch.Tensor
    """
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
