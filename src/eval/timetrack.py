from __future__ import annotations

import torch

from models.backbone import Backbone
from scoring.energy import energy_score
from eval.metrics import auroc, accuracy, geometry_diagnostics, fpr_at_tpr95, oscr


def _diag_forward(
    backbone: Backbone,
    x_id: torch.Tensor,
    x_ood: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """No-grad forward of the diagnostic set in mixed csID+csOOD batches.

    Evaluate the diagnostic set in mixed in-distribution + OOD batches:
    sequential slices of ``x_id`` / ``x_ood`` are concatenated into batches of
    ``batch_size // 2`` ID + ``batch_size // 2`` OOD so BN sees the same mixed
    regime as adaptation, and order is preserved so ``feat_id[i]`` corresponds to
    ``x_id[i]`` (and ``y_id[i]`` in the caller). The whole pass runs under
    ``torch.no_grad()`` and leaves every parameter (incl. BN affine) untouched.

    :returns: ``(feat_id, logits_id, feat_ood, logits_ood)`` assembled across
        batches.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    n_id = max(batch_size // 2, 1)
    n_ood = max(batch_size // 2, 1)
    # Cover all diagnostic samples even when the last batch is partial.
    k_id = (len(x_id) + n_id - 1) // n_id
    k_ood = (len(x_ood) + n_ood - 1) // n_ood
    K = max(k_id, k_ood, 1)

    feat_id_list, logits_id_list = [], []
    feat_ood_list, logits_ood_list = [], []

    with torch.no_grad():
        for k in range(K):
            xb_id = x_id[k * n_id: (k + 1) * n_id]
            xb_ood = x_ood[k * n_ood: (k + 1) * n_ood]
            if len(xb_id) == 0 and len(xb_ood) == 0:
                continue
            x_batch = torch.cat([xb_id, xb_ood], dim=0)
            feat, logits = backbone._forward(x_batch)
            n = len(xb_id)
            feat_id_list.append(feat[:n])
            logits_id_list.append(logits[:n])
            feat_ood_list.append(feat[n:])
            logits_ood_list.append(logits[n:])

    return (
        torch.cat(feat_id_list),
        torch.cat(logits_id_list),
        torch.cat(feat_ood_list),
        torch.cat(logits_ood_list),
    )


def _eval_theta_t(
    backbone: Backbone,
    t: int,
    D,
    batch_size: int,
) -> dict:
    """Evaluate the current (post-update) ``theta_t`` on ``D`` -> one ``m(t)`` dict.

    ``D`` carries the diagnostic tensors ``(x_csid, y_csid, x_csood)``. Returns
    ``{t, auroc, acc, fpr, oscr, norm_gap_l2, norm_gap_l1, maxcos_id, maxcos_ood,
    conf_ood}`` -- ``fpr`` is FPR@TPR95 (csOOD positive, lower better) and ``oscr``
    the open-set classification rate (higher better).
    """
    feat_id, logits_id, feat_ood, logits_ood = _diag_forward(
        backbone, D.x_csid, D.x_csood, batch_size
    )

    # Primary OOD metric: energy-AUROC (higher score = more ID-like).
    scores_id = -energy_score(logits_id)
    scores_ood = -energy_score(logits_ood)

    m = {
        "t": t,
        "auroc": auroc(scores_id, scores_ood),
        "acc": accuracy(logits_id, D.y_csid),
        # FPR@95 expects higher = more OOD, so negate the ID-oriented scores;
        # OSCR expects higher = more ID, which scores_id / scores_ood already are.
        "fpr": fpr_at_tpr95(
            torch.cat([-scores_id, -scores_ood]),
            torch.cat([torch.zeros(len(scores_id)), torch.ones(len(scores_ood))]),
        ),
        "oscr": oscr(scores_id, scores_ood, logits_id.argmax(dim=-1), D.y_csid),
    }
    m.update(
        geometry_diagnostics(feat_id, feat_ood, logits_id, logits_ood, backbone.W)
    )
    return m


def run_timetrack(
    backbone: Backbone,
    adapter,
    stream,
    D,
    T: int,
    batch_size: int = 200,
) -> list[dict]:
    """Trace ``m(t)`` for ``t = 0..T`` along the adaptation stream.

    Evaluates ``theta_0`` on ``D`` first (before any step), then for each of the
    ``T`` stream batches calls ``adapter.step(x_batch)`` and evaluates the
    resulting ``theta_t`` on ``D``. Evaluation is no-grad, train-mode (BN
    batch-stats) and never mutates ``gamma`` / ``beta`` or the adapter's GMM /
    step counter -- it goes straight through ``backbone.features`` / ``logits``.

    :param backbone: the model being adapted (``Backbone``).
    :param adapter: the TTA adapter; only its ``.step`` is called.
    :param stream: an iterable yielding ``(t, x_batch)`` mixed adaptation batches
        (``AdaptationStream``).
    :param D: diagnostic-set carrier exposing ``x_csid``, ``y_csid``, ``x_csood``.
    :param T: number of adaptation steps (the trajectory has ``T + 1`` points).
    :type T: int
    :param batch_size: eval mini-batch size (mixed ``//2`` ID + ``//2`` OOD);
        defaults to 200 -> 20 batches of 100 + 100 over the diagnostic set ``D``.
    :type batch_size: int
    :returns: a list of ``T + 1`` dicts, ``m(t)`` for ``t = 0..T``, each with keys
        ``{t, auroc, acc, fpr, oscr, norm_gap_l2, norm_gap_l1, maxcos_id,
        maxcos_ood, conf_ood}``.
    :rtype: list[dict]
    """
    # BN must stay in train mode (batch statistics) for eval -- the same regime as
    # adaptation. load_backbone leaves the model in train mode; re-assert it
    # defensively (no parameter is touched by this).
    backbone.model.train()

    trajectory: list[dict] = []

    # t = 0: BN-adapt start point, before any gradient step.
    trajectory.append(_eval_theta_t(backbone, 0, D, batch_size))

    # t = 1..T: step, then evaluate the post-update model.
    for _t_stream, x_batch in stream:
        adapter.step(x_batch)
        trajectory.append(_eval_theta_t(backbone, adapter.t, D, batch_size))

    return trajectory


def diag_energy_norm(backbone: Backbone, D, batch_size: int = 200) -> dict:
    """Per-sample energy and L2 feature norm on the diagnostic set ``D``.

    Feeds the energy / feature-norm distribution figure. Returns four 1-D numpy
    arrays: ``energy_id`` / ``energy_ood`` (raw energy ``E``; csID is lower-energy)
    and ``norm_id`` / ``norm_ood`` (L2 penultimate-feature norms). No-grad,
    train-mode (the same BN regime as adaptation and eval); touches no parameter.

    :param backbone: the model at its current ``theta_t`` (``Backbone``).
    :param D: diagnostic-set carrier exposing ``x_csid`` / ``x_csood``.
    :param batch_size: eval mini-batch size (mixed ``//2`` ID + ``//2`` OOD).
    :returns: ``{energy_id, energy_ood, norm_id, norm_ood}`` of numpy arrays.
    :rtype: dict
    """
    backbone.model.train()
    feat_id, logits_id, feat_ood, logits_ood = _diag_forward(
        backbone, D.x_csid, D.x_csood, batch_size
    )
    return {
        "energy_id": energy_score(logits_id).detach().cpu().numpy(),
        "energy_ood": energy_score(logits_ood).detach().cpu().numpy(),
        "norm_id": feat_id.norm(dim=-1).detach().cpu().numpy(),
        "norm_ood": feat_ood.norm(dim=-1).detach().cpu().numpy(),
        # full penultimate embeddings (for the t-SNE latent-space figure)
        "feat_id": feat_id.detach().cpu().numpy(),
        "feat_ood": feat_ood.detach().cpu().numpy(),
    }
