from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.backbone import Backbone
from scoring.alignment import max_cosine
from scoring.energy import energy_score, predictive_entropy
from scoring.gmm import OODPosterior
from methods.base import (
    feature_l1,
    feature_sq_l2,
    softmax_entropy,
    softmax_mean_entropy,
    warmup_factor,
)

_SCORES = ("maxcos", "energy", "entropy")
_GMM_FITS = ("none", "perbatch", "pooled")
_OOD_OPS = ("none", "entropy_max", "l1_norm", "sq_l2_norm", "l1_entmax", "sq_l2_entmax")
_LABELS = ("hard", "soft")


def aggregate_loss(
    id_contrib: torch.Tensor,
    ood_contrib: torch.Tensor,
    pi_ood: torch.Tensor,
    label: str,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine the per-sample ID/OOD contributions into the scalar (pre-marginal) loss.

    ``id_contrib`` and ``ood_contrib`` are the already-weighted, already-signed
    per-sample contributions (``pi_ID * H`` and ``pi_OOD * ood_op``); ``pi_ood``
    is the detached per-sample OOD weight (a hard ``{0,1}`` mask in the ``hard``
    case, soft posterior in the ``soft`` case). The aggregation differs by
    ``label`` -- this is the faithful UniEnt behaviour:

    * ``soft``: a single mean over the FULL batch of ``id_contrib + ood_contrib``.
    * ``hard``: each term averaged over its OWN pseudo-subset --
      ``sum(id_contrib)/max(sum(pi_ID),1) + sum(ood_contrib)/max(sum(pi_ood),1)``.
      The ``max(.,1)`` clamps guard an empty subset against division by zero.

    ``gate`` is the OSTTA reliability mask -- a detached per-sample ``{0,1}``
    tensor (shape ``[N]``) that restricts ONLY the csID entropy term to samples
    whose adapted confidence did not drop below the frozen source's (see
    :meth:`FactorizedAdapter._reliability_gate`). When ``gate is None`` the
    behaviour above is reproduced EXACTLY (no gating). When provided, the csID
    term is renormalised over the gated samples while the OOD term is unchanged:

    * ``soft``: ``(id_contrib * gate).sum() / max(gate.sum(), 1)  +  ood.mean()``.
    * ``hard``: ``(id_contrib * gate).sum() / max(sum(pi_ID * gate), 1)`` plus the
      same (ungated) OOD per-subset mean.

    :param id_contrib: ``[N]`` weighted per-sample ID contributions (``pi_ID * H``).
    :type id_contrib: torch.Tensor
    :param ood_contrib: ``[N]`` weighted per-sample OOD contributions (``pi_OOD * ood_op``).
    :type ood_contrib: torch.Tensor
    :param pi_ood: ``[N]`` detached per-sample OOD weight (hard mask or soft posterior).
    :type pi_ood: torch.Tensor
    :param label: ``hard`` | ``soft`` -- the aggregation mode.
    :type label: str
    :param gate: optional ``[N]`` detached OSTTA reliability mask; ``None`` = no gating.
    :type gate: torch.Tensor | None
    :returns: the scalar loss (grad-carrying through the contribution tensors).
    :rtype: torch.Tensor
    """
    if label == "soft":
        if gate is None:
            return (id_contrib + ood_contrib).mean()
        id_term = (id_contrib * gate).sum() / gate.sum().clamp(min=1.0)
        ood_term = ood_contrib.mean()  # full-batch (unchanged by the gate)
        return id_term + ood_term
    # hard: per-subset means using the {0,1} mask sizes as denominators.
    n_ood = pi_ood.sum().clamp(min=1.0)
    ood_term = ood_contrib.sum() / n_ood  # csOOD term is not gated
    if gate is None:
        n_id = (1.0 - pi_ood).sum().clamp(min=1.0)
        return id_contrib.sum() / n_id + ood_term
    # Gated csID term: averaged over {pseudo-csID AND gated}.
    n_id_gated = ((1.0 - pi_ood) * gate).sum().clamp(min=1.0)
    return (id_contrib * gate).sum() / n_id_gated + ood_term


class FactorizedAdapter:
    """Configurable open-set TTA adapter: every method (Tent, NOVA, UniEnt(+),
    BN-adapt, the ablation cells) is one setting of the axes documented below.

    :param backbone: the model to adapt (``Backbone``); only its BN affine
        params are trained. Assumed to be at theta_0 at construction time.
    :type backbone: Backbone
    :param score: ``maxcos`` | ``energy`` | ``entropy`` -- the frozen-scorer
        score fed to the GMM. Ignored (may be ``None``) when ``gmm_fit='none'``.
    :type score: str | None
    :param gmm_fit: ``none`` | ``perbatch`` | ``pooled`` -- GMM accumulation mode.
    :type gmm_fit: str
    :param ood_op: ``none`` | ``entropy_max`` | ``l1_norm`` -- the OOD objective.
    :type ood_op: str
    :param label: ``hard`` | ``soft`` -- threshold vs. soft posterior weighting.
    :type label: str
    :param lr: base Adam learning rate (before warm-up scaling).
    :type lr: float
    :param lam: weight on the OOD term -- the ``l1_norm`` / ``sq_l2_norm``
        penalty, or (for ``entropy_max``) the ``lambda1`` csOOD
        entropy-maximisation weight. For the two-axis ops (``l1_entmax`` /
        ``sq_l2_entmax``) it weights the NORM part only.
    :type lam: float
    :param mu: weight on the entropy-maximisation part of the two-axis OOD ops
        (``l1_entmax`` / ``sq_l2_entmax``); ignored by every other ``ood_op``.
        ``0.0`` (the default) makes the two-axis ops degenerate to their pure
        norm penalty.
    :type mu: float
    :param marginal_lambda: weight (``lambda2``) on the marginal anti-collapse
        term ``H(f-bar)``, subtracted from the loss. ``0.0`` (the default)
        recovers the plain two-term objective (Tent / NOVA).
    :type marginal_lambda: float
    :param warmup_K: linear LR warm-up length in steps (``<= 0`` disables it).
    :type warmup_K: int
    :param reliability_gate: if ``True`` the csID entropy-minimisation term is
        restricted to samples whose ADAPTED softmax probability (read at the
        frozen-source argmax class) stayed ``>=`` the frozen-source max
        probability -- the OSTTA reliability gate that makes UniEnt faithful. The
        OOD term and the marginal term are NOT gated. ``False`` (the default)
        leaves the objective ungated (NOVA / Tent / BN-adapt). Requires the
        frozen scorer (``gmm_fit != 'none'``) for the source confidence.
    :type reliability_gate: bool
    :param scorer_frozen: if ``True`` (default) the OOD score is read from a
        frozen ``theta_0`` deepcopy of the backbone (NOVA's design -- a
        stationary score distribution). If ``False`` the score is read from the
        *adapting* backbone instead (``self.scorer is self.backbone``), so the
        scored features drift as the BN affine params adapt (the classifier
        ``W`` is frozen regardless). Only consulted when ``gmm_fit != 'none'``.
    :type scorer_frozen: bool
    :param frozen: if ``True`` the adapter never updates ``theta`` -- ``.step``
        only advances the step counter and returns, so ``theta`` stays at
        ``theta_0`` for the whole stream. This is the BN-adapt baseline: the
        model still re-normalises every batch with current-batch BN stats
        (the trajectory is flat at ``theta_0``). No optimizer / scorer is
        built in this mode. (All other axes are ignored.)
    :type frozen: bool
    """

    def __init__(
        self,
        backbone: Backbone,
        score: str | None,
        gmm_fit: str,
        ood_op: str,
        label: str = "soft",
        lr: float = 1e-3,
        lam: float = 0.03,
        mu: float = 0.0,
        marginal_lambda: float = 0.0,
        warmup_K: int = 10,
        reliability_gate: bool = False,
        scorer_frozen: bool = True,
        frozen: bool = False,
    ) -> None:
        if gmm_fit not in _GMM_FITS:
            raise ValueError(f"gmm_fit must be one of {_GMM_FITS}, got {gmm_fit!r}")
        if ood_op not in _OOD_OPS:
            raise ValueError(f"ood_op must be one of {_OOD_OPS}, got {ood_op!r}")
        if label not in _LABELS:
            raise ValueError(f"label must be one of {_LABELS}, got {label!r}")
        # The score axis only matters when a GMM is fit; otherwise it is unused.
        if gmm_fit != "none" and score not in _SCORES:
            raise ValueError(
                f"score must be one of {_SCORES} when gmm_fit != 'none', got {score!r}"
            )
        # The reliability gate reads the frozen-source confidence, which only
        # exists when a frozen scorer is built (gmm_fit != 'none').
        if reliability_gate and gmm_fit == "none":
            raise ValueError(
                "reliability_gate=True requires a frozen scorer (gmm_fit != 'none')"
            )

        self.backbone = backbone
        self.score = score
        self.gmm_fit = gmm_fit
        self.ood_op = ood_op
        self.label = label
        self.base_lr = lr
        self.lam = lam
        self.mu = mu
        self.marginal_lambda = marginal_lambda
        self.warmup_K = warmup_K
        self.reliability_gate = reliability_gate
        self.scorer_frozen = scorer_frozen
        self.frozen = frozen

        # Adapt only the BN affine params (idempotent re-assertion).
        self.backbone.set_bn_trainable_only()

        # Frozen theta_0 scorer + GMM posterior + optimizer: not built for the
        # no-op (BN-adapt) baseline, where theta never moves.
        self.scorer: Backbone | None = None
        self.posterior: OODPosterior | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        if not self.frozen:
            if self.gmm_fit != "none":
                if self.scorer_frozen:
                    self.scorer = self._build_frozen_scorer(backbone)
                else:
                    self.scorer = self.backbone
                self.posterior = OODPosterior(mode=self.gmm_fit)
            # Adam over the BN affine params; the per-step LR is set in `.step`.
            self.optimizer = torch.optim.Adam(
                self.backbone.bn_affine_params(), lr=self.base_lr
            )
        self.t = 0

    @staticmethod
    def _build_frozen_scorer(backbone: Backbone) -> Backbone:
        """Snapshot the backbone at theta_0 as a frozen, no-grad scorer.

        The deepcopy captures the current (theta_0) weights. We freeze every
        parameter and keep the BN layers in train mode with batch statistics, so
        the scorer's score distribution matches how the adapted model *started*
        and stays stationary across steps. The scorer is never updated.
        """
        scorer_model = copy.deepcopy(backbone.model)
        scorer_model.train()  # BN uses current-batch stats
        scorer_model.requires_grad_(False)
        scorer = Backbone(scorer_model, device=backbone.device)
        return scorer

    def reset_detector(self) -> None:
        """Clear the pooled OOD detector at a corruption boundary.

        The continual runner calls this at each of the 15 corruption boundaries
        so the pooled GMM re-pools per corruption (the frozen score is only
        stationary within a corruption). A no-op when ``gmm_fit == 'none'``
        (``self.posterior is None``); harmless for ``perbatch`` (which keeps no
        cross-batch history anyway -- ``reset`` just clears the latest fit).
        """
        if self.posterior is not None:
            self.posterior.reset()

    def _frozen_score(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Compute the chosen score from the FROZEN scorer, no grad. Shape ``[N]``."""
        assert self.scorer is not None
        with torch.no_grad():
            features, logits = self.scorer._forward(batch_x)
            if self.score == "maxcos":
                return max_cosine(features, self.scorer.W)  # higher = more ID
            if self.score == "energy":
                return -energy_score(logits)  # higher = more ID
            if self.score == "entropy":
                return -predictive_entropy(logits)  # higher = more ID
        raise ValueError(f"unknown score {self.score!r}")

    def _reliability_gate(
        self, batch_x: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Detached OSTTA reliability mask, shape ``[N]`` (``{0, 1}`` in ``logits.dtype``).

        A sample passes the gate iff its ADAPTED softmax probability, read at the
        FROZEN-source argmax class, stayed ``>=`` the frozen source's max
        probability -- i.e. adaptation did not erode that sample's confidence. The
        frozen-source logits come from the ``theta_0`` scorer under
        ``torch.no_grad()``; the adapted probabilities are detached before the
        comparison, so the mask carries no gradient. It only selects which
        samples' (grad-carrying) entropy contributes to the csID term.
        """
        assert self.scorer is not None
        with torch.no_grad():
            _, logits0 = self.scorer._forward(batch_x)  # frozen theta_0 source
            probs0 = logits0.softmax(dim=-1)
            values0, indices0 = probs0.max(dim=-1)  # source max prob + argmax
            probs = logits.softmax(dim=-1)  # adapted
            values = probs.detach()[
                torch.arange(probs.shape[0], device=probs.device), indices0
            ]  # adapted prob at the source argmax
            return (values >= values0).to(logits.dtype)  # [N], detached {0,1}

    def _ood_weight(
        self, batch_x: torch.Tensor, device_ref: torch.Tensor
    ) -> torch.Tensor:
        """Detached per-sample OOD weight ``pi_OOD`` in ``[0, 1]``, shape ``[N]``.

        ``gmm_fit='none'`` -> all zeros (pure Tent: pi_ID = 1). Otherwise fit /
        update the GMM on the frozen score and read its OOD posterior. ``hard``
        labelling thresholds the posterior at 0.5 to ``{0, 1}``.
        """
        if self.gmm_fit == "none":
            return torch.zeros(
                batch_x.shape[0], device=device_ref.device, dtype=device_ref.dtype
            )

        assert self.posterior is not None
        scores = self._frozen_score(batch_x)  # [N], no grad
        scores_np = scores.detach().cpu().numpy()
        self.posterior.update(scores_np)  # (re)fit the mixture
        p_ood_np = self.posterior.posterior(scores_np)  # [N] numpy P(OOD)
        pi_ood = torch.as_tensor(
            p_ood_np, device=device_ref.device, dtype=device_ref.dtype
        ).clamp(0.0, 1.0)
        if self.label == "hard":
            pi_ood = (pi_ood >= 0.5).to(device_ref.dtype)
        return pi_ood.detach()

    def _ood_term(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Per-sample OOD objective, shape ``[N]`` (grad flows through the adapted model).

        Both non-trivial ops are scaled by ``lam``: for ``entropy_max`` it is the
        ``lambda1`` csOOD entropy-maximisation weight (``lam * (-H)``); for
        ``l1_norm`` it is the NOVA feature-norm penalty weight.
        """
        if self.ood_op == "none":
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=logits.dtype
            )
        if self.ood_op == "entropy_max":
            return self.lam * (
                -softmax_entropy(logits)
            )  # lambda1 * maximise OOD entropy
        if self.ood_op == "l1_norm":
            return self.lam * feature_l1(features)  # suppress norm inflation
        if self.ood_op == "sq_l2_norm":
            # Radial suppression: the per-sample pull 2*lam*g is parallel to g,
            # so the norm shrinks with no rotation toward the class weights.
            return self.lam * feature_sq_l2(features)
        if self.ood_op == "l1_entmax":
            # Two-axis op: norm suppression + entropy maximisation (the two
            # single-axis levers cure complementary axes -- gap vs alignment).
            return self.lam * feature_l1(features) + self.mu * (
                -softmax_entropy(logits)
            )
        if self.ood_op == "sq_l2_entmax":
            return self.lam * feature_sq_l2(features) + self.mu * (
                -softmax_entropy(logits)
            )
        raise ValueError(f"unknown ood_op {self.ood_op!r}")  # pragma: no cover

    def _components(
        self, batch_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The per-sample loss contributions plus the GMM weight and logits.

        Runs ONE adapted forward and returns ``(id_contrib, ood_contrib, pi_ood,
        logits)``: ``id_contrib = pi_ID * H(p_adapted)``, ``ood_contrib =
        pi_OOD * ood_op`` (both ``[N]``, grad on), the detached OOD weight
        ``pi_ood`` (``[N]``), and the grad-carrying ``logits`` (so the marginal
        term reuses them with no extra forward).
        """
        features, logits = self.backbone._forward(batch_x)  # adapted, grad on
        pi_ood = self._ood_weight(batch_x, device_ref=logits)
        pi_id = 1.0 - pi_ood

        id_contrib = pi_id * softmax_entropy(logits)
        ood_contrib = pi_ood * self._ood_term(logits, features)
        return id_contrib, ood_contrib, pi_ood, logits

    def _loss(self, batch_x: torch.Tensor) -> torch.Tensor:
        """The full scalar objective on ``batch_x`` (grad-carrying).

        ``aggregate_loss`` normalises the two entropy contributions per ``label``
        (per-subset means for ``hard``, full-batch mean for ``soft``), then the
        marginal anti-collapse term ``H(f-bar)`` is subtracted with weight
        ``marginal_lambda``. With ``marginal_lambda == 0`` the loss is exactly the
        pre-marginal aggregation (Tent / NOVA unchanged).

        When ``reliability_gate`` is set, a detached OSTTA mask (see
        :meth:`_reliability_gate`) restricts the csID entropy term to the samples
        whose adapted confidence did not drop below the frozen source's; the OOD
        and marginal terms are untouched. Otherwise the gate is ``None`` and
        ``aggregate_loss`` reproduces the ungated behaviour exactly.
        """
        id_contrib, ood_contrib, pi_ood, logits = self._components(batch_x)
        gate = (
            self._reliability_gate(batch_x, logits) if self.reliability_gate else None
        )
        loss = aggregate_loss(id_contrib, ood_contrib, pi_ood, self.label, gate=gate)
        if self.marginal_lambda != 0.0:
            loss = loss - self.marginal_lambda * softmax_mean_entropy(logits)
        return loss

    def loss_terms(self, batch_x: torch.Tensor) -> dict[str, float]:
        """The scalar loss components for ``batch_x`` (no optimisation step).

        :param batch_x: ``[N, 3, 32, 32]`` mixed input batch.
        :type batch_x: torch.Tensor
        :returns: ``{'id_entropy', 'ood_norm', 'marginal_entropy'}`` -- the batch
            means of the ID-entropy and OOD contributions, and the (unweighted)
            marginal entropy ``H(f-bar)``, as plain (detached) Python floats.
        :rtype: dict[str, float]
        """
        id_contrib, ood_contrib, _pi_ood, logits = self._components(batch_x)
        return {
            "id_entropy": float(id_contrib.mean().item()),
            "ood_norm": float(ood_contrib.mean().item()),
            "marginal_entropy": float(softmax_mean_entropy(logits).item()),
        }

    def step(self, batch_x: torch.Tensor) -> None:
        """One BN-affine gradient step on ``batch_x`` with the LR warm-up applied.

        Increments the 1-indexed step counter, sets the optimizer's LR to
        ``base_lr * warmup_factor(t, K)``, then performs a single
        ``zero_grad / backward / step`` on the full objective ``L`` (see
        :meth:`_loss`: the per-``label`` aggregation minus the marginal term).

        In ``frozen`` (BN-adapt) mode this is a no-op apart from advancing the
        step counter: ``theta`` stays at ``theta_0`` and the trajectory is flat.

        :param batch_x: ``[N, 3, 32, 32]`` mixed input batch to adapt on.
        :type batch_x: torch.Tensor
        """
        self.t += 1
        if self.frozen:
            return
        assert self.optimizer is not None
        eff_lr = self.base_lr * warmup_factor(self.t, self.warmup_K)
        for group in self.optimizer.param_groups:
            group["lr"] = eff_lr

        loss = self._loss(batch_x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# BN-adapt: the shared t=0 baseline. theta is never updated (frozen no-op step);
# the only thing that changes batch-to-batch is the current-batch BN statistic.
BNADAPT = dict(score=None, gmm_fit="none", ood_op="none", frozen=True)

# Tent: entropy minimization only, no scorer / GMM, no OOD term.
TENT = dict(score=None, gmm_fit="none", ood_op="none")

# NOVA-TTA: maxcos GMM (pooled fit), soft labels, L1-norm OOD penalty.
NOVA = dict(score="maxcos", gmm_fit="pooled", ood_op="l1_norm", label="soft")

# UniEnt: maxcos-scored per-batch GMM, HARD ID/OOD split (each entropy term
# averaged over its own pseudo-subset), entropy-max csOOD op weighted by
# lambda1=lam=0.2, plus a marginal anti-collapse term H(f-bar) with lambda2=0.2.
# The reliability gate restricts the csID entropy term to samples whose adapted
# confidence stayed >= the frozen source's (the faithful OSTTA filter).
UNIENT = dict(
    score="maxcos",
    gmm_fit="perbatch",
    ood_op="entropy_max",
    label="hard",
    lam=0.2,
    marginal_lambda=0.2,
    warmup_K=0,  # constant LR (no warm-up)
    reliability_gate=True,
)
# UniEnt+: as UniEnt but SOFT, posterior-weighted ID/OOD contributions averaged
# over the full batch (same lambda1/lambda2 = 0.2), with the same reliability gate.
UNIENT_PLUS = dict(
    score="maxcos",
    gmm_fit="perbatch",
    ood_op="entropy_max",
    label="soft",
    lam=0.2,
    marginal_lambda=0.2,
    warmup_K=0,  # constant LR (no warm-up), per the source paper/code
    reliability_gate=True,
)
