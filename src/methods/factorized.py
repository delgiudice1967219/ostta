"""Factorized test-time-adaptation adapter (score x gmm_fit x ood_op x label).

A single configurable adapter that subsumes several open-set TTA methods as
points in a four-axis design space:

* ``score   in {maxcos, energy, entropy}`` -- the scalar a *frozen* theta_0
  scorer maps each sample to; the input to the OOD GMM.
* ``gmm_fit in {none, perbatch, pooled}``  -- how (and whether) a 2-component
  GMM is fit to the scores to produce a soft OOD posterior. ``none`` skips the
  scorer entirely and yields ``pi_ID = 1, pi_OOD = 0`` (i.e. plain Tent).
* ``ood_op  in {none, entropy_max, l1_norm}`` -- the OOD-regime objective:
  ``0`` (none), ``-H(p)`` (entropy-max), or ``lam * ||features||_1`` (l1_norm).
* ``label   in {hard, soft}`` -- whether the posterior is thresholded to
  ``{0, 1}`` (hard) or used directly as a soft weight (soft).

The objective, on the *adapted* model's outputs, is::

    L = mean( pi_ID * H(p_adapted)  +  pi_OOD * ood_op )

where ``H`` is predictive (softmax) entropy and ``pi`` is detached. One
``.step`` performs a single BN-affine Adam update with a linear LR warm-up.

Two-model design:

* The **adapted** model is the passed ``backbone`` -- only its BN affine
  parameters are trainable (via ``Backbone.set_bn_trainable_only``). Its BN
  layers use current-batch statistics (``track_running_stats=False``).
* The **frozen scorer** is a ``deepcopy`` snapshot of the backbone taken at
  construction (when the backbone is still at theta_0). It has ``requires_grad``
  off everywhere, stays in train mode (batch-stat BN) and is *never* updated. It
  is only built when ``gmm_fit != 'none'`` (Tent needs no scorer).

Config presets ``TENT`` and ``NOVA`` are provided.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.backbone import Backbone
from scoring.alignment import max_cosine
from scoring.energy import energy_score, predictive_entropy
from scoring.gmm import OODPosterior
from methods.base import feature_l1, softmax_entropy, warmup_factor

# ---------------------------------------------------------------- axis vocabularies
_SCORES = ("maxcos", "energy", "entropy")
_GMM_FITS = ("none", "perbatch", "pooled")
_OOD_OPS = ("none", "entropy_max", "l1_norm")
_LABELS = ("hard", "soft")


class FactorizedAdapter:
    """Configurable open-set TTA adapter; see the module docstring for the axes.

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
    :param lam: weight on the ``l1_norm`` OOD term.
    :type lam: float
    :param warmup_K: linear LR warm-up length in steps (``<= 0`` disables it).
    :type warmup_K: int
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
        warmup_K: int = 10,
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

        self.backbone = backbone
        self.score = score
        self.gmm_fit = gmm_fit
        self.ood_op = ood_op
        self.label = label
        self.base_lr = lr
        self.lam = lam
        self.warmup_K = warmup_K
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
                self.scorer = self._build_frozen_scorer(backbone)
                self.posterior = OODPosterior(mode=self.gmm_fit)
            # Adam over the BN affine params; the per-step LR is set in `.step`.
            self.optimizer = torch.optim.Adam(
                self.backbone.bn_affine_params(), lr=self.base_lr
            )

        # 1-indexed adaptation step counter (incremented at the top of `.step`).
        self.t = 0

    # ------------------------------------------------------------------ scorer setup
    @staticmethod
    def _build_frozen_scorer(backbone: Backbone) -> Backbone:
        """Snapshot the backbone at theta_0 as a frozen, no-grad scorer.

        The deepcopy captures the current (theta_0) weights. We freeze every
        parameter and keep the BN layers in train mode with batch statistics, so
        the scorer's score distribution matches how the adapted model *started*
        and stays stationary across steps. The scorer is never updated.
        """
        scorer_model = copy.deepcopy(backbone.model)
        scorer_model.train()              # BN uses current-batch stats
        scorer_model.requires_grad_(False)
        scorer = Backbone(scorer_model, device=backbone.device)
        return scorer

    # --------------------------------------------------------------------- posterior
    def _frozen_score(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Compute the chosen score from the FROZEN scorer, no grad. Shape ``[N]``."""
        assert self.scorer is not None
        with torch.no_grad():
            features, logits = self.scorer._forward(batch_x)
            if self.score == "maxcos":
                return max_cosine(features, self.scorer.W)
            if self.score == "energy":
                return energy_score(logits)
            if self.score == "entropy":
                return predictive_entropy(logits)
        raise ValueError(f"unknown score {self.score!r}")  # pragma: no cover

    def _ood_weight(self, batch_x: torch.Tensor, device_ref: torch.Tensor) -> torch.Tensor:
        """Detached per-sample OOD weight ``pi_OOD`` in ``[0, 1]``, shape ``[N]``.

        ``gmm_fit='none'`` -> all zeros (pure Tent: pi_ID = 1). Otherwise fit /
        update the GMM on the frozen score and read its OOD posterior. ``hard``
        labelling thresholds the posterior at 0.5 to ``{0, 1}``.
        """
        if self.gmm_fit == "none":
            return torch.zeros(batch_x.shape[0], device=device_ref.device, dtype=device_ref.dtype)

        assert self.posterior is not None
        scores = self._frozen_score(batch_x)                 # [N], no grad
        scores_np = scores.detach().cpu().numpy()
        self.posterior.update(scores_np)                     # (re)fit the mixture
        p_ood_np = self.posterior.posterior(scores_np)       # [N] numpy P(OOD)
        pi_ood = torch.as_tensor(
            p_ood_np, device=device_ref.device, dtype=device_ref.dtype
        ).clamp(0.0, 1.0)
        if self.label == "hard":
            pi_ood = (pi_ood >= 0.5).to(device_ref.dtype)
        return pi_ood.detach()

    # ------------------------------------------------------------------- loss / terms
    def _ood_term(self, logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Per-sample OOD objective, shape ``[N]`` (grad flows through the adapted model)."""
        if self.ood_op == "none":
            return torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
        if self.ood_op == "entropy_max":
            return -softmax_entropy(logits)              # maximise OOD entropy
        if self.ood_op == "l1_norm":
            return self.lam * feature_l1(features)       # suppress norm inflation
        raise ValueError(f"unknown ood_op {self.ood_op!r}")  # pragma: no cover

    def _components(self, batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The two weighted, per-sample loss components ``(id_term, ood_term)`` [N].

        ``id_term = pi_ID * H(p_adapted)`` and ``ood_term = pi_OOD * ood_op``,
        both on the *adapted* model's grad-carrying outputs. The OOD weight is
        derived from the frozen scorer and detached.
        """
        features, logits = self.backbone._forward(batch_x)   # adapted, grad on
        pi_ood = self._ood_weight(batch_x, device_ref=logits)
        pi_id = 1.0 - pi_ood

        id_term = pi_id * softmax_entropy(logits)
        ood_term = pi_ood * self._ood_term(logits, features)
        return id_term, ood_term

    def loss_terms(self, batch_x: torch.Tensor) -> dict[str, float]:
        """The two scalar loss components for ``batch_x`` (no optimisation step).

        Returns ``{'id_entropy': float, 'ood_norm': float}`` -- the batch means
        of the ID entropy term and the OOD term respectively. Reported as plain
        Python floats (detached).
        """
        id_term, ood_term = self._components(batch_x)
        return {
            "id_entropy": float(id_term.mean().item()),
            "ood_norm": float(ood_term.mean().item()),
        }

    # -------------------------------------------------------------------------- step
    def step(self, batch_x: torch.Tensor) -> None:
        """One BN-affine gradient step on ``batch_x`` with the LR warm-up applied.

        Increments the 1-indexed step counter, sets the optimizer's LR to
        ``base_lr * warmup_factor(t, K)``, then performs a single
        ``zero_grad / backward / step`` on ``L = mean(id_term + ood_term)``.

        In ``frozen`` (BN-adapt) mode this is a no-op apart from advancing the
        step counter: ``theta`` stays at ``theta_0`` and the trajectory is flat.
        """
        self.t += 1
        if self.frozen:
            return
        assert self.optimizer is not None
        eff_lr = self.base_lr * warmup_factor(self.t, self.warmup_K)
        for group in self.optimizer.param_groups:
            group["lr"] = eff_lr

        id_term, ood_term = self._components(batch_x)
        loss = (id_term + ood_term).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ----------------------------------------------------------------- config presets
# BN-adapt: the shared t=0 baseline. theta is never updated (frozen no-op step);
# the only thing that changes batch-to-batch is the current-batch BN statistic.
BNADAPT = dict(score=None, gmm_fit="none", ood_op="none", frozen=True)

# Tent: entropy minimisation only, no scorer / GMM, no OOD term.
TENT = dict(score=None, gmm_fit="none", ood_op="none")

# NOVA-TTA: maxcos GMM (pooled fit), soft labels, L1-norm OOD penalty.
NOVA = dict(score="maxcos", gmm_fit="pooled", ood_op="l1_norm", label="soft")
