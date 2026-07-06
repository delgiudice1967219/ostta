from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

_VALID_MODES = ("pooled", "perbatch")


def _as_column(scores) -> np.ndarray:
    """Coerce a 1-D score array (or anything array-like) to a float64 ``[n, 1]``."""
    arr = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    return arr


class OODPosterior:
    """2-component 1-D GMM that emits a soft ``P(OOD)`` per score.

    :param mode: ``'pooled'`` (refit on accumulated history) or ``'perbatch'``
        (refit on the current batch only).
    :type mode: str
    :param reg_covar: covariance regularisation passed to ``GaussianMixture``.
    :type reg_covar: float
    :param random_state: optional seed for the mixture's EM initialisation.
    :type random_state: int | None
    """

    def __init__(
        self,
        mode: str = "pooled",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.reg_covar = reg_covar
        self.random_state = random_state
        self._history: list[np.ndarray] = []      # pooled: one [n, 1] block per update
        self.gmm: GaussianMixture | None = None
        self._ood_idx: int | None = None          # column of predict_proba that is OOD

    def reset(self) -> None:
        """Clear the pooled state so the next ``update`` starts a fresh fit.

        Drops the accumulated history and the fitted mixture
        (``self._history = []``, ``self.gmm = None``, ``self._ood_idx = None``).
        The continual runner calls this at each corruption boundary because the
        frozen score is only stationary *within* a corruption -- pooling across
        corruptions would mix non-stationary score distributions.
        """
        self._history = []
        self.gmm = None
        self._ood_idx = None

    def update(self, scores) -> None:
        """Refit the mixture.

        ``pooled``: append ``scores`` to the history and refit on the full pool.
        ``perbatch``: refit on ``scores`` alone.

        :param scores: 1-D array-like of frozen scores from the current batch.
        """
        batch = _as_column(scores)
        if self.mode == "pooled":
            self._history.append(batch)
            data = np.concatenate(self._history, axis=0)
        else:  # perbatch
            data = batch

        gmm = GaussianMixture(
            n_components=2,
            reg_covar=self.reg_covar,
            warm_start=False,
            random_state=self.random_state,
        )
        gmm.fit(data)
        self.gmm = gmm

        # ID = higher-mean component
        # OOD = the lower-mean (other) component.
        means = gmm.means_.ravel()
        id_idx = int(means.argmax())
        self._ood_idx = 1 - id_idx

    def posterior(self, scores) -> np.ndarray:
        """Return ``P(OOD | s)`` for the given ``scores`` via ``predict_proba``.

        :param scores: 1-D array-like of scores to label.
        :returns: ``[n]`` numpy array of OOD posteriors, one per input score.
        :rtype: numpy.ndarray
        """
        if self.gmm is None or self._ood_idx is None:
            raise RuntimeError("OODPosterior.posterior called before update()")
        resp = self.gmm.predict_proba(_as_column(scores))   # [n, 2]
        return resp[:, self._ood_idx]
