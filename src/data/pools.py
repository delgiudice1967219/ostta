import numpy as np


class DataPools:
    """Disjoint split of csID and csOOD data into adaptation and diagnostic pools.

    A single ``np.random.default_rng(seed)`` drives ``rng.permutation`` for each
    of csID and csOOD; the first ``n_diag_*`` permuted indices form the held-out
    diagnostic pool ``D``, the rest form the adaptation pool. Splits are asserted
    disjoint.

    :param n_csid: total number of csID samples to split.
    :type n_csid: int
    :param n_csood: total number of csOOD samples to split.
    :type n_csood: int
    :param n_diag_csid: csID samples held out for the diagnostic pool.
    :type n_diag_csid: int
    :param n_diag_csood: csOOD samples held out for the diagnostic pool.
    :type n_diag_csood: int
    :param seed: seed for the permutation RNG (fixes the split).
    :type seed: int
    """

    def __init__(
        self,
        n_csid: int,
        n_csood: int,
        n_diag_csid: int,
        n_diag_csood: int,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)

        csid_idx = rng.permutation(n_csid)
        csood_idx = rng.permutation(n_csood)

        self.csid_adapt = csid_idx[n_diag_csid:].tolist()
        self.csid_diag = csid_idx[:n_diag_csid].tolist()

        self.csood_adapt = csood_idx[n_diag_csood:].tolist()
        self.csood_diag = csood_idx[:n_diag_csood].tolist()

        self._assert_disjoint()

    def _assert_disjoint(self):
        assert len(set(self.csid_adapt) & set(self.csid_diag)) == 0, \
            "csID adaptation and diagnostic pools overlap"
        assert len(set(self.csood_adapt) & set(self.csood_diag)) == 0, \
            "csOOD adaptation and diagnostic pools overlap"
