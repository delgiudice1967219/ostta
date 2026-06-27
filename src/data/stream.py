from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class BatchMeta:
    t: int
    csid_indices: list[int]
    csood_indices: list[int]


@dataclass
class AdaptationStream:
    batches: list[BatchMeta]
    x_csid: torch.Tensor
    y_csid: torch.Tensor
    x_csood: torch.Tensor

    def __iter__(self) -> Iterator[tuple[int, torch.Tensor]]:
        for meta in self.batches:
            x_id = self.x_csid[meta.csid_indices]
            x_ood = self.x_csood[meta.csood_indices]
            yield meta.t, torch.cat([x_id, x_ood], dim=0)

    def to_meta_list(self) -> list[dict]:
        return [
            {"t": b.t, "csid_indices": b.csid_indices, "csood_indices": b.csood_indices}
            for b in self.batches
        ]


def _sequential_samples(rng: np.random.Generator, pool: list[int], total: int) -> list[int]:
    """Draw `total` items from pool in order; reshuffle when exhausted (no within-epoch repeats)."""
    result: list[int] = []
    arr = np.array(pool)
    while len(result) < total:
        result.extend(rng.permutation(arr).tolist())
    return result[:total]


def build_stream(
    x_csid: torch.Tensor,
    y_csid: torch.Tensor,
    x_csood: torch.Tensor,
    adapt_csid_indices: list[int],
    adapt_csood_indices: list[int],
    N: int,
    T: int,
    open_set: bool,
    seed: int = 0,
    n_ood: int | None = None,
) -> AdaptationStream:
    """Build a frozen adaptation stream of ``T`` batches, each size ``N``.

    ``open_set=True``:  default ``n_ood = N//2``, ``n_id = N - n_ood`` per batch.
    Pass ``n_ood`` to override (e.g. a low-alpha mixing protocol).
    ``open_set=False``: ``n_ood = 0``, ``n_id = N`` per batch (closed-set).

    Examples never repeat within a full pass through the adapt pool. With
    ``T=80, N=200, open_set=True``: ``80*100=8000`` draws = pool size exactly
    (zero repeats).

    :param x_csid: csID image tensor indexed by ``adapt_csid_indices``.
    :type x_csid: torch.Tensor
    :param y_csid: csID labels (carried through for downstream use).
    :type y_csid: torch.Tensor
    :param x_csood: csOOD image tensor indexed by ``adapt_csood_indices``.
    :type x_csood: torch.Tensor
    :param adapt_csid_indices: csID indices forming the adaptation pool.
    :type adapt_csid_indices: list[int]
    :param adapt_csood_indices: csOOD indices forming the adaptation pool.
    :type adapt_csood_indices: list[int]
    :param N: per-batch size.
    :type N: int
    :param T: number of batches in the stream.
    :type T: int
    :param open_set: whether each batch mixes in csOOD samples.
    :type open_set: bool
    :param seed: seed fixing the (frozen) batch order.
    :type seed: int
    :param n_ood: optional override for the per-batch csOOD count.
    :type n_ood: int | None
    :returns: the assembled :class:`AdaptationStream`.
    :rtype: AdaptationStream
    """
    if not open_set:
        n_ood = 0
    elif n_ood is None:
        n_ood = N // 2
    assert 0 <= n_ood <= N, f"n_ood={n_ood} out of range for N={N}"
    n_id = N - n_ood

    rng = np.random.default_rng(seed)
    csid_samples = _sequential_samples(rng, adapt_csid_indices, T * n_id)
    csood_samples = _sequential_samples(rng, adapt_csood_indices, T * n_ood) if n_ood > 0 else []

    batches = []
    for t in range(1, T + 1):
        csid_batch = csid_samples[(t - 1) * n_id: t * n_id]
        csood_batch = csood_samples[(t - 1) * n_ood: t * n_ood] if n_ood > 0 else []
        batches.append(BatchMeta(t=t, csid_indices=csid_batch, csood_indices=csood_batch))

    return AdaptationStream(
        batches=batches,
        x_csid=x_csid,
        y_csid=y_csid,
        x_csood=x_csood,
    )
