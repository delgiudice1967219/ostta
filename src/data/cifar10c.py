import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="robustbench")

from robustbench.data import load_cifar10c

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def load_cifar10c_data(
    corruption: str,
    severity: int = 5,
    n_examples: int = 10000,
    data_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10-C for a single corruption via robustbench.

    :param corruption: corruption name (one of :data:`CORRUPTIONS`).
    :type corruption: str
    :param severity: corruption severity level (1-5).
    :type severity: int
    :param n_examples: number of examples to load.
    :type n_examples: int
    :param data_dir: directory the CIFAR-10-C arrays are read from / cached in.
    :type data_dir: str
    :returns: ``(x[N, 3, 32, 32] in [0, 1], y[N])``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    x, y = load_cifar10c(
        n_examples=n_examples,
        corruptions=[corruption],
        severity=severity,
        data_dir=data_dir,
    )
    return x, y
