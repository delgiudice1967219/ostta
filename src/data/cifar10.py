import torch
from torchvision import datasets, transforms


def load_cifar10_data(
    n_examples: int = 10000,
    data_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the clean CIFAR-10 test set.

    :param n_examples: number of test images to return (default: all 10000).
    :type n_examples: int
    :param data_dir: torchvision root (shared with the other loaders).
    :type data_dir: str
    :returns: ``(x[N, 3, 32, 32] in [0, 1], y[N])``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    x = torch.stack([ds[i][0] for i in range(n_examples)])
    y = torch.tensor([ds[i][1] for i in range(n_examples)])
    return x, y
