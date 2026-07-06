from __future__ import annotations

import torch
import torch.nn as nn
from robustbench.utils import load_model as _rb_load_model

_MODEL_NAME = "Hendrycks2020AugMix_WRN"
_DATASET = "cifar10"
_THREAT_MODEL = "corruptions"
_DEFAULT_MODEL_DIR = "./data/models"


def _last_linear(model: nn.Module) -> nn.Linear:
    """Return the final ``nn.Linear`` of ``model`` (the classifier head)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("no nn.Linear layer found in model")
    return linears[-1]


def _bn_layers(model: nn.Module) -> list[nn.BatchNorm2d]:
    """Return all ``BatchNorm2d`` layers of ``model``."""
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


class Backbone:
    """Wrapper around a RobustBench model exposing features, logits and the head.

    A single persistent forward hook on the last linear layer records its input,
    so that one forward pass yields both the features and the logits from a
    consistent autograd graph (see :meth:`_forward`).
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._head = _last_linear(model)
        self._captured_features: torch.Tensor | None = None
        # Persistent hook: capture the head's input (the embedding) on every
        # forward. No detach -- gradients must flow through features into the
        # BN affine params for downstream L1/entropy adapters.
        self._head.register_forward_hook(self._capture_hook)

    def _capture_hook(self, _module, inputs, _output) -> None:
        """Forward hook on the head: stash its input (the embedding), grad intact."""
        self._captured_features = inputs[0]

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass -> ``(features [N, d], logits [N, K])``.

        Features and logits come from the *same* pass and share one autograd
        graph, so a caller can backprop a loss defined on either (or both).
        """
        x = x.to(self.device)
        logits = self.model(x)
        features = self._captured_features
        assert features is not None, "forward hook did not capture features"
        return features, logits

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding fed to the classifier head.

        :param x: ``[N, 3, 32, 32]`` input images.
        :type x: torch.Tensor
        :returns: features of shape ``[N, d]`` (``d = 128``).
        :rtype: torch.Tensor
        """
        features, _ = self._forward(x)
        return features

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Classifier logits.

        :param x: ``[N, 3, 32, 32]`` input images.
        :type x: torch.Tensor
        :returns: logits of shape ``[N, K]`` (``K = 10``).
        :rtype: torch.Tensor
        """
        _, logits = self._forward(x)
        return logits

    @property
    def W(self) -> torch.Tensor:
        """Final-linear weight matrix, shape ``[K, d]`` (frozen during TTA)."""
        return self._head.weight

    @property
    def b(self) -> torch.Tensor:
        """Final-linear bias, shape ``[K]`` (frozen during TTA)."""
        return self._head.bias

    def bn_affine_params(self) -> list[nn.Parameter]:
        """The BN affine parameters (gamma, beta) -- the only TTA-adapted params."""
        params: list[nn.Parameter] = []
        for bn in _bn_layers(self.model):
            if bn.weight is not None:
                params.append(bn.weight)
            if bn.bias is not None:
                params.append(bn.bias)
        return params

    def set_bn_trainable_only(self) -> None:
        """Freeze every parameter, then enable grad on BN affine (gamma, beta)."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        for bn in _bn_layers(self.model):
            if bn.weight is not None:
                bn.weight.requires_grad_(True)
            if bn.bias is not None:
                bn.bias.requires_grad_(True)


def load_backbone(device: str = "cpu", model_dir: str = _DEFAULT_MODEL_DIR) -> Backbone:
    """Load the RobustBench WRN-40-2 backbone and configure it for BN-adapt TTA.

    Weights are fetched from the RobustBench model zoo on first use and cached
    under ``model_dir``. BN layers are set to train mode with
    ``track_running_stats=False`` (current-batch statistics), and only the BN
    affine parameters are left trainable.

    :param device: torch device the model is moved to (``"cpu"`` | ``"cuda"``).
    :type device: str
    :param model_dir: cache directory for the downloaded checkpoint.
    :type model_dir: str
    :returns: the wrapped backbone, configured for BN-affine TTA.
    :rtype: Backbone
    """
    model = _rb_load_model(
        model_name=_MODEL_NAME,
        dataset=_DATASET,
        threat_model=_THREAT_MODEL,
        model_dir=model_dir,
    )
    model.to(device)
    # BN-adapt default: normalise with current-batch stats on every forward.
    model.train()
    for bn in _bn_layers(model):
        bn.track_running_stats = False
        bn.running_mean = None
        bn.running_var = None

    backbone = Backbone(model, device=device)
    backbone.set_bn_trainable_only()
    return backbone
