"""RobustBench WideResNet-40-2 backbone wrapper.

Loads the ``Hendrycks2020AugMix_WRN`` CIFAR-10 corruptions model from the
RobustBench model zoo and exposes the pieces that test-time adaptation needs:

* ``features(x)`` -- the embedding fed to the final classifier (``d = 128``),
  captured as the *input* to the last ``nn.Linear`` via a forward hook.
* ``logits(x)``   -- the classifier output (``K = 10``).
* ``W`` / ``b``   -- the (frozen) final-linear weight ``[K, d]`` and bias ``[K]``.
* ``bn_affine_params()`` / ``set_bn_trainable_only()`` -- the BN affine
  parameters (gamma, beta), which are the only parameters adapted during TTA.

BN layers are put in train mode with ``track_running_stats=False`` at load time
so every forward pass normalises with the *current batch* statistics.

The feature hook does **not** detach the captured tensor: gradients must flow
through ``features(x)`` into the BN affine parameters so that downstream adapters
(e.g. an L1 feature-norm penalty) can backpropagate through it. Callers that want
a frozen, no-grad embedding are expected to wrap the call in ``torch.no_grad()``
themselves.
"""

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
        """Embedding fed to the classifier head, shape ``[N, d]`` (``d = 128``)."""
        features, _ = self._forward(x)
        return features

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Classifier logits, shape ``[N, K]`` (``K = 10``)."""
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
