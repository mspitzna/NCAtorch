"""
Exhaustive registry tests for the loss factory.

These tests are driven by METRIC_REGISTRY and LOSS_FN_REGISTRY — the single
source of truth for all available losses. Any new entry added to either
registry is automatically covered; a non-compliant loss will fail here.

Compliance contract for every loss:
  - forward(pred, target) returns a dict
  - the dict contains a 'total_loss' key
  - 'total_loss' is a scalar tensor
  - gradients flow back through 'total_loss'
"""

import types
import pytest
import torch

from nca.core.losses.loss_factory import (
    create_metric,
    create_loss_fn,
    METRIC_REGISTRY,
    LOSS_FN_REGISTRY,
)
from nca.utils.config import TrainingConfig

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, C, H, W = 2, 3, 16, 16


def _image_inputs():
    """Standard image tensors suitable for most losses."""
    return torch.rand(B, C, H, W), torch.rand(B, C, H, W)


def _classification_inputs():
    """Logit/one-hot tensors for classification losses (p_ce, i_ce, acc, ta)."""
    num_classes = 10
    pred = torch.randn(B, num_classes, H, W)
    target = torch.zeros(B, num_classes, H, W)
    target[:, 0] = 1.0  # class 0 everywhere
    return pred, target

def _lpips_inputs():
    """LPIPS needs at least 64x64 due to successive max-pool layers in its VGG backbone."""
    return torch.rand(B, C, 64, 64), torch.rand(B, C, 64, 64)

# Maps registry keys that need non-standard inputs to a (pred, target) factory.
# Keys not listed here get standard image tensors.
_METRIC_INPUTS = {
    "p_ce": _classification_inputs,
    "acc":  _classification_inputs,
    "ta":   _classification_inputs,
    "lpips": _lpips_inputs,
}

_LOSS_FN_INPUTS = {
    "p_ce":  _classification_inputs,
    "i_ce":  _classification_inputs,
    "lpips": _lpips_inputs,
}


def _make_config(loss_fn_key, overflow=False):
    return types.SimpleNamespace(
        TRAINING=TrainingConfig(LOSS_FN=loss_fn_key, OVERFLOW_LOSS=overflow),
        DEVICE="cpu",
    )


# ---------------------------------------------------------------------------
# METRIC_REGISTRY — exhaustive compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(METRIC_REGISTRY))
def test_metric_registry_contract(key):
    """Every metric in METRIC_REGISTRY must satisfy the loss contract."""

    loss_fn = create_metric(key, device="cpu")
    pred, target = (_METRIC_INPUTS[key] if key in _METRIC_INPUTS else _image_inputs)()

    out = loss_fn(pred, target)

    assert isinstance(out, dict), f"[{key}] forward() must return a dict"
    assert "total_loss" in out, f"[{key}] dict must contain 'total_loss'"
    assert isinstance(out["total_loss"], torch.Tensor), f"[{key}] 'total_loss' must be a Tensor"
    assert out["total_loss"].ndim == 0, f"[{key}] 'total_loss' must be a scalar"


def test_metric_registry_invalid_key_raises():
    with pytest.raises(ValueError):
        create_metric("not_a_real_metric")


# ---------------------------------------------------------------------------
# LOSS_FN_REGISTRY — exhaustive compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(LOSS_FN_REGISTRY))
def test_loss_fn_registry_contract(key):
    """Every entry in LOSS_FN_REGISTRY must satisfy the loss contract."""

    loss_fn = create_loss_fn(_make_config(key))
    pred, target = (_LOSS_FN_INPUTS[key] if key in _LOSS_FN_INPUTS else _image_inputs)()

    out = loss_fn(pred, target)

    assert isinstance(out, dict), f"[{key}] forward() must return a dict"
    assert "total_loss" in out, f"[{key}] dict must contain 'total_loss'"
    assert isinstance(out["total_loss"], torch.Tensor), f"[{key}] 'total_loss' must be a Tensor"
    assert out["total_loss"].ndim == 0, f"[{key}] 'total_loss' must be a scalar"


@pytest.mark.parametrize("key", sorted(LOSS_FN_REGISTRY))
def test_loss_fn_registry_gradients(key):
    """Every entry in LOSS_FN_REGISTRY must allow gradients to flow."""

    loss_fn = create_loss_fn(_make_config(key))
    pred_raw, target = (_LOSS_FN_INPUTS[key] if key in _LOSS_FN_INPUTS else _image_inputs)()
    pred = pred_raw.detach().requires_grad_(True)

    out = loss_fn(pred, target)
    out["total_loss"].backward()

    assert pred.grad is not None, f"[{key}] gradients must reach the prediction tensor"


def test_loss_fn_registry_invalid_key_raises():
    with pytest.raises(Exception):
        TrainingConfig(LOSS_FN="not_a_real_loss")
