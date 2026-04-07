"""
Exhaustive registry tests for update_model_factory.

These tests are driven by UPDATE_MODEL_REGISTRY.
Any new entry added to the registry is automatically covered;
a non-compliant module will fail here.

Compliance contract for every update model:
  - forward(x) returns a tensor of shape [B, out_channels, H, W]
  - gradients flow back through forward()
"""

import pytest
import torch
from unittest.mock import MagicMock

from nca.core.models.update_model_factory import UPDATE_MODEL_REGISTRY
from nca.utils.config import PerceptionConfig

# ---------------------------------------------------------------------------
# Shared dims
# ---------------------------------------------------------------------------

B, IN_CH, H, W = 2, 16, 16, 16
OUT_CH = 32   # perception output channels (not used by sobel)
UPDATE_IN_CH  = 80
UPDATE_OUT_CH = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percep_cfg(mode: str) -> PerceptionConfig:
    return PerceptionConfig(
        MODE=mode,
        OUT_CHANNEL=OUT_CH,
        KERNEL_SIZE=3,
        NUM_HEADS=4,
        EMBED_DIM=OUT_CH,   # must be divisible by NUM_HEADS
        USE_REL_POS_BIAS=True,
        USE_LAYER_NORM=True,
        INCLUDE_FFN=True,
    )


def _update_model_config():
    cfg = MagicMock()
    cfg.MODEL.HIDDEN_CHANNELS = [32]
    cfg.MODEL.RESNET_BLOCKS = 2
    cfg.MODEL.FINAL_ACTIVATION = False
    return cfg

# ---------------------------------------------------------------------------
# UPDATE_MODEL_REGISTRY — exhaustive compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(UPDATE_MODEL_REGISTRY))
def test_update_model_output_shape(key):
    """forward() must return [B, out_channels, H, W]."""
    cfg = _update_model_config()
    module = UPDATE_MODEL_REGISTRY[key](UPDATE_IN_CH, UPDATE_OUT_CH, cfg)

    x = torch.rand(B, UPDATE_IN_CH, H, W)
    out = module(x)

    assert out.shape == (B, UPDATE_OUT_CH, H, W), (
        f"[{key}] expected shape {(B, UPDATE_OUT_CH, H, W)}, got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("key", sorted(UPDATE_MODEL_REGISTRY))
def test_update_model_gradients_flow(key):
    """Gradients must reach the input tensor through forward()."""
    cfg = _update_model_config()
    module = UPDATE_MODEL_REGISTRY[key](UPDATE_IN_CH, UPDATE_OUT_CH, cfg)

    x = torch.rand(B, UPDATE_IN_CH, H, W, requires_grad=True)
    out = module(x)
    out.sum().backward()

    assert x.grad is not None, f"[{key}] gradients did not reach the input"


def test_update_model_registry_invalid_key_raises():
    with pytest.raises(KeyError):
        UPDATE_MODEL_REGISTRY["not_a_real_model"](UPDATE_IN_CH, UPDATE_OUT_CH, _update_model_config())
