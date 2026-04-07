"""
Exhaustive registry tests for perception_factory.

These tests are driven by PERCEPTION_REGISTRY.
Any new entry added to the registry is automatically covered;
a non-compliant module will fail here.

Compliance contract for every perception:
  - forward(x) returns a tensor of shape [B, get_out_channel(), H, W]
  - get_out_channel() matches the actual output channel dimension
  - gradients flow back through forward()

"""

import pytest
import torch
from unittest.mock import MagicMock

from nca.core.models.perception_factory import PERCEPTION_REGISTRY
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
# PERCEPTION_REGISTRY — exhaustive compliance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(PERCEPTION_REGISTRY))
def test_perception_output_shape(key):
    """forward() must return [B, get_out_channel(), H, W]."""
    cfg = _percep_cfg(key)
    module = PERCEPTION_REGISTRY[key](IN_CH, cfg, "cpu")

    x = torch.rand(B, IN_CH, H, W)
    out = module(x)

    expected_ch = module.get_out_channel()
    assert out.shape == (B, expected_ch, H, W), (
        f"[{key}] expected shape {(B, expected_ch, H, W)}, got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("key", sorted(PERCEPTION_REGISTRY))
def test_perception_get_out_channel_matches_forward(key):
    """get_out_channel() must equal the channel dim of forward()'s output."""
    cfg = _percep_cfg(key)
    module = PERCEPTION_REGISTRY[key](IN_CH, cfg, "cpu")

    x = torch.rand(B, IN_CH, H, W)
    out = module(x)

    assert module.get_out_channel() == out.shape[1], (
        f"[{key}] get_out_channel()={module.get_out_channel()} but forward() returned {out.shape[1]} channels"
    )


@pytest.mark.parametrize("key", sorted(PERCEPTION_REGISTRY))
def test_perception_gradients_flow(key):
    """Gradients must reach the input tensor through forward()."""
    cfg = _percep_cfg(key)
    module = PERCEPTION_REGISTRY[key](IN_CH, cfg, "cpu")

    x = torch.rand(B, IN_CH, H, W, requires_grad=True)
    out = module(x)
    out.sum().backward()

    assert x.grad is not None, f"[{key}] gradients did not reach the input"


def test_perception_registry_invalid_key_raises():
    with pytest.raises((ValueError, KeyError)):
        cfg = _percep_cfg("conv")  # valid config, but we'll call registry directly
        PERCEPTION_REGISTRY["not_a_real_mode"](IN_CH, cfg, "cpu")