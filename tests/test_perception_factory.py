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

from nca.core.models.perception_factory import PERCEPTION_REGISTRY
from nca.utils.config import PerceptionConfig

# ---------------------------------------------------------------------------
# Shared dims
# ---------------------------------------------------------------------------

B, IN_CH, H, W = 2, 16, 16, 16
OUT_CH = 32  # perception output channels (not used by sobel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percep_cfg(mode: str) -> PerceptionConfig:
    return PerceptionConfig(
        MODE=mode,
        OUT_CHANNEL=OUT_CH,
        KERNEL_SIZE=3,
        NUM_HEADS=4,
        EMBED_DIM=OUT_CH,  # must be divisible by NUM_HEADS
        USE_REL_POS_BIAS=True,
        USE_LAYER_NORM=True,
        INCLUDE_FFN=True,
    )


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


# ---------------------------------------------------------------------------
# Configurable kernel size and padding mode (conv-based perceptions)
# ---------------------------------------------------------------------------

CONV_MODES = ["conv", "deformable_conv", "residual_conv"]
PADDING_MODES = ["circular", "zeros", "reflect", "replicate"]
# Only these modes accept a non-square (kh, kw) kernel size.
TUPLE_KERNEL_MODES = ["conv", "residual_conv"]


def _conv_cfg(mode: str, **overrides) -> PerceptionConfig:
    overrides.setdefault("KERNEL_SIZE", 3)
    return PerceptionConfig(MODE=mode, OUT_CHANNEL=OUT_CH, **overrides)


@pytest.mark.parametrize("mode", CONV_MODES)
@pytest.mark.parametrize("padding_mode", PADDING_MODES)
def test_conv_perception_padding_modes(mode, padding_mode):
    """Every conv-based perception must accept each padding mode."""
    cfg = _conv_cfg(mode, PADDING_MODE=padding_mode)
    module = PERCEPTION_REGISTRY[mode](IN_CH, cfg, "cpu")

    x = torch.rand(B, IN_CH, H, W)
    out = module(x)

    assert out.shape == (B, module.get_out_channel(), H, W), (
        f"[{mode}/{padding_mode}] unexpected shape {tuple(out.shape)}"
    )


@pytest.mark.parametrize("mode", TUPLE_KERNEL_MODES)
@pytest.mark.parametrize("kernel_size", [(5, 3), (3, 5), (2, 4)])
def test_conv_perception_tuple_kernel(mode, kernel_size):
    """Non-square kernel sizes must be accepted and keep the spatial size."""
    cfg = _conv_cfg(mode, KERNEL_SIZE=list(kernel_size))
    module = PERCEPTION_REGISTRY[mode](IN_CH, cfg, "cpu")

    x = torch.rand(B, IN_CH, H, W)
    out = module(x)

    assert out.shape == (B, module.get_out_channel(), H, W), (
        f"[{mode}/{kernel_size}] unexpected shape {tuple(out.shape)}"
    )
