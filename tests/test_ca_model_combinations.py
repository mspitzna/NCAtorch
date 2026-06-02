"""
Integration tests for CAModel and LatentWrapper with every
perception × update model combination.

Driven by PERCEPTION_REGISTRY and UPDATE_MODEL_REGISTRY — any new entry is
automatically covered. Tests verify:
  - output shape == input shape for every combination
  - gradients flow end-to-end
  - LatentWrapper.evolve_in_pixel_space preserves shape
  - LatentWrapper.evolve_in_latent_space preserves pixel-space shape
"""

import types
import pytest
import torch
from unittest.mock import MagicMock, patch

from nca.core.models.ca.ca_model import CAModel
from nca.core.models.ca.state_updates import ApplyDelta, StateUpdatePipeline
from nca.core.models.perception_factory import PERCEPTION_REGISTRY
from nca.core.models.update_model_factory import UPDATE_MODEL_REGISTRY
from nca.core.models.latent_wrapper import LatentWrapper
from nca.core.models.auto_encoder.ae import AutoEncoder
from nca.utils.config import PerceptionConfig, ModelConfig

# ---------------------------------------------------------------------------
# Shared dims — CA model (state space)
# ---------------------------------------------------------------------------

B, CH, H, W = 2, 16, 16, 16
PERC_OUT = 32

# Latent / pixel-space dims for LatentWrapper tests
PIXEL_IN_CH  = 4   # AE input/output channels
LATENT_CH    = 8   # channels in latent space (== CA model's CH)
COMPRESSION  = 1   # compression_level=1: 2× spatial downscale (32 → 16)
PIXEL_H, PIXEL_W = 32, 32
LATENT_H = PIXEL_H // (2 ** COMPRESSION)
LATENT_W = PIXEL_W // (2 ** COMPRESSION)


# ---------------------------------------------------------------------------
# Helpers — CA model
# ---------------------------------------------------------------------------

def _percep_cfg(mode: str, out_channel: int = PERC_OUT) -> PerceptionConfig:
    return PerceptionConfig(
        MODE=mode,
        OUT_CHANNEL=out_channel,
        KERNEL_SIZE=3,
        NUM_HEADS=4,
        EMBED_DIM=out_channel,   # must be divisible by NUM_HEADS
        USE_REL_POS_BIAS=True,
        USE_LAYER_NORM=True,
        INCLUDE_FFN=True,
    )


def _update_cfg():
    return types.SimpleNamespace(MODEL=ModelConfig(HIDDEN_CHANNELS=[32], RESNET_BLOCKS=2))


def _make_ca(perc_key: str, update_key: str, in_ch: int = CH,
             out_ch: int = CH, img_h: int = H, img_w: int = W) -> CAModel:
    percep_cfg = _percep_cfg(perc_key)
    perception = PERCEPTION_REGISTRY[perc_key](in_ch, percep_cfg, "cpu")
    update = UPDATE_MODEL_REGISTRY[update_key](
        perception.get_out_channel(), out_ch, _update_cfg()
    )
    return CAModel(
        device="cpu",
        img_height=img_h,
        img_width=img_w,
        perception_module=perception,
        update_model_module=update,
        state_update_pipeline=StateUpdatePipeline([ApplyDelta()]),
    )


# ---------------------------------------------------------------------------
# Helpers — LatentWrapper
# ---------------------------------------------------------------------------

def _make_latent_wrapper(perc_key: str, update_key: str) -> LatentWrapper:
    ae = AutoEncoder(PIXEL_IN_CH, PIXEL_IN_CH, LATENT_CH, COMPRESSION)
    ca = _make_ca(perc_key, update_key,
                  in_ch=LATENT_CH, out_ch=LATENT_CH,
                  img_h=LATENT_H, img_w=LATENT_W)
    # Bypass checkpoint loading — inject AE directly
    with patch.object(LatentWrapper, "_load_encoder_decoder", return_value=ae):
        wrapper = LatentWrapper(ca, MagicMock())
    return wrapper


# ---------------------------------------------------------------------------
# Parametrized combinations
# ---------------------------------------------------------------------------

COMBINATIONS = [
    (perc_key, update_key)
    for perc_key in sorted(PERCEPTION_REGISTRY)
    for update_key in sorted(UPDATE_MODEL_REGISTRY)
]


# ---------------------------------------------------------------------------
# CAModel — output shape == input shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perc_key,update_key", COMBINATIONS)
def test_output_shape_preserved(perc_key, update_key):
    """Output shape must equal input shape for every combination."""
    model = _make_ca(perc_key, update_key)
    x = torch.rand(B, CH, H, W)
    out, _ = model(x)
    assert out.shape == x.shape, (
        f"[{perc_key} + {update_key}] input {tuple(x.shape)}, output {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# CAModel — gradient flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perc_key,update_key", COMBINATIONS)
def test_gradients_flow(perc_key, update_key):
    """Gradients must reach the input for every combination."""
    model = _make_ca(perc_key, update_key)
    x = torch.rand(B, CH, H, W, requires_grad=True)
    model(x)[0].sum().backward()
    assert x.grad is not None, (
        f"[{perc_key} + {update_key}] gradients did not reach the input"
    )


# ---------------------------------------------------------------------------
# LatentWrapper — evolve_in_pixel_space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perc_key,update_key", COMBINATIONS)
def test_latent_wrapper_pixel_space_shape_preserved(perc_key, update_key):
    """evolve_in_pixel_space output shape must equal input shape."""
    wrapper = _make_latent_wrapper(perc_key, update_key)
    x = torch.rand(B, LATENT_CH, LATENT_H, LATENT_W)
    out, _ = wrapper.evolve_in_pixel_space(x)
    assert out.shape == x.shape, (
        f"[{perc_key} + {update_key}] pixel space: input {tuple(x.shape)}, output {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# LatentWrapper — evolve_in_latent_space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perc_key,update_key", COMBINATIONS)
def test_latent_wrapper_latent_space_shape_preserved(perc_key, update_key):
    """evolve_in_latent_space must return the same pixel-space shape as the input."""
    wrapper = _make_latent_wrapper(perc_key, update_key)
    # Input is in pixel space — the wrapper encodes, evolves, then decodes
    wrapper.config.LATENT_TRAINING.LATENT_AE_IN_CHANNEL = PIXEL_IN_CH
    x = torch.rand(B, PIXEL_IN_CH, PIXEL_H, PIXEL_W)
    out, _ = wrapper.evolve_in_latent_space(x)
    assert out.shape == x.shape, (
        f"[{perc_key} + {update_key}] latent space: input {tuple(x.shape)}, output {tuple(out.shape)}"
    )
