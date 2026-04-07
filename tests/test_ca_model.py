"""
Tests for CAModel forward pass.

Covers:
  - output shape matches input state shape
  - gradients flow back through forward()
  - return_residuals returns (state, dx) with correct shapes
  - freeze_channels leaves frozen channels unchanged
  - step_size=0 produces no change to state
  - 2D and 4D condition tensors are handled correctly
  - batch size mismatch between x and cond raises ValueError
  - positional embeddings enabled: forward still produces correct shape
  - input tensor is not mutated in-place
  - missing state_update_pipeline triggers a warning and uses ApplyDelta default
"""

import warnings
import pytest
import torch

from nca.core.models.ca.ca_model import CAModel
from nca.core.models.ca.perceptions import ConvPerception
from nca.core.models.ca.update_models import SimpleMLPUpdate
from nca.core.models.ca.state_updates import (
    ApplyDelta,
    ClampOutput,
    FireRateMask,
    LivingMask,
    NoiseInjection,
    StateUpdatePipeline,
)

# ---------------------------------------------------------------------------
# Shared dims
# ---------------------------------------------------------------------------

B, CH, H, W = 2, 16, 16, 16
COND_DIM  = 4
PERC_OUT  = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ca(
    cond_dim: int = 0,
    use_positional_embeddings: bool = False,
    pipeline=None,
) -> CAModel:
    pos_ch = 2 if use_positional_embeddings else 0
    perception = ConvPerception(CH + cond_dim + pos_ch, PERC_OUT, kernel_size=3)
    update = SimpleMLPUpdate(PERC_OUT, CH, hidden_channels=[32])
    if pipeline is None:
        pipeline = StateUpdatePipeline([ApplyDelta()])
    return CAModel(
        device="cpu",
        use_positional_embeddings=use_positional_embeddings,
        img_height=H,
        img_width=W,
        perception_module=perception,
        update_model_module=update,
        state_update_pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape_no_cond():
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    out = model(x)
    assert out.shape == (B, CH, H, W)


def test_output_shape_2d_cond():
    model = _make_ca(cond_dim=COND_DIM)
    x = torch.rand(B, CH, H, W)
    cond = torch.rand(B, COND_DIM)
    out = model(x, cond=cond)
    assert out.shape == (B, CH, H, W)


def test_output_shape_4d_cond():
    model = _make_ca(cond_dim=COND_DIM)
    x = torch.rand(B, CH, H, W)
    cond = torch.rand(B, COND_DIM, H, W)
    out = model(x, cond=cond)
    assert out.shape == (B, CH, H, W)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow():
    model = _make_ca()
    x = torch.rand(B, CH, H, W, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# return_residuals
# ---------------------------------------------------------------------------

def test_return_residuals_returns_tuple():
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    result = model(x, return_residuals=True)
    assert isinstance(result, tuple) and len(result) == 2


def test_return_residuals_shapes():
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    state, dx = model(x, return_residuals=True)
    assert state.shape == (B, CH, H, W)
    assert dx.shape == (B, CH, H, W)


# ---------------------------------------------------------------------------
# step_size
# ---------------------------------------------------------------------------

def test_step_size_zero_produces_no_change():
    """step_size=0 multiplies dx to zero, so output must equal input."""
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    out = model(x, step_size=0.0)
    assert torch.allclose(out, x, atol=1e-6)


def test_step_size_scales_update():
    """Larger step_size should produce a larger deviation from the input."""
    model = _make_ca()
    model.eval()
    x = torch.rand(B, CH, H, W)
    with torch.no_grad():
        out_small = model(x, step_size=0.01)
        out_large = model(x, step_size=10.0)
    diff_small = (out_small - x).abs().mean()
    diff_large = (out_large - x).abs().mean()
    assert diff_large > diff_small


# ---------------------------------------------------------------------------
# freeze_channels
# ---------------------------------------------------------------------------

def test_freeze_channels_unchanged():
    """Channels 0..freeze_channels (inclusive) must be identical after forward."""
    freeze = 3
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    out = model(x, freeze_channels=freeze)
    assert torch.allclose(out[:, :freeze + 1], x[:, :freeze + 1])


def test_freeze_channels_unfrozen_can_change():
    """Channels beyond freeze_channels should be free to update."""
    freeze = 3
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    out = model(x, step_size=1.0, freeze_channels=freeze)
    # At least some unfrozen channels should differ (with high probability)
    assert not torch.allclose(out[:, freeze + 1:], x[:, freeze + 1:])


# ---------------------------------------------------------------------------
# Condition handling
# ---------------------------------------------------------------------------

def test_cond_batch_mismatch_raises():
    model = _make_ca(cond_dim=COND_DIM)
    x = torch.rand(B, CH, H, W)
    cond = torch.rand(B + 1, COND_DIM)  # wrong batch size
    with pytest.raises(ValueError, match="Batch size mismatch"):
        model(x, cond=cond)


def test_cond_4d_spatial_mismatch_is_interpolated():
    """4D condition with different spatial dims should be interpolated, not crash."""
    model = _make_ca(cond_dim=COND_DIM)
    x = torch.rand(B, CH, H, W)
    cond = torch.rand(B, COND_DIM, H // 2, W // 2)  # half spatial size
    out = model(x, cond=cond)
    assert out.shape == (B, CH, H, W)


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

def test_positional_embeddings_output_shape():
    model = _make_ca(use_positional_embeddings=True)
    x = torch.rand(B, CH, H, W)
    out = model(x)
    assert out.shape == (B, CH, H, W)


def test_positional_embeddings_are_learnable():
    model = _make_ca(use_positional_embeddings=True)
    assert model.positional_embeddings is not None
    assert model.positional_embeddings.requires_grad


# ---------------------------------------------------------------------------
# In-place mutation guard
# ---------------------------------------------------------------------------

def test_input_not_mutated():
    model = _make_ca()
    x = torch.rand(B, CH, H, W)
    x_copy = x.clone()
    model(x)
    assert torch.allclose(x, x_copy), "forward() must not mutate the input tensor"


# ---------------------------------------------------------------------------
# Default pipeline warning
# ---------------------------------------------------------------------------

def test_missing_pipeline_warns():
    perception = ConvPerception(CH, PERC_OUT, kernel_size=3)
    update = SimpleMLPUpdate(PERC_OUT, CH, hidden_channels=[32])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CAModel(
            device="cpu",
            img_height=H,
            img_width=W,
            perception_module=perception,
            update_model_module=update,
            state_update_pipeline=None,
        )
    assert any("state_update_pipeline" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# State update pipeline variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pipeline", [
    StateUpdatePipeline([ApplyDelta()]),
    StateUpdatePipeline([NoiseInjection(0.01), ApplyDelta()]),
    StateUpdatePipeline([FireRateMask(0.5), ApplyDelta()]),
    StateUpdatePipeline([ApplyDelta(), ClampOutput()]),
])
def test_pipeline_variants_output_shape(pipeline):
    """All supported pipeline configurations must produce correct output shape."""
    model = _make_ca(pipeline=pipeline)
    x = torch.rand(B, CH, H, W)
    out = model(x)
    assert out.shape == (B, CH, H, W)
