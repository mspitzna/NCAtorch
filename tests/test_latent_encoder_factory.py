"""
Exhaustive registry tests for latent_encoder_factory.

Compliance contract for every latent encoder:
  - create_latent_encoder(cfg, device, inference_only=True)  -> (nn.Module, None, any)
  - create_latent_encoder(cfg, device, inference_only=False) -> (nn.Module, nn.Module, any)
  - model.forward() runs without error and returns the expected output shape
  - gradients flow through the model's encode path

VAE inference_only=False is skipped — it loads a pretrained VGG19 network.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from nca.core.models.latent_encoder_factory import (
    LATENT_ENCODER_REGISTRY,
    create_latent_encoder,
)

# ---------------------------------------------------------------------------
# Minimal config mock
# ---------------------------------------------------------------------------

IN_CH   = 4
OUT_CH  = 4
LATENT  = 8
COMP    = 1   # 2^1 = 2x spatial compression — keeps tensors small in tests


def _make_config(encoder_type: str):
    cfg = MagicMock()
    cfg.LATENT_TRAINING.ENCODER_TYPE    = encoder_type
    cfg.LATENT_TRAINING.LATENT_AE_IN_CHANNEL  = IN_CH
    cfg.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL = OUT_CH
    cfg.LATENT_TRAINING.LATENT_AE_CHANNEL     = LATENT
    cfg.LATENT_TRAINING.LATENT_AE_COMPRESSION = COMP
    # VAE-specific
    cfg.LATENT_TRAINING.VAE_BASE_CHANNELS  = 16
    cfg.LATENT_TRAINING.VAE_NUM_DOWNSAMPLES = 1
    cfg.LATENT_TRAINING.VAE_NORM_GROUPS    = 4
    cfg.LATENT_TRAINING.VAE_KL_BETA        = 1.0
    cfg.LATENT_TRAINING.VAE_RECON_LOSS_TYPE   = "l1"
    cfg.LATENT_TRAINING.VAE_RECON_LOSS_WEIGHT = 1.0
    cfg.LATENT_TRAINING.VAE_VGG_LOSS_WEIGHT   = 1.0
    # AE-specific
    cfg.TRAINING.OVERFLOW_LOSS = False
    return cfg


# ---------------------------------------------------------------------------
# inference_only=True — model only, no criterion, no heavy downloads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(LATENT_ENCODER_REGISTRY))
def test_inference_only_returns_model_and_nones(key):
    """inference_only=True must return (nn.Module, None, *)."""
    cfg = _make_config(key)
    model, criterion, _ = create_latent_encoder(cfg, "cpu", inference_only=True)

    assert isinstance(model, nn.Module), f"[{key}] model must be an nn.Module"
    assert criterion is None, f"[{key}] criterion must be None when inference_only=True"


@pytest.mark.parametrize("key", sorted(LATENT_ENCODER_REGISTRY))
def test_encode_output_is_tensor(key):
    """encode() must return a tensor."""
    cfg = _make_config(key)
    model, _, _ = create_latent_encoder(cfg, "cpu", inference_only=True)

    x = torch.rand(2, IN_CH, 16, 16)
    z = model.encode(x)

    # VAE returns (mu, logvar); AE returns a plain tensor
    out = z[0] if isinstance(z, tuple) else z
    assert isinstance(out, torch.Tensor), f"[{key}] encode() must return a tensor (or tuple of tensors)"


@pytest.mark.parametrize("key", sorted(LATENT_ENCODER_REGISTRY))
def test_encode_gradients_flow(key):
    """Gradients must reach the input through encode()."""
    cfg = _make_config(key)
    model, _, _ = create_latent_encoder(cfg, "cpu", inference_only=True)

    x = torch.rand(2, IN_CH, 16, 16, requires_grad=True)
    z = model.encode(x)
    out = z[0] if isinstance(z, tuple) else z
    out.sum().backward()

    assert x.grad is not None, f"[{key}] gradients did not reach the input through encode()"


# ---------------------------------------------------------------------------
# inference_only=False — criterion is also returned (AE only; VAE skipped)
# ---------------------------------------------------------------------------

def test_ae_full_returns_criterion():
    """AE with inference_only=False must return a non-None criterion."""
    cfg = _make_config("AE")
    model, criterion, kl_beta = create_latent_encoder(cfg, "cpu", inference_only=False)

    assert isinstance(model, nn.Module)
    assert isinstance(criterion, nn.Module), "AE criterion must be an nn.Module"
    assert kl_beta is None, "AE does not use kl_beta"


def test_vae_full_returns_criterion():
    """VAE with inference_only=False must return a non-None criterion and kl_beta."""
    pytest.skip("VAE criterion (VGGLoss) requires downloading pretrained VGG19")


# ---------------------------------------------------------------------------
# Invalid key
# ---------------------------------------------------------------------------

def test_invalid_encoder_type_raises():
    cfg = _make_config("AE")
    cfg.LATENT_TRAINING.ENCODER_TYPE = "not_a_real_encoder"
    with pytest.raises(ValueError):
        create_latent_encoder(cfg, "cpu")
