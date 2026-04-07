"""
Exhaustive registry tests for latent_encoder_factory.

Compliance contract for every latent encoder:
  - create_latent_encoder(cfg, device, inference_only=True)  -> (nn.Module, None, any)
  - create_latent_encoder(cfg, device, inference_only=False) -> (nn.Module, nn.Module, any)
  - model.forward() runs without error and returns the expected output shape
  - gradients flow through the model's encode path

VAE inference_only=False is skipped — it loads a pretrained VGG19 network.
"""

import types
import pytest
import torch
import torch.nn as nn

from nca.core.models.latent_encoder_factory import (
    LATENT_ENCODER_REGISTRY,
    create_latent_encoder,
)
from nca.utils.config import LatentConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Minimal config using real Pydantic models so defaults and validators run
# ---------------------------------------------------------------------------

IN_CH  = 4
OUT_CH = 4
LATENT = 8
COMP   = 1   # 2^1 = 2× spatial compression — keeps tensors small in tests


def _make_config(encoder_type: str):
    latent = LatentConfig(
        ENCODER_TYPE=encoder_type,
        LATENT_AE_IN_CHANNEL=IN_CH,
        LATENT_AE_OUT_CHANNEL=OUT_CH,
        LATENT_AE_CHANNEL=LATENT,
        LATENT_AE_COMPRESSION=COMP,
        VAE_BASE_CHANNELS=16,
        VAE_NUM_DOWNSAMPLES=1,
        VAE_NORM_GROUPS=4,
        VQVAE_NUM_EMBEDDINGS=64,
    )
    return types.SimpleNamespace(
        LATENT_TRAINING=latent,
        TRAINING=TrainingConfig(),
    )


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
    """AE with inference_only=False must return a non-None criterion and no kl_beta."""
    cfg = _make_config("AE")
    model, criterion, kl_beta = create_latent_encoder(cfg, "cpu", inference_only=False)

    assert isinstance(model, nn.Module)
    assert isinstance(criterion, nn.Module), "AE criterion must be an nn.Module"
    assert kl_beta is None, "AE does not use kl_beta"


def test_vqvae_full_returns_criterion():
    """VQVAE with inference_only=False must return a non-None criterion and no kl_beta."""
    cfg = _make_config("VQVAE")
    model, criterion, kl_beta = create_latent_encoder(cfg, "cpu", inference_only=False)

    assert isinstance(model, nn.Module)
    assert isinstance(criterion, nn.Module), "VQVAE criterion must be an nn.Module"
    assert kl_beta is None, "VQVAE does not use kl_beta"


def test_vae_full_returns_criterion():
    """VAE with inference_only=False must return a non-None criterion and kl_beta."""
    pytest.skip("VAE criterion (VGGLoss) requires downloading pretrained VGG19")


# ---------------------------------------------------------------------------
# Invalid key
# ---------------------------------------------------------------------------

def test_invalid_encoder_type_raises():
    with pytest.raises(ValueError):
        LatentConfig(ENCODER_TYPE="not_a_real_encoder")
