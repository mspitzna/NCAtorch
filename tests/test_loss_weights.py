from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nca.core.losses.loss_factory import create_loss_fn
from nca.core.models.latent_encoder_factory import create_latent_encoder
from nca.utils.config import LatentConfig, TrainingConfig


@pytest.mark.parametrize("key", ["overflow", "mse", "l1", "i_ce", "vgg"])
@pytest.mark.parametrize("weight", [0.0, 0.25, 3.0])
def test_overflow_weight_scales_penalty_and_gradients(key, weight, monkeypatch):
    # Exercise the real style-loss path with a tiny feature extractor, no downloads.
    monkeypatch.setattr(
        "nca.core.losses.loss_functions.models.vgg16",
        lambda **kwargs: SimpleNamespace(features=nn.Sequential(nn.Identity())),
    )
    config = SimpleNamespace(
        DEVICE="cpu",
        TRAINING=TrainingConfig(LOSS_FN=key, OVERFLOW_LOSS=True, OVERFLOW_WEIGHT=weight),
    )
    loss = create_loss_fn(config)
    pred = torch.full((2, 4, 4, 4), 2.0, requires_grad=True)
    target = torch.zeros_like(pred)
    if key == "i_ce":
        target[:, 0] = 1

    # Repeat the same random style projections when removing the overflow term.
    with torch.random.fork_rng(devices=[]):
        rng_state = torch.get_rng_state()
        result = loss(pred, target)
        torch.set_rng_state(rng_state)
        if key == "overflow":
            base_loss = pred.sum() * 0
        else:
            loss.overflow_loss = False
            base_loss = loss(pred, target)["total_loss"]

    expected = base_loss + weight * result["overflow_loss"]
    torch.testing.assert_close(result["total_loss"], expected)
    torch.testing.assert_close(
        torch.autograd.grad(result["total_loss"], pred, retain_graph=True)[0],
        torch.autograd.grad(expected, pred)[0],
    )
    assert result["overflow_loss"].item() > 0  # Metrics remain unweighted.
    if key == "vgg":
        torch.testing.assert_close(result["ot_loss"], base_loss)


@pytest.mark.parametrize("encoder_type", ["AE", "VQVAE"])
def test_encoder_reconstruction_receives_overflow_weight(encoder_type):
    config = SimpleNamespace(
        LATENT_TRAINING=LatentConfig(
            ENCODER_TYPE=encoder_type, LATENT_AE_CHANNEL=4,
            LATENT_AE_COMPRESSION=1, VAE_BASE_CHANNELS=8,
            VAE_NUM_DOWNSAMPLES=1, VAE_NORM_GROUPS=2, VQVAE_NUM_EMBEDDINGS=8,
        ),
        TRAINING=TrainingConfig(OVERFLOW_LOSS=True, OVERFLOW_WEIGHT=3.0),
    )
    _, loss, _ = create_latent_encoder(config, "cpu")
    result = loss(torch.full((2, 4, 4, 4), 2.0), torch.zeros(2, 4, 4, 4))
    # MSE is 4, overflow outside [0, 1] is 1.
    torch.testing.assert_close(result["total_loss"], torch.tensor(7.0))
