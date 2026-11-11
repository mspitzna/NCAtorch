from types import SimpleNamespace

import pytest
import torch

from nca.core.models import latent_encoder_factory
from nca.core.models.latent_wrapper import LatentWrapper
from nca.core.models.model_factory import create_model
from nca.utils.config import Config, LatentConfig


def make_config(encoder_type, **latent):
    return Config(
        DEVICE="cpu",
        DATASET={"NAME": "emoji", "EMOJIS": ["x"]},
        MODEL={"USE_POSITIONAL_EMBEDDINGS": True, "FIRE_RATE": 1.0},
        LATENT_TRAINING={
            "ENABLED": True, "ENCODER_TYPE": encoder_type,
            "LATENT_AE_CHANNEL": 8, "LATENT_AE_COMPRESSION": 1,
            "VAE_NUM_DOWNSAMPLES": 2, "VAE_BASE_CHANNELS": 8,
            "VAE_NORM_GROUPS": 4, "VQVAE_NUM_EMBEDDINGS": 8,
            **latent,
        },
    )


@pytest.mark.parametrize("encoder_type", ["AE", "VAE", "VQVAE"])
@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_loading_and_latent_ca_backward(tmp_path, monkeypatch, encoder_type, explicit):
    config = make_config(encoder_type)
    encoder, _, _ = latent_encoder_factory.create_latent_encoder(config, "cpu", inference_only=True)
    if explicit:
        # Explicit weights must work without a run folder.
        checkpoint = tmp_path / "pretrained.pt"
        config.LATENT_TRAINING.AE_CHECKPOINT = checkpoint
    else:
        config.LOGGING.FOLDER_NAME = str(tmp_path)
        checkpoint = tmp_path / "ae_checkpoints" / f"{encoder_type.lower()}.pt"
        checkpoint.parent.mkdir()
    torch.save(encoder.state_dict(), checkpoint)

    def no_training_loss(*args, **kwargs):
        pytest.fail("Loading frozen weights must not construct a training loss")

    monkeypatch.setattr(latent_encoder_factory, "_vae_criterion", no_training_loss)
    monkeypatch.setattr(latent_encoder_factory, "ReconstructionLoss", no_training_loss)
    ca = create_model(config, cond_dim=0, img_height=16, img_width=24)
    wrapper = LatentWrapper(ca, config)
    for name, weight in encoder.state_dict().items():
        torch.testing.assert_close(wrapper.encoder_decoder.state_dict()[name], weight)
    assert not wrapper.encoder_decoder.training
    assert all(not p.requires_grad for p in wrapper.encoder_decoder.parameters())

    # Pixel seeds have padded CA channels; only encoder input channels are used.
    seed = torch.rand(2, config.MODEL.CHANNEL_N, 16, 24)
    latent = wrapper.encode(seed)
    expected_hw = (8, 12) if encoder_type == "AE" else (4, 6)
    assert latent.shape == (2, 8, *expected_hw)
    assert ca.positional_embeddings.shape[-2:] == expected_hw
    prediction, _ = wrapper(seed)
    assert prediction.shape == (2, 4, 16, 24)
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ca.parameters())
    assert all(p.grad is None for p in wrapper.encoder_decoder.parameters())


def test_explicit_checkpoint_wins_over_folder_checkpoint(tmp_path):
    config = make_config("AE")
    encoder, _, _ = latent_encoder_factory.create_latent_encoder(config, "cpu", inference_only=True)
    config.LATENT_TRAINING.AE_CHECKPOINT = tmp_path / "explicit.pt"
    torch.save(encoder.state_dict(), config.LATENT_TRAINING.AE_CHECKPOINT)
    default = tmp_path / "ae_checkpoints" / "ae.pt"
    default.parent.mkdir()
    default.write_bytes(b"invalid checkpoint that must not be read")
    loaded = latent_encoder_factory.load_latent_encoder(config, "cpu", folder_name=tmp_path)
    for name, weight in encoder.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], weight)


@pytest.mark.parametrize("source", ["unset", "folder", "explicit", "testing"])
def test_missing_checkpoint_reports_path(tmp_path, source):
    config = make_config("AE")
    expected = "AE_CHECKPOINT"
    if source == "folder":
        config.LOGGING.FOLDER_NAME = str(tmp_path)
        expected = "ae_checkpoints/ae.pt"
    elif source == "explicit":
        config.LATENT_TRAINING.AE_CHECKPOINT = tmp_path / "missing.pt"
        expected = "missing.pt"
    elif source == "testing":
        config.LOGGING.FOLDER_NAME = "??testing"
        expected = "ae_checkpoints/ae.pt"
    with pytest.raises(FileNotFoundError, match=expected):
        latent_encoder_factory.load_latent_encoder(config, "cpu")


@pytest.mark.parametrize("encoder_type,factor", [("AE", 2), ("VAE", 4), ("VQVAE", 4)])
@pytest.mark.parametrize("height,width", [(15, 24), (16, 23), (1, 1)])
def test_incompatible_image_dimensions_fail_before_model_creation(encoder_type, factor, height, width):
    config = make_config(encoder_type)
    with pytest.raises(ValueError, match=f"positive multiples of {factor}"):
        create_model(config, 0, height, width)


@pytest.mark.parametrize("encoder_type", ["VAE", "VQVAE"])
def test_zero_downsampling_preserves_image_size(encoder_type):
    config = make_config(encoder_type, VAE_NUM_DOWNSAMPLES=0)
    assert config.LATENT_TRAINING.get_latent_shape(15, 23) == (15, 23)
    encoder, _, _ = latent_encoder_factory.create_latent_encoder(config, "cpu", inference_only=True)
    ca = create_model(config, 0, 15, 23)
    encoded = encoder.encode(torch.rand(2, 4, 15, 23))
    latent = encoded[0] if isinstance(encoded, tuple) else encoded
    assert encoder.decode(ca(latent)[0]).shape == (2, 4, 15, 23)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_encoder_training_initializes_weights_from_checkpoint(tmp_path, monkeypatch, use_checkpoint):
    from scripts import train_ae

    config = make_config("AE", LATENT_AE_STEPS=1, LATENT_AE_WARMUP_STEPS=0)
    config.LOGGING.WANDB = False
    model, criterion, beta = latent_encoder_factory.create_latent_encoder(config, "cpu")
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    if use_checkpoint:
        expected = {name: value + 0.1 for name, value in expected.items()}
        config.LATENT_TRAINING.AE_CHECKPOINT = tmp_path / "initial.pt"
        torch.save(expected, config.LATENT_TRAINING.AE_CHECKPOINT)

    observed = []

    def check_loaded_weights(module, inputs):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[name])
        assert all(p.requires_grad for p in module.parameters())
        observed.append(True)

    model.register_forward_pre_hook(check_loaded_weights)
    batch = torch.rand(2, 4, 16, 24)
    monkeypatch.setattr(train_ae, "parse_args", lambda: SimpleNamespace(config="unused.yaml"))
    monkeypatch.setattr(train_ae, "load_config", lambda _: config)
    monkeypatch.setattr(train_ae, "setup_output_folder", lambda *args: (str(tmp_path), str(tmp_path)))
    monkeypatch.setattr(train_ae, "create_dataset", lambda _: ([(batch, None, batch)], 0, 16, 24))
    monkeypatch.setattr(train_ae, "create_latent_encoder", lambda *args: (model, criterion, beta))
    train_ae.main()
    assert observed == [True]
    assert (tmp_path / "ae.pt").is_file()
    assert any(not torch.equal(value, expected[name]) for name, value in model.state_dict().items())
