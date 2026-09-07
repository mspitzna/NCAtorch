from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nca.utils.config import (
    AdversarialConfig, CFGConfig, Config, DatasetConfig, LatentConfig,
    LoggingConfig, ModelConfig, PerceptionConfig, SamplePoolConfig,
    TorchCompileConfig, TrainingConfig, load_config,
)


def test_ae_decoder_requires_nonzero_channel_widths():
    with pytest.raises(ValidationError, match="LATENT_AE_CHANNEL must be at least"):
        LatentConfig(LATENT_AE_COMPRESSION=4, LATENT_AE_CHANNEL=4)
    LatentConfig(LATENT_AE_COMPRESSION=4, LATENT_AE_CHANNEL=8)


@pytest.mark.parametrize("encoder_type", ["VAE", "VQVAE"])
def test_group_norm_channels_must_be_divisible(encoder_type):
    with pytest.raises(ValidationError, match="VAE_BASE_CHANNELS must be divisible"):
        LatentConfig(ENCODER_TYPE=encoder_type, VAE_BASE_CHANNELS=10, VAE_NORM_GROUPS=4)
    LatentConfig(ENCODER_TYPE=encoder_type, VAE_BASE_CHANNELS=8, VAE_NORM_GROUPS=4)
    # Without downsampling there are no GroupNorm layers.
    LatentConfig(ENCODER_TYPE=encoder_type, VAE_NUM_DOWNSAMPLES=0, VAE_BASE_CHANNELS=10, VAE_NORM_GROUPS=4)


@pytest.mark.parametrize("section, field, value", [
    *[("TRAINING", "STEPS", value) for value in (-2, -1, 0)],
    ("TRAINING", "WARMUP_STEPS", -1),
    ("ADVERSARIAL", "D_WARMUP_STEPS", -1),
    ("LATENT_TRAINING", "LATENT_AE_WARMUP_STEPS", -1),
    ("ADVERSARIAL", "D_DOWNSCALE_FACTOR", 0),
    ("ADVERSARIAL", "D_DOWNSCALE_FACTOR", 1.5),
    *[("ADVERSARIAL", "D_GAMMA", value) for value in (-0.1, 1.1, float("nan"))],
    *[("TRAINING", field, value)
      for field in ("WSD_MIN_LR_RATIO", "WSD_DECAY_RATIO")
      for value in (-0.1, 1.1, float("inf"), float("nan"))],
    *[(section, field, value)
      for section, field in (("TRAINING", "OVERFLOW_WEIGHT"), ("ADVERSARIAL", "LPIPS_WEIGHT"))
      for value in (-1, float("inf"), float("nan"))],
    ("TRAINING", "LOSS_FN", "not_a_registered_loss"),
])
def test_invalid_fields_rejected_when_loading_yaml(tmp_path, section, field, value):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "DATASET": {"NAME": "emoji", "EMOJIS": ["x"]},
        section: {field: value},
    }))

    with pytest.raises(ValidationError) as error:
        load_config(str(config_path))

    assert (section, field) in [entry["loc"] for entry in error.value.errors()]


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_zero_warmup_and_ratio_endpoints_are_valid(tmp_path, ratio):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "DATASET": {"NAME": "emoji", "EMOJIS": ["x"]},
        "TRAINING": {"WARMUP_STEPS": 0, "WSD_MIN_LR_RATIO": ratio, "WSD_DECAY_RATIO": ratio},
        "ADVERSARIAL": {"D_WARMUP_STEPS": 0, "D_GAMMA": ratio},
        "LATENT_TRAINING": {"LATENT_AE_WARMUP_STEPS": 0},
    }))
    load_config(str(config_path))


def test_wsd_phase_lengths_validated_together():
    with pytest.raises(ValidationError, match="WSD warmup and decay phases"):
        TrainingConfig(LR_SCHEDULE_MODE="wsd", STEPS=10, WARMUP_STEPS=8, WSD_DECAY_RATIO=0.3)

    # A zero-length stable phase is allowed.
    TrainingConfig(LR_SCHEDULE_MODE="wsd", STEPS=10, WARMUP_STEPS=7, WSD_DECAY_RATIO=0.3)


@pytest.mark.parametrize("model, field", [
    *[(PerceptionConfig, field) for field in ("KERNEL_SIZE", "DILATION", "OUT_CHANNEL", "NUM_HEADS", "EMBED_DIM")],
    *[(ModelConfig, field) for field in ("CHANNEL_N", "CHANNEL_OUT")],
    *[(TrainingConfig, field) for field in (
        "BATCH_SIZE", "ITER_N_MIN", "ITER_N_MAX", "GRADIENT_CHECKPOINT_SEGMENTS",
        "GRADIENT_ACCUMULATION_STEPS", "VGG_PROJ_N", "LEARNING_RATE",
    )],
    *[(DatasetConfig, field) for field in ("TARGET_SIZE", "HISTORY_N", "SEED_SIZE")],
    *[(SamplePoolConfig, field) for field in ("POOL_SIZE", "POOL_DMG_DELAY")],
    *[(LatentConfig, field) for field in (
        "LATENT_AE_STEPS", "LATENT_AE_LR", "LATENT_AE_IN_CHANNEL", "LATENT_AE_OUT_CHANNEL",
        "LATENT_AE_CHANNEL", "LATENT_AE_COMPRESSION", "LATENT_AE_LOG_INTERVAL",
        "LATENT_AE_SAVE_INTERVAL", "VAE_BASE_CHANNELS", "VAE_NORM_GROUPS",
        "VAE_BATCH_SIZE", "VQVAE_NUM_EMBEDDINGS",
    )],
    *[(AdversarialConfig, field) for field in ("D_IN_CHANNELS", "D_LEARNING_RATE", "D_N_CRITIC")],
    *[(LoggingConfig, field) for field in ("LOG_INTERVAL", "SAVE_INTERVAL")],
])
@pytest.mark.parametrize("value", [0, -1])
def test_counts_dimensions_and_rates_must_be_positive(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize("model, field", [
    (ModelConfig, "RESNET_BLOCKS"), (ModelConfig, "LIVING_MASK_INDEX"),
    (TrainingConfig, "GRADIENT_CLIPPING_NORM"), (TrainingConfig, "LR_GAMMA"),
    (DatasetConfig, "TARGET_PADDING"), (DatasetConfig, "NUM_WORKERS"),
    (SamplePoolConfig, "POOL_DELAY"),
    *[(LatentConfig, field) for field in (
        "VAE_KL_BETA", "VAE_NUM_DOWNSAMPLES", "VAE_KL_WARMUP_STEPS",
        "VAE_RECON_LOSS_WEIGHT", "VAE_VGG_LOSS_WEIGHT", "VQVAE_COMMITMENT_COST",
    )],
    *[(AdversarialConfig, field) for field in ("D_START_TRAINING", "D_GP_WEIGHT", "ADV_WEIGHT", "RECON_WEIGHT")],
])
def test_nonnegative_settings_allow_zero(model, field):
    model(**{field: 0})
    with pytest.raises(ValidationError):
        model(**{field: -1})


@pytest.mark.parametrize("model, field", [
    (ModelConfig, "NOISE_INJECTION"), (ModelConfig, "FIRE_RATE"), (CFGConfig, "DROPOUT_PROB"),
    *[(SamplePoolConfig, field) for field in ("POOL_START_RATIO", "POOL_END_RATIO", "POOL_DMG_RATIO", "POOL_MUTATION_RATIO")],
])
def test_fractions_accept_endpoints_and_reject_out_of_range_values(model, field):
    for value in (0, 1):
        model(**{field: value})
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            model(**{field: value})


@pytest.mark.parametrize("model, settings", [
    (ModelConfig, {"HIDDEN_CHANNELS": [16, 0]}),
    (ModelConfig, {"PERCEPTIONS": []}),
    (AdversarialConfig, {"D_FEATURES": []}),
    (AdversarialConfig, {"D_FEATURES": [16, -1]}),
    (TrainingConfig, {"OPTIMIZER_BETAS": [0.9]}),
    (TrainingConfig, {"OPTIMIZER_BETAS": [0.9, 0.99, 0.999]}),
    (TrainingConfig, {"OPTIMIZER_BETAS": [0.9, 1.0]}),
    (TrainingConfig, {"OPTIMIZER_BETAS": [-0.1, 0.9]}),
    (TrainingConfig, {"OPTIMIZER_BETAS": [0.9, float("nan")]}),
    (TrainingConfig, {"MILESTONES": [-1]}),
    (LoggingConfig, {"INTERMEDIATE_LOGGING_STEPS": [-1]}),
    (CFGConfig, {"PRESERVE_CHANNELS": [-1]}),
    (DatasetConfig, {"EMOJIS": [""]}),
])
def test_invalid_list_lengths_and_elements_rejected(model, settings):
    with pytest.raises(ValidationError):
        model(**settings)


@pytest.mark.parametrize("model, settings", [
    (PerceptionConfig, {"MODE": "attention", "KERNEL_SIZE": 2}),
    (PerceptionConfig, {"MODE": "attention", "OUT_CHANNEL": 7, "NUM_HEADS": 2}),
    (PerceptionConfig, {"MODE": "mh_attention", "EMBED_DIM": 7, "NUM_HEADS": 2}),
    (ModelConfig, {"NAME": "ResNet", "HIDDEN_CHANNELS": []}),
    (ModelConfig, {"CLAMP_OUTPUT_MIN": 2, "CLAMP_OUTPUT_MAX": 1}),
    (ModelConfig, {"CHANNEL_N": 3, "LIVING_MASK": True, "LIVING_MASK_INDEX": 3}),
    (TrainingConfig, {"ITER_N_MIN": 64, "ITER_N_MAX": 32}),
    (TrainingConfig, {"GRADIENT_CHECKPOINTING": True, "GRADIENT_CHECKPOINT_SEGMENTS": 33}),
])
def test_related_settings_are_validated_together(model, settings):
    with pytest.raises(ValidationError):
        model(**settings)


@pytest.mark.parametrize("model, settings", [
    (TrainingConfig, {"LPIPS_NET": "invalid"}),
    (TrainingConfig, {"LR_SCHEDULE_MODE": "invalid"}),
    (LatentConfig, {"VAE_RECON_LOSS_TYPE": "invalid"}),
    (CFGConfig, {"NULL_CONDITION_TYPE": "invalid"}),
    (TorchCompileConfig, {"MODE": "reduce-overhead"}),
    (DatasetConfig, {"NAME": "invalid"}),
])
def test_unknown_fixed_choices_and_registry_keys_rejected(model, settings):
    with pytest.raises(ValidationError):
        model(**settings)


def test_finite_float_rule_covers_unbounded_fields_and_defaults():
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError):
            ModelConfig(CLAMP_OUTPUT_MIN=value)

    class InvalidDefault(ModelConfig):
        CLAMP_OUTPUT_MIN: float = float("inf")

    with pytest.raises(ValidationError):
        InvalidDefault()


def test_nullable_values_and_full_config_round_trip():
    config = Config(
        DATASET={"EMOJIS": ["x"], "DATAROOT": None, "DATASET_SAMPLE_PATH": None},
        MODEL={"CHANNEL_N": 8, "CHANNEL_OUT": None},
        PATTERN_POOL={"POOL_DMG_DELAY": None},
        LATENT_TRAINING={"AE_CHECKPOINT": None},
        COND_DIM=None, IM_HEIGHT=None, IM_WIDTH=None,
    )
    assert config.MODEL.CHANNEL_OUT == 8
    assert Config.model_validate_json(config.model_dump_json()) == config


def test_supported_sentinels_and_empty_optional_lists():
    assert TrainingConfig(STEPS=1).STEPS == 1
    for seed in (-1, 0, 2**32 - 1):
        Config(SEED=seed, DATASET={"EMOJIS": ["x"]})
    for steps in (-2, -1, 0):
        with pytest.raises(ValidationError):
            TrainingConfig(STEPS=steps)
    ModelConfig(NAME="MLP", HIDDEN_CHANNELS=[])
    TrainingConfig(MILESTONES=[], OPTIMIZER_BETAS=[0, 0])
    LoggingConfig(INTERMEDIATE_LOGGING_STEPS=[])
    CFGConfig(PRESERVE_CHANNELS=[])


def test_repository_configs_and_template_are_compatible():
    config_dir = Path(__file__).resolve().parents[1] / "config"
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if path.name == "template_config.yaml":
            # The template intentionally leaves required dataset content to the user.
            raw["DATASET"]["EMOJIS"] = ["x"]
        Config(**raw)
