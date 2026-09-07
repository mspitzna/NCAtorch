from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError

from scripts import train_ca
from nca.utils.config import Config


@pytest.mark.parametrize("use_folder", [False, True])
@pytest.mark.parametrize("configured_device, cli_args, expected_device", [
    ("cpu", [], "cpu"),
    ("cuda:1", [], "cuda:1"),
    (None, [], "cuda"),
    ("cuda", ["--device", "cpu"], "cpu"),
    ("cpu", ["--device", "cuda:1"], "cuda:1"),
    ("cuda", ["-o", "DEVICE=cpu"], "cpu"),
    ("cpu", ["-o", "DEVICE=cuda:1", "--device", "cpu"], "cpu"),
])
def test_training_uses_configured_device_unless_explicitly_overridden(
    tmp_path, monkeypatch, use_folder, configured_device, cli_args, expected_device
):
    raw = {"DATASET": {"EMOJIS": ["x"]}}
    if configured_device is not None:
        raw["DEVICE"] = configured_device
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    source_args = ["--folder", str(tmp_path)] if use_folder else ["--config", str(config_path)]
    monkeypatch.setattr("sys.argv", ["train_ca.py", *source_args, *cli_args])
    monkeypatch.delenv("WANDB_SWEEP", raising=False)

    class DatasetReached(Exception):
        pass

    # Inspect the resolved config before any dataset downloads or device allocation.
    create_dataset = Mock(side_effect=DatasetReached)
    monkeypatch.setattr(train_ca, "create_dataset", create_dataset)
    with pytest.raises(DatasetReached):
        train_ca.main()

    create_dataset.assert_called_once()
    assert create_dataset.call_args.args[0].DEVICE == expected_device


def test_overrides_can_set_and_clear_nullable_fields():
    config = Config(DATASET={"EMOJIS": ["x"]})
    updated = train_ca.apply_overrides(config, {
        "TRAINING.TRAINER_TYPE": "adversarial", "LOGGING.FOLDER_NAME": "runs/example",
        "DATASET.DATAROOT": "datasets", "LATENT_TRAINING.AE_CHECKPOINT": "encoder.pt",
        "PATTERN_POOL.POOL_DMG_DELAY": 10,
    })
    assert updated.TRAINING.TRAINER_TYPE == "adversarial"
    assert str(updated.LATENT_TRAINING.AE_CHECKPOINT) == "encoder.pt"
    assert config.TRAINING.TRAINER_TYPE is None
    cleared = train_ca.apply_overrides(updated, {
        "TRAINING.TRAINER_TYPE": None, "LOGGING.FOLDER_NAME": None,
        "DATASET.DATAROOT": None, "LATENT_TRAINING.AE_CHECKPOINT": None,
        "PATTERN_POOL.POOL_DMG_DELAY": None,
    })
    assert cleared == config


@pytest.mark.parametrize("model", [{}, {"CHANNEL_OUT": None}])
def test_channel_overrides_recompute_automatic_width_across_round_trips(model):
    config = Config(DATASET={"EMOJIS": ["x"]}, MODEL=model)
    for width in (32, 64):
        config = Config.model_validate_json(config.model_dump_json())
        config = train_ca.apply_overrides(config, {"MODEL.CHANNEL_N": width})
        assert config.MODEL.CHANNEL_OUT == width
        assert config.MODEL.channel_out_is_auto


def test_channel_overrides_preserve_explicit_width_and_can_restore_auto():
    config = Config(DATASET={"EMOJIS": ["x"]}, MODEL={"CHANNEL_OUT": 16})
    updated = train_ca.apply_overrides(config, {"MODEL.CHANNEL_N": 32})
    assert updated.MODEL.CHANNEL_OUT == 16
    updated = train_ca.apply_overrides(updated, {"MODEL.CHANNEL_OUT": None})
    assert updated.MODEL.CHANNEL_OUT == 32
    updated = train_ca.apply_overrides(updated, {"MODEL.CHANNEL_N": 64, "MODEL.CHANNEL_OUT": 8})
    assert updated.MODEL.CHANNEL_OUT == 8


def test_list_element_overrides_are_validated_without_changing_original():
    config = Config(DATASET={"EMOJIS": ["x"]})
    updated = train_ca.apply_overrides(config, {"MODEL.PERCEPTIONS.0.KERNEL_SIZE": 5})
    assert updated.MODEL.PERCEPTIONS[0].KERNEL_SIZE == 5
    assert config.MODEL.PERCEPTIONS[0].KERNEL_SIZE == 3


@pytest.mark.parametrize("overrides", [
    {"TRAINING.LEARNIG_RATE": 0.1}, {"TRAINING.TRAINER_TYPE": "invalid"},
    {"MODEL.PERCEPTIONS.0.KERNEL_SIZE": 0}, {"MODEL.PERCEPTIONS.5.MODE": "conv"},
    {"TRAINING.BATCH_SIZE.value": 12},
])
def test_bad_overrides_raise_instead_of_being_silently_skipped(overrides):
    config = Config(DATASET={"EMOJIS": ["x"]})
    with pytest.raises((ValueError, ValidationError)):
        train_ca.apply_overrides(config, overrides)
