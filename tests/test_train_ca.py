from unittest.mock import Mock

import pytest
import yaml

from scripts import train_ca


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
