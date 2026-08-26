# train.py
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.training.trainer_factory import create_trainer
from nca.utils.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Cellular Automata Model")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry training if `trainer.train()` returns -1.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path to the folder containing training data",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable wandb sweep mode: init wandb early and apply wandb.config overrides.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Override config values with dot-path syntax, e.g. TRAINING.LEARNING_RATE=0.001",
    )

    return parser.parse_args()


def _set_by_path(cfg_dict, key_path, value):
    """Set nested value in a dict using dot-delimited keys."""
    keys = key_path.split(".")
    d = cfg_dict
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _flatten_dict(d, parent_key=""):
    """Flatten nested dict into dot-delimited keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def apply_overrides(config, overrides: dict):
    """Return a new Config with overrides applied (validated by pydantic)."""
    if not overrides:
        return config
    # Drop None fields to avoid re-validating optional paths as Path(None)
    cfg_dict = config.model_dump(exclude_none=True)
    # Only allow overrides that exist in the current config structure to avoid pydantic errors
    existing_paths = set(_flatten_dict(cfg_dict).keys())
    filtered = {}
    for k, v in overrides.items():
        if k in existing_paths:
            filtered[k] = v
        else:
            print(f"[override] Skipping unknown key '{k}' (not in config).")
    overrides = filtered
    if not overrides:
        return config
    for key, value in overrides.items():
        _set_by_path(cfg_dict, key, value)
    return config.__class__(**cfg_dict)


def parse_override_strings(override_items):
    """Parse CLI override strings into a dict."""
    parsed = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_val = item.split("=", 1)
        try:
            val = yaml.safe_load(raw_val)
        except Exception:
            val = raw_val
        parsed[key] = val
    return parsed


def setup_seed(seed):
    """Set random seed for reproducibility."""
    if seed != -1:
        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Setting random seed to {seed}")


def main():
    # Parse arguments and load config
    args = parse_args()

    if args.folder:
        # If folder is specified, set it in the config
        if not os.path.exists(f"{args.folder}/config.yaml"):
            print(f"Config file not found in {args.folder}. Exiting.")
            sys.exit(1)
        # Load config from the specified folder
        print(f"Loading config from folder: {args.folder}")
        config = load_config(f"{args.folder}/config.yaml")
        config = config.model_copy(
            update={
                "LOGGING": config.LOGGING.model_copy(
                    update={"FOLDER_NAME": args.folder}
                )
            }
        )
    else:
        # If no folder is specified, load config from the provided path
        if not args.config or not os.path.exists(args.config):
            print("No valid config file provided. Exiting.")
            sys.exit(1)
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)

    # Apply CLI overrides (before device / wandb overrides)
    cli_overrides = parse_override_strings(args.override)
    config = apply_overrides(config, cli_overrides)

    # if device is set per args, update config
    if args.device:
        config = config.model_copy(update={"DEVICE": args.device})

    # If running a wandb sweep, init wandb early and apply sweep overrides
    use_sweep = args.sweep or os.environ.get("WANDB_SWEEP") == "1"
    if use_sweep:
        import wandb

        # Ensure wandb logging is on for sweeps
        if not config.LOGGING.WANDB:
            config = config.model_copy(
                update={"LOGGING": config.LOGGING.model_copy(update={"WANDB": True})}
            )
        base_cfg = config.model_dump(exclude_none=True)
        wandb.init(
            project=config.LOGGING.PROJECT_NAME,
            name=config.LOGGING.TRAIN_NAME,
            config=base_cfg,
        )
        # Convert wandb config to plain dict, drop private keys, then flatten for dot-path overrides
        sweep_cfg = wandb.config.as_dict()
        sweep_cfg = {k: v for k, v in sweep_cfg.items() if not str(k).startswith("_")}
        sweep_overrides = _flatten_dict(sweep_cfg)
        config = apply_overrides(config, sweep_overrides)

    # set seed, needs to happen before dataloader and model creation!
    setup_seed(config.SEED)

    # Prepare dataset
    dataloader, cond_dim, im_height, im_width = create_dataset(config)

    # Attempt training up to (retries + 1) times
    for attempt in range(args.retries + 1):
        # Initialize CA model
        ca_model = create_model(config, cond_dim, im_height, im_width)

        # Update config with discovered shapes
        print(f"Cond dim: {cond_dim}, Im height: {im_height}, Im width: {im_width}")
        config.set_cond_dim(cond_dim)
        config.set_im_height(im_height)
        config.set_im_width(im_width)

        # Create the trainer
        trainer = create_trainer(config, ca_model, dataloader, str(args.config))
        result = trainer.train()
        if result == -1:
            print(
                f"[Attempt {attempt + 1}/{args.retries + 1}] Training returned -1, retrying..."
            )
            # sleep for 5 seconds before retrying
            time.sleep(5)
        else:
            print(f"[Attempt {attempt + 1}] Training succeeded! Exiting script.")
            sys.exit(0)

    # If we exhaust all attempts and always got -1, exit with nonzero code
    print("Training failed after all retries. Exiting with code 1.")
    sys.exit(1)


if __name__ == "__main__":
    main()
