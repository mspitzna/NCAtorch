# train.py
import os
import sys
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.training.trainer_factory import create_trainer
from nca.utils.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Cellular Automata Model")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file"
    )
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
        type=str, default=None,
        help="Path to the folder containing training data"
    )

    return parser.parse_args()


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
        config = config.model_copy(update={"FOLDER_NAME": args.folder})
    else:
        # If no folder is specified, load config from the provided path
        if not args.config or not os.path.exists(args.config):
            print("No valid config file provided. Exiting.")
            sys.exit(1)
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    
    # if device is set per args, update config
    if args.device:
        config = config.model_copy(update={"DEVICE": args.device})

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
            print(f"[Attempt {attempt+1}/{args.retries+1}] Training returned -1, retrying...")
            # sleep for 5 seconds before retrying
            time.sleep(5)
        else:
            print(f"[Attempt {attempt+1}] Training succeeded! Exiting script.")
            sys.exit(0)

    # If we exhaust all attempts and always got -1, exit with nonzero code
    print("Training failed after all retries. Exiting with code 1.")
    sys.exit(1)


if __name__ == "__main__":
    main()
