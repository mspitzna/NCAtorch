"""Helpers shared by the evaluation scripts."""

from pathlib import Path

import torch

from nca.core.models.model_factory import create_model
from nca.utils.config import load_config

from ..evaluation import load_checkpoint


def add_checkpoint_args(parser, default_steps):
    parser.add_argument("--checkpoint", required=True, help="CA checkpoint, e.g. train_log/<run>/ca_final.pt.")
    parser.add_argument(
        "--config", default=None,
        help="Run config; defaults to config.yaml next to the checkpoint. "
             "Use an extension config with matching PERCEPTIONS for checkpoints trained elsewhere.",
    )
    parser.add_argument("--n-steps", type=int, default=default_steps, help="Inference rollout length.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Defaults to the config's DEVICE.")


def load_run(args, cond_dim, size=None):
    """Return ``(config, model)`` for ``--checkpoint``/``--config``/``--device``.

    ``size`` is the square grid size; ``None`` uses ``DATASET.TARGET_SIZE``.
    """
    checkpoint = Path(args.checkpoint)
    config = load_config(str(args.config or checkpoint.parent / "config.yaml"))
    if args.device is not None:
        config = config.model_copy(update={"DEVICE": args.device})
    size = size or config.DATASET.TARGET_SIZE
    model = create_model(config, cond_dim, size, size)
    return config, load_checkpoint(model, checkpoint, config.DEVICE)


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
