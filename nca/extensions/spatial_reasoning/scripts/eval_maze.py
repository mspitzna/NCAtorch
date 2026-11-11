"""Tab. 3 and 9: maze Valid Path / Valid Optimal.

Evaluates the 1K-maze test split of the training benchmark and, without
finetuning, freshly generated Prim mazes (``--prims-sizes``; even sizes are
rounded up to the next odd size).

    python -m nca.extensions.spatial_reasoning.scripts.eval_maze --checkpoint train_log/<run>/ca_final.pt
"""

import argparse

import torch

from ..data.factories import dataroot
from ..data.maze import MazeDataset, generate_prims_mazes
from ..config import settings
from ..evaluation import evaluate_maze
from .common import add_checkpoint_args, load_run, print_table, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_args(parser, default_steps=100)
    parser.add_argument("--prims-sizes", type=int, nargs="*", default=[20, 30], help="Prim maze sizes; empty to skip.")
    parser.add_argument("--n-prims", type=int, default=1000, help="Prim mazes per size.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the Prim generator and rollout noise.")
    args = parser.parse_args()

    # The model is fully convolutional; the grid size only shapes positional embeddings.
    config, model = load_run(args, cond_dim=2, size=30)
    if config.MODEL.USE_POSITIONAL_EMBEDDINGS and args.prims_sizes:
        parser.error("Prim mazes of other sizes require a model without positional embeddings.")

    seed_everything(args.seed)
    hf_dataset_id = settings(config).MAZE.HF_DATASET_ID
    test = MazeDataset(config.MODEL.CHANNEL_N, train=False, hf_dataset_id=hf_dataset_id, dataroot=dataroot(config))
    tensors = tuple(torch.stack(parts) for parts in zip(*(test[i] for i in range(len(test)))))
    m = evaluate_maze(model, *tensors, args.n_steps, args.batch_size, config.DEVICE)
    rows = [(f"{test.grid_size}x{test.grid_size}", hf_dataset_id, len(test),
             f"{m['valid_path']:.2f}", f"{m['valid_optimal']:.2f}")]

    for size in args.prims_sizes:
        seed_everything(args.seed)
        tensors = generate_prims_mazes(size, args.n_prims, config.MODEL.CHANNEL_N, seed=args.seed)
        m = evaluate_maze(model, *tensors, args.n_steps, args.batch_size, config.DEVICE)
        side = tensors[0].shape[-1]
        rows.append((f"{side}x{side}", "Prim", args.n_prims, f"{m['valid_path']:.2f}", f"{m['valid_optimal']:.2f}"))

    print_table(
        f"{args.checkpoint} ({args.n_steps} steps)",
        ("Size", "Generator", "Mazes", "Valid Path", "Valid Optimal"),
        rows,
    )


if __name__ == "__main__":
    main()
