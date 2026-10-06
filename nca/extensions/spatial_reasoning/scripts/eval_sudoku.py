"""Tab. 1: Sudoku ASR/ACP per difficulty bucket on the held-out test split.

ASR is the share of puzzles whose completion is a valid Sudoku, ACP the
share of non-clue cells without a row/column/box conflict. The exact-match
rates against the dataset solution are printed alongside.

    python -m nca.extensions.spatial_reasoning.scripts.eval_sudoku --checkpoint train_log/<run>/ca_final.pt
"""

import argparse

from ..data.factories import dataroot
from ..data.sudoku import GRID_SIZE, SudokuDataset
from ..evaluation import evaluate_sudoku
from .common import add_checkpoint_args, load_run, print_table, seed_everything

# Removed-cell fractions for Easy [1, 27], Medium [28, 54] and Hard [55, 81] removed cells.
BUCKETS = {
    "Easy": [0.01, 0.333],
    "Medium": [0.35, 0.666],
    "Hard": [0.68, 1.0],
    "Full": [0.01, 1.0],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_args(parser, default_steps=1500)
    parser.add_argument("--n-puzzles", type=int, default=1000)
    parser.add_argument("--buckets", nargs="+", default=list(BUCKETS), choices=list(BUCKETS))
    parser.add_argument("--seed", type=int, default=None, help="Test-split seed; default SEED + 10000 of the run.")
    args = parser.parse_args()

    config, model = load_run(args, cond_dim=1, size=GRID_SIZE)
    seed = args.seed if args.seed is not None else (config.SEED + 10_000 if config.SEED != -1 else 99_999)
    rows = []
    for name in args.buckets:
        seed_everything(seed)
        dataset = SudokuDataset(
            config.MODEL.CHANNEL_N, BUCKETS[name], args.n_puzzles, train=False, seed=seed, dataroot=dataroot(config),
        )
        m = evaluate_sudoku(model, dataset, args.n_puzzles, args.n_steps, args.batch_size, config.DEVICE)
        rows.append((name, f"{m['valid']:.2f}", f"{m['acp_valid']:.2f}", f"{m['solved']:.2f}", f"{m['acp']:.2f}"))
    print_table(
        f"{args.checkpoint} ({args.n_puzzles} puzzles, {args.n_steps} steps, seed {seed})",
        ("Bucket", "ASR", "ACP", "ASR(exact)", "ACP(exact)"),
        rows,
    )


if __name__ == "__main__":
    main()
