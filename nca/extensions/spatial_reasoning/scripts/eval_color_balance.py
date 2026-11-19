"""Tab. 4: global color balance — imbalance and exact-ratio accuracy.

Imbalance is the mean absolute difference between the white-pixel count and
the prescribed count; accuracy the share of outputs that match it exactly.
Seeds are the deterministic test samples ``0..n-1``.

    python -m nca.extensions.spatial_reasoning.scripts.eval_color_balance --checkpoint train_log/<run>/ca_final.pt
"""

import argparse

import torch

from ..config import settings
from ..data.color_balance import ColorBalanceDataset
from ..evaluation import rollout, summarize
from ..losses import color_balance_metrics
from .common import add_checkpoint_args, load_run, print_table, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_args(parser, default_steps=200)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the stochastic cell updates.")
    args = parser.parse_args()

    config, model = load_run(args, cond_dim=0)
    balance = settings(config).COLOR_BALANCE
    dataset = ColorBalanceDataset(
        config.MODEL.CHANNEL_N, size=config.DATASET.TARGET_SIZE, train=False,
        bias_min=balance.BIAS_MIN, bias_max=balance.BIAS_MAX, white_ratio=balance.WHITE_RATIO,
    )

    seed_everything(args.seed)
    results = {}
    for start in range(0, args.n_samples, args.batch_size):
        seeds = torch.stack([dataset[i][0] for i in range(start, min(start + args.batch_size, args.n_samples))])
        state = rollout(model, seeds.to(config.DEVICE), None, args.n_steps)
        for key, value in color_balance_metrics(state, balance.WHITE_RATIO).items():
            results.setdefault(key, []).append(value.cpu())
    m = summarize(results)
    print_table(
        f"{args.checkpoint} ({args.n_samples} samples, {args.n_steps} steps)",
        ("Imbalance", "Accuracy"),
        [(f"{m['imbalance']:.2f}", f"{m['exact']:.2f}")],
    )


if __name__ == "__main__":
    main()
