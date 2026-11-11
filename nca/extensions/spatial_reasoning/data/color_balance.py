"""Global color balance: rebalance a biased black/white image to a fixed ratio."""

import numpy as np
import torch

from nca.data.datasets.base_dataset import NCADataset


class ColorBalanceDataset(NCADataset):
    """Unconditioned binary images in ``[-1, 1]`` with a random white fraction.

    The seed's white fraction is drawn from ``[bias_min, bias_max]``; hidden
    channels get small noise. The target is a random arrangement with the
    prescribed white ratio and is only used for logging, since the loss has no
    per-pixel supervision. Training uses the framework's NumPy seed (including
    DataLoader worker seeds); test samples are deterministic in their index.
    """

    def __init__(self, channel_n, size=32, train=True, bias_min=0.1, bias_max=0.9, white_ratio=0.5, total_samples=None):
        self.channel_n = channel_n
        self.size = size
        self.train = train
        self.bias_min = bias_min
        self.bias_max = bias_max
        self.white_ratio = white_ratio
        self.total_samples = total_samples or (1_000_000 if train else 10_000)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        rng = np.random if self.train else np.random.default_rng(idx)
        n = self.size * self.size

        target = -np.ones(n, dtype=np.float32)
        target[: round(n * self.white_ratio)] = 1.0
        rng.shuffle(target)

        white_fraction = rng.uniform(self.bias_min, self.bias_max)
        white = rng.random(n) < white_fraction
        seed = torch.zeros(self.channel_n, self.size, self.size)
        seed[0] = torch.from_numpy(np.where(white, 1.0, -1.0).astype(np.float32).reshape(self.size, self.size))
        seed[1:] = torch.from_numpy(
            rng.normal(0.0, 0.05, (self.channel_n - 1, self.size, self.size)).astype(np.float32)
        )
        return seed, torch.zeros(0), torch.from_numpy(target.reshape(1, self.size, self.size))

    def batch_to_rgb(self, x0, x, target, cond=None):
        def gray(t):
            return ((t[:, :1].detach().cpu().clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)
        return gray(x0), gray(x), gray(target)
