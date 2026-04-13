import numpy as np
import torch
from torch.utils.data import Dataset

from nca.utils.image_utils import to_rgb


class NCADataset(Dataset):
    """Base class for all NCA datasets.

    Subclasses must implement ``__getitem__`` and ``__len__``.
    Override ``state_to_img`` and/or ``batch_to_rgb`` when the default
    RGB extraction (first ``color_dim`` channels) is not appropriate.
    """

    def batch_to_rgb(self, x0, x, target, cond=None):
        """
        Convert a batch of raw NCA tensors to RGB for logging.
        Returns (x0_rgb, x_rgb, target_rgb) — each (B, 3, H, W).
        Default: identity pass-through. Override per dataset.
        """
        return x0, x, target

    def state_to_img(self, x_tensor: torch.Tensor, color_dim: int) -> np.ndarray:
        """Convert a CA state tensor to a uint8 RGB numpy image for the UI.

        Args:
            x_tensor: State tensor ``[1, C, H, W]`` (single sample, on CPU).
            color_dim: Number of visible colour channels stored in the state.

        Returns:
            ``np.ndarray`` of shape ``(H, W, 3)`` with dtype ``uint8``.

        Override this method in dataset subclasses that need custom
        visualisation (e.g. per-pixel label colouring for MNIST).
        """
        img = to_rgb(x_tensor[:, :color_dim].clone())
        return (
            img.numpy().clip(0, 1).squeeze(0).transpose(1, 2, 0) * 255
        ).astype(np.uint8)
