import numpy as np
import torch
from torch.utils.data import Dataset

from nca.utils.image_utils import to_rgb


class NCADataset(Dataset):
    """Base class for all NCA datasets.

    Subclasses must implement ``__getitem__`` and ``__len__``.
    Override ``_colorize`` to customise how a single CA state is turned into
    RGB.  Override ``batch_to_rgb`` only when x0, x, and target require
    *different* coloring logic from each other (e.g. CIFAR-10).
    """

    def _colorize(self, x: torch.Tensor, x0=None, target=None, cond=None) -> torch.Tensor:
        """Colorize a single CA state.

        Contract:
            Input  ``x``: ``(1, C, H, W)`` float32.
            Returns:      ``(1, 3, H, W)`` float32, values in ``[0, 1]``.

        ``x0``, ``target``, and ``cond`` are optional context tensors
        (each ``(1, *, H, W)``) available for datasets whose coloring depends
        on information outside the current state.
        """
        return to_rgb(x[:, :min(4, x.shape[1])])

    def batch_to_rgb(self, x0, x, target, cond=None):
        """Convert a batch of raw NCA tensors to RGB for logging.

        Contract:
            Returns ``(x0_rgb, x_rgb, target_rgb)`` — each
            ``(B, 3, H, W)`` float32, values in ``[0, 1]``.
        """
        def colorize_batch(t):
            return torch.cat([
                self._colorize(
                    t[i:i+1],
                    x0[i:i+1],
                    target[i:i+1],
                    cond[i:i+1] if cond is not None else None,
                )
                for i in range(t.shape[0])
            ])
        return colorize_batch(x0), colorize_batch(x), colorize_batch(target)

    def state_to_img(self, x: torch.Tensor, x0=None, target=None, cond=None) -> np.ndarray:
        """Convert a single CA state tensor to a uint8 RGB image for the UI.

        Contract:
            Input  ``x``: ``(1, C, H, W)`` float32, on CPU.
            Returns:      ``(H, W, 3)`` uint8, values in ``[0, 255]``.
        """
        rgb = self._colorize(x, x0, target, cond)
        return (rgb.cpu().numpy().clip(0, 1).squeeze(0).transpose(1, 2, 0) * 255).astype(np.uint8)
