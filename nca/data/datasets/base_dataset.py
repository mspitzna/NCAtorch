from torch.utils.data import Dataset


class NCADataset(Dataset):
    def batch_to_rgb(self, x0, x, target, cond=None):
        """
        Convert a batch of raw NCA tensors to RGB for logging.
        Returns (x0_rgb, x_rgb, target_rgb) — each (B, 3, H, W).
        Default: identity pass-through. Override per dataset.
        """
        return x0, x, target
