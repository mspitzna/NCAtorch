import torch
from torchvision import transforms
from torchvision.datasets import MovingMNIST
from nca.data.datasets.base_dataset import NCADataset
from nca.utils.image_utils import to_rgb


class MovingMNISTDataset(NCADataset):
    """
    Wraps torchvision's MovingMNIST dataset so that each index returns a
    sub-sequence of length `history_n` (possibly reversed),
    plus the immediately following target frame, plus an 'empty_condition' tensor.

    Each underlying sample from MovingMNIST is typically shaped [20, 64, 64]
    or [20, 1, 64, 64]. We'll produce seeds of shape [history_n, 1, 64, 64]
    (or [channel_n, 1, 64, 64] if channel_n > history_n) and a single target frame [1, 64, 64].
    """

    def __init__(
        self,
        root: str,
        download: bool = False,
        history_n: int = 8,
        channel_n: int = 32,
        reverse_seed: bool = False,
        img_size: int = 64,
        train: bool = True,
    ):
        """
        Args:
            root (str): Root directory of dataset where MovingMNIST will be (or is) stored.
            train (bool): Whether to load the train or test split.
            transform (callable, optional): Optional transform to apply to each sub-sequence.
            download (bool): If True, downloads the dataset from the internet and
                             puts it in root directory.
            history_n (int): Number of frames used as seed input.
            channel_n (int): Number of channels to return. If channel_n > history_n,
                             zero-channels will be appended after the seed frames.
            reverse_seed (bool): If True, reverse the order of the history window.
        """
        super().__init__()
        self.history_n = history_n
        self.channel_n = channel_n
        self.reverse_seed = reverse_seed
        self.img_size = img_size

        # Actual dataset from torchvision
        self.dataset = MovingMNIST(
            root=root,
            download=download,
            #split="train" if train else "test",
        )

        # Each item from self.dataset[i] is typically shape (20, 64, 64).
        # But let's store them in memory (or we could re-fetch from the dataset in __getitem__).
        self.samples = []
        for i in range(len(self.dataset))[:200]:  # TODO Limit due to sample pool size
            # The original dataset returns a tuple (video, label),
            # but label is always None or dummy for MovingMNIST. We'll ignore it.
            video = self.dataset[i]  # shape: [20, 64, 64] or [20, 1, 64, 64]

            # If still 2D per frame, unsqueeze to get [20, 1, 64, 64]
            if video.ndim == 3:  # [20, 64, 64]
                video = video.unsqueeze(1)  # becomes [20, 1, 64, 64]

            # Convert to float Tensor right away
            video = video.float()
            self.samples.append(video)

        # Build an index map so we can iterate through sub-sequences
        # For each 20-frame video, the last valid start is `20 - history_n - 1`
        # if we need an immediate next frame.
        self.index_map = []
        for vid_idx, vid_data in enumerate(self.samples):
            # We have 20 frames: indices 0..19
            # We want to create sub-sequences: [0..history_n-1] -> target=history_n
            # then [1..history_n], target=history_n+1, and so on.
            for start in range(0, 20 - self.history_n):
                # The next frame is at `start + history_n`, so the last valid start is 19 - history_n.
                if (start + self.history_n) < 20:
                    self.index_map.append((vid_idx, start))

        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: x / 255.0
                ),  # manually scale your float Tensor to [0..1]
            ]
        )

    def _colorize(self, x, x0=None, target=None, cond=None):
        return to_rgb(x[:, :1])

    def __len__(self):
        return len(self.index_map)

    def get_alpha_mask(self, frame):
        """
        Compute the alpha mask for the frame where pixels are black
        """
        mask = torch.zeros(frame.shape[1], frame.shape[2])
        mask[frame[0] == 0] = 1
        return mask

    def __getitem__(self, idx):
        """
        Returns:
            seed_frames: shape [history_n, 1, 64, 64] (or [channel_n, 1, 64, 64] if channel_n>history_n)
            idx_condition: shape [1] (just a tensor of zeros)
            target_frame: shape [1, 64, 64]
        """
        vid_idx, start_idx = self.index_map[idx]
        video = self.samples[vid_idx]  # shape: [20, 1, 64, 64]

        seed_frames = video[
            start_idx : start_idx + self.history_n
        ]  # [history_n, 1, 64, 64]
        target_frame = video[
            start_idx + 1 : start_idx + self.history_n + 1
        ]  # [1, 64, 64]

        # Optionally reverse the seed frames
        if self.reverse_seed is False:
            seed_frames = torch.flip(seed_frames, dims=[0])  # [::-1] equivalent
            target_frame = torch.flip(target_frame, dims=[0])

        # Create empty_condition as in your original code
        idx_condition = torch.tensor((vid_idx, start_idx + self.history_n))

        # create living mask for the target frame and seed frames where pixels are black
        target_frame = target_frame.squeeze(1)
        seed_frames = seed_frames.squeeze(1)

        # If channel_n > history_n, fill the difference with zeros
        if self.channel_n > self.history_n:
            extra = torch.zeros(
                (self.channel_n - self.history_n, 64, 64), dtype=seed_frames.dtype
            )
            seed_frames = torch.cat([seed_frames, extra], dim=0)
        # else it remains shape [history_n, 1, 64, 64]

        # Apply user-defined transforms if provided
        if self.transform is not None:
            seed_frames = self.transform(seed_frames)
            target_frame = self.transform(target_frame)

        return seed_frames, idx_condition, target_frame