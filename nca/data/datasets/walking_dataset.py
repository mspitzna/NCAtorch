import os
import numpy as np
import torch
import cv2
from nca.data.datasets.base_dataset import NCADataset


class WalkingDataset(NCADataset):
    def __init__(
        self,
        root_dir="/home/mspitzna/NNCellA/growing_ca/datasets/walking",
        persons=range(1, 10),  # person01 to person10
        d_versions=["d1"],
        target_size=(64, 64),
        channel_n=32,
        history_n=8,
        transform=None,
        cutoff=10,
        difference_threshold=100.0,
    ):  # Adjust as needed
        self.root_dir = root_dir
        self.persons = persons
        self.d_versions = d_versions
        self.transform = transform
        self.target_size = target_size
        self.channel_n = channel_n
        self.history_n = history_n
        self.cutoff = cutoff
        self.difference_threshold = difference_threshold

        self.video_frames = []
        self.index_map = []

        for p in self.persons:
            person_str = f"person{p:02d}"
            for d in self.d_versions:
                video_name = f"{person_str}_walking_{d}_uncomp.avi"
                video_path = os.path.join(self.root_dir, video_name)
                if not os.path.exists(video_path):
                    continue

                frames = self._load_video_frames(video_path)
                # Apply cutoff
                frames = frames[self.cutoff : -self.cutoff]
                frames = frames[:100]

                if len(frames) <= self.history_n:
                    continue

                video_idx = len(self.video_frames)
                self.video_frames.append(frames)

                num_samples = len(frames) - self.history_n
                for start_idx in range(num_samples):
                    # Check difference between first and last history frame
                    first_frame = frames[start_idx]
                    last_frame = frames[start_idx + self.history_n - 1]

                    # Compute mean absolute difference
                    diff = np.mean(np.abs(first_frame - last_frame))
                    if diff < self.difference_threshold:
                        # Skip this sample
                        continue

                    self.index_map.append((video_idx, start_idx))

    def _load_video_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
            success, frame = cap.read()
        cap.release()
        return frames

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        video_idx, start_idx = self.index_map[idx]
        frames = self.video_frames[video_idx]

        seed_frames = frames[start_idx : start_idx + self.history_n]
        # invert the order of seed_frames
        seed_frames = seed_frames[::-1]
        target_frame = frames[start_idx + self.history_n]

        seed_frames = np.stack(seed_frames, axis=0)  # shape: (history_n, H, W)

        # If channel_n > history_n, add zero channels
        if self.channel_n > self.history_n:
            extra_channels = np.zeros(
                (self.channel_n - self.history_n, *seed_frames.shape[1:]),
                dtype=seed_frames.dtype,
            )
            seed_frames = np.concatenate([seed_frames, extra_channels], axis=0)

        seed_frames = torch.from_numpy(seed_frames).float()
        target_frame = torch.from_numpy(np.expand_dims(target_frame, 0)).float()
        empty_condition = torch.zeros(1)

        if self.transform:
            seed_frames = self.transform(seed_frames)
            target_frame = self.transform(target_frame)

        return seed_frames, empty_condition, target_frame