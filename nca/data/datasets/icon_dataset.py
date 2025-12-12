import io

import numpy as np
import PIL.Image
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class IconDataset(Dataset):
    def __init__(self, emojis, target_size, target_padding, channel_n, condition_size=None, total_samples=None, device="cpu", rectangle=False):
        """
        Args:
            emojis (list): List of emoji characters to use as targets.
            target_size (int): Size of the target images.
            target_padding (int): Padding around the target images.
            channel_n (int): Number of channels in the CA state.
            condition_size (int): The size of the one-hot condition vector. If None, uses len(emojis).
            total_samples (int): Total number of samples in the dataset.
            device (str): Device to store the tensors ('cpu' or 'cuda').
            rectangle (bool): Whether to use rectangular seed (default: False for single pixel).
        """
        self.emojis = emojis
        self.target_size = target_size
        self.target_padding = target_padding
        self.channel_n = channel_n
        self.condition_size = condition_size if condition_size is not None else len(emojis)
        self.total_samples = total_samples
        self.device = device
        self.rectangle = rectangle
        
        # Prepare targets
        self.targets = [self._prepare_target(emoji) for emoji in emojis]
        self.seed = self._create_seed()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Generate one sample of data
        condition_vector = torch.zeros(self.condition_size, device=self.device)
        target, condition_idx = self.get_random_target()
        condition_vector[condition_idx] = 1  # One-hot encoding of the index

        # send to device
        condition_vector = condition_vector.to(self.device)
        target = target.to(self.device)

        # Return condition_vector and the corresponding target
        return self.seed.clone(), condition_vector, target

    def get_random_target(self):
        idx = np.random.randint(0, len(self.targets))
        return self.targets[idx], idx

    def get_condition_tensor(self, idx):
        condition_vector = torch.zeros(self.condition_size, device=self.device)
        condition_vector[idx] = 1
        return condition_vector
    
    def _prepare_target(self, emoji):
        """Prepare a single emoji target."""
        target_img = load_emoji(emoji, self.target_size)
        target_img = torch.tensor(target_img).permute(2, 0, 1)
        # Pad the image
        pad_target = F.pad(target_img, (self.target_padding, self.target_padding, self.target_padding, self.target_padding), "constant", 0)
        return pad_target
    
    def _create_seed(self):
        """Create the seed tensor based on the first emoji's dimensions."""
        # Get dimensions from first target
        first_target = self.targets[0]
        h, w = first_target.shape[1:]
        
        # Create the seed tensor
        seed = torch.zeros((self.channel_n, h, w), dtype=torch.float32)
        
        # Apply random noise to the higher channels
        seed[4:, :, :] = torch.randn_like(seed[4:, :, :]) * 0.05
        
        if self.rectangle:
            # Create a rectangular seed
            rect_size = int(min(h, w) * 0.1)  # 10% of the smaller dimension
            rect_half = rect_size // 2
            y_start, y_end = h // 2 - rect_half, h // 2 + rect_half + 1
            x_start, x_end = w // 2 - rect_half, w // 2 + rect_half + 1
            
            # Set the rectangular region to 0.5
            seed[:4, y_start:y_end, x_start:x_end] = 0.5
        else:
            # Set the center pixel for a single point seed
            seed[:4, h // 2, w // 2] = 1.0
        
        return seed
    

def load_image(url, max_size=None):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download emoji from {url}") from e

    try:
        img = PIL.Image.open(io.BytesIO(r.content)).convert("RGBA")
    except PIL.Image.UnidentifiedImageError as e:
        content_type = r.headers.get("content-type", "unknown")
        raise RuntimeError(
            f"Downloaded emoji from {url} is not a valid image (content-type: {content_type})"
        ) from e

    if max_size is not None:
        img = img.resize((max_size, max_size), PIL.Image.LANCZOS)

    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def make_seed(size, channel_n):
    x = np.zeros([channel_n, size, size], np.float32)
    x[:, :, size // 2, size // 2] = 1.0
    return x


def load_emoji(emoji, target_size=128):
    codepoints = "_".join(hex(ord(char))[2:] for char in emoji)
    url = (
        "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/512/emoji_u%s.png"
        % codepoints
    )
    return load_image(url, target_size)
