import io
import base64
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import Image, display
import random


def np2pil(a):
    """
    Convert a numpy array to a PIL Image.

    Args:
      a (numpy.ndarray): The input array.

    Returns:
      PIL.Image.Image: The converted PIL Image.
    """
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    """
    Save an image array to a file.

    Args:
      f (str or file-like object): The file path or file-like object to save the image to.
      a (numpy.ndarray): The image array.
      fmt (str, optional): The image format. If not specified, it will be inferred from the file extension.

    Returns:
      None
    """
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        with open(f, "wb") as file:
            np2pil(a).save(file, fmt, quality=95)


def imencode(a, fmt="jpeg"):
    """
    Encode an image array to a byte string.

    Args:
      a (numpy.ndarray): The image array.
      fmt (str, optional): The image format. Default is 'jpeg'.

    Returns:
      bytes: The encoded image as a byte string.
    """
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt="jpeg"):
    """
    Convert an image array to a data URL.

    Args:
      a (numpy.ndarray): The image array.
      fmt (str, optional): The image format. Default is 'jpeg'.

    Returns:
      str: The data URL of the image.
    """
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode("ascii")
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def imshow(a, fmt="jpeg"):
    """
    Display an image array in Jupyter Notebook.

    Args:
      a (numpy.ndarray): The image array.
      fmt (str, optional): The image format. Default is 'jpeg'.

    Returns:
      None
    """
    display(Image(data=imencode(a, fmt)))


def tile2d(a, w=None):
    """
    Tile a 2D array into a grid.

    Args:
      a (numpy.ndarray): The input array.
      w (int, optional): The width of the grid. If not specified, it will be calculated based on the length of the array.

    Returns:
      numpy.ndarray: The tiled array.
    """
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), "constant")
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a

def make_seed(size, channel_n, n=1):
    x = np.zeros([n, channel_n, size, size], np.float32)
    x[:, 3:, size // 2, size // 2] = 1.0
    return x

def zoom(img, scale=4):
    # handle batches if there
    if len(img.shape) == 4:
        # if batched, iterate over each image
        return torch.stack([zoom(img[i], scale) for i in range(img.shape[0])])
    # img expected to be in the shape [Channels, Height, Width]
    # Repeat img along the height and width dimensions
    # First, repeat along the height (dimension 1)
    img = img.repeat_interleave(scale, dim=1)
    # Then, repeat along the width (dimension 2)
    img = img.repeat_interleave(scale, dim=2)
    return img


def to_rgba(x):
    x_clone = x.clone()
    if len(x_clone.shape) == 3:
        x_clone = x_clone.unsqueeze_(0)
    return x_clone[:, :4, ...]


def to_grayscale(x):
    return x[:, :1, ...].clone()

def grayscale_to_rgb(x):
    if x.dim() == 3:
        # If the input tensor does not have a batch dimension, add one
        x = x.unsqueeze(0)
    
    # Repeat the grayscale channel 3 times along the channel dimension
    return x.repeat(1, 3, 1, 1)


def to_alpha(x):
    # Check if input has a batch dimension
    if x.dim() == 3:
        # Input is [Channels, Height, Width]
        return x[-1, :, :]
    elif x.dim() == 4:
        # Input is [Batch, Channels, Height, Width]
        return x[:, -1, :, :]
    else:
        raise ValueError("Input tensor must be 3D or 4D")

def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    Converts `x` to an RGB representation, dropping any alpha channel if present.
    
    Supported shapes:
      - [H, W]: single grayscale image (2D). -> [3, H, W]
      - [C, H, W]: single 3D image. C can be 1 (grayscale), 2 (gray+alpha),
                   3 (RGB), or 4 (RGBA).
      - [B, C, H, W]: batch of images (4D). Same channel logic as above.

    Returns:
      - Tensor in shape [3, H, W] for unbatched (2D/3D) input, or
      - [B, 3, H, W] for batched (4D) input.
    """
    dim = x.dim()

    # --------------------
    # Case 1: 2D (no batch, no channel)
    # Shape [H, W]
    # --------------------
    if dim == 2:
        # Interpret as single grayscale image: [H, W]
        # Expand to [1, H, W] then replicate to [3, H, W]
        return x.unsqueeze(0).repeat(3, 1, 1)

    # --------------------
    # Case 2: 3D (no batch)
    # Shape [C, H, W]
    # --------------------
    elif dim == 3:
        C, H, W = x.shape
        if C == 1:
            # Grayscale -> replicate to RGB
            return x.repeat(3, 1, 1)
        elif C == 2:
            # Grayscale + Alpha -> composite over white background
            gray = x[0:1, :, :]   # shape [1, H, W]
            alpha = x[1:2, :, :]  # shape [1, H, W]
            # Blend: out = alpha * gray + (1-alpha) * white
            white = 1.0  # in [0..1] range
            out_gray = alpha * gray + (1 - alpha) * white
            return out_gray.repeat(3, 1, 1)  # now [3, H, W]
        elif C == 3:
            # Already RGB
            return x
        elif C >= 4:
            # RGBA -> drop alpha => x[:3, :, :]
            return x[:3, :, :]
        else:
            raise ValueError(
                f"Unsupported 3D shape [C,H,W] with C={C}. "
                "Expected 1/2/3/4 channels."
            )

    # --------------------
    # Case 3: 4D (batch)
    # Shape [B, C, H, W]
    # --------------------
    elif dim == 4:
        B, C, H, W = x.shape
        if C == 1:
            # Batch of grayscale images
            # x shape: [B, 1, H, W] => replicate to [B, 3, H, W]
            return x.repeat(1, 3, 1, 1)
        elif C == 2:
            # Batch of grayscale+alpha -> composite over white background
            gray = x[:, 0:1, :, :]   # shape [B, 1, H, W]
            alpha = x[:, 1:2, :, :]  # shape [B, 1, H, W]
            # Blend: out = alpha * gray + (1-alpha) * white
            out_gray = alpha * gray
            return out_gray.repeat(1, 3, 1, 1)  # shape [B, 3, H, W]
        elif C == 3:
            # Batch of RGB images
            return x
        elif C >= 4:
            # RGBA -> Composite over white
            alpha = x[:, 3:4, :, :]
            rgb   = x[:, :3, :, :]
            # Blend: out = alpha * rgb + (1-alpha) * white
            white = 1.0  # in [0..1] range
            out_rgb = alpha * rgb + (1 - alpha) * white
            return out_rgb
        else:
            raise ValueError(
                f"Unsupported 4D shape [B,C,H,W] with C={C}. "
                "Expected 1/2/3/4 channels."
            )
    else:
        raise ValueError(
            f"Expected input to be 2D ([H,W]), 3D ([C,H,W]) or 4D ([B,C,H,W]). "
            f"Got shape {tuple(x.shape)}."
        )



def get_living_mask(x):
    alpha = x[:, 3:4, :, :]  # Select alpha channel
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

def prepare_condition_vector(class_contributions, condition_size, device="cpu"):
    """
    Prepares a condition vector with specified class contributions.

    Args:
        class_contributions (dict): A dictionary where keys are class indices and values are their contributions.
        condition_size (int): The total number of classes (size of the condition vector).
        device (str): The device to store the tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A condition vector with specified contributions.
    """
    # Initialize the condition vector with zeros
    condition_vector = torch.zeros(condition_size, device=device)

    # Set the specified contributions
    for class_index, contribution in class_contributions.items():
        if class_index < condition_size:
            condition_vector[class_index] = contribution

    # Optional: Normalize the vector to sum to 1
    # condition_vector /= condition_vector.sum()

    return condition_vector


def make_circle_damage_mask(n, size, dmg_scale=0.5):
    x = torch.linspace(-1.0, 1.0, size).unsqueeze(0).unsqueeze(1)
    y = torch.linspace(-1.0, 1.0, size).unsqueeze(0).unsqueeze(2)

    # Generate random centers and radii for each mask in the batch
    centers = (torch.rand(n, 2, 1, 1) * 2 - 1.0) / 2.0
    radii = torch.rand(n, 1, 1, 1) * dmg_scale

    # Initialize the batch of masks
    masks = torch.zeros(n, 1, size, size)

    for i in range(n):
        x_shifted = (x - centers[i, 0]) / radii[i]
        y_shifted = (y - centers[i, 1]) / radii[i]
        mask = (x_shifted * x_shifted + y_shifted * y_shifted < 1.0).float()
        masks[i] = 1 - mask

    return masks


def make_rectangular_damage_mask(n, size, orientation=None):
    """
    Generate a rectangular damage mask that randomly covers either the top/bottom half (horizontal)
    or the left/right half (vertical) of the image.
    
    Args:
        n (int): Number of masks in the batch.
        size (int): Size of the mask (height and width).
        orientation (str): 'horizontal' or 'vertical' to specify the orientation of the mask.
    
    Returns:
        torch.Tensor: A batch of damage masks with shape (n, 1, size, size).
    """
    masks = torch.zeros(n, 1, size, size)  # Initialize the batch of masks
    split = size // 2  # Midpoint for splitting the image

    if orientation is None:
        # Randomly choose between horizontal or vertical orientation
        orientation = random.choice(['horizontal', 'vertical'])

    for i in range(n):
        if orientation == 'horizontal':
            # Randomly choose between top half or bottom half
            if random.random() < 0.5:
                masks[i, :, :split, :] = 1  # Cover the top half
            else:
                masks[i, :, split:, :] = 1  # Cover the bottom half
        elif orientation == 'vertical':
            # Randomly choose between left half or right half
            if random.random() < 0.5:
                masks[i, :, :, :split] = 1  # Cover the left half
            else:
                masks[i, :, :, split:] = 1  # Cover the right half
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

    return masks

