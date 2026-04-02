from PIL import Image
import os
import random
import numpy as np
import torch
from torchvision import transforms
from nca.data.datasets.base_dataset import NCADataset


def get_params(size, resize_size, crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    params,
    resize_size,
    crop_size,
    method=Image.BICUBIC,
    flip=True,
    crop=True,
    totensor=True,
):
    transform_list = []
    transform_list.append(
        transforms.Lambda(lambda img: __scale(img, crop_size, method))
    )

    if flip:
        transform_list.append(
            transforms.Lambda(lambda img: __flip(img, params["flip"]))
        )
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


def is_image_file(filename):
    IMG_EXTENSIONS = [
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
        ".ppm",
        ".PPM",
        ".bmp",
        ".BMP",
        ".tif",
        ".TIF",
        ".tiff",
        ".TIFF",
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(target_width, target_width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)


def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class EdgesDataset(NCADataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(
        self,
        dataroot,
        train=True,
        img_size=256,
        channel_n=16,
        random_crop=False,
        random_flip=False,
        invertible=False,
        device="cpu",
    ):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, "train")  # get the image directory
            self.train_paths = make_dataset(self.train_dir)  # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:

            self.test_dir = os.path.join(dataroot, "val")  # get the image directory

            self.AB_paths = make_dataset(self.test_dir)  # get image paths

        self.crop_size = img_size
        self.resize_size = img_size
        self.img_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train
        self.channel_n = channel_n
        self.invertible = invertible
        self.device = device

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params = get_params(A.size, self.resize_size, self.crop_size)

        transform_image = get_transform(
            params,
            self.resize_size,
            self.crop_size,
            crop=self.random_crop,
            flip=self.random_flip,
        )

        # Compute alpha channels: 0 for white pixels (all RGB values are max), 1 otherwise
        A_np = np.array(A.resize((self.resize_size, self.resize_size)))
        B_np = np.array(B.resize((self.resize_size, self.resize_size)))

        A_alpha = torch.from_numpy((~(A_np > 220).all(axis=2)).astype(np.float32))
        B_alpha = torch.from_numpy((~(B_np > 220).all(axis=2)).astype(np.float32))

        # Move alpha channels to shape (1, H, W)
        A_alpha = A_alpha.unsqueeze(0)
        B_alpha = B_alpha.unsqueeze(0)

        # Transform images and move to tensors
        A = transform_image(A)
        B = transform_image(B)

        # Add the alpha channel to A and B
        A = torch.cat((A[:3], A_alpha), dim=0)
        B = torch.cat((B[:3], B_alpha), dim=0)

        # Apply alpha channel to RGB channels in-place
        A[:3] *= A_alpha
        B[:3] *= B_alpha

        # Extend A to the desired number of channels if needed
        if self.channel_n > 4:
            extended_A = torch.zeros(
                (self.channel_n, self.img_size, self.img_size),
                dtype=A.dtype,
                device=A.device,
            )
            extended_A[4:] = torch.randn_like(extended_A[4:]) * 0.05
            extended_A[:4] = A
        else:
            extended_A = A

        alpha_cond = A_alpha.clone()


        # If invertible randomly swap A and B
        if self.invertible:

            # extend target B to match the channel size
            if self.channel_n > 4:
                extended_B = torch.cat(
                    (
                        B[:3],
                        B_alpha,
                        torch.randn_like(extended_A[4:, :, :]) * 0.05
                    ),
                    dim=0,
                )
            if random.random() > 0.65:
                extended_A, extended_B = extended_B, extended_A
                invertible_cond = torch.tensor([0.0, 1.0], device=self.device)  # Backward condition
            else:
                invertible_cond = torch.tensor([1.0, 0.0], device=self.device)  # Forward condition

            return extended_A.to(self.device), invertible_cond.to(self.device), extended_B.to(self.device)

        return extended_A.to(self.device), alpha_cond.to(self.device), B.to(self.device)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)