import os
import random
from PIL import Image
from torchvision import transforms
from nca.data.datasets.base_dataset import NCADataset
from nca.utils.image_utils import make_circle_damage_mask


class CelebADataset(NCADataset):
    """
    A wrapper for the official torchvision CelebA dataset that provides a clean
    interface and handles automatic downloads.
    """
    def __init__(
        self,
        root_dir,  # e.g., './datasets/celeba'
        split='train',
        img_size=512,
        seed=42  # A fixed seed to ensure the split is always the same
    ):
        """
        Args:
            root_dir (str): The root directory of the CelebA dataset.
            split (str): The dataset split: 'train' or 'test'.
            img_size (int): The final output image size.
            seed (int): The seed for the random shuffle to ensure a
                        deterministic train/test split.
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.seed = seed
        
        self.image_dir = os.path.join(self.root_dir, 'img_align_celeba')
        
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")

        # --- File Discovery and Splitting Logic ---
        # 1. Get all image filenames from the directory.
        # We sort them to ensure the initial order is always the same
        # before shuffling. This is crucial for reproducibility.
        all_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        # 2. Seed the random number generator and shuffle the list.
        # Because the seed is fixed, the shuffle will be identical every time.
        random.seed(self.seed)
        #random.shuffle(all_filenames)
        
        # 3. Calculate the split point.
        split_idx = int(len(all_filenames) * 0.95)
        
        # 4. Assign filenames based on the requested split.
        if self.split == 'train':
            self.image_filenames = all_filenames[:split_idx]
        elif self.split == 'test':
            self.image_filenames = all_filenames[split_idx:]
        else:
            raise ValueError("Split must be 'train' or 'test'")

        # --- Define the transformation pipeline ---
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5), (0.5)),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Construct the full path to the image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])

        # Load image and apply the transformation
        image = Image.open(img_path).convert("RGB")
        target_image = self.transform(image)

        # apply dmg to the target image
        seed_image = target_image.clone()
        dmg_mask = make_circle_damage_mask(seed_image.shape[0] if len(seed_image.shape) == 4 else 1, seed_image.shape[-1], dmg_scale=0.5).squeeze(1)
        seed_image = (seed_image * dmg_mask) # Apply the damage mask to the seed image
        seed_image = seed_image + (1 - dmg_mask) * 0.5  # Fill the damaged area with gray

        # Return only the image tensor
        return seed_image, (1 - dmg_mask), target_image