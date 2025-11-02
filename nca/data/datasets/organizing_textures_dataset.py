from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class OrganizingTexturesDataset(Dataset):
    def __init__(
        self,
        sample_path,
        channel_n,
        img_size,
        condition_size,
        total_samples,
        device="cpu",
    ):
        """
        Args:
            seed (Tensor): The seed tensor to be used for the dataset.
            condition_size (int): The size of the one-hot condition vector.
            total_samples (int): Total number of samples in the dataset.
            device (str): Device to store the tensors ('cpu' or 'cuda').
        """
        self.sample_path = sample_path
        self.condition_size = condition_size
        self.total_samples = total_samples
        self.device = device
        self.ot_transform = transforms.Compose(
            [
                # Resize while keeping aspect ratio
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                # Center crop to the desired size
                transforms.CenterCrop(img_size),
                # rotate image, flip image, etc.
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        self.channel_n = channel_n
        seed = torch.zeros((channel_n, img_size, img_size), dtype=torch.float32)

        # Set a specific position to 1.0 (similar to what was done in TensorFlow)
        # Adjusting for 'channels first' in typical PyTorch but keeping 'channels last' as your input suggests
        seed[:] = 0.5
        self.seed = seed

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Generate one sample of data
        condition_vector = torch.zeros(self.condition_size, device=self.device)
        target = self.get_random_target()

        # send to device
        condition_vector = condition_vector.to(self.device)
        target = target.to(self.device)

        # Return condition_vector and the corresponding target
        return self.seed.clone(), condition_vector, target

    def get_random_target(self):
        target = Image.open(self.sample_path).convert("RGB")
        target = self.ot_transform(target)
        return target