import torch
from torchvision import transforms, datasets
from nca.data.datasets.base_dataset import NCADataset
from nca.utils.image_utils import to_rgba, to_rgb

def compute_saliency(image_tensor):
    # Placeholder: convert to grayscale and threshold as a simple saliency approximation.
    gray = image_tensor.mean(dim=0)  # shape: [H, W]
    threshold = gray.mean()
    mask = (gray > threshold).float()
    return mask


class CIFAR10Dataset(NCADataset):
    def __init__(self, root, channel_n, target_size, train=True, device="cpu"):
        self.cifar10_data = datasets.CIFAR10(root=root, train=train, download=True)
        self.channel_n = channel_n
        self.img_size = target_size  # e.g., 32 for CIFAR10
        self.device = device
        self.color_lookup = (
            torch.tensor(
                [
                    [128, 0, 0],  # airplane
                    [0, 0, 128],  # automobile
                    [0, 128, 0],  # bird
                    [128, 165, 0],  # cat
                    [128, 0, 128],  # deer
                    [128, 20, 147],  # dog
                    [69, 69, 0],  # frog
                    [128, 69, 0],  # horse
                    [75, 0, 130],  # ship
                    [0, 128, 127],  # truck
                ],
                dtype=torch.float32,
                device=device,
            ) / 255.0
        ).to(device)

    def __len__(self):
        return len(self.cifar10_data)

    def __getitem__(self, idx):
        image_pil, target_label = self.cifar10_data[idx]
        image_pil = transforms.Resize(self.img_size)(image_pil)
        image_tensor = transforms.ToTensor()(image_pil).float()  # [3, H, W]
    
        # Prepare input tensor.
        seed_tensor = torch.zeros(self.channel_n, self.img_size, self.img_size)
        seed_tensor[:3] = image_tensor

        # apply random noise to seed
        seed_tensor[3:] = torch.randn_like(seed_tensor[3:]) * 0.05
    
        # Create a 10-channel target mask, where the channel corresponding to the target is all ones.
        target_mask = torch.zeros(10, self.img_size, self.img_size)
        target_mask[target_label] = torch.ones(self.img_size, self.img_size)
    
        # Return input, dummy condition, and target mask.
        return seed_tensor, torch.tensor(0), target_mask
    
    def batch_to_rgb(self, x0, x, target, cond=None):
        """
        Convert a batch of raw NCA tensors to RGB for logging.
        Returns (x0_rgb, x_rgb, target_rgb) — each (B, 3, H, W).
        """
        x0_rgb = self.apply_per_pixel_coloring(to_rgba(x0), target)
        x_rgb = self.apply_per_pixel_coloring(to_rgba(x), x[:, -10:])
        target_rgb = to_rgb(x0[:, :3])  # Just show the original image for the target, without coloring
        return x0_rgb, x_rgb, target_rgb

    def apply_per_pixel_coloring(self, image_tensor, prediction_tensor):
        """
        Applies color to each pixel of the CIFAR-10 image based on the prediction tensor.

        Args:
            image_tensor (Tensor): The original CIFAR-10 image tensor of shape [3, 32, 32].
            prediction_tensor (Tensor): The model's prediction per pixel of shape [10, 32, 32].

        Returns:
            Tensor: The colored image of shape [3, 32, 32].
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if prediction_tensor.dim() == 3:
            prediction_tensor = prediction_tensor.unsqueeze(0)

        batch_size = image_tensor.size(0)
        colored_images = []

        for i in range(batch_size):
            img = image_tensor[i]  # Shape: [3, 32, 32]
            label = prediction_tensor[i]  # Shape: [10, 32, 32]

            # Get label indices by taking argmax over label channels
            label_indices = torch.argmax(label, dim=0)  # Shape: [32, 32]

            # Check shape of color_lookup
            assert (
                self.color_lookup.dim() == 2
            ), f"Expected color_lookup to have 2 dimensions (num_classes, 3), but got {self.color_lookup.dim()}"

            # Map indices to RGB colors using color lookup table
            color_overlay = self.color_lookup[label_indices]  # Shape: [32, 32, 3]

            # Check if the color_overlay has 3 dimensions
            if color_overlay.dim() != 3:
                raise ValueError(
                    f"color_overlay should have shape [32, 32, 3], but got {color_overlay.shape}"
                )

            color_overlay = color_overlay.permute(2, 0, 1)  # Shape: [3, 32, 32]

            # Blend the CIFAR-10 image with the color overlay using alpha blending
            alpha = 0.4
            blended_image = (1 - alpha) * img[:3] + alpha * color_overlay

            # Append to the list
            colored_images.append(blended_image)

        # Stack all images to return as a batch
        return torch.stack(colored_images)