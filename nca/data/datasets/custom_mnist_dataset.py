import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset


class CustomMNISTDataset(Dataset):
    def __init__(self, root, channel_n, train=True, target_padding=0, device="cpu"):
        # Load MNIST data
        self.mnist_data = datasets.MNIST(
            root=root, train=train, download=True
        )
        self.channel_n = channel_n
        self.target_padding = target_padding
        self.img_size = int(28 + target_padding * 2)

        # Define color lookup table
        self.color_lookup = (
            torch.tensor(
                [
                    [128, 0, 0],
                    [230, 25, 75],
                    [70, 240, 240],
                    [210, 245, 60],
                    [250, 190, 190],
                    [170, 110, 40],
                    [170, 255, 195],
                    [165, 163, 159],
                    [0, 128, 128],
                    [128, 128, 0],
                    [255, 255, 255],  # This is the default for digits.
                ],
                dtype=torch.float32,
                device=device,
            )
            / 255.0
        )
        self.device = device

    def __len__(self):
        return len(self.mnist_data)

    def create_per_pixel_label(self, image_tensor, digit_label):
        """
        Creates a per-pixel label tensor where each pixel belonging to the digit
        has a one-hot encoding corresponding to the digit label.

        Args:
            image_tensor (Tensor): The grayscale image tensor of shape [1, 28, 28].
            digit_label (int): The digit label (0-9).

        Returns:
            Tensor: Per-pixel label tensor of shape [10, 28, 28].
        """
        per_pixel_label = torch.zeros(10, self.img_size, self.img_size)
        digit_mask = image_tensor[0] >= 0.1
        per_pixel_label[digit_label, digit_mask] = 1.0
        return per_pixel_label

    def generate_colored_image(self, image_tensor, per_pixel_label):
        """
        Generates a colored image by mapping per-pixel labels to RGB colors.

        Args:
            image_tensor (Tensor): Grayscale image tensor of shape [B, 1, 28, 28] or [1, 28, 28].
            per_pixel_label (Tensor): Per-pixel label tensor of shape [B, 10, 28, 28] or [10, 28, 28].

        Returns:
            Tensor: Colored image tensor of shape [B, 3, 28, 28] or [3, 28, 28].
        """
        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if per_pixel_label.dim() == 3:
            per_pixel_label = per_pixel_label.unsqueeze(0)

        batch_size = image_tensor.size(0)
        colored_images = []

        for i in range(batch_size):
            img = image_tensor[i]  # Shape: [C, 28, 28] where C = channel_n
            label = per_pixel_label[i]  # Shape: [10, 28, 28]

            # Create digit mask (identify where the digit exists)
            is_digit = (img[0] > 0.1).long()  # Threshold > 0.1 for "digit" pixels
            is_background = 1 - is_digit  # Pixels where there is no digit

            # Get label indices by taking argmax over label channels
            label = label.permute(1, 2, 0)  # Shape: [28, 28, 10]
            _, label_indices = label.max(dim=-1)  # Shape: [28, 28], digit indices

            # Assign background index (10) where not digit
            background_index = 10  # Index for white background
            label_indices = label_indices * is_digit + background_index * is_background
            label_indices = label_indices.long()  # Ensure indices are integers

            # Map indices to RGB colors using color lookup table
            rgb_image = self.color_lookup[label_indices.cpu()]  # Shape: [28, 28, 3]

            # Permute to [3, 28, 28] for consistency
            colored_images.append(rgb_image.permute(2, 0, 1))  # Shape: [3, 28, 28]

        # Stack all images to return as a batch
        return torch.stack(colored_images)  # Shape: [B, 3, 28, 28]

    def __getitem__(self, idx):
        image_pil, digit_label = self.mnist_data[idx]

        # apply padding to the image
        image_pil = transforms.Pad(self.target_padding)(image_pil)

        # Convert image to tensor and normalize to [0, 1]
        image_tensor = transforms.ToTensor()(image_pil).float()  # Shape: [1, 28, 28]

        # Create per-pixel label tensor (Seed)
        per_pixel_label = self.create_per_pixel_label(
            image_tensor, digit_label
        )  # Shape: [10, 28, 28]

        # Initialize a tensor with zeros (background) and assign the digit mask to channel 0 (alpha)
        image_with_alpha = torch.zeros(self.channel_n, self.img_size, self.img_size)
        image_with_alpha[0] = (
            image_tensor[0] > 0.1
        ).float()  # Alpha channel: digit mask
        image_with_alpha[1:] = image_tensor  # Copy grayscale image to other channels

        # print(image_tensor)
        # Return Seed, Empty Condition, Target Image
        return image_with_alpha, torch.tensor(0), per_pixel_label
