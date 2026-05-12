import torch
import random
from torchvision import transforms, datasets
from nca.data.datasets.base_dataset import NCADataset
import torch.nn.functional as F

class GrowingMNISTDataset(NCADataset):
    """
    Creates a dataset for training an NCA to grow MNIST digits.
    Each sample provides a starting seed, a condition vector, and the target image.
    Includes methods for getting deterministic forward/backward samples.
    """
    def __init__(self, root, channel_n=18, train=True, target_padding=4, device="cpu", seed_size=1, goal_channels=False, enable_rotation=False, enable_zoom=False, latent_noise_channel=False):
        self.root = root if root is not None else "datasets/mnist"
        self.channel_n = channel_n
        self.target_padding = target_padding
        self.device = device
        self.img_size = 28 + self.target_padding * 2
        self.seed_size = seed_size
        self.goal_channels = goal_channels
        self.enable_rotation = enable_rotation
        self.enable_zoom = enable_zoom
        self.latent_noise_channel = latent_noise_channel

        self.target_dim = 2
        if self.enable_rotation:
            self.target_dim += 1
        if self.enable_zoom:
            self.target_dim += 1


        # Load the underlying MNIST dataset
        self.mnist_data = datasets.MNIST(
            root=self.root, train=train, download=True,
            transform=transforms.Compose([
                transforms.Pad(self.target_padding),
                transforms.ToTensor()
            ])
        )
        self.blur_transform = transforms.GaussianBlur(kernel_size=21, sigma=10.0)

        # --- PERFORMANCE FIX: Pre-compute indices for fast sampling ---
        print("Pre-computing digit indices for fast sampling...")
        self.digit_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist_data):
            self.digit_indices[label].append(idx)

        self.active_indices = []
        for digit in [5, 6, 9]:
            self.active_indices.extend(self.digit_indices[digit])
        
        print("Done.")

    def _apply_transforms(self, image_tensor, rotation_angle=0.0, zoom_factor=1.0):
        """Apply rotation and zoom transformations to the image tensor."""
        # image_tensor shape: (1, H, W)
        transformed = image_tensor.clone()
        
        if self.enable_rotation and rotation_angle != 0.0:
            # Apply rotation using transforms.functional.rotate
            transformed = transforms.functional.rotate(
                transformed,
                angle=rotation_angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0.0
            )
        
        if self.enable_zoom and zoom_factor != 1.0:
            # Apply zoom using transforms.functional.affine
            transformed = transforms.functional.affine(
                transformed,
                angle=0,
                translate=[0, 0],
                scale=zoom_factor,
                shear=[0, 0],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0.0
            )
            
        return transformed

    def _create_seed(self, digit_label, rotation_angle=0.0, zoom_factor=1.0):
        seed = torch.zeros(self.channel_n, self.img_size, self.img_size, device=self.device)
        h, w = self.img_size, self.img_size
        rect_size = int(self.seed_size)
        
        # Grayscale value based on class label (0->0.0, 1->0.1, ..., 9->0.9)
        grayscale_value = digit_label * 0.1
        
        rect_half = rect_size // 2
        y_start, y_end = h // 2 - rect_half, h // 2 + rect_half + 1
        x_start, x_end = w // 2 - rect_half, w // 2 + rect_half + 1
        
        # Channel 0: Grayscale (digit label encoding)
        seed[0, y_start:y_end, x_start:x_end] = grayscale_value
        
        # Channel 1: Alpha channel
        seed[1, y_start:y_end, x_start:x_end] = 1.0
        
        channel_idx = 2
        
        # Channel 2: Rotation encoding (0.0 = 0°, 1.0 = 360°)
        if self.enable_rotation:
            seed[channel_idx, y_start:y_end, x_start:x_end] = rotation_angle / 360.0
            channel_idx += 1
            
        # Next channel: Zoom encoding (0.5 = 50% scale, 1.0 = 100% scale, etc.)
        if self.enable_zoom:
            seed[channel_idx, y_start:y_end, x_start:x_end] = zoom_factor / 2.0
            channel_idx += 1
        
        # Remaining channels: Random noise
        seed[channel_idx:, :, :] = torch.randn_like(seed[channel_idx:, :, :]) * 0.05
        return seed

    def __len__(self):
        #return len(self.mnist_data)
        return len(self.active_indices)

    def _create_sample(self, idx, direction='random', rotation_angle=None, zoom_factor=None):
        """
        Internal helper method to create a sample.
        Args:
            idx (int): The index of the MNIST digit to use.
            direction (str): 'random', 'forward', or 'backward'.
            rotation_angle (float, optional): Specific rotation angle. If None, uses random.
            zoom_factor (float, optional): Specific zoom factor. If None, uses random.
        """
        image_tensor, digit_label = self.mnist_data[idx]
        image_tensor = image_tensor.to(self.device)
        
        # Generate transformation parameters (random if not specified)
        if rotation_angle is None:
            rotation_angle = random.choice([0, 90, 180, 270]) if self.enable_rotation else 0.0
        
        if zoom_factor is None:
            zoom_factor = random.uniform(0.5, 2.0) if self.enable_zoom else 1.0
        
        # Apply transformations to the original image
        transformed_image = self._apply_transforms(image_tensor, rotation_angle, zoom_factor)
        
        target = torch.zeros(self.target_dim, self.img_size, self.img_size, device=self.device)
        digit_mask = (transformed_image > 0.1).float().squeeze(0)
        target[0] = transformed_image
        target[1] = digit_mask
        
        channel_idx = 2
        
        # Rotation channel
        if self.enable_rotation:
            target[channel_idx] = torch.zeros(self.img_size, self.img_size, device=self.device)
            channel_idx += 1
            
        # Zoom channel  
        if self.enable_zoom:
            target[channel_idx] = torch.zeros(self.img_size, self.img_size, device=self.device)
            channel_idx += 1

        digit_encoded_seed = self._create_seed(digit_label, rotation_angle, zoom_factor)

        condition_vector = self.blur_transform(target.clone()) if self.goal_channels else torch.tensor(0.0, device=self.device)
        return digit_encoded_seed, condition_vector, target

    def _convert_states(self, initial_state: torch.Tensor, final_state: torch.Tensor, is_forward_pass: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_forward_pass:
           direction_channel = torch.ones(1, final_state.shape[1], final_state.shape[2], device=self.device)
        else:
            target_dim = 2
            if self.enable_rotation:
                target_dim += 1
            if self.enable_zoom:
                target_dim += 1

            original_seed = initial_state.clone()  # This is the encoded seed
            hidden_state = torch.randn_like(initial_state[target_dim:, :, :]) * 0.05
            initial_state = torch.cat((final_state[:target_dim, :, :], hidden_state), dim=0)
            final_state = original_seed[:target_dim, :, :]
            direction_channel = -torch.ones(1, final_state.shape[1], final_state.shape[2], device=self.device)
        return initial_state, direction_channel, final_state

    def __getitem__(self, idx):
        """
        Gets a sample for training.
        """
        mnist_idx = self.active_indices[idx] # TODO
        return self._create_sample(mnist_idx, direction='random' if self.latent_noise_channel is False else 'forward')

    def get_forward_sample(self, digit):
        """
        Gets a random sample for a specific digit, forced to be a forward pass.
        (seed -> digit)
        """
        if not self.digit_indices[digit]:
            raise ValueError(f"Digit {digit} not found in dataset")
        random_idx = random.choice(self.digit_indices[digit])
        return self._create_sample(random_idx, direction='forward')

    def get_backward_sample(self, digit):
        """
        Gets a random sample for a specific digit, forced to be a backward pass.
        (digit -> seed)
        """
        if not self.digit_indices[digit]:
            raise ValueError(f"Digit {digit} not found in dataset")
        random_idx = random.choice(self.digit_indices[digit])
        return self._create_sample(random_idx, direction='backward')

    def get_sample_with_transforms(self, digit, rotation_angle=0.0, zoom_factor=1.0, direction='forward'):
        """
        Gets a sample with specific transformation parameters.
        Args:
            digit (int): The digit class (0-9)
            rotation_angle (float): Rotation angle in degrees
            zoom_factor (float): Zoom factor (1.0 = no zoom)
            direction (str): 'forward' or 'backward'
        """
        if not self.digit_indices[digit]:
            raise ValueError(f"Digit {digit} not found in dataset")
        random_idx = random.choice(self.digit_indices[digit])
        return self._create_sample(random_idx, direction, rotation_angle, zoom_factor)