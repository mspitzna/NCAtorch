import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
from nca.utils.image_utils import to_rgb, zoom, to_grayscale
from nca.data.datasets import CustomMNISTDataset, CIFAR10Dataset
from abc import ABC, abstractmethod
import math

class BaseVideoGenerator(ABC):
    """Base abstract class for video generation."""
    
    def __init__(self, dataset_name: str, device="cpu", zoom_magnitude=4):
        self.device = device
        self.zoom_magnitude = zoom_magnitude
        self.dataset_name = dataset_name
        
        if dataset_name == "mnist":
            self.dataset = CustomMNISTDataset(root="datasets", channel_n=3, device=device)
        elif dataset_name == "cifar10":
            self.dataset = CIFAR10Dataset(root="datasets", channel_n=3, device=device, target_size=32)
        else:
            self.dataset = None
    
    @abstractmethod
    def commit_frame(self, x):
        """Process and store a frame."""
        pass
    
    @abstractmethod
    def save_video(self, file_name="output.mp4", fps=30):
        """Save the stored frames as a video."""
        pass
    
    def _process_rgb(self, x):
        """Process tensor to extract RGB representation based on dataset."""
        if self.dataset_name == "mnist":
            return self.dataset.generate_colored_image(to_grayscale(x), x[:, -10:])
        elif self.dataset_name == "cifar10":
            return self.dataset.apply_per_pixel_coloring(x, x[:, -10:])
        elif self.dataset_name == "ot":
            return x[:, :3]
        elif self.dataset_name == "e2h":
            return x[:, :3]
        else:
            return to_rgb(x[:, :3])


class VideoGenerator(BaseVideoGenerator):
    """Video generator for batch visualization."""
    
    def __init__(self, dataset_name: str, device="cpu", zoom_magnitude=4):
        super().__init__(dataset_name, device, zoom_magnitude)
        self.frames = []  # List to store processed frames

    def _process_tensor(self, x, columns=6):
        """Helper function to process a tensor into a frame."""
        # Process RGB channels based on dataset
        rgb_vis = self._process_rgb(x)

        # If the original tensor has an alpha channel, use it for blending on a white background.
        if x.shape[1] >= 4:
            alpha = x[:, 3:4]
            vis = rgb_vis * alpha + (1.0 - alpha)
        else:
            vis = to_rgb(rgb_vis)

        vis = vis.detach().cpu()
        # Upsample for better visualization
        vis = zoom(vis, self.zoom_magnitude)
        # Create a grid of images
        grid = vutils.make_grid(vis, nrow=columns, padding=2, normalize=False)
        grid = grid.permute(1, 2, 0)  # Reorder to (H, W, C)
        grid = np.clip(grid.numpy(), 0, 1)  # Ensure values are within [0, 1]
        # Convert to PIL Image
        img = Image.fromarray((grid * 255).astype(np.uint8))
        return img

    def commit_frame(self, x):
        """
        Commit a tensor as a frame. The tensor will be processed and added to the frames list.
        Args:
            x (torch.Tensor): The tensor to commit as a frame.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        processed_frame = self._process_tensor(x)
        self.frames.append(processed_frame)

    def save_video(self, file_name="output.mp4", fps=30):
        """
        Save all committed frames as a video.
        Args:
            file_name (str): The name of the output video file.
            fps (int): Frames per second for the video.
        """
        if not self.frames:
            raise ValueError("No frames to save. Commit frames first.")
        
        with imageio.get_writer(file_name, fps=fps) as writer:
            for frame in self.frames:
                writer.append_data(np.array(frame))
        print(f"Video saved as {file_name}")
        self.frames = []  # Clear the frames list after saving


class VideoGeneratorPerSample(BaseVideoGenerator):
    """Video generator for per-sample visualization with feature channels."""
    
    def __init__(self, dataset_name: str, device="cpu", zoom_magnitude=4, feature_channels=None, visualization_mode="grayscale", grid_rows=10):
        """
        Initialize the video generator for per-sample visualization.
        
        Args:
            dataset_name (str): Name of the dataset
            device (str): Device to use for computation
            zoom_magnitude (int): Magnification factor for visualization
            feature_channels (tuple, optional): Tuple specifying the range of feature channels to visualize (start, end).
                                               If None, all extra channels will be visualized.
            visualization_mode (str): Mode for visualizing feature channels - "grayscale" or "heatmap"
        """
        super().__init__(dataset_name, device, zoom_magnitude)
        self.feature_channels = feature_channels
        self.visualization_mode = visualization_mode
        # List to store frames where each frame contains all samples in the batch
        self.frames = []
        # Track the batch size to ensure consistency
        self.batch_size = None
        self.grid_rows = grid_rows

    def _process_extra_channel(self, x):
        """
        Process a single extra channel (tensor shape [1,1,H,W]) into an image.
        
        Returns a grayscale or heatmap image based on visualization_mode.
        """
        # Remove batch and channel dims.
        tensor = x[0, 0]  # shape: [H, W]
        # Upsample using your zoom function.
        tensor = zoom(tensor.unsqueeze(0).unsqueeze(0), self.zoom_magnitude)
        arr = tensor.squeeze().detach().cpu().numpy()  # shape: [newH, newW]
        arr = np.clip(arr, 0, 1)
        
        if self.visualization_mode == "heatmap":
            # Create a heatmap (blue to red)
            cmap = plt.cm.get_cmap('jet')
            colored_arr = cmap(arr)
            colored_arr = (colored_arr[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(colored_arr, mode="RGB")
        else:  # grayscale
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        
        return img

    def _process_sample_frame(self, sample_tensor):
        """
        Given a tensor of shape [C, H, W] for one sample, create a composite PIL image
        that concatenates the RGB visualization and the selected feature channels.
        """
        # Process main RGB image.
        if self.dataset_name == "mnist":
            rgb_out = self.dataset.generate_colored_image(
                to_grayscale(sample_tensor.unsqueeze(0)),
                sample_tensor[-10:].unsqueeze(0)
            )
        elif self.dataset_name == "cifar10":
            rgb_out = self.dataset.apply_per_pixel_coloring(
                sample_tensor.unsqueeze(0),
                sample_tensor[-10:].unsqueeze(0)
            )
        else:
            if sample_tensor.shape[0] >= 3:
                rgb_tensor = sample_tensor[:4].unsqueeze(0)
            else:
                rgb_tensor = sample_tensor.repeat(3, 1, 1).unsqueeze(0)
            rgb_out = to_rgb(rgb_tensor)

        # Ensure rgb_out is a PIL image.
        if isinstance(rgb_out, torch.Tensor):
            rgb_vis = zoom(rgb_out, self.zoom_magnitude)
            rgb_vis = vutils.make_grid(rgb_vis, nrow=1, padding=2, normalize=False)
            rgb_vis = rgb_vis.permute(1, 2, 0)
            rgb_vis = np.clip(rgb_vis.cpu().numpy(), 0, 1)
            rgb = Image.fromarray((rgb_vis * 255).astype(np.uint8))
        else:
            rgb = rgb_out

        extra = sample_tensor[3:] if sample_tensor.shape[0] > 3 else None

        extra_imgs = []
        if extra is not None:
            # If feature_channels is specified, use only those channels in the range
            if self.feature_channels is not None:
                start, end = self.feature_channels
                start, end = start - 3, end - 3
                # Ensure end is within the valid range
                end = min(end, extra.shape[0])
                channel_indices = range(start, end)
            else:
                channel_indices = range(extra.shape[0])
                
            for i in channel_indices:
                # Process each selected extra channel.
                ch_tensor = extra[i:i+1].unsqueeze(0)  # shape: [1,1,H,W]
                extra_img = self._process_extra_channel(ch_tensor)
                # Convert to RGB if it's grayscale
                if self.visualization_mode == "grayscale":
                    extra_img = extra_img.convert("RGB")
                extra_imgs.append(extra_img)

        # Concatenate the RGB image and extra channel images in a grid layout.
        imgs = [rgb] + extra_imgs
        
        # Default to 1 row (horizontal layout) if not specified
        rows = self.grid_rows
        
        # Define the black line width
        line_width = 6
        
        if rows == 1:
            # Original horizontal layout with black lines between images
            total_width = sum(im.width for im in imgs) + line_width * (len(imgs) - 1)
            max_height = max(im.height for im in imgs)
            composite = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
            x_offset = 0
            for i, im in enumerate(imgs):
                composite.paste(im, (x_offset, 0))
                x_offset += im.width
                
                # Add black line after each image except the last one
                if i < len(imgs) - 1:
                    for x in range(x_offset, x_offset + line_width):
                        for y in range(max_height):
                            composite.putpixel((x, y), (0, 0, 0))
                    x_offset += line_width
        else:
            # Grid layout with specified number of rows and black lines between images
            num_imgs = len(imgs)
            cols = math.ceil(num_imgs / rows)
            
            # Find maximum dimensions for grid cells
            max_width = max(im.width for im in imgs)
            max_height = max(im.height for im in imgs)
            
            # Calculate total dimensions including black lines
            total_width = max_width * cols + line_width * (cols - 1)
            total_height = max_height * rows + line_width * (rows - 1)
            
            # Create a new image with grid layout
            composite = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            
            for i, im in enumerate(imgs):
                row = i // cols
                col = i % cols
                
                # Calculate offsets including space for black lines
                x_offset = col * (max_width + line_width)
                y_offset = row * (max_height + line_width)
                
                composite.paste(im, (x_offset, y_offset))
                
                # Draw horizontal black line after each row except the last
                if row < rows - 1 and i + cols < len(imgs):
                    for y in range(y_offset + max_height, y_offset + max_height + line_width):
                        for x in range(x_offset, x_offset + max_width):
                            composite.putpixel((x, y), (0, 0, 0))
                
                # Draw vertical black line after each column except the last
                if col < cols - 1 and i + 1 < len(imgs):
                    for x in range(x_offset + max_width, x_offset + max_width + line_width):
                        for y in range(y_offset, y_offset + max_height):
                            composite.putpixel((x, y), (0, 0, 0))
        return composite

    def commit_frame(self, x):
        """
        Process a batch tensor x (shape [B, C, H, W]) and create a single frame
        with all samples arranged vertically.
        """
        batch_size = x.size(0)
        
        # Set batch size on first commit or verify it's consistent
        if self.batch_size is None:
            self.batch_size = batch_size
        elif self.batch_size != batch_size:
            raise ValueError(f"Inconsistent batch size: expected {self.batch_size}, got {batch_size}")
        
        # Process each sample in the batch
        sample_images = []
        for i in range(batch_size):
            sample_tensor = x[i]  # shape: [C, H, W]
            composite = self._process_sample_frame(sample_tensor)
            sample_images.append(composite)
        
        # Combine all sample images vertically into a single frame
        total_height = sum(img.height for img in sample_images)
        max_width = max(img.width for img in sample_images)
        
        # Create a new image with all samples stacked vertically
        combined_frame = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
        y_offset = 0
        for img in sample_images:
            combined_frame.paste(img, (0, y_offset))
            y_offset += img.height
        
        # Add the combined frame to our frames list
        self.frames.append(combined_frame)

    def save_video(self, file_name="sample_video", fps=30):
        """
        Save a single video with all samples from each batch arranged vertically.
        """
        if not self.frames:
            raise ValueError("No frames have been committed.")
        
        filename = f"{file_name}.mp4"
        with imageio.get_writer(filename, fps=fps) as writer:
            for frame in self.frames:
                writer.append_data(np.array(frame))
        
        print(f"Video saved as {filename}")
        self.frames = []
        self.batch_size = None

    def save_frame(self, index=0, file_name="frame.png"):
        """Save the frame at the given index as a PIL image."""
        if not self.frames:
            raise ValueError("No frames have been committed.")
        return self.frames[index].save(file_name)
    
    def get_frame(self, index=0):
        """Get the frame at the given index as a PIL image."""
        if not self.frames:
            raise ValueError("No frames have been committed.")
        return self.frames[index]
