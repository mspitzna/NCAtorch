import torch
import torch.nn as nn

# =================================================================================
# Helper Modules for a Stronger Critic
# =================================================================================

class MiniBatchStdDev(nn.Module):
    """
    A layer that calculates and concatenates the standard deviation of features across a batch.
    
    This is a powerful technique from the ProGAN paper to prevent mode collapse. 
    It gives the critic a direct statistical cue about the diversity of the current batch,
    making it much easier to distinguish between varied real batches and less varied fake ones.
    
    Args:
        group_size (int): The number of samples to group for stddev calculation.
        num_new_features (int): The number of new feature channels to add.
    """
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        N, C, H, W = x.shape
        group_size = min(N, self.group_size)
        # Ensure group_size divides N to avoid view errors (e.g., batch_size not multiple of 4)
        if N % group_size != 0:
            group_size = N
        
        # Reshape to [G, M, C, H, W] where G=group_size, M=N/G
        # Note: C // self.num_new_features requires C to be divisible by num_new_features
        y = x.view(group_size, -1, self.num_new_features, C // self.num_new_features, H, W)
        
        # Calculate stddev over the group dimension 'G'
        y = torch.sqrt(y.var(0, unbiased=False) + 1e-8)
        
        # [M, F, C/F, H, W] -> [M, F, 1, 1]
        y = y.mean([2,3,4], keepdim=True).squeeze(2)
        
        # [M, F, 1, 1] -> [N, F, H, W]
        # This is the corrected line. It now repeats across H and W as well.
        y = y.repeat(group_size, 1, H, W)
        
        # Concatenate the new feature map to the input
        return torch.cat([x, y], dim=1)

# =================================================================================
# Advanced Critic Architecture
# =================================================================================

class CNNBlock(nn.Module):
    """Convolutional block, now with an option to disable normalization."""
    def __init__(self, in_channels, out_channels, stride, use_norm=True, norm_groups=32):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=not use_norm, padding_mode="reflect"
            )
        ]
        if use_norm:
            # GroupNorm is a good choice as it's independent of batch size
            layers.append(nn.GroupNorm(norm_groups, out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Critic(nn.Module):
    """
    A more powerful Critic for WGAN training.
    
    Key Improvements:
    1.  **MiniBatchStdDev:** Incorporates batch-wide statistics to fight mode collapse.
    2.  **Deeper by Default:** Uses a slightly deeper default architecture.
    3.  **Optional Normalization:** Allows you to easily disable normalization layers,
        which can sometimes interfere with the gradient penalty in WGAN-GP.
    """
    def __init__(self, image_channels=4, condition_channels=4, features=[64, 128, 256, 512], use_norm=True, norm_groups=32):
        super().__init__()
        
        in_channels = image_channels + condition_channels

        # Initial block without normalization
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        current_in_channels = features[0]
        
        # Middle blocks
        for feature in features[1:-1]:
            layers.append(
                CNNBlock(current_in_channels, feature, stride=2, use_norm=use_norm, norm_groups=norm_groups)
            )
            current_in_channels = feature
        
        # --- The key addition: MiniBatch Standard Deviation ---
        # It's added before the last main convolutional block.
        # The input channels for the next layer increase by 1.
        layers.append(MiniBatchStdDev())
        
        # Last main block
        layers.append(
            CNNBlock(current_in_channels + 1, features[-1], stride=1, use_norm=use_norm, norm_groups=norm_groups)
        )
        current_in_channels = features[-1]

        # Final output convolution. This layer should NOT have activation or normalization.
        layers.append(
            nn.Conv2d(current_in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x is expected to be the concatenated (image, condition) tensor
        x = self.initial(x)
        return self.model(x)
