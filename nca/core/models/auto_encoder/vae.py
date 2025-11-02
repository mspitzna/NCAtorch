import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Keep for potential future use, though not strictly needed in this version

# --- Helper Blocks ---

class ResBlock(nn.Module):
    """Simple Residual Block with GroupNorm and optional downsampling."""
    def __init__(self, in_channels, out_channels, norm_groups=32, activation_fn=nn.LeakyReLU(0.2, inplace=True), downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # Bias false if using Norm
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.act1 = activation_fn
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Need projection shortcut if changing dims or downsampling
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        # Note: Activation applied AFTER residual connection in forward

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity # Add shortcut
        out = self.act1(out) # Apply final activation for the block
        return out

class UpsampleConvBlock(nn.Module):
    """Upsamples (Nearest) then applies Conv + Norm + Activation."""
    def __init__(self, in_channels, out_channels, norm_groups=32, activation_fn=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Use bias=False because GroupNorm has affine=True by default
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.act = activation_fn

    def forward(self, x):
        x_up = self.upsample(x)
        out = self.conv(x_up)
        out = self.norm(out)
        out = self.act(out)
        return out

# --- Main VAE Class ---

class VAE(nn.Module): # Renamed to avoid clash if keeping old one
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 latent_channels=128, # Increased default latent channels
                 base_channels=64,   # Base channel count
                 num_downsamples=5,  # Number of downsampling stages (256 -> 128 -> 64 -> 32 -> 16 -> 8)
                 norm_groups=32,     # Number of groups for GroupNorm
                 activation_fn=nn.LeakyReLU(0.2, inplace=True),
                 final_activation=nn.Sigmoid() # Sigmoid for [0,1], Tanh for [-1,1], None otherwise
                ):
        """
        VAE with ResNet-style blocks, GroupNorm, and Upsample+Conv. Suitable for 256x256.
        Latent space spatial size will be input_size / (2**num_downsamples).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_downsamples = num_downsamples
        self.norm_groups = norm_groups
        self.activation_fn = activation_fn
        self.final_activation = final_activation

        # --- Build Encoder ---
        encoder_blocks = []
        # Initial convolution
        encoder_blocks.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1))
        # encoder_blocks.append(activation_fn) # Optional activation after first conv

        ch = base_channels
        for i in range(num_downsamples):
            out_ch = ch * 2
            encoder_blocks.append(ResBlock(ch, out_ch, norm_groups, activation_fn, downsample=True))
            ch = out_ch
        # At this point, ch = base_channels * (2**num_downsamples)
        self.encoder = nn.Sequential(*encoder_blocks)

        final_encoder_channels = ch

        # --- Latent Distribution Layers ---
        # Use 1x1 conv here as spatial dimension is already small and ResBlocks handled features
        self.fc_mu = nn.Conv2d(final_encoder_channels, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(final_encoder_channels, latent_channels, kernel_size=1)

        # --- Build Decoder ---
        decoder_blocks = []
        # Initial layer: project latent_channels back to the channel depth of the last encoder stage
        decoder_blocks.append(nn.Conv2d(latent_channels, final_encoder_channels, kernel_size=3, stride=1, padding=1))
        decoder_blocks.append(activation_fn) # Activation after initial projection

        ch = final_encoder_channels
        for i in range(num_downsamples):
            out_ch = ch // 2
            # Use Upsample + Conv block
            decoder_blocks.append(UpsampleConvBlock(ch, out_ch, norm_groups, activation_fn))
            # Or could use ResUpBlock if defined and preferred
            # decoder_blocks.append(ResUpBlock(ch, out_ch, norm_groups, activation_fn))
            ch = out_ch
        # At this point, ch = base_channels

        # Final convolution to match out_channels
        decoder_blocks.append(nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if self.final_activation is not None:
            decoder_blocks.append(final_activation)

        self.decoder = nn.Sequential(*decoder_blocks)


    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample from N(0, I)
        return mu + eps * std

    def forward(self, x):
        # Encode
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstructed_x = self.decoder(z)

        # If using Tanh activation, scale from [-1, 1] to [0, 1]
        if self.final_activation is not None and isinstance(self.final_activation, nn.Tanh):
            reconstructed_x = (reconstructed_x + 1.0) / 2.0

        return reconstructed_x, mu, logvar, z # Return z for CA

    def encode(self, x):
        """Encodes input to latent distribution parameters."""
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def sample_latent(self, x):
        """Encodes input and samples from the latent distribution."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        """Decodes latent sample z back to image space."""
        reconstructed_x = self.decoder(z)
        # If using Tanh activation, scale from [-1, 1] to [0, 1]
        if self.final_activation is not None and isinstance(self.final_activation, nn.Tanh):
            reconstructed_x = (reconstructed_x + 1.0) / 2.0
        return reconstructed_x

    def get_input_channels(self):
        return self.in_channels