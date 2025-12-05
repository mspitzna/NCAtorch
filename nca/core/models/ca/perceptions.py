from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca.core.models.ca.deformable_conv import DeformableConv2d


class Perception(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_out_channel(self):
        pass

class SobelPerception(nn.Module):
    def __init__(self, in_channel=16, device="cpu") -> None:
        super().__init__()
        self.in_channel = in_channel

        # Identity filter
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)

        # Sobel filters (dx, dy)
        sobel_x = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            / 8.0
        )
        sobel_y = sobel_x.T  # just transpose

        # Laplacian filter
        laplacian = (
            torch.tensor([[1, 2, 1], [2, -6, 2], [1, 2, 1]], dtype=torch.float32) / 8.0
        )

        # Register as non-learnable buffers, but still move with .to(device)
        self.register_buffer(
            "f_identity",
            identity.view(1, 1, 3, 3).repeat(in_channel, 1, 1, 1).to(device),
        )
        self.register_buffer(
            "f_sobel_x", sobel_x.view(1, 1, 3, 3).repeat(in_channel, 1, 1, 1).to(device)
        )
        self.register_buffer(
            "f_sobel_y", sobel_y.view(1, 1, 3, 3).repeat(in_channel, 1, 1, 1).to(device)
        )
        self.register_buffer(
            "f_laplacian",
            laplacian.view(1, 1, 3, 3).repeat(in_channel, 1, 1, 1).to(device),
        )

    def forward(self, x):
        """
        x shape: [B, in_channel, H, W]
        Output shape: [B, 4*in_channel, H, W]
        """
        # Apply circular padding manually for wraparound behavior
        x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')

        # pylint: disable=not-callable
        identity_out = F.conv2d(x_padded, self.f_identity, padding=0, groups=self.in_channel)
        grad_x = F.conv2d(x_padded, self.f_sobel_x, padding=0, groups=self.in_channel)
        grad_y = F.conv2d(x_padded, self.f_sobel_y, padding=0, groups=self.in_channel)
        lap_out = F.conv2d(x_padded, self.f_laplacian, padding=0, groups=self.in_channel)

        # Concatenate along channel dimension
        return torch.cat([identity_out, grad_x, grad_y, lap_out], dim=1)

    def get_out_channel(self):
        # 4 outputs (identity + dx + dy + lap) per input channel
        return self.in_channel * 4


class ConvPerception(Perception):
    def __init__(
        self, in_channel=16, out_channel=80, kernel_size=3, slope=0.2, dilation=1, device="cpu"
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.device = device
        self.dilation = dilation

        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel_size, padding="same", stride=1, dilation=dilation, padding_mode="circular"
        )
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        # print("Pre perception", x.shape)
        c = self.conv(x)
        c = self.lrelu(c)
        # print("Post perception", c.shape)
        return c

    def get_out_channel(self):
        return self.out_channel

class DeformableConvPerception(Perception):
    def __init__(self, in_channel=16, out_channel=80, kernel_size=3, slope=0.2, device="cpu"):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.device = device

        # Compute padding for "same" spatial dimensions
        self.padding = (kernel_size - 1) // 2

        self.deform_conv = DeformableConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        c = self.deform_conv(x)
        c = self.lrelu(c)
        return c

    def get_out_channel(self):
        return self.out_channel
    
    def visualize_deformation(self, x):
        return self.deform_conv.visualize_deformation(x)


class ResidualConvPerception(Perception):
    def __init__(self, in_channel=16, out_channel=80, kernel_size=3, device="cpu"):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel_size, padding="same", padding_mode="circular"
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=kernel_size, padding="same", padding_mode="circular"
        )
        # Use a 1x1 convolution to match channels if necessary.
        self.skip = (
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
            if in_channel != out_channel
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + identity)

    def get_out_channel(self):
        return self.out_channel


class AttentionPerception(Perception):
    """
    Local attention-based perception for cellular automata.
    Each pixel attends to its local neighborhood (e.g., 3x3, 5x5) instead of the entire image.

    Unlike ConvPerception which uses fixed filter weights, AttentionPerception computes
    dynamic attention weights based on the content (features) of each position and its neighbors.
    """
    def __init__(
        self,
        in_channel=16,
        out_channel=80,
        kernel_size=3,  # Defines the local neighborhood size
        num_heads=4,    # Number of attention heads
        slope=0.2,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for 'same' padding behavior.")
        if out_channel % num_heads != 0:
            raise ValueError("out_channel must be divisible by num_heads.")

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = out_channel // num_heads
        self.padding = kernel_size // 2
        self.scale = self.head_dim ** -0.5
        self.neighborhood_size = kernel_size * kernel_size

        # Project input to Q, K, V
        self.to_qkv = nn.Conv2d(in_channel, out_channel * 3, kernel_size=1, bias=False)

        # Output projection
        self.project_out = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        # Activation
        self.lrelu = nn.LeakyReLU(slope)

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        # Initialize QKV projection with small weights
        nn.init.xavier_uniform_(self.to_qkv.weight, gain=0.1)

        # Initialize output projection to near-zero (residual-like behavior)
        nn.init.xavier_uniform_(self.project_out.weight, gain=0.01)
        if self.project_out.bias is not None:
            nn.init.zeros_(self.project_out.bias)

    def forward(self, x):
        """
        x shape: (B, in_channel, H, W)
        Output: (B, out_channel, H, W)

        Process:
        1. Generate Q, K, V from input
        2. Extract local neighborhoods for K, V using circular padding
        3. Compute attention scores: each pixel's Q attends to its neighborhood's K
        4. Apply attention weights to neighborhood's V
        5. Combine multi-head outputs and project
        """
        B, C, H, W = x.shape

        # 1. Generate Q, K, V and split into heads
        qkv = self.to_qkv(x)  # (B, out_channel*3, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # Each: (B, out_channel, H, W)

        # 2. Apply circular padding to K and V BEFORE reshaping (need 4D for circular padding)
        k_padded = F.pad(k, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        v_padded = F.pad(v, (self.padding, self.padding, self.padding, self.padding), mode='circular')

        # Now reshape for multi-head attention: (B, num_heads, head_dim, H_pad, W_pad)
        H_pad, W_pad = H + 2 * self.padding, W + 2 * self.padding
        q = q.view(B, self.num_heads, self.head_dim, H, W)
        k_padded = k_padded.view(B, self.num_heads, self.head_dim, H_pad, W_pad)
        v_padded = v_padded.view(B, self.num_heads, self.head_dim, H_pad, W_pad)

        # Extract all neighborhood patches efficiently
        k_patches = []
        v_patches = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                k_patches.append(k_padded[:, :, :, i:i+H, j:j+W])
                v_patches.append(v_padded[:, :, :, i:i+H, j:j+W])

        # Stack: (B, num_heads, head_dim, neighborhood_size, H, W)
        k_local = torch.stack(k_patches, dim=3)
        v_local = torch.stack(v_patches, dim=3)

        # 3. Compute attention scores
        # Q broadcasts with K_local for dot product over each neighborhood
        # q shape: (B, num_heads, head_dim, 1, H, W)
        # k_local shape: (B, num_heads, head_dim, neighborhood_size, H, W)
        q = q.unsqueeze(3)  # Add neighborhood dimension

        # Attention scores: dot product between Q and each neighbor's K
        # (B, num_heads, head_dim, 1, H, W) * (B, num_heads, head_dim, neighborhood_size, H, W)
        # -> sum over head_dim -> (B, num_heads, neighborhood_size, H, W)
        attn_scores = (q * k_local).sum(dim=2) * self.scale

        # Softmax over neighborhood dimension
        attn_weights = F.softmax(attn_scores, dim=2)  # (B, num_heads, neighborhood_size, H, W)

        # 4. Apply attention to values
        # attn_weights: (B, num_heads, 1, neighborhood_size, H, W)
        # v_local: (B, num_heads, head_dim, neighborhood_size, H, W)
        attn_weights = attn_weights.unsqueeze(2)

        # Weighted sum over neighborhood
        out = (attn_weights * v_local).sum(dim=3)  # (B, num_heads, head_dim, H, W)

        # 5. Combine heads and project
        out = out.reshape(B, self.out_channel, H, W)
        out = self.project_out(out)
        out = self.lrelu(out)

        return out

    def get_out_channel(self):
        return self.out_channel

class MultiHeadAttentionPerception(Perception):
    def __init__(
        self,
        in_channel: int = 16,
        out_channel: int = 80,
        embed_dim: int = 128, # Increased default example
        num_heads: int = 4,   # Example
        kernel_size: int = 3,
        slope: float = 0.2,
        use_layer_norm: bool = True,
        include_ffn: bool = True, # Flag to include FFN
        ffn_expand_ratio: int = 4  # How much to expand channels in FFN
    ) -> None:
        super().__init__()

        # ... (Input validation for kernel_size, embed_dim, num_heads) ...
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for 'same' padding behavior.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.scale = self.head_dim ** -0.5
        self.use_layer_norm = use_layer_norm
        self.include_ffn = include_ffn

        # --- Attention Layers ---
        self.query_conv = nn.Conv2d(in_channel, embed_dim, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channel, embed_dim, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channel, embed_dim, kernel_size=1, bias=False)
        # Remove unfold - we'll use manual circular padding instead
        self.proj_conv = nn.Conv2d(embed_dim, out_channel, kernel_size=1)
        self.neighborhood_size = kernel_size * kernel_size

        if in_channel != out_channel:
            self.input_residual = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.input_residual = nn.Identity()

        # --- Optional LayerNorm before FFN/Final Activation ---
        if self.use_layer_norm:
            # Apply LayerNorm on the out_channel dimension after attention projection
            self.norm_attn = nn.LayerNorm(out_channel)

        # --- Optional FFN Layers ---
        if self.include_ffn:
            ffn_hidden_dim = int(out_channel * ffn_expand_ratio)
            self.ffn_conv1 = nn.Conv2d(out_channel, ffn_hidden_dim, kernel_size=1)
            # Using GELU which is common in transformers, but ReLU/LeakyReLU also work
            self.ffn_activation = nn.GELU()
            self.ffn_conv2 = nn.Conv2d(ffn_hidden_dim, out_channel, kernel_size=1)
            # Optional: LayerNorm after FFN (before final activation)
            if self.use_layer_norm:
                 self.norm_ffn = nn.LayerNorm(out_channel)

        # --- Final Activation ---
        self.lrelu = nn.LeakyReLU(slope)


    def forward(self, x):
        b, c, h, w = x.shape
        L = h * w

        # 1. QKV Projection
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        # 2. Apply circular padding and manual unfold for K, V
        k_padded = F.pad(k, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        v_padded = F.pad(v, (self.padding, self.padding, self.padding, self.padding), mode='circular')

        # Manual unfold with circular padding
        k_patches = []
        v_patches = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                k_patch = k_padded[:, :, i:i+h, j:j+w]
                v_patch = v_padded[:, :, i:i+h, j:j+w]
                k_patches.append(k_patch)
                v_patches.append(v_patch)

        # Stack patches and reshape for MHSA
        k_stacked = torch.stack(k_patches, dim=2)  # (B, embed_dim, neighborhood_size, H, W)
        v_stacked = torch.stack(v_patches, dim=2)  # (B, embed_dim, neighborhood_size, H, W)

        # 3. Reshape for MHSA
        q = q.view(b, self.num_heads, self.head_dim, h, w)
        k_reshaped = k_stacked.view(b, self.num_heads, self.head_dim, self.neighborhood_size, h, w)
        v_reshaped = v_stacked.view(b, self.num_heads, self.head_dim, self.neighborhood_size, h, w)
        q_reshaped = q.unsqueeze(3) # (B, num_heads, head_dim, 1, H, W)

        # 4. Attention Calculation
        attn_scores = (q_reshaped * k_reshaped).sum(dim=2) * self.scale # Sum over head_dim
        attn_weights = F.softmax(attn_scores, dim=2) # Softmax over neighborhood
        attn_weights_expanded = attn_weights.unsqueeze(2) # Add head_dim dim

        # 5. Value Aggregation
        attended_v_heads = (attn_weights_expanded * v_reshaped).sum(dim=3) # Sum over neighborhood
        attended_v = attended_v_heads.reshape(b, self.embed_dim, h, w) # Combine heads

        # 6. Final Attention Projection
        attn_output = self.proj_conv(attended_v) # Shape: (B, out_channel, H, W)
        attn_output = attn_output + self.input_residual(x)

        # --- Optional: Post-Attention LayerNorm & Residual ---
        # Note: Standard Transformers often apply residual connection HERE before FFN
        # x_res = x # Need to ensure x channels match out_channel if adding here
        # attn_output = attn_output + x_res # Example residual (needs channel matching)
        if self.use_layer_norm:
            attn_output_norm = self.norm_attn(attn_output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            attn_output_norm = attn_output


        # --- Optional: FFN Block ---
        if self.include_ffn:
            # Store input for residual connection
            ffn_input = attn_output_norm
            # Apply FFN layers
            out_ffn = self.ffn_conv1(ffn_input)
            out_ffn = self.ffn_activation(out_ffn)
            out_ffn = self.ffn_conv2(out_ffn)
            # Add residual connection
            ffn_output = ffn_input + out_ffn # Add input to FFN output
            # Optional: LayerNorm after FFN residual
            if self.use_layer_norm:
                ffn_output_norm = self.norm_ffn(ffn_output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                 ffn_output_norm = ffn_output
            final_output = ffn_output_norm
        else:
             # If no FFN, use the output from attention/norm stage
             final_output = attn_output_norm

        # --- Final Activation ---
        out = self.lrelu(final_output)

        return out

    def get_out_channel(self):
        return self.out_channel

class MultiPerception(Perception):
    def __init__(self, perceptions):
        super().__init__()
        # perceptions is a list of instantiated Perception modules.
        self.perceptions = nn.ModuleList(perceptions)
    
    def forward(self, x):
        # Apply each perception to the input and concatenate along the channel dimension.
        outputs = [p(x) for p in self.perceptions]
        return torch.cat(outputs, dim=1)
    
    def get_out_channel(self):
        # Total output channels is the sum of each perception's output channels.
        return sum(p.get_out_channel() for p in self.perceptions)
    
