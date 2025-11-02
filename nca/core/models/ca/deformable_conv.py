import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DeformableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular'
    ):

        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding if padding_mode != 'circular' else 0,
            padding_mode='zeros' if padding_mode == 'circular' else padding_mode,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding if padding_mode != 'circular' else 0,
            padding_mode='zeros' if padding_mode == 'circular' else padding_mode,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding if padding_mode != 'circular' else 0,
            padding_mode='zeros' if padding_mode == 'circular' else padding_mode,
            bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0

        # Apply circular padding if needed
        if self.padding_mode == 'circular':
            x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
            offset = self.offset_conv(x_padded).clamp(-max_offset, max_offset)
            modulator = 2.0 * torch.sigmoid(self.modulator_conv(x_padded))
        else:
            offset = self.offset_conv(x).clamp(-max_offset, max_offset)
            modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))

        # For deform_conv2d, we need to apply circular padding to the input
        if self.padding_mode == 'circular':
            x_for_deform = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
            x_out = torchvision.ops.deform_conv2d(
                input=x_for_deform,
                offset=offset,
                weight=self.regular_conv.weight,
                bias=self.regular_conv.bias,
                padding=0,  # Padding already applied
                mask=modulator,
            )
        else:
            x_out = torchvision.ops.deform_conv2d(
                input=x,
                offset=offset,
                weight=self.regular_conv.weight,
                bias=self.regular_conv.bias,
                padding=self.padding,
                mask=modulator,
            )
        return x_out
    
    def visualize_deformation(self, x, grid_size=8, scale_factor=5.0):
        """
        Visualize the learned deformation field of the deformable convolution.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            grid_size: Size of the visualization grid
            scale_factor: Factor to scale the offset vectors for better visibility
            
        Returns:
            Dictionary containing:
            - 'grid': Regular grid points [B, H, W, 2]
            - 'deformed_grid': Grid points after deformation [B, H, W, 2]
            - 'offsets': Raw offset values [B, 2*K*K, H, W]
            - 'offset_map': Reshaped offset values for visualization [B, H, W, 2]
            - 'modulation': Modulation values [B, K*K, H, W]
            - 'modulation_map': Average modulation per position [B, H, W]
        """
        batch_size, _, h, w = x.shape
        k = self.kernel_size
        
        # Get offsets and modulation
        offsets = self.offset_conv(x)  # [B, 2*K*K, H, W]
        modulation = torch.sigmoid(self.modulator_conv(x))  # [B, K*K, H, W]
        
        # Create visualization grid (subsample if needed)
        if h > grid_size or w > grid_size:
            h_step = max(1, h // grid_size)
            w_step = max(1, w // grid_size)
            grid_h = torch.arange(0, h, h_step, device=x.device)
            grid_w = torch.arange(0, w, w_step, device=x.device)
        else:
            grid_h = torch.arange(0, h, device=x.device)
            grid_w = torch.arange(0, w, device=x.device)
        
        # Create meshgrid
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Calculate average offset at each position
        # Reshape offsets from [B, 2*K*K, H, W] to [B, K*K, 2, H, W]
        offsets_reshaped = offsets.view(batch_size, 2, k*k, h, w).permute(0, 2, 1, 3, 4)
        
        # Average across kernel positions to get a single offset per pixel
        offset_map = offsets_reshaped.mean(dim=1)  # [B, 2, H, W]
        offset_map = offset_map.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Subsample offset map to match grid size
        if h > grid_size or w > grid_size:
            offset_map = offset_map[:, ::h_step, ::w_step, :]
        
        # Create deformed grid by adding scaled offsets
        deformed_grid = grid.clone().float()
        deformed_grid += offset_map * scale_factor
        
        # Calculate average modulation per position
        modulation_map = modulation.mean(dim=1)  # [B, H, W]
        if h > grid_size or w > grid_size:
            modulation_map = modulation_map[:, ::h_step, ::w_step]
        
        return {
            'grid': grid,
            'deformed_grid': deformed_grid,
            'offsets': offsets,
            'offset_map': offset_map,
            'modulation': modulation,
            'modulation_map': modulation_map
        }