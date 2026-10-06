"""Perception branches of the aNCA paper.

``anca_deform``     — DCNv2 head: learned offsets and per-tap modulation, sampled
                      on a torus (the aNCA perception, stack several for H heads).
``anca_deform_v1``  — DCNv1 ablation: offsets only, no modulation.
``row_conv``        — learnable ``1 x KERNEL_SIZE`` convolution along rows.
``column_conv``     — learnable ``KERNEL_SIZE x 1`` convolution along columns.

Set ``KERNEL_SIZE`` of ``row_conv``/``column_conv`` to the grid size so the
kernel spans a whole row or column. The paper's R/C/B variant combines these
two with a host ``conv`` 3x3 branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from nca.core.models.ca.perceptions import Perception


def _circular_deform_conv2d(x, offset, weight, bias, stride, padding, mask=None):
    """Sample on a torus, including bilinear interpolation across its edges.

    Padding by the kernel radius alone is insufficient for displaced samples.
    Wrap absolute sampling coordinates into the original image, then translate
    them back to offsets in a periodically padded image. At least one halo pixel
    is needed for interpolation between the last and first pixels. Remainder
    preserves offset gradients away from the usual interpolation knots.
    """
    batch, _, height, width = x.shape
    out_height, out_width = offset.shape[-2:]
    kernel_height, kernel_width = weight.shape[-2:]
    expected_shape = (
        (height + 2 * padding - kernel_height) // stride + 1,
        (width + 2 * padding - kernel_width) // stride + 1,
    )
    if (out_height, out_width) != expected_shape:
        raise ValueError(f"Offset spatial shape must be {expected_shape}, got {(out_height, out_width)}.")
    kernel_y, kernel_x = torch.meshgrid(
        torch.arange(kernel_height, device=offset.device, dtype=offset.dtype),
        torch.arange(kernel_width, device=offset.device, dtype=offset.dtype),
        indexing="ij",
    )
    output_y = torch.arange(out_height, device=offset.device, dtype=offset.dtype) * stride - padding
    output_x = torch.arange(out_width, device=offset.device, dtype=offset.dtype) * stride - padding
    base_y, base_x = torch.broadcast_tensors(
        kernel_y.reshape(-1, 1, 1) + output_y.reshape(1, -1, 1),
        kernel_x.reshape(-1, 1, 1) + output_x.reshape(1, 1, -1),
    )
    base = torch.stack((base_y, base_x), dim=1).unsqueeze(0)
    coordinates = offset.reshape(batch, -1, 2, out_height, out_width) + base
    period = offset.new_tensor((height, width)).reshape(1, 1, 2, 1, 1)
    halo = max(1, padding)
    wrapped_offset = (coordinates.remainder(period) - base + halo - padding).flatten(1, 2)
    padded_x = F.pad(x, (halo, halo, halo, halo), mode="circular")

    # With padding=0 the interpolation halo enlarges the native operator's
    # output. Supply dummy sampling locations and discard those extra outputs.
    native_height = (height + 2 * halo - kernel_height) // stride + 1
    native_width = (width + 2 * halo - kernel_width) // stride + 1
    extra_height, extra_width = native_height - out_height, native_width - out_width
    if extra_height or extra_width:
        wrapped_offset = F.pad(wrapped_offset, (0, extra_width, 0, extra_height))
        if mask is not None:
            mask = F.pad(mask, (0, extra_width, 0, extra_height))

    result = torchvision.ops.deform_conv2d(
        input=padded_x, offset=wrapped_offset, weight=weight, bias=bias,
        stride=stride, padding=0, mask=mask,
    )
    return result[..., :out_height, :out_width]


class TorusDeformableConv2d(nn.Module):
    """Deformable convolution on a torus (DCNv2, or DCNv1 with ``version='v1'``).

    Offsets and modulation are predicted from the circularly padded input by
    zero-initialised convolutions, so the layer starts as a regular convolution.
    Offsets are clamped to a quarter of the grid; modulation is ``2 * sigmoid``.
    Sampling coordinates wrap around the grid, so every offset reads a real cell.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, version="v2"):
        super().__init__()
        if version not in ("v1", "v2"):
            raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
        self.version = version
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        if version == "v2":
            self.modulator_conv = nn.Conv2d(
                in_channels, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, bias=True,
            )
            nn.init.constant_(self.modulator_conv.weight, 0.0)
            nn.init.constant_(self.modulator_conv.bias, 0.0)

        # Holds the kernel weights; applied through deform_conv2d only.
        self.regular_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias,
        )

    def predict_sampling(self, x):
        """Return ``(offset, modulator)``; ``modulator`` is ``None`` for DCNv1."""
        max_offset = max(x.shape[-2:]) / 4.0
        x_padded = F.pad(x, (self.padding,) * 4, mode="circular")
        offset = self.offset_conv(x_padded).clamp(-max_offset, max_offset)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x_padded)) if self.version == "v2" else None
        return offset, modulator

    def forward(self, x):
        offset, modulator = self.predict_sampling(x)
        return _circular_deform_conv2d(
            x=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
            stride=self.stride, padding=self.padding, mask=modulator,
        )


class AncaDeformablePerception(Perception):
    """One aNCA head: torus deformable convolution followed by leaky ReLU."""

    def __init__(self, in_channel, out_channel, kernel_size=3, version="v2", slope=0.2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Deformable perception KERNEL_SIZE must be odd.")
        self.out_channel = out_channel
        self.deform_conv = TorusDeformableConv2d(
            in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, version=version,
        )
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        return self.lrelu(self.deform_conv(x))

    def get_out_channel(self):
        return self.out_channel


class RowConvPerception(Perception):
    """Learnable ``1 x length`` convolution along each row."""

    def __init__(self, in_channel, out_channel, length, slope=0.2):
        super().__init__()
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, length), padding="same")
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        return self.lrelu(self.conv(x))

    def get_out_channel(self):
        return self.out_channel


class ColumnConvPerception(Perception):
    """Learnable ``length x 1`` convolution along each column."""

    def __init__(self, in_channel, out_channel, length, slope=0.2):
        super().__init__()
        self.out_channel = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(length, 1), padding="same")
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x):
        return self.lrelu(self.conv(x))

    def get_out_channel(self):
        return self.out_channel
