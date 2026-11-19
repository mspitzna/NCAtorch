"""MNIST Sudoku codec: pixel encoder, conditional-VAE decoder and digit classifier.

The encoder ``E_psi`` maps a ``(B, 1, 9s, 9s)`` handwritten Sudoku to
``(B, 9, 9, 9)`` digit logits; the decoder ``D_omega`` renders a board back to
pixels with one style latent per cell. Both are trained separately and frozen
around an array-trained NCA. ``MNISTCNN`` is the frozen classifier used for
the decoder's perceptual losses and for scoring decoded images.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_sudoku_cells(images: torch.Tensor, grid_size: int, cell_size: int) -> torch.Tensor:
    """Split ``(B, 1, H, W)`` Sudoku images into ``(B*81, 1, cell, cell)`` cells."""
    batch_size = images.shape[0]
    cells = images.reshape(batch_size, 1, grid_size, cell_size, grid_size, cell_size)
    cells = cells.permute(0, 2, 4, 1, 3, 5).contiguous()
    return cells.view(batch_size * grid_size * grid_size, 1, cell_size, cell_size)


def tile_sudoku_cells(cells: torch.Tensor, batch_size: int, grid_size: int, cell_size: int) -> torch.Tensor:
    """Tile ``(B*81, 1, cell, cell)`` cells back into ``(B, 1, H, W)`` Sudoku images."""
    cells = cells.view(batch_size, grid_size, grid_size, 1, cell_size, cell_size)
    cells = cells.permute(0, 3, 1, 4, 2, 5).contiguous()
    return cells.view(batch_size, 1, grid_size * cell_size, grid_size * cell_size)


class PixelSudokuBoardEncoder(nn.Module):
    """Cell-wise CNN encoder from pixel Sudoku images to board logits."""

    def __init__(self, grid_size=9, cell_size=28, num_digits=9, base_channels=32, hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_digits = num_digits
        self.cell_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_digits),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return ``(B, D, 9, 9)`` digit logits."""
        batch_size = pixels.shape[0]
        logits = self.cell_encoder(extract_sudoku_cells(pixels, self.grid_size, self.cell_size))
        logits = logits.view(batch_size, self.grid_size, self.grid_size, self.num_digits)
        return logits.permute(0, 3, 1, 2).contiguous()


class SudokuBoardDecoder(nn.Module):
    """Board-to-pixel decoder; a conditional VAE when ``style_dim > 0``.

    Each cell is decoded independently from its normalized digit vector and a
    style latent. Blank cells stay white because occupancy scales the ink.
    Training infers the style posterior from target pixels; inference uses
    zero latents (deterministic) or samples from the prior.
    """

    def __init__(self, grid_size=9, cell_size=28, num_digits=9, base_channels=64, hidden_dim=128, style_dim=8):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.num_digits = num_digits
        self.start_hw = max(4, cell_size // 4)
        self.style_dim = style_dim
        self.base_channels = base_channels
        mid_channels = max(base_channels // 2, 16)
        low_channels = max(base_channels // 4, 8)

        self.fc = nn.Sequential(
            nn.Linear(num_digits + style_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, base_channels * self.start_hw * self.start_hw),
            nn.ReLU(inplace=True),
        )
        self.cell_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, low_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_channels, 1, kernel_size=3, padding=1),
        )
        if style_dim > 0:
            self.style_encoder = nn.Sequential(
                nn.Conv2d(1, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(mid_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.style_posterior = nn.Sequential(
                nn.Linear(base_channels * 4 * 4 + num_digits, hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.style_mu = nn.Linear(hidden_dim, style_dim)
            self.style_logvar = nn.Linear(hidden_dim, style_dim)

    def _decode_cells(self, board_vectors, style_latents):
        occupancy = board_vectors.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        features = board_vectors / board_vectors.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.style_dim > 0:
            features = torch.cat([features, style_latents], dim=1)
        features = self.fc(features).view(-1, self.base_channels, self.start_hw, self.start_hw)
        ink_logits = self.cell_decoder(features)
        if ink_logits.shape[-2:] != (self.cell_size, self.cell_size):
            ink_logits = F.interpolate(ink_logits, size=(self.cell_size, self.cell_size), mode="bilinear", align_corners=False)
        return 1.0 - occupancy.view(-1, 1, 1, 1) * torch.sigmoid(ink_logits)

    def forward(self, board, target_pixels=None, sample_prior=False):
        """Decode ``(B, D, 9, 9)`` boards to ``(pixels, kl_loss)``.

        With ``target_pixels`` the style comes from the posterior (sampled in
        training mode, its mean otherwise); else from the prior if
        ``sample_prior``, else zeros.
        """
        batch_size = board.shape[0]
        board_vectors = board.permute(0, 2, 3, 1).reshape(-1, self.num_digits)
        kl_loss = board.new_zeros(())
        style = board_vectors.new_zeros(board_vectors.shape[0], self.style_dim)
        if self.style_dim > 0 and target_pixels is not None:
            cells = extract_sudoku_cells(target_pixels, self.grid_size, self.cell_size)
            normalized = board_vectors / board_vectors.sum(dim=1, keepdim=True).clamp_min(1e-6)
            hidden = self.style_posterior(torch.cat([normalized, self.style_encoder(cells)], dim=1))
            mu, logvar = self.style_mu(hidden), self.style_logvar(hidden)
            style = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
            occupied = (board_vectors.sum(dim=1) > 1e-6).float()
            kl_per_cell = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            kl_loss = (kl_per_cell * occupied).sum() / occupied.sum().clamp_min(1.0)
        elif self.style_dim > 0 and sample_prior:
            style = torch.randn_like(style)
        cells = self._decode_cells(board_vectors, style)
        return tile_sudoku_cells(cells, batch_size, self.grid_size, self.cell_size), kl_loss


class MNISTCNN(nn.Module):
    """Small digit classifier (classes 1..9 -> 0..8) for inverted MNIST glyphs."""

    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class DigitClassifier(nn.Module):
    """Frozen ``MNISTCNN`` that resizes cells to its training resolution."""

    def __init__(self, path):
        super().__init__()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Digit classifier not found at {path}. Train it with "
                "`python -m nca.extensions.spatial_reasoning.scripts.train_digit_classifier --out {path}`."
            )
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.input_size = ckpt.get("input_size") if isinstance(ckpt, dict) else None
        self.model = MNISTCNN(num_classes=ckpt.get("num_classes", 9) if isinstance(ckpt, dict) else 9)
        self.model.load_state_dict(state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def train(self, mode=True):
        # Stay in eval mode when a parent module switches to training.
        return super().train(False)

    def resize(self, cells):
        if self.input_size is None or cells.shape[-1] == self.input_size:
            return cells
        return F.interpolate(cells, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

    def forward(self, cells):
        """``(N, 1, s, s)`` cells in [0, 1] -> ``(N, 9)`` logits."""
        return self.model(self.resize(cells))

    def features(self, cells):
        """Pooled convolutional features for the decoder's perceptual loss."""
        return self.model.pool(self.model.features(self.resize(cells)))

    @torch.no_grad()
    def classify_grid(self, images, grid_size=9, cell_size=28):
        """``(B, 1, H, W)`` Sudoku images -> ``(digits 0..8, confidence)``, each ``(B, 9, 9)``."""
        cells = extract_sudoku_cells(images.clamp(0, 1), grid_size, cell_size)
        probs = torch.softmax(self(cells), dim=1)
        conf, digits = probs.max(dim=1)
        shape = (images.shape[0], grid_size, grid_size)
        return digits.view(shape), conf.view(shape)
