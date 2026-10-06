"""MNIST Sudoku: the Sudoku puzzles of ``sudoku.py`` rendered with MNIST glyphs.

Every cell becomes an inverted (black on white) MNIST digit of its solution;
blank cells stay white. Train puzzles use MNIST train glyphs, the held-out test
puzzles MNIST test glyphs. Glyphs are drawn with the global torch RNG.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .sudoku import GRID_SIZE, SudokuPuzzles, board_to_onehot


class MNISTSudokuDataset(Dataset):
    """Pixel and board views of one Sudoku per sample.

    Returns a dict with ``seed_pixels``/``target_pixels`` ``(1, 9s, 9s)``,
    ``given_mask_pixels`` ``(1, 9s, 9s)``, ``seed_board``/``target_board``
    ``(9, 9, 9)`` (empty cells all zero) and ``given_mask_board`` ``(1, 9, 9)``.
    """

    def __init__(self, difficulty, total_samples, train=True, seed=None, cell_size=28, dataroot="datasets"):
        self.total_samples = total_samples
        self.cell_size = cell_size
        self.puzzles = SudokuPuzzles(difficulty, seed, train, dataroot)
        tfm = [transforms.ToTensor()]
        if cell_size != 28:
            tfm.insert(0, transforms.Resize((cell_size, cell_size)))
        mnist = datasets.MNIST(root=str(dataroot), train=train, download=True, transform=transforms.Compose(tfm))
        images = torch.stack([img for img, _ in mnist])
        labels = mnist.targets
        self.glyphs = {d: images[labels == d] for d in range(1, 10)}

    def __len__(self):
        return self.total_samples

    def _glyph(self, digit):
        pool = self.glyphs[digit]
        return 1.0 - pool[torch.randint(len(pool), (1,)).item()]

    def __getitem__(self, idx):
        puzzle, solution = self.puzzles[idx]
        cs, size = self.cell_size, GRID_SIZE * self.cell_size
        target = torch.ones(1, size, size)
        seed = torch.ones(1, size, size)
        given_pixels = torch.zeros(1, size, size)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = (slice(None), slice(r * cs, (r + 1) * cs), slice(c * cs, (c + 1) * cs))
                glyph = self._glyph(int(solution[r, c]))
                target[cell] = glyph
                if puzzle[r, c] > 0:
                    seed[cell] = glyph
                    given_pixels[cell] = 1.0
        return {
            "seed_pixels": seed,
            "target_pixels": target,
            "given_mask_pixels": given_pixels,
            "seed_board": board_to_onehot(puzzle),
            "target_board": board_to_onehot(solution),
            "given_mask_board": (puzzle > 0).float().unsqueeze(0),
        }
