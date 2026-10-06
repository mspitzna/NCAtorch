"""Sudoku puzzles from the 1M solution grids of ``BartekPog/mnist-sudoku``.

The first 990K grids are the training pool: every sample picks a grid and
removes a random fraction of cells (``difficulty``), without a uniqueness check.
The last 10K grids are held out; a fixed 10K-puzzle test split is drawn from
them once per (difficulty, seed) and cached under ``<dataroot>/sudoku_test``.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nca.data.datasets.base_dataset import NCADataset

GRID_SIZE = 9
BOX_SIZE = 3
HF_REPO = "BartekPog/mnist-sudoku"
HF_FILE = "sudoku-mnist.safetensors"
TRAIN_SPLIT_END = 990_000
TEST_SPLIT_SIZE = 10_000


def parse_difficulty(difficulty):
    """Return ``(min, max)`` fraction of removed cells from a float or a pair."""
    if isinstance(difficulty, (list, tuple)):
        low, high = (float(v) for v in difficulty)
    else:
        low = high = float(difficulty)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"difficulty must satisfy 0 <= min <= max <= 1, got {difficulty!r}.")
    return low, high


def load_solution_grids() -> np.ndarray:
    """Memory-map the ``(1M, 9, 9)`` int32 ``sudokus`` tensor of the safetensors file.

    A safetensors file is an 8-byte header length, a JSON header and raw
    little-endian data. Mapping only ``sudokus`` skips the MNIST images, and
    DataLoader workers share the page cache instead of copying ~300 MB each.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(header_len))["sudokus"]
    if info["dtype"] != "I32" or info["shape"][1:] != [GRID_SIZE, GRID_SIZE]:
        raise ValueError(f"Expected int32 sudoku grids of shape (N, 9, 9), got {info}.")
    start, _ = info["data_offsets"]
    return np.memmap(path, dtype="<i4", mode="r", offset=8 + header_len + start, shape=tuple(info["shape"]))


class SudokuPuzzles:
    """``(puzzle, solution)`` long boards of shape (9, 9); 0 marks an empty cell."""

    def __init__(self, difficulty, seed, train, dataroot="datasets"):
        self.difficulty = parse_difficulty(difficulty)
        # A scalar difficulty draws no random number, keeping puzzles identical to earlier runs.
        self.is_range = isinstance(difficulty, (list, tuple))
        self.seed = None if seed in (None, -1) else int(seed)
        self.train = train
        self.dataroot = Path(dataroot)
        self._grids = load_solution_grids()
        if not train:
            self._test_puzzles, self._test_solutions = self._load_or_build_test_split()

    def _remove_cells(self, solution_flat, rng):
        puzzle = solution_flat.copy()
        fraction = rng.uniform(*self.difficulty) if self.is_range else self.difficulty[0]
        n_remove = int(fraction * puzzle.size)
        if n_remove > 0:
            puzzle[rng.choice(puzzle.size, size=n_remove, replace=False)] = 0
        return puzzle

    def _cache_path(self) -> Path:
        low, high = self.difficulty
        diff_tag = f"{low:.2f}_{high:.2f}" if self.is_range else f"{low:.2f}"
        seed_tag = "unseeded" if self.seed is None else str(self.seed)
        return self.dataroot / "sudoku_test" / f"test_{TEST_SPLIT_SIZE}_{diff_tag}_seed{seed_tag}.pt"

    def _load_or_build_test_split(self):
        cache = self._cache_path()
        if cache.exists():
            data = torch.load(cache, weights_only=True)
            return data["puzzles"], data["solutions"]

        print(f"Building {TEST_SPLIT_SIZE} held-out test puzzles (difficulty={self.difficulty}), cached to {cache}")
        test_grids = self._grids[TRAIN_SPLIT_END:]
        puzzles = torch.zeros(TEST_SPLIT_SIZE, GRID_SIZE, GRID_SIZE, dtype=torch.int8)
        solutions = torch.zeros_like(puzzles)
        for i in tqdm(range(TEST_SPLIT_SIZE), desc="Building test split"):
            rng = np.random.default_rng(i if self.seed is None else self.seed + 20_000_000 + i)
            solution = np.array(test_grids[int(rng.integers(0, len(test_grids)))]).flatten()
            puzzles[i] = torch.from_numpy(self._remove_cells(solution, rng).reshape(GRID_SIZE, GRID_SIZE))
            solutions[i] = torch.from_numpy(solution.reshape(GRID_SIZE, GRID_SIZE))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"puzzles": puzzles, "solutions": solutions}, cache)
        return puzzles, solutions

    def __getitem__(self, idx):
        if not self.train:
            i = idx % TEST_SPLIT_SIZE
            return self._test_puzzles[i].long(), self._test_solutions[i].long()
        rng = np.random.default_rng() if self.seed is None else np.random.default_rng(self.seed + idx)
        solution = np.array(self._grids[int(rng.integers(0, TRAIN_SPLIT_END))]).flatten()
        puzzle = self._remove_cells(solution, rng)
        return (
            torch.from_numpy(puzzle.reshape(GRID_SIZE, GRID_SIZE)).long(),
            torch.from_numpy(solution.reshape(GRID_SIZE, GRID_SIZE)).long(),
        )


def board_to_onehot(board: torch.Tensor) -> torch.Tensor:
    """``(9, 9)`` digits 0..9 (0 = empty) -> ``(9, 9, 9)`` one-hot, empty cells all zero."""
    onehot = torch.nn.functional.one_hot(board.long(), GRID_SIZE + 1)[..., 1:]
    return onehot.permute(2, 0, 1).float()


class SudokuDataset(NCADataset):
    """Array Sudoku for NCAs.

    Returns per sample:
        seed:      (channel_n, 9, 9) — one-hot clues in channels 0..8, hidden noise after
        condition: (1, 9, 9)         — given-cell mask (1 = clue)
        target:    (9, 9, 9)         — one-hot solution
    """

    num_digits = GRID_SIZE
    grid_size = GRID_SIZE
    box_size = BOX_SIZE

    def __init__(self, channel_n, difficulty, total_samples, train=True, seed=None, dataroot="datasets"):
        if channel_n < GRID_SIZE:
            raise ValueError(f"Sudoku requires MODEL.CHANNEL_N >= {GRID_SIZE}, got {channel_n}.")
        self.channel_n = channel_n
        self.total_samples = total_samples
        self.puzzles = SudokuPuzzles(difficulty, seed, train, dataroot)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        puzzle, solution = self.puzzles[idx]
        seed = torch.zeros(self.channel_n, GRID_SIZE, GRID_SIZE)
        seed[:GRID_SIZE] = board_to_onehot(puzzle)
        seed[GRID_SIZE:] = torch.randn(self.channel_n - GRID_SIZE, GRID_SIZE, GRID_SIZE) * 0.1
        condition = (puzzle > 0).float().unsqueeze(0)
        return seed, condition, board_to_onehot(solution)

    def batch_to_rgb(self, x0, x, target, cond=None):
        mask = cond[:, 0] if cond is not None else None
        return self.to_rgb(x0, mask), self.to_rgb(x, mask), self.to_rgb(target, mask)

    def to_rgb(self, state, given_mask=None, cell_px=24):
        """Render argmax digits; clues blue, rule violations red, low confidence gray."""
        logits = state[:, :GRID_SIZE].detach().float().cpu()
        digits = logits.argmax(dim=1) + 1
        confident = logits.max(dim=1).values >= 0.5
        violations = sudoku_violations(digits - 1)
        font = _font(int(cell_px * 0.7))
        images = []
        for b in range(digits.shape[0]):
            size = GRID_SIZE * cell_px
            img = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(img)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    given = given_mask is not None and given_mask[b, r, c] > 0.5
                    fill = "#ff9999" if violations[b, r, c] else ("#d4edff" if given else "#d4ffd4")
                    box = (c * cell_px, r * cell_px, (c + 1) * cell_px - 1, (r + 1) * cell_px - 1)
                    draw.rectangle(box, fill=fill, outline="#999999")
                    color = "blue" if given else ("black" if confident[b, r, c] else "#888888")
                    draw.text(
                        ((c + 0.5) * cell_px, (r + 0.5) * cell_px), str(int(digits[b, r, c])),
                        fill=color, font=font, anchor="mm",
                    )
            for i in range(0, GRID_SIZE + 1, BOX_SIZE):
                p = min(i * cell_px, size - 1)
                draw.line([(p, 0), (p, size)], fill="black", width=2)
                draw.line([(0, p), (size, p)], fill="black", width=2)
            images.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(images)


def sudoku_violations(digits: torch.Tensor) -> torch.Tensor:
    """``(B, 9, 9)`` digits 0..8 -> bool mask of cells sharing a digit with their row, column or box."""
    b = digits.shape[0]
    onehot = torch.nn.functional.one_hot(digits.long(), GRID_SIZE).float()  # (B, R, C, D)
    idx = digits.long().unsqueeze(-1)
    row = onehot.sum(dim=2, keepdim=True).expand_as(onehot).gather(3, idx).squeeze(-1)
    col = onehot.sum(dim=1, keepdim=True).expand_as(onehot).gather(3, idx).squeeze(-1)
    s = BOX_SIZE
    box = onehot.reshape(b, s, s, s, s, GRID_SIZE).sum(dim=(2, 4), keepdim=True)
    box = box.expand(b, s, s, s, s, GRID_SIZE).reshape(b, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    box = box.gather(3, idx).squeeze(-1)
    return (row > 1) | (col > 1) | (box > 1)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()
