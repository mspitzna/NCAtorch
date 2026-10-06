"""Maze shortest-path data: the HF benchmark and a Prim's-algorithm generator.

Encoding shared by both sources (``#`` wall, ``S`` start, ``G`` goal, ``o`` path):
    seed:      (channel_n, H, W) — 1 at the start cell in channel 0, hidden noise after
    condition: (2, H, W)         — wall mask; endpoints (+1 start, -1 goal)
    target:    (1, H, W)         — shortest path including start and goal
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from nca.data.datasets.base_dataset import NCADataset

HF_DATASET_ID = "sapientinc/maze-30x30-hard-1k"


def _string_to_grid(maze_str: str) -> np.ndarray:
    side = int(round(len(maze_str) ** 0.5))
    if side * side != len(maze_str):
        raise ValueError(f"Expected a square flat maze string, got length {len(maze_str)}.")
    return np.array(list(maze_str), dtype="U1").reshape(side, side)


def maze_to_tensors(grid, path_grid, channel_n, noise=None):
    """Build ``(seed, condition, target)`` from a character grid and its solution grid."""
    wall = torch.from_numpy(grid == "#").float()
    start = torch.from_numpy(grid == "S").float()
    goal = torch.from_numpy(grid == "G").float()
    path = torch.from_numpy(np.isin(path_grid, ("o", "S", "G"))).float()
    h, w = grid.shape
    seed = torch.zeros(channel_n, h, w)
    seed[0] = start
    seed[1:] = (torch.randn(channel_n - 1, h, w) if noise is None else noise) * 0.1
    return seed, torch.stack([wall, start - goal]), path.unsqueeze(0)


class MazeDataset(NCADataset):
    """``sapientinc/maze-30x30-hard-1k`` (1K train / 1K test mazes).

    Training samples get a random D4 symmetry. The parsed split is cached at
    ``<dataroot>/maze/<id>_<split>.pt``.
    """

    # The 8 symmetries of the square.
    _D4 = [
        lambda t: t,
        lambda t: torch.rot90(t, 1, [-2, -1]),
        lambda t: torch.rot90(t, 2, [-2, -1]),
        lambda t: torch.rot90(t, 3, [-2, -1]),
        lambda t: torch.flip(t, [-1]),
        lambda t: torch.flip(t, [-2]),
        lambda t: t.transpose(-2, -1),
        lambda t: torch.flip(t.transpose(-2, -1), [-2, -1]),
    ]

    def __init__(self, channel_n, train=True, augment=None, hf_dataset_id=HF_DATASET_ID, dataroot="datasets"):
        self.channel_n = channel_n
        self.augment = train if augment is None else augment
        split = "train" if train else "test"
        cache = Path(dataroot) / "maze" / f"{hf_dataset_id.replace('/', '_')}_{split}.pt"
        if cache.exists():
            self.samples = torch.load(cache, weights_only=False)
        else:
            from datasets import load_dataset

            rows = load_dataset(hf_dataset_id, split=split)
            self.samples = [(row["question"], row["answer"]) for row in rows]
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.samples, cache)
        self.grid_size = _string_to_grid(self.samples[0][0]).shape[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        maze_str, solution_str = self.samples[idx]
        tensors = maze_to_tensors(_string_to_grid(maze_str), _string_to_grid(solution_str), self.channel_n)
        if self.augment:
            op = self._D4[torch.randint(0, len(self._D4), (1,)).item()]
            tensors = tuple(op(t) for t in tensors)
        return tensors

    def batch_to_rgb(self, x0, x, target, cond=None):
        return maze_to_rgb(x0, cond), maze_to_rgb(x, cond, target), maze_to_rgb(target, cond)


def maze_to_rgb(state, cond, target=None, cell_px=8):
    """Walls black, start red, goal green, predicted path blue, off-target path yellow."""
    state, cond = state.detach().cpu(), cond.detach().cpu()
    b, _, h, w = state.shape
    pred = state[:, 0] > 0.5
    color = torch.full((b, h, w, 3), 0.8)
    color[pred] = torch.tensor([0.23, 0.53, 1.0])
    if target is not None:
        color[pred & (target.detach().cpu()[:, 0] < 0.5)] = torch.tensor([1.0, 1.0, 0.0])
    color[cond[:, 1] > 0.5] = torch.tensor([1.0, 0.0, 0.0])
    color[cond[:, 1] < -0.5] = torch.tensor([0.0, 0.5, 0.0])
    color[cond[:, 0] > 0.5] = 0.0
    color = color.permute(0, 3, 1, 2)
    return color.repeat_interleave(cell_px, dim=2).repeat_interleave(cell_px, dim=3)


# ---------------------------------------------------------------------------
# Out-of-distribution mazes (Prim's algorithm)
# ---------------------------------------------------------------------------

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def prims_maze(size, rng):
    """Randomized-Prim perfect maze with 1-cell corridors; ``size`` must be odd."""
    grid = np.full((size, size), "#", dtype="U1")
    grid[1::2, 1::2] = " "
    n = size // 2
    visited = np.zeros((n, n), dtype=bool)
    r0, c0 = int(rng.integers(n)), int(rng.integers(n))
    visited[r0, c0] = True
    frontier = [(r0, c0, r0 + dr, c0 + dc) for dr, dc in _NEIGHBOURS if 0 <= r0 + dr < n and 0 <= c0 + dc < n]
    while frontier:
        r, c, r2, c2 = frontier.pop(int(rng.integers(len(frontier))))
        if visited[r2, c2]:
            continue
        grid[2 * r + 1 + (r2 - r), 2 * c + 1 + (c2 - c)] = " "
        visited[r2, c2] = True
        frontier += [
            (r2, c2, r2 + dr, c2 + dc) for dr, dc in _NEIGHBOURS
            if 0 <= r2 + dr < n and 0 <= c2 + dc < n and not visited[r2 + dr, c2 + dc]
        ]
    return grid


def shortest_path(grid, start, goal):
    """BFS shortest path as a list of cells, or ``None`` if unreachable."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return path[::-1]
        r, c = cell
        for dr, dc in _NEIGHBOURS:
            nxt = (r + dr, c + dc)
            if (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]
                    and grid[nxt] != "#" and nxt not in parent):
                parent[nxt] = cell
                queue.append(nxt)
    return None


def generate_prims_mazes(size, n, channel_n, seed=0):
    """``n`` Prim mazes, start in the left third and goal in the right third.

    Returns stacked ``(seeds, conditions, targets)``. Even sizes are rounded up.
    """
    size = size if size % 2 else size + 1
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n:
        grid = prims_maze(size, rng)
        left = [(r, c) for r in range(size) for c in range(1, size // 3) if grid[r, c] != "#"]
        right = [(r, c) for r in range(size) for c in range(2 * size // 3, size - 1) if grid[r, c] != "#"]
        start = left[int(rng.integers(len(left)))]
        goal = right[int(rng.integers(len(right)))]
        path = shortest_path(grid, start, goal)
        if path is None or len(path) < 3:
            continue
        solution = grid.copy()
        for cell in path:
            solution[cell] = "o"
        grid[start], grid[goal] = "S", "G"
        noise = torch.from_numpy(rng.standard_normal((channel_n - 1, size, size)).astype(np.float32))
        samples.append(maze_to_tensors(grid, solution, channel_n, noise))
    return tuple(torch.stack(parts) for parts in zip(*samples))
