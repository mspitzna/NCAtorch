"""Paper metrics and rollout helpers shared by trainers and evaluation scripts."""

from collections import deque
from pathlib import Path

import torch

from .data.sudoku import GRID_SIZE, sudoku_violations


@torch.no_grad()
def rollout(model, state, cond, n_steps):
    """Apply ``n_steps`` CA updates; the host model returns ``(state, dx)``."""
    for _ in range(n_steps):
        state, _ = model(state, cond)
    return state


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def sudoku_metrics(pred_digits, target_digits, given_mask):
    """Per-puzzle Sudoku metrics; digits are ``(B, 9, 9)`` in 0..8.

    Clue cells are taken from the target, so only the solved cells are judged.
        solved:  every non-clue cell equals the dataset solution (exact match)
        valid:   the completed grid is a valid Sudoku (ASR in the paper)
        acp:     % of non-clue cells equal to the dataset solution
        acp_valid: % of non-clue cells without a row/column/box conflict (ACP in the paper)

    Puzzles without empty cells count as solved and are left out of the ACP means.
    """
    to_solve = given_mask < 0.5
    grid = torch.where(to_solve, pred_digits, target_digits)
    conflict = sudoku_violations(grid)
    n_to_solve = to_solve.sum(dim=(1, 2)).float()
    correct = ((pred_digits == target_digits) & to_solve).sum(dim=(1, 2)).float()
    no_conflict = (to_solve & ~conflict).sum(dim=(1, 2)).float()
    return {
        "solved": (correct == n_to_solve).float(),
        "valid": (~conflict.flatten(1).any(dim=1)).float(),
        "acp": 100.0 * correct / n_to_solve,  # NaN for puzzles without empty cells
        "acp_valid": 100.0 * no_conflict / n_to_solve,
    }


def summarize(per_sample: dict) -> dict:
    """Concatenate per-batch metric tensors and average (ignoring NaN); rates in percent."""
    out = {}
    for key, values in per_sample.items():
        mean = torch.cat(values).float().nanmean().item()
        out[key] = mean * 100.0 if key in ("solved", "valid", "valid_path", "valid_optimal", "exact") else mean
    return out


@torch.no_grad()
def evaluate_sudoku(model, dataset, n_puzzles, n_steps, batch_size, device):
    """ASR/ACP of the array NCA on the first ``n_puzzles`` of ``dataset``."""
    results = {}
    for start in range(0, n_puzzles, batch_size):
        batch = [dataset[i] for i in range(start, min(start + batch_size, n_puzzles))]
        seed, cond, target = (torch.stack(parts).to(device) for parts in zip(*batch))
        state = rollout(model, seed, cond, n_steps)
        metrics = sudoku_metrics(state[:, :GRID_SIZE].argmax(dim=1), target.argmax(dim=1), cond[:, 0])
        for key, value in metrics.items():
            results.setdefault(key, []).append(value.cpu())
    return summarize(results)


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------

def _connected(path, start, goal):
    """4-connectivity of ``start`` and ``goal`` through the boolean ``path`` grid."""
    if not (path[start] and path[goal]):
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nxt[0] < path.shape[0] and 0 <= nxt[1] < path.shape[1] and path[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def maze_metrics(state, cond, target):
    """Per-maze ``valid_path`` (start and goal connected through predicted
    non-wall path cells) and ``valid_optimal`` (additionally the shortest-path length)."""
    free = (cond[:, 0] <= 0.5).cpu()
    pred = (state[:, 0].clamp(0, 1) > 0.5).cpu() & free
    true = (target[:, 0] > 0.5).cpu() & free
    endpoints = cond[:, 1].cpu()
    valid, optimal = [], []
    for b in range(pred.shape[0]):
        starts, goals = (endpoints[b] > 0.5).nonzero(), (endpoints[b] < -0.5).nonzero()
        # A maze without both endpoints cannot be solved.
        ok = len(starts) > 0 and len(goals) > 0 and _connected(
            pred[b].numpy(), tuple(starts[0].tolist()), tuple(goals[0].tolist()),
        )
        valid.append(ok)
        optimal.append(ok and int(pred[b].sum()) == int(true[b].sum()))
    return {"valid_path": torch.tensor(valid).float(), "valid_optimal": torch.tensor(optimal).float()}


@torch.no_grad()
def evaluate_maze(model, seeds, conds, targets, n_steps, batch_size, device):
    """Valid Path / Valid Optimal over stacked maze tensors."""
    results = {}
    for start in range(0, seeds.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        cond = conds[sl].to(device)
        state = rollout(model, seeds[sl].to(device), cond, n_steps)
        for key, value in maze_metrics(state, cond, targets[sl]).items():
            results.setdefault(key, []).append(value)
    return summarize(results)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """Load a CA state dict, dropping a ``torch.compile`` prefix if present."""
    state = torch.load(Path(path), map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    getattr(model, "_orig_mod", model).load_state_dict(state)
    return model.eval().to(device)
