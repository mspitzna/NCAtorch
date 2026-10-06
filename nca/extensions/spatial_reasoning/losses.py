"""Constraint-based training objectives of the spatial reasoning tasks.

Each loss returns a dict with a differentiable ``total_loss`` plus its terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SudokuConstraintLoss(nn.Module):
    """Row/column/box constraint, one-hot tightening and given-cell cross-entropy.

    With ``p = softmax(logits / temperature)`` over the first ``D`` channels,
    every digit must sum to one in each row, column and box. Any valid
    completion has zero constraint loss, so puzzles with several solutions
    give consistent gradients. Clue cells are anchored by cross-entropy.

    The constraint alone is degenerate: a uniform ``1/D`` prediction satisfies
    it with zero gradient. The one-hot term asks ``p^2`` to sum to one per unit
    as well, which together with the constraint holds only for one-hot units.
    """

    def __init__(
        self, constraint_weight=1.0, given_weight=1.0, onehot_weight=0.1, softmax_temperature=0.25,
        grid_size=9, box_size=3,
    ):
        super().__init__()
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0.")
        self.constraint_weight = constraint_weight
        self.given_weight = given_weight
        self.onehot_weight = onehot_weight
        self.softmax_temperature = softmax_temperature
        self.grid_size = grid_size
        self.box_size = box_size

    def forward(self, predictions, targets, condition):
        """
        Args:
            predictions: (B, C, D, D) NCA state; channels ``:D`` are digit logits.
            targets:     (B, D, D, D) one-hot solution.
            condition:   (B, 1, D, D) given-cell mask (1 = clue).
        """
        d, b = self.grid_size, predictions.shape[0]
        if predictions.shape[1] < d or targets.shape[1] != d or targets.shape[-2:] != (d, d):
            raise ValueError(
                f"Expected predictions (B, >={d}, {d}, {d}) and targets (B, {d}, {d}, {d}), "
                f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
            )
        logits = predictions[:, :d] / self.softmax_temperature
        probs = torch.softmax(logits, dim=1)
        s = self.box_size

        def unit_losses(p):
            """Mean squared deviation of per-digit row, column and box sums from one."""
            # (rows, cols) -> (box_row, cell_row, box_col, cell_col)
            boxes = p.reshape(b, d, d // s, s, d // s, s).sum(dim=(3, 5))
            return tuple(((sums - 1.0) ** 2).mean() for sums in (p.sum(dim=3), p.sum(dim=2), boxes))

        row_loss, col_loss, box_loss = unit_losses(probs)
        constraint_loss = row_loss + col_loss + box_loss
        onehot_loss = sum(unit_losses(probs ** 2)) if self.onehot_weight > 0 else probs.new_zeros(())

        given = condition[:, 0] > 0.5
        given_loss = predictions.new_zeros(())
        if given.any():
            given_logits = logits.permute(0, 2, 3, 1)[given]
            given_loss = F.cross_entropy(given_logits, targets.argmax(dim=1)[given])

        total_loss = (
            self.constraint_weight * constraint_loss
            + self.given_weight * given_loss
            + self.onehot_weight * onehot_loss
        )
        return {
            "total_loss": total_loss,
            "constraint_loss": constraint_loss,
            "row_loss": row_loss,
            "col_loss": col_loss,
            "box_loss": box_loss,
            "given_loss": given_loss,
            "onehot_loss": onehot_loss,
        }


class MazeDiceBCELoss(nn.Module):
    """BCE + Dice + normalized path-length penalty on non-wall cells.

    Walls are trivially predictable from the condition and are excluded.
    Dice handles the path/non-path imbalance, BCE pushes binary commitment and
    the length term penalizes paths longer or shorter than the shortest path.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, length_weight=0.05, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.length_weight = length_weight
        self.smooth = smooth

    def forward(self, predictions, targets, condition):
        """
        Args:
            predictions: (B, C, H, W) NCA state; channel 0 is the path.
            targets:     (B, 1, H, W) binary shortest path.
            condition:   (B, 2, H, W) channel 0 is the wall mask.
        """
        # binary_cross_entropy is rejected inside autocast, so stay in float32.
        pred = predictions[:, :1].float().clamp(1e-6, 1.0 - 1e-6)
        true = targets[:, :1].float()
        free = (condition[:, :1] <= 0.5).float()

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            bce_map = F.binary_cross_entropy(pred, true, reduction="none")
        bce_loss = (bce_map * free).sum() / free.sum().clamp(min=1.0)

        pred_free, true_free = pred * free, true * free
        intersection = (pred_free * true_free).sum(dim=(1, 2, 3))
        denom = pred_free.sum(dim=(1, 2, 3)) + true_free.sum(dim=(1, 2, 3))
        dice_loss = (1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()

        n_free = free.sum(dim=(1, 2, 3)).clamp(min=1.0)
        length_loss = ((pred_free.sum(dim=(1, 2, 3)) - true_free.sum(dim=(1, 2, 3))) / n_free).abs().mean()

        total_loss = (
            self.bce_weight * bce_loss + self.dice_weight * dice_loss + self.length_weight * length_loss
        )
        return {
            "total_loss": total_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "length_loss": length_loss,
        }


class ColorBalanceLoss(nn.Module):
    """Global count term plus per-pixel saturation term, no per-pixel targets.

    Pixels live in ``[-1, 1]`` (black/white). The count term pushes the image
    mean toward ``2 r - 1`` for white fraction ``r``; the saturation term
    ``1 - x^2`` vanishes only at fully committed pixels.
    """

    def __init__(self, count_weight=1.0, saturation_weight=0.01, white_ratio=0.5):
        super().__init__()
        self.count_weight = count_weight
        self.saturation_weight = saturation_weight
        self.target_mean = 2.0 * white_ratio - 1.0
        self.white_ratio = white_ratio

    def forward(self, predictions, targets=None, condition=None):
        gray = predictions[:, :1].clamp(-1.0, 1.0)
        count_loss = ((gray.mean(dim=(1, 2, 3)) - self.target_mean) ** 2).mean()
        saturation_loss = (1.0 - gray.square()).mean()
        total_loss = self.count_weight * count_loss + self.saturation_weight * saturation_loss

        metrics = color_balance_metrics(predictions, self.white_ratio)
        return {
            "total_loss": total_loss,
            "count_loss": count_loss,
            "saturation_loss": saturation_loss,
            "imbalance": metrics["imbalance"].mean(),
            "accuracy": metrics["exact"].mean(),
        }


def color_balance_metrics(predictions, white_ratio=0.5):
    """Per-sample absolute white-count error and exact-ratio indicator."""
    gray = predictions[:, :1].detach().float()
    white = (gray >= 0.0).sum(dim=(1, 2, 3)).float()
    target = round(gray.shape[-2] * gray.shape[-1] * white_ratio)
    imbalance = (white - target).abs()
    return {"imbalance": imbalance, "exact": (imbalance == 0).float()}


def create_loss(config):
    """The loss named by ``EXTENSIONS.SPATIAL_REASONING.LOSS``, weighted by its task settings."""
    from .config import settings

    section = settings(config)
    if section.LOSS == "sudoku_constraint":
        s = section.SUDOKU
        return SudokuConstraintLoss(
            constraint_weight=s.CONSTRAINT_WEIGHT, given_weight=s.GIVEN_WEIGHT,
            onehot_weight=s.ONEHOT_WEIGHT, softmax_temperature=s.SOFTMAX_TEMPERATURE,
        )
    if section.LOSS == "maze_path":
        s = section.MAZE
        return MazeDiceBCELoss(bce_weight=s.BCE_WEIGHT, dice_weight=s.DICE_WEIGHT, length_weight=s.LENGTH_WEIGHT)
    if section.LOSS == "color_balance":
        s = section.COLOR_BALANCE
        return ColorBalanceLoss(count_weight=s.COUNT_WEIGHT, saturation_weight=s.SATURATION_WEIGHT, white_ratio=s.WHITE_RATIO)
    raise ValueError("EXTENSIONS.SPATIAL_REASONING.LOSS is not set.")
