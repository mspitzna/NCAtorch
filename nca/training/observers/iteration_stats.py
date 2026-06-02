"""Minimal :class:`LoggingObserver`: state value range across CA iterations.

For every CA rollout step it records four summary statistics of the state fed
into that step — ``min``, ``max``, ``mean``, ``std`` — **per channel**. On the
logging phase it logs **one W&B panel per statistic** as an interactive
``wandb.Plotly`` line chart: x = CA iteration, one line per channel (with a
legend) plus a bold mean-over-channels line. Logging the same keys every
``LOG_INTERVAL`` gives each panel a media step-slider, so the per-iteration
curves can be inspected at step 500, 1000, … to see whether the value range
stays stable across iterations or drifts.

W&B-only: no disk output.
"""

from __future__ import annotations

import plotly.graph_objects as go
import torch
import wandb

from nca.training.observers.base import LoggingObserver


# Order matters: it is reused for the per-iteration records and the per-stat panels.
_STATS = ("min", "max", "mean", "std")


def _per_channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduce ``[B, C, H, W]`` to a per-channel ``[C]`` vector for each stat."""
    return {
        "min": x.amin(dim=(0, 2, 3)),
        "max": x.amax(dim=(0, 2, 3)),
        "mean": x.mean(dim=(0, 2, 3)),
        "std": x.std(dim=(0, 2, 3)),
    }


class IterationStatsObserver(LoggingObserver):
    """Track per-channel min/max/mean/std of the state per CA iteration."""

    def __init__(self):
        self._iterations: list[int] = []
        # stat -> list of per-channel [C] tensors, one per observed iteration
        self._per_channel: dict[str, list[torch.Tensor]] = {s: [] for s in _STATS}

    def reset(self) -> None:
        self._iterations = []
        self._per_channel = {s: [] for s in _STATS}

    def observe(self, context) -> None:
        # previous_state is the input fed into this iteration: [B, C, H, W].
        x = context.previous_state.detach().float()
        self._iterations.append(context.step_index)
        for stat, values in _per_channel_stats(x).items():
            self._per_channel[stat].append(values)

    def log(self, logger, step: int) -> None:
        if not self._iterations or not logger.use_wandb:
            self.reset()
            return
        # One interactive panel per statistic; same keys every logging phase
        # => media step-slider over training steps (inspect at 500, 1000, ...).
        panels = {
            f"IterationStats/{stat}": wandb.Plotly(self._figure(stat))
            for stat in _STATS
        }
        logger.wandb_log(panels, step=step)
        self.reset()

    def _figure(self, stat: str) -> go.Figure:
        # [n_iters, C]: rows = iterations, columns = channels.
        matrix = torch.stack(self._per_channel[stat], dim=0).cpu().numpy()
        n_channels = matrix.shape[1]
        fig = go.Figure()
        for channel in range(n_channels):
            fig.add_scatter(
                x=self._iterations,
                y=matrix[:, channel],
                mode="lines",
                name=f"ch{channel}",
            )
        # Bold mean-over-channels line on top of the per-channel curves.
        fig.add_scatter(
            x=self._iterations,
            y=matrix.mean(axis=1),
            mode="lines",
            name="mean(ch)",
            line=dict(color="black", width=3),
        )
        fig.update_layout(
            title=f"state {stat} over CA iterations",
            xaxis_title="CA iteration",
            yaxis_title=stat,
        )
        return fig
