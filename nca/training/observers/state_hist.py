from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from nca.training.observers.base import LoggingObserver


class StateHistogramObserver(LoggingObserver):
    """
    Log per-channel histograms of an NCA state.

    The state is expected to have shape [B, C, H, W].

    During ``observe()``, one histogram is calculated for every channel and
    stored internally. During ``log()``, one W&B image is created for every
    channel. Each image contains the histograms from the collected CA
    iterations in the same overlapping/ridgeline format as the reference
    implementation.
    """

    def __init__(
        self,
        n_bins: int = 100,
        hist_range: tuple[float, float] | None = None,
        max_histograms: int = 30,
    ) -> None:
        self.n_bins = n_bins
        self.hist_range = hist_range
        self.max_histograms = max_histograms

        # Indexed as:
        #   self._histograms[channel][iteration]
        #
        # Every histogram is:
        #   {
        #       "step": int,
        #       "counts": np.ndarray,
        #       "limits": np.ndarray,
        #   }
        self._histograms: list[list[dict[str, Any]]] = []
        self._iterations: list[int] = []

    def reset(self) -> None:
        self._histograms = []
        self._iterations = []

    def observe(self, context) -> None:
        """
        Calculate and store one histogram for every channel.

        ``context.previous_state`` is expected to have shape [B, C, H, W].
        Batch and spatial dimensions are flattened, leaving one distribution
        per channel.
        """
        x = context.previous_state.detach().float()

        if x.ndim != 4:
            raise ValueError(
                "WeightHistogramObserver expects state with shape "
                f"[B, C, H, W], got {tuple(x.shape)}"
            )

        _, n_channels, _, _ = x.shape

        # Initialize the per-channel storage on the first observation.
        if not self._histograms:
            self._histograms = [[] for _ in range(n_channels)]
        elif len(self._histograms) != n_channels:
            raise ValueError(
                "Number of channels changed during observation: "
                f"{len(self._histograms)} -> {n_channels}"
            )

        step = int(context.step_index)
        self._iterations.append(step)

        # Move only the data needed for histogram calculation to CPU.
        x = x.cpu()

        for channel in range(n_channels):
            values = x[:, channel, :, :].reshape(-1).numpy()

            # Remove NaN/Inf values so they cannot corrupt the histogram.
            values = values[np.isfinite(values)]

            if self.hist_range is None:
                # A fixed range is needed for comparable histograms.
                # The implementation uses the range of the current observation here.
                # If you want exactly the same x-axis across all logging phases, provide hist_range explicitly.
                if values.size == 0:
                    limits = np.linspace(-1.0, 1.0, self.n_bins + 1)
                else:
                    value_min = float(values.min())
                    value_max = float(values.max())

                    if value_min == value_max:
                        delta = max(abs(value_min) * 0.05, 1e-6)
                        value_min -= delta
                        value_max += delta

                    limits = np.linspace(
                        value_min,
                        value_max,
                        self.n_bins + 1,
                    )
            else:
                limits = np.linspace(
                    self.hist_range[0],
                    self.hist_range[1],
                    self.n_bins + 1,
                )

            counts, _ = np.histogram(
                values,
                bins=limits,
                density=False,
            )

            self._histograms[channel].append(
                {
                    "step": step,
                    "counts": counts,
                    "limits": limits,
                }
            )

    def log(self, logger, step: int) -> None:
        """
        Create and log one histogram image per state channel.
        """
        if not self._histograms or not logger.use_wandb:
            self.reset()
            return

        panels = {}

        for channel, histograms in enumerate(self._histograms):


            fig = self._plot_histograms(
                histograms,
                channel=channel,
            )

            panels[f"StateHistogram/channel_{channel}"] = wandb.Image(fig)

            plt.close(fig)

        logger.wandb_log(
            panels,
            step=step,
        )

        self.reset()

    def _plot_histograms(
        self,
        histograms: list[dict[str, Any]],
        channel: int,
    ):
        """
        Create the overlapping density/ridgeline plot used for logging.
        """
        n_steps = len(histograms)

        if n_steps == 0:
            raise ValueError("Cannot plot an empty histogram list.")

        # Orange palette
        import seaborn as sns

        sns.set_theme(
            style="white",
            rc={
                "axes.facecolor": (0, 0, 0, 0),
            },
        )

        palette = sns.color_palette(
            "Oranges",
            max(n_steps, 20),
        )

        if len(palette) > 10:
            palette = palette[5:-5]

        fig, axes = plt.subplots(
            nrows=n_steps,
            ncols=1,
            figsize=(15, 0.4 * n_steps),
            sharex=True,
        )

        axes = np.atleast_1d(axes)

        for index, (ax, histogram) in enumerate(
            zip(axes, histograms)
        ):
            step = histogram["step"]
            counts = histogram["counts"]
            limits = histogram["limits"]

            bin_widths = np.diff(limits)
            total = counts.sum()

            if total > 0:
                density = (
                    counts
                    / total
                    / bin_widths
                )
            else:
                density = np.zeros_like(
                    counts,
                    dtype=float,
                )

            x = (
                limits[:-1]
                + limits[1:]
            ) / 2.0

            color = palette[
                int(
                    index
                    * (len(palette) - 1)
                    / max(n_steps - 1, 1)
                )
            ]

            # Distribution outline.
            ax.plot(
                x,
                density,
                color="white",
                linewidth=2,
                clip_on=False,
            )

            # Filled distribution.
            ax.fill_between(
                x,
                density,
                color=color,
            )

            # CA iteration label.
            ax.text(
                0.005,
                0.15,
                str(step),
                transform=ax.transAxes,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
            )

            ax.axhline(
                0,
                linewidth=2,
                color=color,
            )

            # Remove y-axis.
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_title("")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Only the final subplot gets an x-axis.
            if index != n_steps - 1:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False,
                )
                ax.spines["bottom"].set_visible(False)

        fig.subplots_adjust(
            hspace=-0.9,
            left=0.03,
            right=0.99,
            top=0.92,
            bottom=0.1,
        )

        fig.suptitle(
            f"State value distribution — channel {channel}",
            fontweight="bold",
        )

        return fig