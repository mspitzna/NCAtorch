from __future__ import annotations

import functools
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Evolver(nn.Module, ABC):
    """Base interface for CA rollout strategies.

    Evolvers own the per-step rollout loop. Different implementations can
    change how conditions are prepared at each step, add clocks, alter update
    schedules, or customize checkpointing without changing trainer logic.
    """

    @abstractmethod
    def forward(
        self,
        ca_model: nn.Module,
        state_in: torch.Tensor,
        conds: torch.Tensor | None,
        iter_n: int,
        logger=None,
        freeze_channels: int | None = None,
        logging: bool = False,
    ) -> torch.Tensor:
        pass


class BaseEvolver(Evolver):
    """Default NCA rollout.

    This preserves the previous ``BaseTrainer._evolve`` behavior: reuse the
    same condition for every step, optionally log intermediate states, and use
    activation checkpointing when configured.
    """

    def __init__(
        self,
        gradient_checkpointing: bool = False,
        checkpoint_segments: int = 16,
        intermediate_logging_steps: list[int] | None = None,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_segments = checkpoint_segments
        self.intermediate_logging_steps = set(intermediate_logging_steps or [])

    def forward(
        self,
        ca_model: nn.Module,
        state_in: torch.Tensor,
        conds: torch.Tensor | None,
        iter_n: int,
        logger=None,
        freeze_channels: int | None = None,
        logging: bool = False,
    ) -> torch.Tensor:
        state = state_in

        with torch.set_grad_enabled(ca_model.training):
            def evolve_step(step, current_state):
                new_state = ca_model(
                    current_state,
                    conds,
                    freeze_channels=freeze_channels,
                )
                if (
                    logging
                    and logger is not None
                    and step in self.intermediate_logging_steps
                ):
                    logger.add_state_log(step, new_state)
                return new_state

            if not self.gradient_checkpointing:
                for step in range(iter_n):
                    state = evolve_step(step, state)
            else:
                layers = [functools.partial(evolve_step, i) for i in range(iter_n)]
                state = torch.utils.checkpoint.checkpoint_sequential(
                    layers,
                    self.checkpoint_segments,
                    state,
                    use_reentrant=False,
                )

        return state
