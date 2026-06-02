from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
import torch.nn as nn


RolloutValue = torch.Tensor | float | int


@dataclass(frozen=True)
class StepContext:
    """Structured data exposed to rollout observers for one CA step."""

    step_index: int
    previous_state: torch.Tensor
    next_state: torch.Tensor
    condition: torch.Tensor | None
    ca_model: nn.Module
    freeze_channels: int | None = None
    dx: torch.Tensor | None = None


@dataclass
class StepObserverOutput:
    """Optional losses, metrics, or state snapshots produced by a step observer."""

    losses: Mapping[str, RolloutValue] = field(default_factory=dict)
    metrics: Mapping[str, RolloutValue] = field(default_factory=dict)
    collected_state: torch.Tensor | None = None


@dataclass
class RolloutOutput:
    """Structured metadata collected during rollout."""

    final_state: torch.Tensor
    step_losses: list[dict[str, RolloutValue]] = field(default_factory=list)
    losses: dict[str, RolloutValue] = field(default_factory=dict)
    step_metrics: list[dict[str, RolloutValue]] = field(default_factory=list)
    metrics: dict[str, RolloutValue] = field(default_factory=dict)
    states: list[torch.Tensor] = field(default_factory=list)

    def add_step_output(self, output: StepObserverOutput) -> None:
        if output.losses:
            self.step_losses.append(dict(output.losses))
        if output.metrics:
            self.step_metrics.append(dict(output.metrics))
        if output.collected_state is not None:
            self.states.append(output.collected_state)

    def aggregate(self) -> None:
        self.losses = _aggregate_named_values(self.step_losses)
        self.metrics = _aggregate_named_values(self.step_metrics)


StepObserver = Callable[[StepContext], StepObserverOutput | None]


def _aggregate_named_values(
    per_step_values: list[dict[str, RolloutValue]],
) -> dict[str, RolloutValue]:
    values_by_name: dict[str, list[RolloutValue]] = {}
    for step_values in per_step_values:
        for name, value in step_values.items():
            values_by_name.setdefault(name, []).append(value)
    return {
        name: _mean_rollout_values(values)
        for name, values in values_by_name.items()
        if values
    }


def _mean_rollout_values(values: list[RolloutValue]) -> RolloutValue:
    tensor_values = [value for value in values if torch.is_tensor(value)]
    if tensor_values:
        ref = tensor_values[0]
        converted_values = [
            value
            if torch.is_tensor(value)
            else torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
            for value in values
        ]
        return torch.stack(converted_values).mean()
    return sum(float(value) for value in values) / len(values)


class Evolver(nn.Module, ABC):
    """Base interface for CA rollout strategies (template method).

    ``Evolver`` owns the entire rollout *scaffolding* — the iteration loop,
    activation checkpointing, intermediate-state logging, building the
    :class:`StepContext`, invoking step observers, and aggregating the
    :class:`RolloutOutput`. Concrete strategies implement only :meth:`step`,
    the single CA update, so they can change how conditions are prepared per
    step, add clocks, or alter the update schedule without re-implementing any
    of the observer/logging machinery.

    Override :meth:`forward` directly only for a fundamentally different loop
    structure; in that case reuse :meth:`_invoke_observers` to keep the
    observer contract centralized.
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

    @abstractmethod
    def step(
        self,
        ca_model: nn.Module,
        state: torch.Tensor,
        conds: torch.Tensor | None,
        step_index: int,
        iter_n: int,
        freeze_channels: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one CA update and return ``(new_state, dx)``.

        This is the only method a rollout strategy must implement. ``dx`` is
        the per-step update delta exposed to observers via ``StepContext.dx``.
        """

    def _invoke_observers(
        self,
        observers: list[StepObserver],
        rollout_output: RolloutOutput | None,
        *,
        step_index: int,
        previous_state: torch.Tensor,
        new_state: torch.Tensor,
        conds: torch.Tensor | None,
        ca_model: nn.Module,
        freeze_channels: int | None,
        dx: torch.Tensor,
    ) -> None:
        """Build the StepContext and dispatch it to every observer (centralized)."""
        if not observers:
            return
        context = StepContext(
            step_index=step_index,
            previous_state=previous_state,
            next_state=new_state,
            condition=conds,
            ca_model=ca_model,
            freeze_channels=freeze_channels,
            dx=dx,
        )
        for observer in observers:
            observer_output = observer(context)
            if observer_output is None:
                continue
            if not isinstance(observer_output, StepObserverOutput):
                raise TypeError("Step observers must return None or StepObserverOutput.")
            rollout_output.add_step_output(observer_output)

    def forward(
        self,
        ca_model: nn.Module,
        state_in: torch.Tensor,
        conds: torch.Tensor | None,
        iter_n: int,
        logger=None,
        freeze_channels: int | None = None,
        logging: bool = False,
        step_observers: list[StepObserver] | None = None,
        return_rollout: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RolloutOutput]:
        state = state_in
        observers = list(step_observers or [])

        if observers and self.gradient_checkpointing:
            raise ValueError(
                "Step observers require TRAINING.GRADIENT_CHECKPOINTING=false "
                "for now because checkpoint_sequential does not expose each "
                "rollout step to observers."
            )

        rollout_output = (
            RolloutOutput(final_state=state_in)
            if return_rollout or observers
            else None
        )

        with torch.set_grad_enabled(ca_model.training):
            def evolve_step(step, current_state):
                previous_state = current_state
                new_state, dx = self.step(
                    ca_model, current_state, conds, step, iter_n, freeze_channels
                )
                if (
                    logging
                    and logger is not None
                    and step in self.intermediate_logging_steps
                ):
                    logger.add_state_log(step, new_state)
                self._invoke_observers(
                    observers,
                    rollout_output,
                    step_index=step,
                    previous_state=previous_state,
                    new_state=new_state,
                    conds=conds,
                    ca_model=ca_model,
                    freeze_channels=freeze_channels,
                    dx=dx,
                )
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

        if rollout_output is not None:
            rollout_output.final_state = state
            rollout_output.aggregate()

        if return_rollout:
            return state, rollout_output
        return state


class BaseEvolver(Evolver):
    """Default NCA rollout: reuse the same condition for every step.

    All rollout scaffolding (loop, checkpointing, intermediate logging,
    observers, aggregation) lives in :class:`Evolver`; this strategy only
    defines the per-step CA update.
    """

    def step(self, ca_model, state, conds, step_index, iter_n, freeze_channels):
        return ca_model(state, conds, freeze_channels=freeze_channels)
