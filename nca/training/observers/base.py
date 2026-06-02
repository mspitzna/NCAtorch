"""Base class for config-driven diagnostic logging observers.

A :class:`LoggingObserver` is a small, self-contained diagnostic that hooks into
the CA rollout, collects whatever it needs, and logs it itself. Observers are
declared in ``LOGGING.OBSERVERS`` and instantiated through the observer registry
(see :mod:`nca.training.observers.registry`), so new diagnostics can be added
without touching the trainer or the logger.

The trainer drives the following lifecycle, **only on logging steps**:

1. ``reset()``                    — clear any per-rollout buffers
2. ``__call__(context)`` per step — collect data during the CA rollout
3. ``log(logger, step)``          — emit to W&B / console, then clear

Because ``__call__`` matches the ``StepObserver`` protocol
(``Callable[[StepContext], StepObserverOutput | None]``), an observer plugs
straight into the existing evolver step-observer mechanism. Observers are a
pure side channel: ``__call__`` returns ``None`` so nothing is aggregated into
the rollout loss/metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nca.training.evolve import StepContext, StepObserverOutput


class LoggingObserver(ABC):
    """Interface for diagnostic observers that log during the logging phase.

    Subclasses override :meth:`observe` (collection) and :meth:`log` (emission).
    Every ``StepContext`` already carries both ``previous_state``/``next_state``
    and the per-step update delta ``dx``, so observers can use any of them
    without extra plumbing.
    """

    def __call__(self, context: StepContext) -> StepObserverOutput | None:
        self.observe(context)
        return None  # side channel only — nothing aggregated into the rollout

    def observe(self, context: StepContext) -> None:
        """Collect data from a single CA rollout step. Override as needed."""

    def reset(self) -> None:
        """Clear per-rollout buffers before a fresh logging rollout."""

    @abstractmethod
    def log(self, logger, step: int) -> None:
        """Emit collected diagnostics to W&B / console, then clear buffers.

        Args:
            logger: The active :class:`~nca.training.logger.Logger`; use
                ``logger.use_wandb`` / ``logger.wandb_log(...)`` for W&B and
                ``print(...)`` for console output.
            step: The global training step the diagnostics belong to.
        """
