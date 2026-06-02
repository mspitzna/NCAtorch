"""Registry and factory for config-driven logging observers.

Maps a ``LOGGING.OBSERVERS[].TYPE`` key to a ``LoggingObserver`` subclass.

Adding a new observer:
  1. Implement it as a ``LoggingObserver`` subclass in this package.
  2. Add one entry to ``LOGGING_OBSERVER_REGISTRY``.
  3. Reference it from ``LOGGING.OBSERVERS`` in the YAML config, passing any
     constructor arguments under ``PARAMS``.
"""

from __future__ import annotations

from nca.training.observers.base import LoggingObserver
from nca.training.observers.iteration_stats import IterationStatsObserver


# key -> LoggingObserver subclass; instantiated as cls(**observer_cfg.PARAMS)
LOGGING_OBSERVER_REGISTRY: dict[str, type[LoggingObserver]] = {
    "iteration_stats": IterationStatsObserver,
}


def create_logging_observers(config) -> list[LoggingObserver]:
    """Instantiate the enabled logging observers declared in ``LOGGING.OBSERVERS``.

    Returns a fresh list each call; raises if an observer's ``PARAMS`` do not
    match its constructor (surfacing config mistakes early).
    """
    observers: list[LoggingObserver] = []
    for observer_cfg in config.LOGGING.OBSERVERS:
        observer_cls = LOGGING_OBSERVER_REGISTRY[observer_cfg.TYPE]
        observers.append(observer_cls(**observer_cfg.PARAMS))
    return observers
