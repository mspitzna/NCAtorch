"""Tests for the config-driven logging-observer framework."""

import pytest
import torch
import torch.nn as nn

from nca.training.evolve import StepContext
from nca.training.observers import (
    LOGGING_OBSERVER_REGISTRY,
    IterationStatsObserver,
    LoggingObserver,
    create_logging_observers,
)
from nca.training.trainers.base_trainer import BaseTrainer
from nca.utils.config import Config


class ParamAddCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = nn.Parameter(torch.ones(()))

    def forward(self, x, cond=None, freeze_channels=None):
        dx = self.delta.expand_as(x)
        return x + dx, dx


class DummyLoader:
    def __iter__(self):
        return iter([])


class ForwardOnlyTrainer(BaseTrainer):
    def _initialize_additional_components(self):
        pass

    def _compute_losses(self, initial_state, cond, target, logging=False):
        raise NotImplementedError


class FakeLogger:
    def __init__(self):
        self.use_wandb = True
        self.logged = {}

    def wandb_log(self, data, step):
        self.logged.setdefault(step, {}).update(data)


def make_config(tmp_path, observers):
    return Config(
        DATASET={"NAME": "cifar10"},
        DEVICE="cpu",
        LOGGING={
            "FOLDER_NAME": str(tmp_path / "logs"),
            "INTERMEDIATE_LOGGING_STEPS": [],
            "OBSERVERS": observers,
        },
        COND_DIM=0,
        TRAINING={"ITER_N_MIN": 3, "ITER_N_MAX": 3},
    )


def _context(step_index, value, channels=4):
    state = torch.full((1, channels, 8, 8), float(value))
    return StepContext(
        step_index=step_index,
        previous_state=state,
        next_state=torch.zeros_like(state),
        condition=None,
        ca_model=nn.Identity(),
        dx=torch.zeros_like(state),
    )


def test_registry_holds_observer_subclasses():
    assert "iteration_stats" in LOGGING_OBSERVER_REGISTRY
    assert issubclass(LOGGING_OBSERVER_REGISTRY["iteration_stats"], LoggingObserver)


def test_factory_builds_observers_from_config(tmp_path):
    cfg = make_config(tmp_path, [{"TYPE": "iteration_stats"}])
    observers = create_logging_observers(cfg)
    assert len(observers) == 1
    assert isinstance(observers[0], IterationStatsObserver)


def test_unknown_observer_type_rejected_at_config_time(tmp_path):
    with pytest.raises(Exception):
        make_config(tmp_path, [{"TYPE": "does_not_exist"}])


def test_observer_collects_per_channel_and_logs_then_clears(monkeypatch):
    import wandb

    channels = 4
    observer = IterationStatsObserver()
    for i in range(3):
        assert observer(_context(i, value=i, channels=channels)) is None  # side channel
    assert observer._iterations == [0, 1, 2]
    # one per-channel [C] vector stored per iteration, for each stat
    assert len(observer._per_channel["mean"]) == 3
    assert observer._per_channel["mean"][1].shape == (channels,)

    monkeypatch.setattr(wandb, "Plotly", lambda fig: ("plotly", fig))
    logger = FakeLogger()
    observer.log(logger, step=10)

    assert set(logger.logged[10]) == {
        "IterationStats/min",
        "IterationStats/max",
        "IterationStats/mean",
        "IterationStats/std",
    }
    kind, fig = logger.logged[10]["IterationStats/mean"]
    assert kind == "plotly"
    # one trace per channel plus the bold mean-over-channels line
    assert len(fig.data) == channels + 1
    assert fig.data[-1].name == "mean(ch)"
    assert observer._iterations == []  # buffer cleared after logging


def test_observer_is_noop_without_wandb():
    observer = IterationStatsObserver()
    observer.observe(_context(0, value=1.0))

    class NoWandb:
        use_wandb = False

        def wandb_log(self, data, step):  # pragma: no cover - must not be called
            raise AssertionError("wandb_log called while W&B disabled")

    observer.log(NoWandb(), step=1)
    assert observer._iterations == []  # still cleared


def test_trainer_attaches_observers_only_on_logging_steps(tmp_path):
    cfg = make_config(tmp_path, [{"TYPE": "iteration_stats"}])
    trainer = ForwardOnlyTrainer(
        ca_model=ParamAddCA(),
        dataloader=DummyLoader(),
        config=cfg,
        config_path=__file__,
    )
    assert len(trainer.logging_observers) == 1
    state = torch.zeros(1, 4, 8, 8)

    # Non-logging step: observers are not attached, nothing collected.
    trainer.forward(state, cond=None, target=None, logging=False)
    assert trainer.logging_observers[0]._iterations == []

    # Logging step: observer collects one record per CA iteration (iter_n=3).
    trainer.forward(state, cond=None, target=None, logging=True)
    assert len(trainer.logging_observers[0]._iterations) == 3
