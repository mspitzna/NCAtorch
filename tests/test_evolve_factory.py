import pytest
import torch
import torch.nn as nn

from nca.training.evolve import BaseEvolver, RolloutOutput, StepObserverOutput
from nca.training.evolve_factory import create_evolver
from nca.training.trainers.base_trainer import BaseTrainer
from nca.training.trainers.image_gen_trainer import ImageGenTrainer
from nca.utils.config import Config


class AddOneCA(nn.Module):
    def forward(self, x, cond=None, freeze_channels=None):
        return x + 1.0


class ParamAddCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = nn.Parameter(torch.ones(()))

    def forward(self, x, cond=None, freeze_channels=None):
        return x + self.delta


class DummyLoader:
    def __iter__(self):
        return iter([])


class ForwardOnlyTrainer(BaseTrainer):
    def _initialize_additional_components(self):
        pass

    def _compute_losses(self, initial_state, cond, target, logging=False):
        raise NotImplementedError


def make_test_config(tmp_path, gradient_checkpointing=False):
    return Config(
        DATASET={"NAME": "cifar10"},
        DEVICE="cpu",
        FOLDER_NAME=str(tmp_path / "logs"),
        COND_DIM=0,
        TRAINING={
            "ITER_N_MIN": 3,
            "ITER_N_MAX": 3,
            "INTERMEDIATE_LOGGING_STEPS": [],
            "GRADIENT_CHECKPOINTING": gradient_checkpointing,
        },
    )


def make_forward_trainer(tmp_path):
    return ForwardOnlyTrainer(
        ca_model=ParamAddCA(),
        dataloader=DummyLoader(),
        config=make_test_config(tmp_path),
        config_path=__file__,
    )


class RecordingLogger:
    def __init__(self):
        self.logs = {}

    def add_state_log(self, step, state):
        self.logs[step] = state.detach().clone()


def test_base_evolver_matches_plain_rollout():
    evolver = BaseEvolver(intermediate_logging_steps=[1])
    ca_model = AddOneCA()
    state = torch.zeros(2, 4, 8, 8)
    logger = RecordingLogger()

    out = evolver(
        ca_model=ca_model,
        state_in=state,
        conds=None,
        iter_n=3,
        logger=logger,
        logging=True,
    )

    assert torch.allclose(out, torch.full_like(state, 3.0))
    assert 1 in logger.logs
    assert torch.allclose(logger.logs[1], torch.full_like(state, 2.0))


def test_create_evolver_uses_default_base_mode():
    config = Config(DATASET={"NAME": "cifar10"})

    evolver = create_evolver(config)

    assert isinstance(evolver, BaseEvolver)


def test_base_evolver_return_rollout_returns_structured_output():
    evolver = BaseEvolver()
    ca_model = AddOneCA()
    state = torch.zeros(1, 1, 2, 2)

    final_state, rollout = evolver(
        ca_model=ca_model,
        state_in=state,
        conds=None,
        iter_n=3,
        return_rollout=True,
    )

    assert torch.allclose(final_state, torch.full_like(state, 3.0))
    assert isinstance(rollout, RolloutOutput)
    assert torch.allclose(rollout.final_state, final_state)
    assert rollout.losses == {}


def test_step_observers_are_called_and_losses_are_aggregated():
    evolver = BaseEvolver()
    ca_model = AddOneCA()
    state = torch.zeros(1, 1, 2, 2)
    calls = []

    def observer(context):
        calls.append(context.step_index)
        return StepObserverOutput(
            losses={"aux_loss": context.next_state.mean()},
            metrics={"step_index": context.step_index},
            collected_state=context.next_state.detach().clone(),
        )

    final_state, rollout = evolver(
        ca_model=ca_model,
        state_in=state,
        conds=None,
        iter_n=3,
        step_observers=[observer],
        return_rollout=True,
    )

    assert calls == [0, 1, 2]
    assert torch.allclose(final_state, torch.full_like(state, 3.0))
    assert len(rollout.step_losses) == 3
    assert torch.allclose(rollout.losses["aux_loss"], torch.tensor(2.0))
    assert rollout.metrics["step_index"] == pytest.approx(1.0)
    assert len(rollout.states) == 3


def test_step_observers_with_gradient_checkpointing_raise_clear_error():
    evolver = BaseEvolver(gradient_checkpointing=True)
    ca_model = AddOneCA()
    state = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match="Step observers require.*GRADIENT_CHECKPOINTING=false"):
        evolver(
            ca_model=ca_model,
            state_in=state,
            conds=None,
            iter_n=3,
            step_observers=[lambda context: None],
        )


def test_trainer_forward_existing_api_returns_two_tuple(tmp_path):
    trainer = make_forward_trainer(tmp_path)
    state = torch.zeros(1, 1, 2, 2)

    result = trainer.forward(state, cond=None, target=state)

    assert isinstance(result, tuple)
    assert len(result) == 2
    prediction_image, final_state = result
    assert torch.allclose(prediction_image, torch.full_like(state, 3.0))
    assert torch.allclose(final_state, prediction_image)


def test_trainer_forward_return_rollout_returns_three_tuple(tmp_path):
    trainer = make_forward_trainer(tmp_path)
    state = torch.zeros(1, 1, 2, 2)

    result = trainer.forward(state, cond=None, target=state, return_rollout=True)

    assert isinstance(result, tuple)
    assert len(result) == 3
    prediction_image, final_state, rollout = result
    assert torch.allclose(prediction_image, torch.full_like(state, 3.0))
    assert torch.allclose(final_state, prediction_image)
    assert isinstance(rollout, RolloutOutput)
    assert torch.allclose(rollout.final_state, final_state)


def test_existing_image_gen_trainer_train_step_still_returns_two_tensors(tmp_path):
    trainer = ImageGenTrainer(
        ca_model=ParamAddCA(),
        dataloader=DummyLoader(),
        config=make_test_config(tmp_path),
        config_path=__file__,
    )
    state = torch.zeros(1, 1, 2, 2)
    target = torch.ones_like(state)

    prediction_image, final_state = trainer._run_train_step(
        state,
        cond=None,
        target=target,
    )

    assert torch.allclose(prediction_image, torch.full_like(state, 3.0))
    assert torch.allclose(final_state, prediction_image)
