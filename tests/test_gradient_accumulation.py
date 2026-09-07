from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nca.training.trainers.base_trainer import BaseTrainer
from nca.training.trainers.adversarial_trainer import AdversarialTrainer
from nca.utils.config import Config


class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, state):
        return self.weight * state


class ScalarCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images):
        return self.weight * images.mean((1, 2, 3))


class QuietTraining:
    def commit_logs(self, *args, **kwargs):
        pass

    def add_img_logs(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass


class RegressionTrainer(QuietTraining, BaseTrainer):
    def _initialize_additional_components(self):
        pass

    def _compute_losses(self, initial_state, cond, target, logging=False):
        prediction = self.ca_model(initial_state)
        return prediction, prediction, {"total_loss": (prediction - target).square().mean()}


class GANTrainer(QuietTraining, AdversarialTrainer):
    def forward(self, state, condition, target, logging=False):
        prediction = self.ca_model(state)
        return prediction, prediction

    def _run_train_step(self, state, condition, target, logging=False):
        contributes = self._should_train_generator(self.current_step)
        weight_before = self.ca_model.weight.item()
        result = super()._run_train_step(state, condition, target, logging=logging)
        if contributes:
            # For this scalar model with unit seeds and zero targets: d(MSE)/dw = 2w.
            derivative = 2 * weight_before
            if self.current_step - 1 >= self.d_start_training and self.adv_weight > 0:
                derivative -= self.adv_weight * self.critic.weight.item()
            self.expected_derivatives.append(derivative)
        return result


@pytest.fixture
def make_trainer(monkeypatch, tmp_path):
    logger = Mock()
    logger.get_output_folder.return_value = str(tmp_path)
    monkeypatch.setattr("nca.training.trainers.base_trainer.Logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr("nca.training.trainers.adversarial_trainer.Critic", ScalarCritic)

    def make(targets, accumulation, *, adversarial=None, training=None):
        config = Config(
            DEVICE="cpu", COND_DIM=0, DATASET={"EMOJIS": ["x"]},
            MODEL={"CHANNEL_N": 4}, LOGGING={"INTERMEDIATE_LOGGING_STEPS": []},
            TRAINING={
                "STEPS": len(targets), "GRADIENT_ACCUMULATION_STEPS": accumulation,
                "GRADIENT_CLIPPING_NORM": 0, "LEARNING_RATE": 0.05,
                "LR_SCHEDULE_MODE": "constant", "WARMUP_STEPS": 0,
                **(training or {}),
            },
            ADVERSARIAL=adversarial or {},
        )
        batches = [
            (torch.ones(2, 4, 4, 4), torch.zeros(2), torch.full((2, 4, 4, 4), float(target)))
            for target in targets
        ]
        cls = RegressionTrainer if adversarial is None else GANTrainer
        trainer = cls(ScalarModel(), batches, config, __file__)
        trainer.expected_derivatives = []
        trainer.applied_gradients = []
        trainer.rates_after_batches = []
        trainer.optimizer.register_step_pre_hook(
            lambda optimizer, args, kwargs: trainer.applied_gradients.append(trainer.ca_model.weight.grad.item())
        )
        trainer._on_step_end = lambda step: trainer.rates_after_batches.append(trainer.optimizer.param_groups[0]["lr"])
        return trainer

    return make


@pytest.mark.parametrize("accumulation, targets", [(1, [0, 1, 2]), (3, [0, 1, 2, 3, 4]), (10, [0, 1, 2])])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_accumulation_matches_explicit_group_averages(make_trainer, accumulation, targets, mixed_precision):
    trainer = make_trainer(targets, accumulation, training={"MIXED_PRECISION": mixed_precision})
    reference = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.Adam([reference], lr=0.05)
    expected_gradients = []
    for start in range(0, len(targets), accumulation):
        optimizer.zero_grad()
        group = torch.tensor(targets[start:start + accumulation], dtype=torch.float32)
        (reference - group).square().mean().backward()
        expected_gradients.append(reference.grad.item())
        optimizer.step()

    assert trainer.train() == 0
    assert trainer.applied_gradients == pytest.approx(expected_gradients)
    torch.testing.assert_close(trainer.ca_model.weight, reference)
    assert trainer.ca_model.weight.grad is None or trainer.ca_model.weight.grad == 0


@pytest.mark.parametrize("n_critic, accumulation, start, adv_weight", [
    (2, 2, 0, 1.0), (3, 2, 2, 1.0), (3, 10, 0, 1.0), (3, 2, 0, 0.0),
])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_gan_accumulates_only_generator_batches_and_flushes_tail(
    make_trainer, n_critic, accumulation, start, adv_weight, mixed_precision
):
    trainer = make_trainer(
        [0] * 8, accumulation,
        adversarial={"D_N_CRITIC": n_critic, "D_START_TRAINING": start, "ADV_WEIGHT": adv_weight},
        training={"MIXED_PRECISION": mixed_precision},
    )
    critic_updates = []
    trainer.d_optimizer.register_step_pre_hook(lambda *args: critic_updates.append(True))
    assert trainer.train() == 0

    eligible = [i for i in range(8) if adv_weight == 0 or i < start or i % n_critic == 0]
    assert len(trainer.expected_derivatives) == len(eligible)
    expected_gradients = [
        sum(trainer.expected_derivatives[i:i + accumulation]) / len(trainer.expected_derivatives[i:i + accumulation])
        for i in range(0, len(eligible), accumulation)
    ]
    assert trainer.applied_gradients == pytest.approx(expected_gradients, rel=1e-5)
    assert len(critic_updates) == (8 - start if adv_weight else 0)
    last_update_batch = 8 if len(eligible) % accumulation else eligible[-1] + 1
    assert trainer.lr_scheduler.last_epoch == last_update_batch


@pytest.mark.parametrize("mode, extra, expected_rates", [
    ("step", {"MILESTONES": [3, 5], "LR_GAMMA": 0.1}, [0.05, 0.005, 0.0005, 0.0005]),
    ("constant", {"WARMUP_STEPS": 4}, [0.025, 0.05, 0.05, 0.05]),
    ("cosine", {"WARMUP_STEPS": 2}, [0.05, 0.0375, 0.0125, 0.0]),
    ("wsd", {"WARMUP_STEPS": 2, "WSD_DECAY_RATIO": 0.5, "WSD_MIN_LR_RATIO": 0.2}, [0.05, 0.05, 0.03, 0.01]),
])
def test_scheduler_uses_training_batch_clock_with_accumulation(make_trainer, mode, extra, expected_rates):
    trainer = make_trainer([0] * 8, 2, training={"LR_SCHEDULE_MODE": mode, **extra})
    assert trainer.train() == 0
    assert trainer.rates_after_batches[1::2] == pytest.approx(expected_rates)
    assert trainer.lr_scheduler.last_epoch == 8


def test_amp_skipped_update_clears_gradients_without_advancing_scheduler(make_trainer):
    trainer = make_trainer([0, 0], 2, training={"MIXED_PRECISION": True})
    trainer.scaler.scale(trainer.ca_model.weight * float("inf")).backward()
    scale_before = trainer.scaler.get_scale()

    assert not trainer._step_generator(1, 1)
    assert trainer.lr_scheduler.last_epoch == 0
    assert trainer.ca_model.weight.item() == 2.0
    assert trainer.scaler.get_scale() < scale_before

    trainer.scaler.scale(trainer.ca_model.weight / 2).backward()
    assert trainer._step_generator(1, 2)
    assert trainer.applied_gradients == pytest.approx([1.0])
    assert trainer.lr_scheduler.last_epoch == 2
