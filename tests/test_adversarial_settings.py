from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nca.core.losses.loss_factory import LOSS_FN_REGISTRY
from nca.training.trainers.adversarial_trainer import AdversarialTrainer
from nca.training.training_utils import create_warmup_cosine_scheduler
from nca.utils.config import Config


class ScaleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, state):
        return state * self.scale


def make_trainer(*, training=None, adversarial=None, cond_dim=0):
    config = Config(
        DEVICE="cpu", COND_DIM=cond_dim,
        DATASET={"NAME": "emoji", "EMOJIS": ["x"]},
        MODEL={"CHANNEL_N": 4},
        TRAINING={"STEPS": 12, **(training or {})},
        ADVERSARIAL={"D_FEATURES": [4, 8], **(adversarial or {})},
    )
    # Avoid datasets, disk logging and a full CA rollout; use the real trainer step.
    trainer = AdversarialTrainer.__new__(AdversarialTrainer)
    trainer.config = config
    trainer.device = "cpu"
    trainer.ca_model = ScaleGenerator()
    trainer.optimizer = torch.optim.SGD(trainer.ca_model.parameters(), lr=0.01)
    trainer.accumulation_steps = config.TRAINING.GRADIENT_ACCUMULATION_STEPS
    trainer.current_step = 0
    trainer.logger = Mock()
    trainer._initialize_additional_components()

    def forward(state, condition, target, logging=False):
        prediction = trainer.ca_model(state)
        return prediction, prediction

    trainer.forward = forward
    return trainer


@pytest.fixture
def tiny_lpips(monkeypatch):
    created = []

    class TinyLPIPS(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.calls = []
            created.append(self)

        def forward(self, prediction, target):
            self.calls.append(prediction.shape)
            return (prediction - target).square().mean((1, 2, 3), keepdim=True)

    monkeypatch.setattr("nca.core.losses.loss_functions.lpips.LPIPS", TinyLPIPS)
    return created


@pytest.mark.parametrize("loss_key", ["mse", "l1"])
@pytest.mark.parametrize("weight", [0.0, 3.0])
def test_adversarial_reconstruction_honors_overflow_weight(loss_key, weight):
    trainer = make_trainer(training={
        "LOSS_FN": loss_key, "OVERFLOW_LOSS": True, "OVERFLOW_WEIGHT": weight,
    })
    result = trainer.recon_loss(torch.full((2, 4, 4, 4), 2.0), torch.zeros(2, 4, 4, 4))
    reconstruction = 4.0 if loss_key == "mse" else 2.0
    torch.testing.assert_close(result["total_loss"], torch.tensor(reconstruction + weight))


@pytest.mark.parametrize("gamma", [0.0, 0.1, 0.5, 1.0])
def test_critic_gamma_controls_cosine_endpoint_after_delayed_start(gamma):
    trainer = make_trainer(adversarial={
        "D_LEARNING_RATE": 0.01, "D_START_TRAINING": 2,
        "D_WARMUP_STEPS": 2, "D_GAMMA": gamma,
    })
    rates = [trainer.d_optimizer.param_groups[0]["lr"]]
    for _ in range(10):  # Twelve batches minus two batches before critic starts.
        trainer.d_optimizer.step()
        trainer.d_scheduler.step()
        rates.append(trainer.d_optimizer.param_groups[0]["lr"])
    assert rates[0] == 0
    assert rates[1] == pytest.approx(0.005)
    assert rates[2] == pytest.approx(0.01)
    assert rates[6] == pytest.approx(0.01 * (1 + gamma) / 2)
    assert rates[10] == pytest.approx(0.01 * gamma)


def test_default_cosine_scheduler_still_decays_to_zero():
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=0.01)
    scheduler = create_warmup_cosine_scheduler(optimizer, 0, 2)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0


@pytest.mark.parametrize("weight", [0.0, 0.25, 2.0])
@pytest.mark.parametrize("loss_key", ["mse", "lpips"])
def test_auxiliary_lpips_scales_generator_loss_and_gradient(weight, loss_key, tiny_lpips):
    trainer = make_trainer(
        training={"LOSS_FN": loss_key, "LPIPS_NET": "squeeze", "GRADIENT_ACCUMULATION_STEPS": 2},
        adversarial={"LPIPS_WEIGHT": weight, "D_START_TRAINING": 5, "RECON_WEIGHT": 0.5},
    )
    trainer._run_train_step(torch.ones(2, 4, 16, 16), None, torch.zeros(2, 4, 16, 16))
    expected_weight = 0.5 + weight
    # Both MSE and the tiny LPIPS equal scale**2; backward includes accumulation scaling.
    torch.testing.assert_close(trainer.ca_model.scale.grad, torch.tensor(2 * expected_weight))
    metrics = dict(call.args for call in trainer.logger.add_metric.call_args_list)
    assert metrics["g_total_loss"] == pytest.approx(4 * expected_weight)
    assert ("lpips_loss" in metrics) == (weight > 0)
    expected_networks = int(weight > 0 or loss_key == "lpips")
    assert len(tiny_lpips) == expected_networks
    if tiny_lpips:
        assert tiny_lpips[0].net == "squeeze"
        assert tiny_lpips[0].calls == [torch.Size([2, 3, 16, 16])]


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("condition_kind, seed_to_critic", [(None, False), ("vector", True), ("spatial", True)])
def test_all_critic_paths_use_downscaled_inputs(factor, condition_kind, seed_to_critic, tiny_lpips):
    trainer = make_trainer(
        adversarial={"D_DOWNSCALE_FACTOR": factor, "SEED_TO_CRITIC": seed_to_critic, "LPIPS_WEIGHT": 0.25},
        cond_dim=2 if condition_kind else 0,
    )
    inputs = []
    trainer.critic.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
    seed = torch.ones(2, 4, 32, 40)
    target = torch.full_like(seed, 0.5)
    condition = None
    if condition_kind == "vector":
        condition = torch.full((2, 2), 0.25)
    elif condition_kind == "spatial":
        condition = torch.full((2, 2, 16, 20), 0.25)
    before = [p.detach().clone() for p in trainer.critic.parameters()]
    trainer._run_train_step(seed, condition, target)

    channels = 4 + (2 if condition_kind else 0) + (4 if seed_to_critic else 0)
    assert len(inputs) == 4  # Real, fake, gradient penalty, generator adversarial loss.
    for critic_input in inputs:
        assert critic_input.shape == (2, channels, 32 // factor, 40 // factor)
        if condition_kind:
            assert torch.all(critic_input[:, 4:6] == 0.25)
        if seed_to_critic:
            assert torch.all(critic_input[:, -4:] == 1)
    assert torch.all(inputs[0][:, :4] == 0.5)
    assert torch.all(inputs[1][:, :4] == 2)
    assert torch.isfinite(trainer.ca_model.scale.grad)
    assert trainer.ca_model.scale.grad != 0
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.critic.parameters()))
    assert tiny_lpips[0].calls == [torch.Size([2, 3, 32, 40])]


@pytest.mark.parametrize("loss_key", sorted(LOSS_FN_REGISTRY))
def test_adversarial_step_supports_every_registered_loss(loss_key, tiny_lpips, monkeypatch):
    monkeypatch.setattr(
        "nca.core.losses.loss_functions.models.vgg16",
        lambda **kwargs: SimpleNamespace(features=nn.Sequential(nn.Identity())),
    )
    trainer = make_trainer(training={"LOSS_FN": loss_key})
    seed = torch.linspace(0.1, 1.2, 4)[None, :, None, None].expand(2, -1, 32, 32)
    target = torch.zeros_like(seed)
    target[:, 0] = 1  # Valid one-hot targets for the classification losses, too.

    trainer._run_train_step(seed, None, target)

    metrics = dict(call.args for call in trainer.logger.add_metric.call_args_list)
    assert torch.isfinite(torch.tensor(metrics["g_total_loss"]))
    assert trainer.ca_model.scale.grad is not None
    assert torch.isfinite(trainer.ca_model.scale.grad)
    assert trainer.ca_model.scale.grad != 0


def test_adversarial_trainer_accepts_new_registry_entries(monkeypatch):
    class CustomLoss(nn.Module):
        def forward(self, prediction, target):
            return {"total_loss": (prediction - target).square().mean()}

    monkeypatch.setitem(LOSS_FN_REGISTRY, "custom_test_loss", lambda config: CustomLoss())
    trainer = make_trainer(training={"LOSS_FN": "custom_test_loss"})
    assert isinstance(trainer.recon_loss, CustomLoss)
    trainer._run_train_step(torch.ones(2, 4, 32, 32), None, torch.zeros(2, 4, 32, 32))
    assert torch.isfinite(trainer.ca_model.scale.grad)
