from types import SimpleNamespace

import pytest
import torch

from nca.training.trainers.base_trainer import BaseTrainer
from nca.utils.config import TrainingConfig


def trainer_with_bounds(minimum, maximum):
    return SimpleNamespace(config=SimpleNamespace(
        TRAINING=TrainingConfig(ITER_N_MIN=minimum, ITER_N_MAX=maximum)
    ))


@pytest.mark.parametrize("minimum, maximum", [(2, 3), (1, 4), (32, 64)])
def test_rollout_length_includes_both_endpoints(minimum, maximum):
    trainer = trainer_with_bounds(minimum, maximum)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        lengths = {BaseTrainer.get_iter_range(trainer) for _ in range(1024)}

    assert lengths == set(range(minimum, maximum + 1))


def test_fixed_rollout_length_does_not_consume_randomness():
    trainer = trainer_with_bounds(3, 3)
    rng_state = torch.get_rng_state()
    assert BaseTrainer.get_iter_range(trainer) == 3
    assert torch.equal(torch.get_rng_state(), rng_state)
