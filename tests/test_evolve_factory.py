import torch
import torch.nn as nn

from nca.training.evolve import BaseEvolver
from nca.training.evolve_factory import create_evolver
from nca.utils.config import Config


class AddOneCA(nn.Module):
    def forward(self, x, cond=None, freeze_channels=None):
        return x + 1.0


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
