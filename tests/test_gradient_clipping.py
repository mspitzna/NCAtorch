from types import SimpleNamespace

import torch

from nca.training.trainers.base_trainer import BaseTrainer


def _trainer_with_clip_norm(max_norm):
    return SimpleNamespace(
        config=SimpleNamespace(
            TRAINING=SimpleNamespace(GRADIENT_CLIPPING_NORM=max_norm)
        )
    )


def test_zero_gradient_clipping_norm_leaves_gradients_unchanged():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.tensor([3.0, 4.0, 5.0])
    original_gradient = parameter.grad.clone()

    result = BaseTrainer._clip_gradients(
        _trainer_with_clip_norm(0), [parameter]
    )

    assert result is None
    torch.testing.assert_close(parameter.grad, original_gradient)


def test_positive_gradient_clipping_norm_clips_gradients():
    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.tensor([3.0, 4.0, 5.0])

    total_norm = BaseTrainer._clip_gradients(
        _trainer_with_clip_norm(1.0), [parameter]
    )

    torch.testing.assert_close(total_norm, torch.sqrt(torch.tensor(50.0)))
    assert torch.linalg.vector_norm(parameter.grad) <= 1.0
