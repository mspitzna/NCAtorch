from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateUpdater(nn.Module):
    """Base interface for state update modifiers.

    Each updater receives the current state and delta (dx), and returns the
    possibly modified state and dx.
    """

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state, dx


class StateUpdatePipeline(StateUpdater):
    def __init__(self, updaters: Iterable[StateUpdater]):
        super().__init__()
        self.updaters = nn.ModuleList(list(updaters))

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for updater in self.updaters:
            state, dx = updater(state, dx, **kwargs)
        return state, dx


class NoiseInjection(StateUpdater):
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sigma > 0.0:
            dx = dx + torch.randn_like(dx) * self.sigma
        return state, dx


class FireRateMask(StateUpdater):
    def __init__(self, fire_rate: float = 1.0):
        super().__init__()
        if not (0.0 <= fire_rate <= 1.0):
            raise ValueError("fire_rate must be between 0 and 1")
        self.fire_rate = fire_rate

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fire_rate < 1.0:
            update_mask = (torch.rand_like(state[:, :1, :, :]) <= self.fire_rate).float()
            dx = dx * update_mask
        return state, dx


class ApplyDelta(StateUpdater):
    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return state + dx, dx


class LivingMask(StateUpdater):
    def __init__(
        self,
        threshold: float = 0.1,
        max_pool_size: int = 3,
        alpha_channel: int = 3,
    ):
        super().__init__()
        self.threshold = threshold
        self.max_pool_size = max_pool_size
        self.alpha_channel = alpha_channel

    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        alpha = x[:, self.alpha_channel, :, :]
        max_pool = F.max_pool2d(
            alpha,
            kernel_size=self.max_pool_size,
            stride=1,
            padding=self.max_pool_size // 2,
        )
        return max_pool > self.threshold

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_state = kwargs.get("original_state")
        frozen_layers = kwargs.get("frozen_layers")

        if frozen_layers is not None:
            full_state = torch.cat([frozen_layers, state], dim=1)
        else:
            full_state = state

        if original_state is None:
            original_state = full_state

        pre_life = self._get_mask(original_state)
        post_life = self._get_mask(full_state)
        life_mask = pre_life & post_life

        life_mask = life_mask.unsqueeze(1).expand(-1, state.shape[1], -1, -1)
        state = state * life_mask.float()
        return state, dx


class ClampOutput(StateUpdater):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(
        self,
        state: torch.Tensor,
        dx: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.clamp(state, min=self.min_val, max=self.max_val), dx
