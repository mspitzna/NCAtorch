import torch.nn as nn
from abc import ABC, abstractmethod


class UpdateModelBase(nn.Module, ABC):
    """Abstract Base Class for all update models."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x):
        pass


class SimpleMLPUpdate(UpdateModelBase):
    """A simple multi-layer perceptron (MLP) using 1x1 convolutions."""
    def __init__(self, in_channels, out_channels, hidden_channels=[80], activation=nn.LeakyReLU, final_activation=False, **kwargs):
        super().__init__(in_channels, out_channels)
        layers = []
        current_in = in_channels
        for h_dim in hidden_channels:
            layers.append(nn.Conv2d(current_in, h_dim, 1))
            layers.append(activation(0.2))
            current_in = h_dim
        layers.append(nn.Conv2d(current_in, out_channels, 1))
        if final_activation:
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ResNetUpdate(UpdateModelBase):
    """An update model using residual connections for deeper, stable networks."""
    class ResidualBlock(nn.Module):
        def __init__(self, channels, activation=nn.LeakyReLU):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 1)
            self.act = activation(0.2)
            self.conv2 = nn.Conv2d(channels, channels, 1)
        def forward(self, x):
            residual = x
            out = self.act(self.conv1(x))
            out = self.conv2(out)
            return self.act(out + residual)

    def __init__(self, in_channels, out_channels, hidden_channels=128, num_blocks=2, activation=nn.LeakyReLU, final_activation=False, **kwargs):
        super().__init__(in_channels, out_channels)

        # check if hidden_channels is a list or a single integer
        if isinstance(hidden_channels, list):
            hidden_channels = hidden_channels[0]
        elif not isinstance(hidden_channels, int):
            raise ValueError("hidden_channels must be an integer or a list of integers.")

        self.initial_conv = nn.Conv2d(in_channels, hidden_channels, 1)
        self.res_blocks = nn.Sequential(*[self.ResidualBlock(hidden_channels, activation) for _ in range(num_blocks)])
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, 1)
        self.final_activation = nn.Tanh() if final_activation else nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return self.final_activation(x)