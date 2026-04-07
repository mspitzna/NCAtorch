import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
import warnings

from .state_updates import (
    ApplyDelta,
    StateUpdatePipeline,
)


class CAModel(nn.Module, ABC):
    def __init__(
        self,
        device="cpu",
        use_positional_embeddings=False,
        img_height=32,
        img_width=32,
        perception_module=None,
        update_model_module=None,
        state_update_pipeline=None,
    ):
        super().__init__()
        self.device = device
        self.use_positional_embeddings = use_positional_embeddings

        # Learnable Positional Embeddings
        if self.use_positional_embeddings:
            # Create meshgrid for positional coordinates
            y_coords = torch.linspace(0, 1, img_height).view(1, 1, img_height, 1).expand(1, 1, img_height, img_width)
            x_coords = torch.linspace(0, 1, img_width).view(1, 1, 1, img_width).expand(1, 1, img_height, img_width)

            # Combine x and y into positional embeddings and set as learnable parameters
            pos_emb_init = torch.cat([x_coords, y_coords], dim=1)  # Shape: [1, 2, img_height, img_width]
            pos_emb_init = 2 * (pos_emb_init - 0.5)
            self.positional_embeddings = nn.Parameter(pos_emb_init)  # Set as learnable

            self.positional_scaling = nn.Parameter(torch.tensor(1.0))
        else:
            self.positional_embeddings = None

        # Initialize the perception module
        self.perception = perception_module

        # Initialize the delta model
        self.update_model = update_model_module

        # Initialize state update pipeline
        if state_update_pipeline is None:
            warnings.warn(
                "No state_update_pipeline provided; using default ApplyDelta-only pipeline.",
                stacklevel=2,
            )
            self.state_updater = self._build_default_state_updater()
        else:
            self.state_updater = state_update_pipeline

        self._init_weights()

        num_perception_params = sum(p.numel() for p in self.perception.parameters() if p.requires_grad)
        print(f"Perception has {num_perception_params:,} learnable parameters")


        num_dmodel_params = sum(p.numel() for p in self.update_model.parameters() if p.requires_grad)
        print(f"Update model has {num_dmodel_params:,} learnable parameters")

    def _init_weights(self):
        # Find the last Conv2d layer and reinitialize it with small random weights
        last_conv = None
        for _, module in self.update_model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module  # this will eventually be the last one encountered

        if last_conv is not None:
            print(f"Re-initializing last convolutional layer: {last_conv}")
            # Initialize with small random weights so the first updates have non-zero variance.
            torch.nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)
            if last_conv.bias is not None:
                torch.nn.init.normal_(last_conv.bias, mean=0.0, std=1e-3)

    def _build_default_state_updater(self):
        updaters = []
        updaters.append(ApplyDelta())
        return StateUpdatePipeline(updaters)


    def _prepare_input(self, x, cond):
        """
        Prepare the input tensor by handling the condition tensor.
        x: [batch, dim, height, width]
        cond: [batch, cond_dim] or [batch, cond_dim, height, width]
        """
        if cond is not None:
            if x.shape[0] != cond.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: x.shape[0]={x.shape[0]}, cond.shape[0]={cond.shape[0]}"
                )
            # If cond is already an image (4D), check spatial dims
            if cond.ndim == 4:
                if cond.shape[2:] != x.shape[2:]:
                    # adjust spation dimensions of cond to match x
                    cond = F.interpolate(cond, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                # Concatenate condition image to the input
                x = torch.cat([x, cond], dim=1)
            # If cond is a vector (2D), expand to image
            elif cond.ndim == 2:
                cond_expanded = cond.unsqueeze(-1).unsqueeze(-1)  # [B, cond_dim, 1, 1]
                cond_expanded = cond_expanded.expand(-1, -1, x.shape[2], x.shape[3])
                x = torch.cat([x, cond_expanded], dim=1)
            else:
                raise ValueError(
                    f"Condition tensor must be 2D or 4D, got shape {cond.shape}"
                )
        return x

    def forward(self, x, cond=None, step_size=1.0, freeze_channels=None, return_residuals=False):
        original_state_full = x
        frozen_layers = None
        if freeze_channels is not None:
            frozen_layers, state = x[:, :freeze_channels + 1].clone(), x[:, freeze_channels + 1:].clone()
        else:
            state = x.clone()  # No channels are frozen

        # 1) Prepare the input
        x = self._prepare_input(x, cond)

        # 2) Compute "grads"
        if self.use_positional_embeddings and self.positional_embeddings is not None:
            grads = self._perception_with_positional_embeddings(x)
        else:
            grads = self.perception(x)

        # 3) Concatenate current state + grads, then pass through dmodel
        dx = self.update_model(grads) * step_size

        # When some channels are frozen the state only contains the unfrozen portion,
        # so dx must be sliced to match before entering the update pipeline.
        if freeze_channels is not None:
            dx = dx[:, freeze_channels + 1:]

        # 4) Apply update pipeline (noise, fire-rate, add, living mask, clamp, etc.)
        state, dx = self.state_updater(
            state,
            dx,
            original_state=original_state_full,
            frozen_layers=frozen_layers,
        )

        if freeze_channels is not None:
            state = torch.cat([frozen_layers, state], dim=1)
        if return_residuals:
            return state, dx
        return state

    def _perception_with_positional_embeddings(self, x):
        """Generate positional embeddings and pass them only to perception."""
        # Append learnable positional embeddings to the input for perception
        constrained_scaling = torch.tanh(self.positional_scaling)
        x_pos_emb = torch.cat(
            [x, self.positional_embeddings.expand(x.shape[0], -1, -1, -1) * constrained_scaling], dim=1
        )
        return self.perception(x_pos_emb)

    def get_positional_embeddings(self):
        """Get the learnable positional embeddings."""
        return self.positional_embeddings
