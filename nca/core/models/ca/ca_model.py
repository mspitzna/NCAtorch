import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from .perceptions import ConvPerception

class CAModel(nn.Module, ABC):
    def __init__(
        self,
        channel_n=16,
        channel_out=16,
        cond_channel_n=0,
        fire_rate=0.5,
        device="cpu",
        use_positional_embeddings=False,
        img_height=32,
        img_width=32,
        perception_module=None,
        perception_params=None,
        update_model_module=None,
        living_mask=False,
        living_mask_indice=3,
        noise_injection=0.0,
        normalize_output=False,
        **kwargs,
    ):
        super().__init__()
        self.channel_n = channel_n
        self.channel_out = channel_out
        self.fire_rate = fire_rate
        self.device = device
        self.use_positional_embeddings = use_positional_embeddings
        self.cond_channel_n = cond_channel_n
        self.living_mask = living_mask
        self.living_mask_indice = living_mask_indice
        self.noise_injection = noise_injection
        self.normalize_output = normalize_output

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
        self.perception_params = (
            perception_params if perception_params is not None else {}
        )
        self.perception = self._build_perception(
            perception_module, **self.perception_params
        )

        # Initialize the delta model
        self.update_model = update_model_module

        self._init_weights()

        num_perception_params = sum(p.numel() for p in self.perception.parameters() if p.requires_grad)
        print(f"Perception has {num_perception_params:,} learnable parameters")


        num_dmodel_params = sum(p.numel() for p in self.update_model.parameters() if p.requires_grad)
        print(f"Update model has {num_dmodel_params:,} learnable parameters")

    def _init_weights(self):
        # Find the last Conv2d layer and reinitialize it to near zero weights
        last_conv = None
        for _, module in self.update_model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module  # this will eventually be the last one encountered

        if last_conv is not None:
            print(f"Re-initializing last convolutional layer: {last_conv}")
            # Initialize weights to a uniform distribution around 0 with a small range.
            torch.nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                torch.nn.init.xavier_uniform_(last_conv.weight, gain=0.1)

    def _call_update_model(self, x, current_state=None):
        """
        Call the update model with the input tensor x.
        For ReversibleUpdate, also pass the current state if available.
        x: [batch, dim, height, width] (perception vector)
        current_state: [batch, channel_out, height, width] (optional, for reversible updates)
        Returns: dx: [batch, channel_out, height, width]
        """
        # Check if this is a PathInvertibleCA that needs current state
        if hasattr(self.update_model, '__class__') and 'PathInvertibleCA' in self.update_model.__class__.__name__:
            if current_state is not None:
                return self.update_model(x, current_state)
        
        # Standard update model call
        return self.update_model(x)

    def _build_perception(self, perception_module, **perception_params):
        """Abstract method for building the perception module."""
        if perception_module is not None:
            return perception_module
        else:
            in_channel_n = (
                self.channel_n + self.cond_channel_n + 2
                if self.use_positional_embeddings
                else self.channel_n + self.cond_channel_n
            )
            print(
                f"ConvPerception with in_channel_n: {in_channel_n}, out_channel: 80, kernel_size: 3"
            )
            return ConvPerception(
                in_channel_n,
                80,
                3,
                self.device,
                **perception_params,
            )

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
        if freeze_channels is not None:
            frozen_layers, state = x[:, :freeze_channels + 1].clone(), x[:, freeze_channels + 1:].clone()
        else:
            state = x.clone()  # No channels are frozen

        pre_life_mask = self.get_living_mask(x) if self.living_mask else None
        # 1) Prepare the input
        x = self._prepare_input(x, cond)

        # 2) Compute "grads"
        if self.use_positional_embeddings and self.positional_embeddings is not None:
            grads = self._perception_with_positional_embeddings(x)
        else:
            grads = self.perception(x)

        # 3) Concatenate current state + grads, then pass through dmodel
        dx = self._call_update_model(grads, state) * step_size

        # 4) Add noise to the update
        if self.noise_injection > 0.0:
            noise = torch.randn_like(dx) * self.noise_injection
            dx = dx + noise

        # 5) Randomly mask updates (fire_rate)
        if self.fire_rate < 1.0:
            update_mask = (torch.rand_like(x[:, :1, :, :]) <= self.fire_rate).float()
            dx = dx * update_mask

        # 6) Update the state using masked dx
        state = state + dx

        # 7) If using a living mask, apply it after the update
        if self.living_mask:
            if freeze_channels is not None:
                post_life_mask = self.get_living_mask(torch.cat([frozen_layers, state], dim=1))
            else:
                post_life_mask = self.get_living_mask(state)
            life_mask = pre_life_mask & post_life_mask

            # Expand life_mask to match [B, channel_n, H, W]
            life_mask = life_mask.unsqueeze(1)  # -> [B, 1, H, W]
            life_mask = life_mask.expand(-1, dx.size(1), -1, -1)  # -> [B, channel_n, H, W]

            state = state * life_mask.float()  # Apply the living mask to the update

        # 8) Normalize each batch item
        if self.normalize_output:
            state = state / (state.std(dim=(1, 2, 3), keepdim=True) + 1e-6)

        if freeze_channels is not None:
            state = torch.cat([frozen_layers, state], dim=1)
        if return_residuals:
            return state, dx
        return state

    def get_living_mask(self, x):
        """Get the living mask based on the alpha channel."""
        alpha = x[:, self.living_mask_indice, :, :]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

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