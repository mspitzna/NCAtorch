<div align="center">
  <h1>Custom Update Module Guide</h1>
  <p><em>Extend NCAtorch with custom update models that transform perception outputs into state changes.</em></p>
</div>

---

## 🛠️ Overview

Custom update modules let you experiment with novel architectures for computing state updates (dx) from perception vectors. The update module is the "brain" of the CA that decides how each cell evolves based on its neighborhood information. Build the module, register it with the factory, then expose it through the config system.

## Step 1 – Implement the update module ([nca/core/models/ca/update_models.py](../nca/core/models/ca/update_models.py))

1. Create a class that inherits from `UpdateModelBase` and implements the `forward` method.
2. Accept runtime kwargs such as `in_channels`, `out_channels`, `hidden_channels`, and `device`. The base class already stores `in_channels` and `out_channels` as attributes.
3. Use the following skeleton as a reference:

```python
class MyCustomUpdate(UpdateModelBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        activation=nn.LeakyReLU,
        final_activation=False,
        device="cpu",
        **kwargs
    ):
        super().__init__(in_channels, out_channels)

        self.hidden_channels = hidden_channels
        self.activation = activation(0.2)

        # Example: Simple two-layer architecture
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        # Optional final activation (e.g., Tanh to bound outputs)
        self.final_activation = nn.Tanh() if final_activation else nn.Identity()

    def forward(self, x):
        """
        x shape: [B, in_channels, H, W] (perception vector)
        Returns: dx shape: [B, out_channels, H, W] (state update)
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return self.final_activation(x)
```

**Key considerations:**
- The `in_channels` parameter corresponds to `perception_module.get_out_channel()` – the total output from all perception modules combined.
- The `out_channels` parameter matches the CA's state channels (either `CHANNEL_N` or `LATENT_AE_CHANNEL` if using latent training).
- Use 1x1 convolutions (`kernel_size=1`) for pixel-wise operations, as perception already handles spatial information.
- The output represents the *change* (dx) to be added to the current state, not the new state itself.
- Initialize the final layer with small weights (see `CAModel._init_weights()` at [ca_model.py:74-86](../nca/core/models/ca/ca_model.py#L74-L86)) for stable training.

## Step 2 – Register it in the factory ([nca/core/models/model_factory.py](../nca/core/models/model_factory.py))

1. Import your class near the existing update model imports:

```python
from nca.core.models.ca.update_models import SimpleMLPUpdate, ResNetUpdate, MyCustomUpdate
```

2. Extend the `get_update_model` function to handle your new model name:

```python
def get_update_model(config: Config, in_channel_n, channel_out, device):
    """
    Create the update model based on the configuration.
    """
    model_name = config.MODEL.NAME
    print(f"Creating {model_name}. The value of in_channel_n is: {in_channel_n}")

    if model_name == "MLP":
        model = SimpleMLPUpdate(
            in_channels=in_channel_n,
            out_channels=channel_out,
            hidden_channels=config.MODEL.HIDDEN_CHANNELS,
            final_activation=config.MODEL.FINAL_ACTIVATION,
            device=device,
        )
        return model.to(device)
    elif model_name == "ResNet":
        model = ResNetUpdate(
            in_channels=in_channel_n,
            out_channels=channel_out,
            hidden_channels=config.MODEL.HIDDEN_CHANNELS,
            num_blocks=config.MODEL.RESNET_BLOCKS,
            final_activation=config.MODEL.FINAL_ACTIVATION,
            device=device,
        )
        return model.to(device)
    elif model_name == "MyCustom":
        model = MyCustomUpdate(
            in_channels=in_channel_n,
            out_channels=channel_out,
            hidden_channels=config.MODEL.HIDDEN_CHANNELS,
            final_activation=config.MODEL.FINAL_ACTIVATION,
            device=device,
        )
        return model.to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
```

3. The factory automatically passes the perception output channels as `in_channel_n`, so your module will receive the correct input dimensions.

## Step 3 – Expose the model in the config ([nca/utils/config.py](../nca/utils/config.py))

1. Update the `ModelConfig` validator to include your new model name in the valid options:

```python
@validator('NAME')
def validate_name(cls, v):
    valid_names = ['MLP', 'ResNet', 'MyCustom']
    if v not in valid_names:
        raise ValueError(f"MODEL.NAME must be one of {valid_names}, got '{v}'")
    return v
```

2. Add any custom hyperparameters your update module needs to `ModelConfig`. For example, if your module uses a specific parameter:

```python
class ModelConfig:
    NAME: str = "MLP"
    HIDDEN_CHANNELS: List[int] = [80]
    FINAL_ACTIVATION: bool = False
    MY_CUSTOM_PARAM: float = 0.5  # Your custom parameter
    # ... other fields ...
```

3. Reference your update module in the YAML config:

```yaml
MODEL:
  NAME: "MyCustom"
  HIDDEN_CHANNELS: [128, 128]
  FINAL_ACTIVATION: false
  MY_CUSTOM_PARAM: 0.7
```

## Understanding the Update Flow

The update module is called within the CA's forward pass ([ca_model.py:88-102](../nca/core/models/ca/ca_model.py#L88-L102)):

1. **Input**: The perception vector `grads` with shape `[B, perception_out_channel, H, W]`
2. **Output**: The state update `dx` with shape `[B, channel_out, H, W]`
3. **Application**: The CA adds `dx` to the current state: `state = state + dx`

Additional processing after the update module:
- Step size scaling: `dx = dx * step_size`
- Noise injection: `dx = dx + noise` (if enabled)
- Fire rate masking: Stochastic dropout of updates
- Living mask: Zero out updates for dead cells

---

With those three touchpoints wired up, rerun your training script and the new update module will replace the default MLP/ResNet architecture. The module will automatically receive the correct input dimensions based on your perception configuration and output the appropriate number of channels for your CA state.
