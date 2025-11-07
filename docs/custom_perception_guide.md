<div align="center">
  <h1>Custom Perception Guide</h1>
  <p><em>Extend NCAtorch with custom perceptions that plug straight into the training pipeline.</em></p>
</div>

---

## 🛠️ Overview

Custom perceptions let you experiment with novel neighborhood operators while keeping the rest of the CA stack unchanged. Build the module, register it with the factory, then expose it through the config system.

## Step 1 – Implement the perception (`nca/core/models/ca/perceptions.py`)

1. Create a class that inherits from `Perception` (or `nn.Module`) and implements `forward` plus `get_out_channel`.
2. Accept runtime kwargs such as `in_channel`, `out_channel`, `kernel_size`, and `device`. Most built-in perceptions store these values as attributes so they can size convolutions or buffers correctly.
3. Use the following skeleton as a reference:

```python
class MyCustomPerception(Perception):
    def __init__(self, in_channel=16, out_channel=64, kernel_size=3, device="cpu"):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.device = device

        self.filter = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="circular",
        )

    def forward(self, x):
        return torch.relu(self.filter(x))

    def get_out_channel(self):
        return self.out_channel
```

Need extra utilities like buffers or visualization hooks? Mirror the patterns used by `SobelPerception` or `DeformableConvPerception`.

## Step 2 – Register it in the factory (`nca/core/models/model_factory.py`)

1. Import your class near the existing perception imports:

```python
from .ca.perceptions import MyCustomPerception
```

2. Extend the `create_perception` helper inside `get_perception` so the new `MODE` string instantiates your class:

```python
elif mode == "my_custom":
    return MyCustomPerception(
        in_channel=in_channel_n,
        out_channel=out_channel,
        kernel_size=kernel_size,
        device=device,
    )
```

3. Nothing else is required: `get_perception` already wraps multiple entries in `MultiPerception`, so your branch will automatically concatenate with others when listed together.

## Step 3 – Expose the mode in the config (`nca/utils/config.py`)

1. Update the `PerceptionConfig` validator so your identifier passes validation:

```python
valid_modes = [
    "conv",
    "attention",
    "sobel",
    "deformable_conv",
    "residual_conv",
    "my_custom",
]
```

2. Introduce additional fields (and validators) in `PerceptionConfig` if your module needs custom hyperparameters.
3. Reference the perception via the `MODE` field:

```yaml
MODEL:
  PERCEPTIONS:
    - MODE: "my_custom"
      KERNEL_SIZE: 5
      OUT_CHANNEL: 96
```

Listing multiple entries will run each perception in parallel and concatenate their outputs. The factory automatically sums their `OUT_CHANNEL` values via `perception_module.get_out_channel()` and feeds that number into the update model, so you only need to decide how many filters each branch should emit.

---

With those three touchpoints wired up, rerun your training script and the new perception should slot into the CA pipeline just like the built-in modules.
