<div align="center">
  <h1>Custom Perception Guide</h1>
  <p><em>Extend NCAtorch with custom perceptions that plug straight into the training pipeline.</em></p>
</div>

---

## Overview

Custom perceptions let you experiment with novel neighborhood operators while keeping the rest of the CA stack unchanged. There are only two steps: implement the module and add one line to the registry. The config validator and tests update themselves automatically.

## Step 1 – Implement the perception ([nca/core/models/ca/perceptions.py](../nca/core/models/ca/perceptions.py))

Create a class that inherits from `Perception` (or `nn.Module`) and implements `forward` and `get_out_channel`.

```python
class MyCustomPerception(Perception):
    def __init__(self, in_channel=16, out_channel=64, kernel_size=3, device="cpu"):
        super().__init__()
        self.out_channel = out_channel
        self.filter = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel_size,
            padding="same", padding_mode="circular",
        )

    def forward(self, x):
        return torch.relu(self.filter(x))

    def get_out_channel(self):
        return self.out_channel
```

The `in_channel` argument is the CA state channel count (e.g. `CHANNEL_N`). `get_out_channel()` must match the actual channel dimension returned by `forward()` — this is verified by the exhaustive registry tests.

For more complex patterns (fixed Sobel-style filters, learnable deformable offsets, etc.) see `SobelPerception` or `DeformableConvPerception` in the same file.

## Step 2 – Add one entry to the registry ([nca/core/models/perception_factory.py](../nca/core/models/perception_factory.py))

Import your class and add a lambda to `PERCEPTION_REGISTRY`:

```python
from nca.core.models.ca.perceptions import ..., MyCustomPerception

PERCEPTION_REGISTRY = {
    # ... existing entries ...
    "my_custom": lambda in_ch, cfg, dev: MyCustomPerception(
        in_channel=in_ch,
        out_channel=cfg.OUT_CHANNEL,
        kernel_size=cfg.KERNEL_SIZE,
        device=dev,
    ),
}
```

The lambda receives `(in_channels: int, perception_cfg: PerceptionConfig, device: str)`. Use `cfg.*` to access any `PerceptionConfig` field — add new fields to `PerceptionConfig` in [nca/utils/config.py](../nca/utils/config.py) if your module needs custom hyperparameters.

**That's it.** Two things happen automatically:

- The `PerceptionConfig.MODE` validator imports `PERCEPTION_REGISTRY` at runtime, so `"my_custom"` becomes a valid value immediately — no manual list to update.
- The exhaustive perception tests in `tests/test_perception_factory.py` are driven by `PERCEPTION_REGISTRY`, so your new entry is covered on the next test run. If `forward()` returns the wrong shape or gradients don't flow, a test will fail.

## Step 3 – Use it in a config YAML

```yaml
MODEL:
  PERCEPTIONS:
    - MODE: "my_custom"
      KERNEL_SIZE: 5
      OUT_CHANNEL: 96
```

Listing multiple entries runs each perception in parallel and concatenates their outputs. The factory sums `get_out_channel()` across all branches and passes the total to the update model automatically.

```yaml
MODEL:
  PERCEPTIONS:
    - MODE: "conv"
      OUT_CHANNEL: 48
    - MODE: "my_custom"
      OUT_CHANNEL: 48   # update model receives 96 total
```

---

With those two touchpoints wired up, rerun your training script and the new perception slots into the CA pipeline like any built-in module.
