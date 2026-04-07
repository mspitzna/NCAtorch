<div align="center">
  <h1>Custom Update Module Guide</h1>
  <p><em>Extend NCAtorch with custom update models that transform perception outputs into state changes.</em></p>
</div>

---

## Overview

The update module is the "brain" of the CA — it maps perception vectors to per-cell state deltas. There are only two steps: implement the module and add one line to the registry. The config validator and tests update themselves automatically.

## Step 1 – Implement the update module ([nca/core/models/ca/update_models.py](../nca/core/models/ca/update_models.py))

Create a class that inherits from `UpdateModelBase` and implements `forward`.

```python
class MyCustomUpdate(UpdateModelBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        final_activation=False,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            nn.Tanh() if final_activation else nn.Identity(),
        )

    def forward(self, x):
        """
        x:      [B, in_channels,  H, W]  — perception vector
        return: [B, out_channels, H, W]  — state delta (dx)
        """
        return self.net(x)
```

Key points:
- `in_channels` = `perception_module.get_out_channel()` — the total output across all perception branches.
- `out_channels` = CA state channels (`CHANNEL_N`, or `LATENT_AE_CHANNEL` for latent training).
- Use `kernel_size=1` convolutions — spatial context is already captured by the perception stage.
- The output is a *delta* added to the current state, not the new state itself.
- Keep the final layer's initial weights small; `CAModel._init_weights()` handles this automatically.

## Step 2 – Add one entry to the registry ([nca/core/models/update_model_factory.py](../nca/core/models/update_model_factory.py))

Import your class and add a lambda to `UPDATE_MODEL_REGISTRY`:

```python
from nca.core.models.ca.update_models import SimpleMLPUpdate, ResNetUpdate, MyCustomUpdate

UPDATE_MODEL_REGISTRY = {
    # ... existing entries ...
    "MyCustom": lambda in_ch, out_ch, cfg: MyCustomUpdate(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=cfg.MODEL.HIDDEN_CHANNELS,
        final_activation=cfg.MODEL.FINAL_ACTIVATION,
    ),
}
```

The lambda receives `(in_channels: int, out_channels: int, config: Config)`. Use `cfg.MODEL.*` to access `ModelConfig` fields — add new fields to `ModelConfig` in [nca/utils/config.py](../nca/utils/config.py) if your module needs custom hyperparameters.

**That's it.** Two things happen automatically:

- The `ModelConfig.NAME` validator imports `UPDATE_MODEL_REGISTRY` at runtime, so `"MyCustom"` becomes a valid value immediately — no manual list to update.
- The exhaustive update model tests in `tests/test_update_model_factory.py` are driven by `UPDATE_MODEL_REGISTRY`, so your new entry is covered on the next test run. If `forward()` returns the wrong shape or gradients don't flow, a test will fail.

## Step 3 – Use it in a config YAML

```yaml
MODEL:
  NAME: "MyCustom"
  HIDDEN_CHANNELS: [128, 128]
  FINAL_ACTIVATION: false
```

If you added a custom config field (e.g. `MY_DROPOUT: float = 0.0`):

```yaml
MODEL:
  NAME: "MyCustom"
  HIDDEN_CHANNELS: [128]
  MY_DROPOUT: 0.1
```

## Update flow in the CA forward pass

```
perception(state)          →  grads  [B, perception_out_ch, H, W]
update_model(grads)        →  dx     [B, channel_out, H, W]
dx * step_size             →  scaled delta
+ noise (optional)
* fire_rate_mask           →  stochastic dropout
state + dx                 →  new state
```

---

With those two touchpoints wired up, rerun your training script and the new update module replaces the default MLP/ResNet architecture.
