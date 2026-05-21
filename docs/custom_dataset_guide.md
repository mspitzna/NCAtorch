<div align="center">
  <h1>Custom Dataset Guide</h1>
  <p><em>Extend NCAtorch with custom datasets that plug straight into the training pipeline.</em></p>
</div>

---

## Overview

Custom datasets let you train NCAs on your own data. Build the dataset class, wire it into the factory, and optionally extend the config.

## Step 1 – Implement the dataset ([nca/data/datasets/](../nca/data/datasets/))

Create a class that inherits from `Dataset` and implements `__len__` and `__getitem__`. The `__getitem__` method must return a `(seed, condition, target)` tuple.

```python
class MyCustomDataset(Dataset):
    def __init__(self, root_dir, img_size=512):
        self.filenames = self._load_filenames(root_dir)
        self.transform = transforms.Compose([...])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target = self.load_and_transform(self.filenames[idx])
        seed = target.clone()          # or build a different seed
        condition = torch.zeros(0)     # or your condition vector
        return seed, condition, target
```

## Step 2 – Register it in the factory ([nca/data/dataset_factory.py](../nca/data/dataset_factory.py))

Import your class and add a factory function, then register it in `DATASET_REGISTRY`:

```python
from nca.data.datasets.my_custom_dataset import MyCustomDataset

def _create_my_custom(config: Config, train: bool):
    dataset = MyCustomDataset(
        root_dir=config.DATASET.DATAROOT,
        img_size=config.DATASET.TARGET_SIZE,
    )
    size = config.DATASET.TARGET_SIZE
    return dataset, 0, size, size  # (dataset, cond_dim, im_height, im_width)

DATASET_REGISTRY = {
    ...
    "my_custom": _create_my_custom,
}
```

Each factory function receives `(config, train)` and must return `(dataset, cond_dim, im_height, im_width)`. Make sure `cond_dim`, `im_height`, and `im_width` are set correctly for your data — they are used to size the CA model and condition embedding.

## Step 3 – Export from `__init__.py` ([nca/data/datasets/\_\_init\_\_.py](../nca/data/datasets/__init__.py))

```python
from nca.data.datasets.my_custom_dataset import MyCustomDataset

__all__ = [..., "MyCustomDataset"]
```

## Step 4 – Add custom config fields (optional) ([nca/utils/config.py](../nca/utils/config.py))

If your dataset needs extra parameters, add them to `DatasetConfig`:

```python
class DatasetConfig(BaseModel):
    NAME: str = "emoji"
    DATAROOT: Path = None
    TARGET_SIZE: int = 64
    MY_CUSTOM_PARAM: int = 10  # your parameter
```

## Step 5 – YAML config

```yaml
DATASET:
  NAME: "my_custom"
  DATAROOT: "/path/to/dataset"
  TARGET_SIZE: 512
  MY_CUSTOM_PARAM: 20
```

## Step 6 – Customize visualization (optional)

By default the pipeline reads the first 1–4 channels of the CA state and converts them to RGB via `to_rgb`. Override `_colorize` in your dataset class to change this — it is the single hook consumed by both the training logger and the web UI.

The example below assumes channels 0–2 carry RGB colour, channel 3 is an alive/alpha mask, and the optional `cond` tensor is a 3-element RGB tint. Adapt the channel indexing, compositing, and conditioning logic to match your own state and condition layout.

```python
import torch
from nca.data.datasets.base_dataset import NCADataset

class MyCustomDataset(NCADataset):

    def _colorize(self, x, x0=None, target=None, cond=None):
        """
        x       : (1, C, H, W) float32 — current CA state, single sample
        x0      : (1, C, H, W) optional seed for context
        target  : (1, C, H, W) optional target for context
        cond    : optional condition tensor

        Must return (1, 3, H, W) float32 in [0, 1].
        """
        # Channels 0-2: learned RGB colour; channel 3: alive/alpha mask.
        rgb   = x[:, :3].clamp(0.0, 1.0)   # (1, 3, H, W)
        alpha = x[:, 3:4].clamp(0.0, 1.0)  # (1, 1, H, W)

        # Composite over a white background so dead cells appear white.
        rgb = alpha * rgb + (1.0 - alpha)

        # Tint the output with the condition colour when a condition is available.
        if cond is not None:
            # cond assumed to be a (1, 3) RGB tint in [0, 1]
            tint = cond.view(1, 3, 1, 1)
            rgb  = (rgb * tint).clamp(0.0, 1.0)

        return rgb
```

`_colorize` is called per sample. The base `batch_to_rgb` loops over the batch and stacks the results; `state_to_img` calls it once and converts to uint8 for the web UI. In most cases overriding `_colorize` is all you need.

Override `batch_to_rgb` only when `x0`, `x`, and `target` require *different* coloring logic from each other — for example, fixing the target column to always show the ground-truth appearance rather than re-running `_colorize` on it:

```python
def batch_to_rgb(self, x0, x, target, cond=None):
    x0_rgb, x_rgb, _ = super().batch_to_rgb(x0, x, target, cond)
    # Show the target as a plain RGB composite, independent of _colorize.
    target_rgb = target[:, :3].clamp(0.0, 1.0)
    return x0_rgb, x_rgb, target_rgb
```

### Visualization contract

| Method | Input | Returns | dtype | Range | Shape |
|---|---|---|---|---|---|
| `_colorize` | single sample | `torch.Tensor` | float32 | `[0, 1]` | `(1, 3, H, W)` |
| `batch_to_rgb` | full batch | tuple of 3 tensors | float32 | `[0, 1]` | `(B, 3, H, W)` each |
| `state_to_img` | single sample | `np.ndarray` | uint8 | `[0, 255]` | `(H, W, 3)` |

---

With those steps wired up, your dataset slots into the training pipeline. The trainer, sample pool, and logging all consume `(seed, condition, target)` tuples, so no further changes are needed elsewhere.
