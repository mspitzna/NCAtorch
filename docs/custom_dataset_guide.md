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

By default the pipeline visualizes the first 1–4 channels of the CA state and composites them over a white background using `to_rgb`. Override `_colorize` in your dataset class to change this — it is the single hook used by both the training logger and the web UI.

```python
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
        # Example: color pixels by their argmax class channel
        return self.apply_coloring(x, x[:, -self.num_classes:])
```

`_colorize` is called per sample. The base `batch_to_rgb` loops over the batch and stacks the results; `state_to_img` calls it once and converts to uint8 for the UI. You never need to override either of those methods unless `x0`, `x`, and `target` each need *different* coloring logic — for example, showing the seed colored by the ground-truth labels while `x` is colored by its own predicted labels:

```python
def batch_to_rgb(self, x0, x, target, cond=None):
    x0_rgb, x_rgb, _ = super().batch_to_rgb(x0, x, target, cond)
    # target column: x0's appearance with target's class labels
    target_rgb = self.apply_coloring(x0, target[:, -self.num_classes:])
    return x0_rgb, x_rgb, target_rgb
```

### Visualization contract

| Method | Returns | dtype | Range | Shape |
|---|---|---|---|---|
| `_colorize` | `torch.Tensor` | float32 | `[0, 1]` | `(1, 3, H, W)` |
| `batch_to_rgb` | tuple of 3 tensors | float32 | `[0, 1]` | `(B, 3, H, W)` each |
| `state_to_img` | `np.ndarray` | uint8 | `[0, 255]` | `(H, W, 3)` |

---

With those steps wired up, your dataset slots into the training pipeline. The trainer, sample pool, and logging all consume `(seed, condition, target)` tuples, so no further changes are needed elsewhere.
