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

Import your class and add a branch in `create_dataset()`:

```python
from nca.data.datasets.my_custom_dataset import MyCustomDataset

# inside create_dataset():
elif config.DATASET.NAME == "my_custom":
    dataset = MyCustomDataset(
        root_dir=config.DATASET.DATAROOT,
        img_size=config.DATASET.TARGET_SIZE,
    )
    cond_dim = 0
    im_height = im_width = config.DATASET.TARGET_SIZE
```

`create_dataset` returns `(dataloader, cond_dim, im_height, im_width)`. Make sure `cond_dim`, `im_height`, and `im_width` are set correctly for your data — they are used to size the CA model and condition embedding.

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

---

With those steps wired up, your dataset slots into the training pipeline. The trainer, sample pool, and logging all consume `(seed, condition, target)` tuples, so no further changes are needed elsewhere.
