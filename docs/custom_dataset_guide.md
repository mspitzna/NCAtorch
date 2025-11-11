<div align="center">
  <h1>Custom Dataset Guide</h1>
  <p><em>Extend NCAtorch with custom datasets that plug straight into the training pipeline.</em></p>
</div>

---

## 🛠️ Overview

Custom datasets let you train NCAs on your own data. Build the dataset class, register it with the factory, then expose it through the config system.

## Step 1 – Implement the dataset ([nca/data/datasets/](../nca/data/datasets/))

1. Create a class that inherits from `Dataset` and implements `__len__` and `__getitem__`.
2. `__getitem__` must return `(seed, condition, target)` tuple.
3. Use the following skeleton as a reference:

```python
class MyCustomDataset(Dataset):
    def __init__(self, root_dir, img_size=512):
        self.filenames = self._load_filenames(root_dir)
        self.transform = transforms.Compose([...])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target = self.load_and_transform(self.filenames[idx])
        seed = target.clone()  # or apply modifications
        condition = torch.zeros(0)  # or your condition
        return seed, condition, target
```

## Step 2 – Register it in the factory ([nca/data/dataset_factory.py](../nca/data/dataset_factory.py))

1. Import your class and add a branch in `create_dataset()`:

```python
elif config.DATASET.NAME == "my_custom":
    dataset = MyCustomDataset(root_dir=config.DATASET.DATAROOT, img_size=config.DATASET.TARGET_SIZE)
    cond_dim = 0
    im_height = im_width = config.DATASET.TARGET_SIZE
```

## Step 3 – Register in `__init__.py` ([nca/data/datasets/\_\_init\_\_.py](../nca/data/datasets/__init__.py))

Add your dataset to imports:

```python
from nca.data.datasets.my_custom_dataset import MyCustomDataset

__all__ = ["MyCustomDataset", ...]
```

## Step 4 – Add custom config fields (optional) ([nca/utils/config.py](../nca/utils/config.py))

If your dataset needs custom parameters, add them to `DatasetConfig`:

```python
class DatasetConfig(BaseModel):
    NAME: str = "emoji"
    DATAROOT: Path = None
    TARGET_SIZE: int = 64
    MY_CUSTOM_PARAM: int = 10  # Your custom parameter
```

## Step 5 – Expose in config YAML

```yaml
DATASET:
  NAME: "my_custom"
  DATAROOT: "/path/to/dataset"
  TARGET_SIZE: 512
  MY_CUSTOM_PARAM: 20
```

---

With those steps wired up, your dataset slots into the training pipeline.