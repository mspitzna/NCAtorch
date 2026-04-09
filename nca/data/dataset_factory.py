from nca.utils.config import Config
from nca.data.datasets import (
    IconDataset,
    EdgesDataset,
    CustomMNISTDataset,
    OrganizingTexturesDataset,
    CIFAR10Dataset,
    MovingMNISTDataset,
    CelebADataset,
    GrowingMNISTDataset,
)
from nca.data.data_wrapper import DataWrapper
from nca.data.cfg_dataset_wrapper import CFGDatasetWrapper
from torch.utils.data import DataLoader

# Registry of all available datasets.
#
# Maps a NAME string to a factory function with signature:
#   (config: Config, train: bool) -> (dataset, cond_dim, im_height, im_width)
#
# Adding a new dataset:
#   1. Implement the dataset class in nca/data/datasets/.
#   2. Add one entry here — the config validator and tests pick it up automatically.
#
# Available datasets:
#   "emoji"         — icon/emoji generation
#   "e2h"           — edge-to-handbag conditional generation
#   "ot"            — organizing textures synthesis
#   "mnist"         — self-classifying MNIST digits
#   "cifar10"       — CIFAR-10 classification
#   "growing_mnist" — invertible growing MNIST
#   "moving_mnist"  — temporal video prediction on Moving MNIST
#   "celeba"        — CelebA face generation


def _create_emoji(config: Config, _train: bool):
    dataset = IconDataset(
        emojis=config.DATASET.EMOJIS,
        target_size=config.DATASET.TARGET_SIZE,
        target_padding=config.DATASET.TARGET_PADDING,
        channel_n=config.MODEL.CHANNEL_N,
        condition_size=len(config.DATASET.EMOJIS),
        total_samples=(
            config.TRAINING.STEPS * config.TRAINING.BATCH_SIZE
            if config.TRAINING.STEPS != -1
            else float("inf")
        ),
        device="cpu",
        rectangle=False,
    )
    cond_dim = len(config.DATASET.EMOJIS)
    size = config.DATASET.TARGET_SIZE + config.DATASET.TARGET_PADDING * 2
    return dataset, cond_dim, size, size


def _create_e2h(config: Config, train: bool):
    dataset = EdgesDataset(
        dataroot=config.DATASET.DATAROOT,
        train=train,
        img_size=config.DATASET.TARGET_SIZE,
        channel_n=config.MODEL.CHANNEL_N,
        device="cpu",
        invertible=config.DATASET.INVERTIBLE,
    )
    cond_dim = 2 if config.DATASET.INVERTIBLE else 0
    size = config.DATASET.TARGET_SIZE
    return dataset, cond_dim, size, size


def _create_ot(config: Config, _train: bool):
    dataset = OrganizingTexturesDataset(
        sample_path=config.DATASET.DATASET_SAMPLE_PATH,
        channel_n=config.MODEL.CHANNEL_N,
        img_size=config.DATASET.TARGET_SIZE,
        device="cpu",
        condition_size=0,
        total_samples=(
            config.TRAINING.STEPS * config.TRAINING.BATCH_SIZE
            if config.TRAINING.STEPS != -1
            else float("inf")
        ),
    )
    size = config.DATASET.TARGET_SIZE
    return dataset, 0, size, size


def _create_mnist(config: Config, train: bool):
    dataset = CustomMNISTDataset(
        root=config.DATASET.DATAROOT,
        channel_n=config.MODEL.CHANNEL_N,
        train=train,
        target_padding=config.DATASET.TARGET_PADDING,
        device="cpu",
    )
    size = int(28 + config.DATASET.TARGET_PADDING * 2)
    return dataset, 0, size, size


def _create_cifar10(config: Config, train: bool):
    dataset = CIFAR10Dataset(
        root=config.DATASET.DATAROOT,
        channel_n=config.MODEL.CHANNEL_N,
        target_size=config.DATASET.TARGET_SIZE,
        train=train,
        device=config.DEVICE,
    )
    size = config.DATASET.TARGET_SIZE
    return dataset, 0, size, size


def _create_growing_mnist(config: Config, train: bool):
    dataset = GrowingMNISTDataset(
        root=config.DATASET.DATAROOT,
        channel_n=config.MODEL.CHANNEL_N,
        train=train,
        target_padding=config.DATASET.TARGET_PADDING,
        device="cpu",
        invertible=config.DATASET.INVERTIBLE,
        seed_size=config.DATASET.SEED_SIZE,
        goal_channels=config.CFG.GOAL_CHANNELS and config.CFG.ENABLED,
        enable_rotation=config.DATASET.ENABLE_ROTATION,
        enable_zoom=config.DATASET.ENABLE_ZOOM,
        latent_noise_channel=config.DATASET.Z_LATENT_NOISE_CHANNEL,
    )
    cond_dim = 1 if config.DATASET.INVERTIBLE else 0
    if config.CFG.ENABLED and config.CFG.GOAL_CHANNELS:
        cond_dim += 2
    size = int(28 + config.DATASET.TARGET_PADDING * 2)
    return dataset, cond_dim, size, size


def _create_moving_mnist(config: Config, _train: bool):
    dataset = MovingMNISTDataset(
        root=config.DATASET.DATAROOT,
        download=True,
        history_n=config.DATASET.HISTORY_N,
        channel_n=config.MODEL.CHANNEL_N,
        reverse_seed=config.DATASET.REVERSE_HISTORY_SEED,
    )
    size = config.DATASET.TARGET_SIZE
    return dataset, 0, size, size


def _create_celeba(config: Config, train: bool):
    dataset = CelebADataset(
        root_dir=config.DATASET.DATAROOT,
        split="train" if train else "test",
        img_size=config.DATASET.TARGET_SIZE,
    )
    size = config.DATASET.TARGET_SIZE
    return dataset, 1, size, size


DATASET_REGISTRY = {
    "emoji":         _create_emoji,
    "e2h":           _create_e2h,
    "ot":            _create_ot,
    "mnist":         _create_mnist,
    "cifar10":       _create_cifar10,
    "growing_mnist": _create_growing_mnist,
    "moving_mnist":  _create_moving_mnist,
    "celeba":        _create_celeba,
}


def create_dataset(config: Config, train=True):
    name = config.DATASET.NAME
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Invalid DATASET.NAME '{name}'. "
            f"Registered datasets: {sorted(DATASET_REGISTRY)}"
        )

    dataset, cond_dim, im_height, im_width = DATASET_REGISTRY[name](config, train)

    if config.CFG.ENABLED and cond_dim > 0:
        dataset = CFGDatasetWrapper(
            dataset,
            config.CFG.DROPOUT_PROB,
            config.CFG.NULL_CONDITION_TYPE,
            config.CFG.PRESERVE_CHANNELS,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=config.DATASET.DROP_LAST_BATCH if train else False,
    )
    handler = DataWrapper(dataloader, config)
    return handler, cond_dim, im_height, im_width
