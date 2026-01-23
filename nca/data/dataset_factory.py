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

def create_dataset(config: Config, train=True):
    if config.DATASET.NAME == "emoji":
        if config.DATASET.INVERTIBLE:
            # Create InvertibleIconDataset with refactored seed creation
            dataset = IconDataset(
                emojis=config.DATASET.EMOJIS,
                target_size=config.DATASET.TARGET_SIZE,
                target_padding=config.DATASET.TARGET_PADDING,
                channel_n=config.MODEL.CHANNEL_N,
                total_samples=(
                    config.TRAINING.STEPS * config.TRAINING.BATCH_SIZE
                    if config.TRAINING.STEPS != -1
                    else float("inf")
                ),
                device="cpu",
                rectangle=True  # Invertible emoji dataset uses rectangular seed by default
            )
            cond_dim = 1
        else:
            # Create IconDataset with refactored seed creation
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
                rectangle=False  # Standard emoji dataset uses single pixel seed
            )
            cond_dim = len(config.DATASET.EMOJIS)
        im_height = config.DATASET.TARGET_SIZE + config.DATASET.TARGET_PADDING * 2
        im_width = config.DATASET.TARGET_SIZE + config.DATASET.TARGET_PADDING * 2
    elif config.DATASET.NAME == "e2h":
        dataset = EdgesDataset(
            dataroot=config.DATASET.DATAROOT,
            train=train,
            img_size=config.DATASET.TARGET_SIZE,
            channel_n=config.MODEL.CHANNEL_N,
            device="cpu",
            invertible=config.DATASET.INVERTIBLE
        )
        cond_dim = 2 if config.DATASET.INVERTIBLE else 0
        im_height = config.DATASET.TARGET_SIZE
        im_width = config.DATASET.TARGET_SIZE
    elif config.DATASET.NAME == "ot":
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
        cond_dim = 0
        im_height = config.DATASET.TARGET_SIZE
        im_width = config.DATASET.TARGET_SIZE
    elif config.DATASET.NAME == "mnist":
        dataset = CustomMNISTDataset(
            root=config.DATASET.DATAROOT,
            channel_n=config.MODEL.CHANNEL_N,
            train=train,
            target_padding=config.DATASET.TARGET_PADDING,
            device="cpu",
        )
        cond_dim = 0
        im_height = int(28 + config.DATASET.TARGET_PADDING * 2)
        im_width = int(28 + config.DATASET.TARGET_PADDING * 2)
    elif config.DATASET.NAME == "cifar10":
        dataset = CIFAR10Dataset(
            root=config.DATASET.DATAROOT,
            channel_n=config.MODEL.CHANNEL_N,
            target_size=config.DATASET.TARGET_SIZE,
            train=train,
            device=config.DEVICE,
        )
        cond_dim = 0
        im_height = config.DATASET.TARGET_SIZE
        im_width = config.DATASET.TARGET_SIZE
    elif config.DATASET.NAME == "growing_mnist":
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
            latent_noise_channel=config.DATASET.Z_LATENT_NOISE_CHANNEL
        )
        # Condition dimension: 1 for direction + 1 for z latent if invertible
        cond_dim = 1 + 0 if config.DATASET.INVERTIBLE else 0
        if config.CFG.ENABLED and config.CFG.GOAL_CHANNELS:
            cond_dim += 2
        im_height = int(28 + config.DATASET.TARGET_PADDING * 2)
        im_width = int(28 + config.DATASET.TARGET_PADDING * 2)
    elif config.DATASET.NAME == "moving_mnist":
        dataset = MovingMNISTDataset(
            root=config.DATASET.DATAROOT,
            download=True,
            history_n=config.DATASET.HISTORY_N,
            channel_n=config.MODEL.CHANNEL_N,
            reverse_seed=config.DATASET.REVERSE_HISTORY_SEED,
        )
        cond_dim = 0
        im_height = config.DATASET.TARGET_SIZE
        im_width = config.DATASET.TARGET_SIZE
    elif config.DATASET.NAME == "celeba":
        dataset = CelebADataset(
            root_dir=config.DATASET.DATAROOT,
            split="train" if train else "test",
            img_size=config.DATASET.TARGET_SIZE,
        )
        cond_dim = 1
        im_height = config.DATASET.TARGET_SIZE
        im_width = config.DATASET.TARGET_SIZE
    else:
        raise ValueError("Invalid dataset DATASET_NAME")

    # Apply CFG wrapper if enabled and dataset is conditional
    if config.CFG.ENABLED and cond_dim > 0:
        cfg_dropout_prob = config.CFG.DROPOUT_PROB
        null_condition_type = config.CFG.NULL_CONDITION_TYPE
        preserve_channels = config.CFG.PRESERVE_CHANNELS
        dataset = CFGDatasetWrapper(dataset, cfg_dropout_prob, null_condition_type, preserve_channels)

    dataloader = DataLoader(
        dataset, batch_size=config.TRAINING.BATCH_SIZE, shuffle=True, num_workers=config.DATASET.NUM_WORKERS, pin_memory=True, drop_last=config.DATASET.DROP_LAST_BATCH if train else False
    )
    handler = DataWrapper(dataloader, config)
    return handler, cond_dim, im_height, im_width