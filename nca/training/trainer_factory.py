from nca.utils.config import Config
from nca.core.losses.loss_factory import create_loss_fn
from nca.training.trainers.image_gen_trainer import ImageGenTrainer
from nca.training.trainers.adversarial_trainer import AdversarialTrainer
from nca.training.trainers.classification_trainer import ClassificationTrainer

# Registry of all available trainers.
#
# Maps a TRAINING.TRAINER_TYPE string to a trainer class.
# To add a new trainer:
#   1. Implement the class in nca/training/trainers/.
#   2. Add one entry here.
#   3. If your trainer only makes sense for specific datasets, add an entry to
#      _DATASET_REQUIRED_TRAINER below so users get a clear error early.
#
# Available trainers:
#   "image_gen"      — standard image generation / reconstruction training
#   "adversarial"    — GAN training with a patch discriminator (WGAN-GP)
#   "classification" — self-classifying NCA (mnist, cifar10)
TRAINER_REGISTRY = {
    "image_gen":      ImageGenTrainer,
    "adversarial":    AdversarialTrainer,
    "classification": ClassificationTrainer,
}

# Datasets that are only compatible with a specific trainer.
# Using a different trainer with these datasets raises a ValueError.
# Datasets not listed here accept any trainer.
_DATASET_REQUIRED_TRAINER = {
    "mnist":   "classification",
    "cifar10": "classification",
}


def create_trainer(config: Config, model, dataloader, config_path: str):
    """Instantiate the correct trainer for the given config.

    Trainer selection works in two modes:

    **Auto (default):** ``TRAINING.TRAINER_TYPE`` is ``None``.
    The trainer is chosen based on the dataset and adversarial flag:

    **Explicit:** Set ``TRAINING.TRAINER_TYPE`` in the config YAML to one of
    the keys in ``TRAINER_REGISTRY``. A ``ValueError`` is raised if the chosen
    trainer is incompatible with the dataset (e.g. ``"image_gen"`` with mnist).

    Args:
        config: Full ``Config`` object.
        model: Instantiated ``CAModel``.
        dataloader: Training dataloader.
        config_path: Path to the YAML config file (stored in the trainer for
            checkpointing and reproducibility).

    Returns:
        An instantiated trainer ready to call ``.train()``.

    Raises:
        ValueError: If ``TRAINER_TYPE`` is not in ``TRAINER_REGISTRY`` or is
            incompatible with the configured dataset.
    """
    trainer_key = config.TRAINING.TRAINER_TYPE

    if trainer_key is None:
        # Auto-select based on dataset and adversarial flag
        dataset = config.DATASET.NAME
        if dataset in _DATASET_REQUIRED_TRAINER:
            trainer_key = _DATASET_REQUIRED_TRAINER[dataset]
        elif config.ADVERSARIAL.ENABLED:
            trainer_key = "adversarial"
        else:
            trainer_key = "image_gen"
    else:
        # Validate explicit choice against dataset constraints
        if trainer_key not in TRAINER_REGISTRY:
            raise ValueError(
                f"Unknown TRAINER_TYPE: '{trainer_key}'. "
                f"Valid options: {sorted(TRAINER_REGISTRY)}"
            )
        dataset = config.DATASET.NAME
        required = _DATASET_REQUIRED_TRAINER.get(dataset)
        if required is not None and trainer_key != required:
            raise ValueError(
                f"Dataset '{dataset}' requires TRAINER_TYPE='{required}', "
                f"but got '{trainer_key}'. Either change the dataset or the trainer."
            )

    loss_fn = create_loss_fn(config)
    use_latent = config.LATENT_TRAINING.ENABLED
    trainer_cls = TRAINER_REGISTRY[trainer_key]

    print(f"Trainer: {trainer_key}")

    # AdversarialTrainer manages its own loss internally
    if trainer_key == "adversarial":
        return trainer_cls(model, dataloader, config, config_path, use_latent=use_latent)

    return trainer_cls(model, dataloader, config, config_path, loss_fn=loss_fn, use_latent=use_latent)
