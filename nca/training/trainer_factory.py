from nca.utils.config import Config
from nca.core.losses.loss_factory import create_loss_fn
from nca.training.trainers.image_gen_trainer import ImageGenTrainer
from nca.training.trainers.adversarial_trainer import AdversarialTrainer
from nca.training.trainers.classification_trainer import ClassificationTrainer


def create_trainer(config: Config, model, dataloader, config_path):

    loss_fn = create_loss_fn(config)
    use_latent = config.LATENT_TRAINING.ENABLED

    # Determine which trainer to use based on config
    if config.DATASET.NAME == "mnist":
        trainer = ClassificationTrainer(
            model, dataloader, config, config_path, loss_fn=loss_fn, use_latent=use_latent
        )
    elif config.DATASET.NAME == "cifar10":
        trainer = ClassificationTrainer(
            model, dataloader, config, config_path, loss_fn=loss_fn, use_latent=use_latent
        )
    elif config.ADVERSARIAL.ENABLED:
        trainer = AdversarialTrainer(model, dataloader, config, config_path, use_latent=use_latent)
    else:
        trainer = ImageGenTrainer(model, dataloader, config, config_path, loss_fn=loss_fn, use_latent=use_latent)
    return trainer