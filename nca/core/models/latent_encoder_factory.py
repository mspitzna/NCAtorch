import torch.nn as nn
from nca.utils.config import Config
from nca.core.models.auto_encoder.ae import AutoEncoder
from nca.core.models.auto_encoder.vae import VAE
from nca.core.models.auto_encoder.vqvae import VQVAE
from nca.core.losses.loss_functions import VGGLoss, ReconstructionLoss

# Registry for VAE reconstruction loss types.
_VAE_RECON_LOSS_REGISTRY = {
    "l1":  lambda: nn.L1Loss(reduction='sum'),
    "mse": lambda: nn.MSELoss(reduction='sum'),
}


def _vae_criterion(cfg: Config, device: str):
    recon_loss_type = cfg.LATENT_TRAINING.VAE_RECON_LOSS_TYPE
    if recon_loss_type not in _VAE_RECON_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown VAE_RECON_LOSS_TYPE: '{recon_loss_type}'. "
            f"Valid options: {sorted(_VAE_RECON_LOSS_REGISTRY)}"
        )
    return VGGLoss(
        device=device,
        vgg_loss_weight=cfg.LATENT_TRAINING.VAE_VGG_LOSS_WEIGHT,
        l1_loss_weight=cfg.LATENT_TRAINING.VAE_RECON_LOSS_WEIGHT,
        loss_fn=_VAE_RECON_LOSS_REGISTRY[recon_loss_type](),
    )


# Registry of all available latent encoder types.
# Key -> factory function (config, device, inference_only) -> (model, criterion, kl_beta)
# To add a new encoder: add an entry here. Tests and the config validator pick it up automatically.
LATENT_ENCODER_REGISTRY = {
    "AE": lambda cfg, device, inference_only: (
        AutoEncoder(
            in_channels=cfg.LATENT_TRAINING.LATENT_AE_IN_CHANNEL,
            out_channels=cfg.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL,
            latent_channels=cfg.LATENT_TRAINING.LATENT_AE_CHANNEL,
            compression_level=cfg.LATENT_TRAINING.LATENT_AE_COMPRESSION,
        ).to(device),
        None if inference_only else ReconstructionLoss(overflow_loss=cfg.TRAINING.OVERFLOW_LOSS),
        None,
    ),
    "VAE": lambda cfg, device, inference_only: (
        VAE(
            in_channels=cfg.LATENT_TRAINING.LATENT_AE_IN_CHANNEL,
            out_channels=cfg.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL,
            latent_channels=cfg.LATENT_TRAINING.LATENT_AE_CHANNEL,
            base_channels=cfg.LATENT_TRAINING.VAE_BASE_CHANNELS,
            num_downsamples=cfg.LATENT_TRAINING.VAE_NUM_DOWNSAMPLES,
            norm_groups=cfg.LATENT_TRAINING.VAE_NORM_GROUPS,
            activation_fn=nn.LeakyReLU(0.2, inplace=True),
            final_activation=nn.Tanh(),
        ).to(device),
        None if inference_only else _vae_criterion(cfg, device),
        cfg.LATENT_TRAINING.VAE_KL_BETA,
    ),
    "VQVAE": lambda cfg, device, inference_only: (
        VQVAE(
            in_channels=cfg.LATENT_TRAINING.LATENT_AE_IN_CHANNEL,
            out_channels=cfg.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL,
            latent_channels=cfg.LATENT_TRAINING.LATENT_AE_CHANNEL,
            num_embeddings=cfg.LATENT_TRAINING.VQVAE_NUM_EMBEDDINGS,
            commitment_cost=cfg.LATENT_TRAINING.VQVAE_COMMITMENT_COST,
            base_channels=cfg.LATENT_TRAINING.VAE_BASE_CHANNELS,
            num_downsamples=cfg.LATENT_TRAINING.VAE_NUM_DOWNSAMPLES,
            norm_groups=cfg.LATENT_TRAINING.VAE_NORM_GROUPS,
        ).to(device),
        None if inference_only else ReconstructionLoss(overflow_loss=cfg.TRAINING.OVERFLOW_LOSS),
        None,
    ),
}


def create_latent_encoder(config: Config, device: str, inference_only: bool = False):
    key = config.LATENT_TRAINING.ENCODER_TYPE
    if key not in LATENT_ENCODER_REGISTRY:
        raise ValueError(
            f"Invalid ENCODER_TYPE: '{key}'. Valid options: {sorted(LATENT_ENCODER_REGISTRY)}"
        )
    return LATENT_ENCODER_REGISTRY[key](config, device, inference_only)
