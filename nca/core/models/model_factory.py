import torch.nn as nn
from nca.utils.config import Config
from nca.core.models.auto_encoder.ae import AutoEncoder
from nca.core.models.auto_encoder.vae import VAE
from .ca.state_updates import (
    ApplyDelta,
    ClampOutput,
    FireRateMask,
    LivingMask,
    NoiseInjection,
    StateUpdatePipeline,
)
from nca.core.models.perception_factory import create_perception_module
from nca.core.models.update_model_factory import create_update_model
from nca.core.models.ca.ca_model import CAModel
from nca.core.losses.loss_functions import VGGLoss, ReconstructionLoss


def create_model(config: Config, cond_dim, img_height, img_width):
    device = config.DEVICE

    def get_img_dims(height, width, compression):
        factor = pow(2, compression)
        return int(height / factor), int(width / factor)

    if config.LATENT_TRAINING.ENABLED:
        channel_out = config.LATENT_TRAINING.LATENT_AE_CHANNEL
        img_height, img_width = get_img_dims(
            img_height, img_width, config.LATENT_TRAINING.LATENT_AE_COMPRESSION
        )
    else:
        channel_out = config.MODEL.CHANNEL_OUT

    perception_module = create_perception_module(config, cond_dim, device)
    update_model = create_update_model(config, perception_module.get_out_channel(), channel_out, device)
    state_update_pipeline = get_state_update_pipeline(config, device)

    ca = CAModel(
        device=device,
        use_positional_embeddings=config.MODEL.USE_POSITIONAL_EMBEDDINGS,
        img_height=img_height,
        img_width=img_width,
        perception_module=perception_module,
        update_model_module=update_model,
        state_update_pipeline=state_update_pipeline
    )
    return ca.to(device)


def get_state_update_pipeline(config: Config, device) -> StateUpdatePipeline:
    pipeline = []

    if config.MODEL.NOISE_INJECTION > 0.0:
        pipeline.append(NoiseInjection(config.MODEL.NOISE_INJECTION))
    if config.MODEL.FIRE_RATE < 1.0:
        pipeline.append(FireRateMask(config.MODEL.FIRE_RATE))

    pipeline.append(ApplyDelta())

    if not config.LATENT_TRAINING.ENABLED and config.MODEL.LIVING_MASK:
        pipeline.append(LivingMask(alpha_channel=config.MODEL.LIVING_MASK_INDEX))
    if config.MODEL.CLAMP_OUTPUT:
        pipeline.append(ClampOutput())

    return StateUpdatePipeline(pipeline)


def get_latent_encoder(config: Config, device, inference_only=False):
    # --- Model Initialization ---
    common_args = {
        "in_channels": config.LATENT_TRAINING.LATENT_AE_IN_CHANNEL,
        "out_channels": config.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL,
        "latent_channels": config.LATENT_TRAINING.LATENT_AE_CHANNEL,
        "compression_level": config.LATENT_TRAINING.LATENT_AE_COMPRESSION,
    }

    model_type = config.LATENT_TRAINING.ENCODER_TYPE

    if model_type == "VAE":
        print("Initializing VAE_ResNet model...")
        # --- Get VAE specific args from config ---
        base_channels = config.LATENT_TRAINING.VAE_BASE_CHANNELS
        num_downsamples = config.LATENT_TRAINING.VAE_NUM_DOWNSAMPLES
        norm_groups = config.LATENT_TRAINING.VAE_NORM_GROUPS
        activation_fn = nn.LeakyReLU(0.2, inplace=True)  # Default or configure
        # Use Tanh which doesn't saturate as hard as Sigmoid
        # Tanh outputs [-1, 1], better gradient flow than Sigmoid [0, 1]
        final_activation = nn.Tanh()

        model = VAE(
            in_channels=config.LATENT_TRAINING.LATENT_AE_IN_CHANNEL,
            out_channels=config.LATENT_TRAINING.LATENT_AE_OUT_CHANNEL,
            latent_channels=config.LATENT_TRAINING.LATENT_AE_CHANNEL,
            base_channels=base_channels,
            num_downsamples=num_downsamples,
            norm_groups=norm_groups,
            activation_fn=activation_fn,
            final_activation=final_activation,
        )
        model = model.to(device)

        if inference_only:
            return model, None, None

        if config.LATENT_TRAINING.VAE_RECON_LOSS_TYPE == "l1":
            loss_fn = nn.L1Loss(reduction='sum')
        elif config.LATENT_TRAINING.VAE_RECON_LOSS_TYPE == "mse":
            loss_fn = nn.MSELoss(reduction='sum')
        else:
            raise ValueError(f"Unknown VAE_RECON_LOSS_TYPE: {config.LATENT_TRAINING.VAE_RECON_LOSS_TYPE}")

        reconstruction_criterion = VGGLoss(
            device=device,
            vgg_loss_weight=config.LATENT_TRAINING.VAE_VGG_LOSS_WEIGHT,
            l1_loss_weight=config.LATENT_TRAINING.VAE_RECON_LOSS_WEIGHT,
            loss_fn=loss_fn
        )

        vae_kl_beta = config.LATENT_TRAINING.VAE_KL_BETA
        print(f"Using VAE with KL Beta: {vae_kl_beta}")
        return model, reconstruction_criterion, vae_kl_beta
    elif model_type == "AE":
        print("Initializing AutoEncoder model...")
        model = AutoEncoder(**common_args)
        model = model.to(device)
        if inference_only:
            return model, None, None
        # Use standard MSE loss for AE (mean reduction is default)
        ae_criterion = ReconstructionLoss(overflow_loss=config.TRAINING.OVERFLOW_LOSS)
        return model, ae_criterion, None
    else:
        raise ValueError(f"Unknown LATENT_TRAINING.MODEL_TYPE: {model_type}")
