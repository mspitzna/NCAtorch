import torch.nn as nn
from nca.utils.config import Config, PerceptionConfig
from nca.core.models.auto_encoder.ae import AutoEncoder
from nca.core.models.auto_encoder.vae import VAE
from .ca.perceptions import (
    AttentionPerception,
    ConvPerception,
    SobelPerception,
    DeformableConvPerception,
    ResidualConvPerception,
    MultiPerception,
)
from nca.core.models.ca.update_models import SimpleMLPUpdate, ResNetUpdate
from nca.core.models.ca.ca_model import CAModel
from nca.core.losses.loss_functions import VGGLoss, ReconstructionLoss, L1Loss


def create_model(config: Config, cond_dim, img_height, img_width):
    device = config.DEVICE
    use_positional_embeddings = config.MODEL.USE_POSITIONAL_EMBEDDINGS
    noise_injection = config.MODEL.NOISE_INJECTION
    perception_output_dim = config.MODEL.HIDDEN_CHANNELS[0]
    fire_rate = config.MODEL.FIRE_RATE

    def get_img_dims(height, width, compression):
        factor = pow(2, compression)
        return int(height / factor), int(width / factor)

    if config.LATENT_TRAINING.ENABLED:
        channel_n = config.LATENT_TRAINING.LATENT_AE_CHANNEL
        channel_out = config.LATENT_TRAINING.LATENT_AE_CHANNEL
        img_height, img_width = get_img_dims(
            img_height, img_width, config.LATENT_TRAINING.LATENT_AE_COMPRESSION
        )
        living_mask = False
        living_mask_index = None
    else:
        channel_n = config.MODEL.CHANNEL_N
        channel_out = config.MODEL.CHANNEL_OUT
        living_mask = config.MODEL.LIVING_MASK
        living_mask_index = config.MODEL.LIVING_MASK_INDEX


    perception_module = get_perception(config, cond_dim, perception_output_dim, device)
    update_model = get_update_model(config, perception_module.get_out_channel(), channel_out, device)

    ca = CAModel(
        channel_n=channel_n,
        channel_out=channel_out,
        cond_channel_n=cond_dim,
        device=device,
        use_positional_embeddings=use_positional_embeddings,
        img_height=img_height,
        img_width=img_width,
        perception_module=perception_module,
        living_mask=living_mask,
        living_mask_indice=living_mask_index,
        update_model_module=update_model,
        noise_injection=noise_injection,
        fire_rate=fire_rate,
        normalize_output=config.MODEL.NORMALIZE_OUTPUT,
    )
    
    # Move to device once after full construction
    return ca.to(device)


def get_update_model(config: Config, in_channel_n, channel_out, device):
    """
    Create the update model based on the configuration.
    """
    model_name = config.MODEL.NAME
    print(f"Creating {model_name}. The value of in_channel_n is: {in_channel_n}")
    if model_name == "ConvCA":
        model = SimpleMLPUpdate(
            in_channels=in_channel_n,
            out_channels=channel_out,
            hidden_channels=config.MODEL.HIDDEN_CHANNELS,
            final_activation=config.MODEL.FINAL_ACTIVATION,
            device=device,
        )
        return model.to(device)
    elif model_name == "ResNetCA":
        model = ResNetUpdate(
            in_channels=in_channel_n,
            out_channels=channel_out,
            hidden_channels=config.MODEL.HIDDEN_CHANNELS,
            num_blocks=config.MODEL.RESNET_BLOCKS,
            final_activation=config.MODEL.FINAL_ACTIVATION,
            device=device,
        )
        return model.to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def get_perception(config: Config, cond_dim, perception_output_dim, device):
    # Determine the number of channels in the state.
    channel_n = (
        config.MODEL.CHANNEL_N
        if not config.LATENT_TRAINING.ENABLED
        else config.LATENT_TRAINING.LATENT_AE_CHANNEL
    )
    in_channel_n = (
        channel_n
        #+ (cond_dim if config.LATENT_TRAINING.ENABLED is False else 0)
        + cond_dim
        + (2 if config.MODEL.USE_POSITIONAL_EMBEDDINGS else 0)
    )

    # Helper to create a single perception module from a PerceptionConfig.
    def create_perception(percep_cfg: PerceptionConfig):
        mode = percep_cfg.MODE
        kernel_size = percep_cfg.KERNEL_SIZE
        # For modes that use dilation, use the provided value.
        dilation = percep_cfg.DILATION

        if mode == "attention":
            return AttentionPerception(
                in_channel=in_channel_n,
                out_channel=perception_output_dim,
                kernel_size=kernel_size,
            )
        elif mode == "sobel":
            return SobelPerception(
                in_channel=in_channel_n,
                device=device,
            )
        elif mode == "deformable_conv":
            return DeformableConvPerception(
                in_channel=in_channel_n,
                out_channel=perception_output_dim,
                kernel_size=kernel_size,
                device=device,
            )
        elif mode == "residual_conv":
            return ResidualConvPerception(
                in_channel=in_channel_n,
                out_channel=perception_output_dim,
                kernel_size=kernel_size,
                device=device,
            )
        else:
            # Default to regular convolution perception.
            return ConvPerception(
                in_channel=in_channel_n,
                out_channel=perception_output_dim,
                kernel_size=kernel_size,
                device=device,
                dilation=dilation,
            )

    # Create perception modules based on the list in config.MODEL.PERCEPTIONS.
    perception_modules = [create_perception(pc) for pc in config.MODEL.PERCEPTIONS]

    # Create a single module or wrap multiple modules in a MultiPerception.
    if len(perception_modules) == 1:
        perception_module = perception_modules[0]
    else:
        perception_module = MultiPerception(perception_modules)

    # Log the details of the created perception module.
    modes = ", ".join([pc.MODE for pc in config.MODEL.PERCEPTIONS])
    kernel_sizes = ", ".join([str(pc.KERNEL_SIZE) for pc in config.MODEL.PERCEPTIONS])
    print(
        f"Perception: {modes} with in_channel_n: {in_channel_n}, "
        f"out_channel: {perception_module.get_out_channel()}, kernel_sizes: {kernel_sizes}"
    )
    return perception_module


def get_latent_encoder(config: Config, device):
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
        # Use standard MSE loss for AE (mean reduction is default)
        ae_criterion = ReconstructionLoss(overflow_loss=config.TRAINING.OVERFLOW_LOSS)
        return model, ae_criterion, None
    else:
        raise ValueError(f"Unknown LATENT_TRAINING.MODEL_TYPE: {model_type}")

