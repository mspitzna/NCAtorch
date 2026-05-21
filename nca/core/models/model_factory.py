from nca.utils.config import Config
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
        pipeline.append(ClampOutput(config.MODEL.CLAMP_OUTPUT_MIN, config.MODEL.CLAMP_OUTPUT_MAX))

    return StateUpdatePipeline(pipeline)


