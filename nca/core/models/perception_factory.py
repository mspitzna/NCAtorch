from nca.utils.config import Config, PerceptionConfig
from nca.core.models.ca.perceptions import (
    AttentionPerception,
    ConvPerception,
    SobelPerception,
    DeformableConvPerception,
    ResidualConvPerception,
    MultiPerception,
    MultiHeadAttentionPerception,
)

# Registry of all available perception modules.
#
# Maps a MODE string to a factory lambda with signature:
#   (in_channels: int, cfg: PerceptionConfig, device: str) -> Perception
#
# Adding a new perception module:
#   1. Implement the class in nca/core/models/ca/perceptions.py.
#   2. Add one entry here — the config validator and tests pick it up automatically.
#
# Available modules:
#   "conv"           — standard dilated convolution
#   "attention"      — single-head local attention with optional relative pos bias
#   "mh_attention"   — multi-head attention with optional layer-norm and FFN
#   "sobel"          — fixed Sobel edge-detection filters (no learned parameters)
#   "deformable_conv"— deformable convolution with learned offsets
#   "residual_conv"  — residual convolutional block
PERCEPTION_REGISTRY = {
    "conv": lambda in_ch, cfg, dev: ConvPerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE, dilation=cfg.DILATION, device=dev
    ),
    "attention": lambda in_ch, cfg, dev: AttentionPerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE,
        num_heads=cfg.NUM_HEADS, use_rel_pos_bias=cfg.USE_REL_POS_BIAS,
    ),
    "mh_attention": lambda in_ch, cfg, dev: MultiHeadAttentionPerception(
        in_ch, cfg.OUT_CHANNEL, cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.KERNEL_SIZE,
        use_layer_norm=cfg.USE_LAYER_NORM, include_ffn=cfg.INCLUDE_FFN,
        use_rel_pos_bias=cfg.USE_REL_POS_BIAS,
    ),
    "sobel": lambda in_ch, cfg, dev: SobelPerception(in_ch, dev),
    "deformable_conv": lambda in_ch, cfg, dev: DeformableConvPerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE, device=dev
    ),
    "residual_conv": lambda in_ch, cfg, dev: ResidualConvPerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE, device=dev
    ),
}


def create_perception_module(config: Config, cond_dim: int, device: str):
    channel_n = (
        config.MODEL.CHANNEL_N
        if not config.LATENT_TRAINING.ENABLED
        else config.LATENT_TRAINING.LATENT_AE_CHANNEL
    )
    in_channel = (
        channel_n
        + cond_dim
        + (2 if config.MODEL.USE_POSITIONAL_EMBEDDINGS else 0)
    )

    modules = [
        PERCEPTION_REGISTRY[pc.MODE](in_channel, pc, device)
        for pc in config.MODEL.PERCEPTIONS
    ]

    perception = modules[0] if len(modules) == 1 else MultiPerception(modules)

    modes = ", ".join(pc.MODE for pc in config.MODEL.PERCEPTIONS)
    kernel_sizes = ", ".join(str(pc.KERNEL_SIZE) for pc in config.MODEL.PERCEPTIONS)
    print(
        f"Perception: {modes} with in_channel: {in_channel}, "
        f"out_channel: {perception.get_out_channel()}, kernel_sizes: {kernel_sizes}"
    )
    return perception
