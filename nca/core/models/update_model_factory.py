from nca.utils.config import Config
from nca.core.models.ca.update_models import SimpleMLPUpdate, ResNetUpdate

# Registry of all available update models.
#
# Maps a MODEL.NAME string to a factory lambda with signature:
#   (in_channels: int, out_channels: int, config: Config) -> UpdateModelBase
#
# Adding a new update model:
#   1. Implement the class in nca/core/models/ca/update_models.py.
#   2. Add one entry here — the config validator and tests pick it up automatically.
#
# Available models:
#   "MLP"    — stack of 1×1 convolutions with configurable hidden channels
#   "ResNet" — residual blocks with configurable depth
UPDATE_MODEL_REGISTRY = {
    "MLP": lambda in_ch, out_ch, cfg: SimpleMLPUpdate(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=cfg.MODEL.HIDDEN_CHANNELS,
        final_activation=cfg.MODEL.FINAL_ACTIVATION,
    ),
    "ResNet": lambda in_ch, out_ch, cfg: ResNetUpdate(
        in_channels=in_ch,
        out_channels=out_ch,
        hidden_channels=cfg.MODEL.HIDDEN_CHANNELS,
        num_blocks=cfg.MODEL.RESNET_BLOCKS,
        final_activation=cfg.MODEL.FINAL_ACTIVATION,
    ),
}


def create_update_model(config: Config, in_channels: int, out_channels: int, device: str):
    key = config.MODEL.NAME
    if key not in UPDATE_MODEL_REGISTRY:
        raise ValueError(f"Invalid model name: '{key}'. Valid options: {sorted(UPDATE_MODEL_REGISTRY)}")
    print(f"Creating {key}. in_channels: {in_channels}, out_channels: {out_channels}")
    return UPDATE_MODEL_REGISTRY[key](in_channels, out_channels, config).to(device)
