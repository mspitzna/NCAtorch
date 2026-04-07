from nca.utils.config import Config
from nca.core.models.ca.update_models import SimpleMLPUpdate, ResNetUpdate

# Registry of all available update models.
# Key -> factory function (in_channels, out_channels, config) -> UpdateModelBase
# To add a new update model: add an entry here. Tests and the config validator pick it up automatically.
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
