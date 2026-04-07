import torch.nn as nn
from nca.core.losses.loss_functions import (
    ReconstructionLoss,
    VGGStyleOTLoss,
    PixelCrossEntropyLoss,
    PixelAccuracyLoss,
    LPIPSLoss,
    TotalAgreementLoss,
    OverflowLoss,
    L1Loss,
    ImageCrossEntropyLoss
)
from nca.utils.config import Config

# Registry of all available metrics.
# Key -> factory function (device: str) -> Loss
# To add a new metric: add an entry here. create_metric and tests pick it up automatically.
METRIC_REGISTRY = {
    "mse":   lambda device: ReconstructionLoss(),
    "mae":   lambda device: ReconstructionLoss(loss_fn=nn.L1Loss()),
    "vgg":   lambda device: VGGStyleOTLoss(device=device),
    "p_ce":  lambda device: PixelCrossEntropyLoss(),
    "acc":   lambda device: PixelAccuracyLoss(),
    "ta":    lambda device: TotalAgreementLoss(),
    "lpips": lambda device: LPIPSLoss(device=device),
}

# Registry of all available training loss functions.
# Key -> factory function (config: Config) -> Loss
# To add a new loss: add an entry here. create_loss_fn and tests pick it up automatically.
LOSS_FN_REGISTRY = {
    "mse":      lambda cfg: ReconstructionLoss(overflow_loss=cfg.TRAINING.OVERFLOW_LOSS),
    "l1":       lambda cfg: L1Loss(overflow_loss=cfg.TRAINING.OVERFLOW_LOSS, config=cfg),
    "lpips":    lambda cfg: LPIPSLoss(device=cfg.DEVICE),
    "vggstyle": lambda cfg: VGGStyleOTLoss(device=cfg.DEVICE, overflow_loss=cfg.TRAINING.OVERFLOW_LOSS),
    "p_ce":     lambda cfg: PixelCrossEntropyLoss(),
    "i_ce":     lambda cfg: ImageCrossEntropyLoss(overflow_loss=cfg.TRAINING.OVERFLOW_LOSS),
    "overflow": lambda cfg: OverflowLoss(overflow_weight=cfg.TRAINING.OVERFLOW_LOSS),
}


def create_metric(metric: str, device: str = "cuda"):
    if metric not in METRIC_REGISTRY:
        raise ValueError(f"Invalid metric: '{metric}'. Valid options: {sorted(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[metric](device)


def create_loss_fn(config: Config):
    key = config.TRAINING.LOSS_FN
    if key not in LOSS_FN_REGISTRY:
        raise ValueError(f"Invalid loss function: '{key}'. Valid options: {sorted(LOSS_FN_REGISTRY)}")
    return LOSS_FN_REGISTRY[key](config)
