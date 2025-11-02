from nca.core.losses.loss_functions import (
    ReconstructionLoss,
    VGGStyleOTLoss,
    PixelCrossEntropyLoss,
    PixelAccuracyLoss,
    LPIPSLoss,
    TotalAgreementLoss,
    OverflowLoss,
    SeedPreservingReconstructionLoss,
    L1Loss,
    ImageCrossEntropyLoss
)
from nca.utils.config import Config

def create_metric(metric: str, device: str = "cuda"):
    if metric == "mse":
        return ReconstructionLoss()
    elif metric == "mae":
        return ReconstructionLoss(loss_fn="mae")
    elif metric == "vgg":
        return VGGStyleOTLoss(device=device)
    elif metric == "p_ce":
        return PixelCrossEntropyLoss()
    elif metric == "acc":
        return PixelAccuracyLoss()
    elif metric == "ta":
        return TotalAgreementLoss()
    elif metric == "lpips":
        return LPIPSLoss(device=device)
    else:
        raise ValueError(f"Invalid metric: {metric}")
    


def create_loss_fn(config: Config):
    if config.TRAINING.LOSS_FN == "mse":
        return ReconstructionLoss(overflow_loss=config.TRAINING.OVERFLOW_LOSS)
    elif config.TRAINING.LOSS_FN == "seed_preserving_l1":
        import torch.nn as nn
        return SeedPreservingReconstructionLoss(
            loss_fn=nn.L1Loss(reduction='none'),
            seed_weight=getattr(config.TRAINING, 'SEED_WEIGHT', 10.0),
            seed_radius=getattr(config.TRAINING, 'SEED_RADIUS', 3),
            overflow_loss=config.TRAINING.OVERFLOW_LOSS,
            backward_seed_weight=getattr(config.TRAINING, 'BACKWARD_SEED_WEIGHT', None)
        )
    elif config.TRAINING.LOSS_FN == "seed_preserving_mse":
        import torch.nn as nn
        return SeedPreservingReconstructionLoss(
            loss_fn=nn.MSELoss(reduction='none'),  # Default to MSE if not specified
            seed_weight=getattr(config.TRAINING, 'SEED_WEIGHT', 10.0),
            seed_radius=getattr(config.TRAINING, 'SEED_RADIUS', 3),
            overflow_loss=config.TRAINING.OVERFLOW_LOSS
        )
    elif config.TRAINING.LOSS_FN == "l1":
        return L1Loss(overflow_loss=config.TRAINING.OVERFLOW_LOSS, config=config)
    elif config.TRAINING.LOSS_FN == "p_ce":
        return PixelCrossEntropyLoss()
    elif config.TRAINING.LOSS_FN == "lpips":
        return LPIPSLoss(device=config.DEVICE)
    elif config.TRAINING.LOSS_FN == "vggstyle":
        return VGGStyleOTLoss(device=config.DEVICE, overflow_loss=config.TRAINING.OVERFLOW_LOSS)
    elif config.TRAINING.LOSS_FN == "i_ce":
        return ImageCrossEntropyLoss(overflow_loss=config.TRAINING.OVERFLOW_LOSS)
    elif config.TRAINING.LOSS_FN == "overflow":
        return OverflowLoss(overflow_weight=config.TRAINING.OVERFLOW_LOSS)
    else:
        raise ValueError("Invalid loss function")