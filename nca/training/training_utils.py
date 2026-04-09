import math
import torch
from torch.optim import lr_scheduler


def export_model(ca, base_fn):
    torch.save(ca.state_dict(), base_fn)


def create_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a learning rate scheduler with linear warmup followed by cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear ramp-up)
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for cosine decay.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    def lr_lambda_warmup_cosine(current_step):
        """
        Calculates the LR multiplicative factor: linear warmup then cosine decay.
        Assumes scheduler.step() is called PER BATCH/STEP.
        """
        # Ensure current_step is an integer
        current_step = int(current_step)
        if warmup_steps > 0 and current_step < warmup_steps:
            # Linear warmup phase: factor increases from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay phase
            decay_steps = total_steps - warmup_steps
            # Prevent division by zero or issues if warmup >= total steps
            if decay_steps <= 0:
                return 0.0  # End of training, LR should be minimal
            # Calculate progress within the decay phase (from 0 to 1)
            # Ensure step doesn't exceed total steps for calculation
            effective_step = min(current_step, total_steps)
            progress = float(effective_step - warmup_steps) / float(decay_steps)
            # Calculate cosine annealing factor (ranges from 1 to 0)
            # 0.5 * (1 + cos(pi * progress))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Assuming eta_min = 0, the factor scales from 1 down to 0
            return cosine_factor

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_warmup_cosine)


def create_warmup_constant_scheduler(optimizer, warmup_steps):
    """
    Creates a learning rate scheduler with linear warmup followed by a constant LR.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps (linear ramp-up).

    Returns:
        LambdaLR scheduler.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    def lr_lambda_warmup_constant(current_step):
        current_step = int(current_step)
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_warmup_constant)


def create_wsd_scheduler(optimizer, warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.0):
    """
    Creates a WSD (Warmup-Stable-Decay) learning rate scheduler.

    This is the current state-of-the-art schedule used in LLM training:
    1. Linear warmup from 0 to peak LR
    2. Constant LR for the stable phase (majority of training)
    3. Cosine decay to min_lr at the end

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps (linear ramp-up)
        stable_steps: Number of steps at constant peak LR
        decay_steps: Number of steps for final cosine decay
        min_lr_ratio: Minimum LR as ratio of peak LR (default 0.0)

    Returns:
        LambdaLR scheduler
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")
    if stable_steps < 0:
        raise ValueError("stable_steps cannot be negative.")
    if decay_steps < 0:
        raise ValueError("decay_steps cannot be negative.")

    total_steps = warmup_steps + stable_steps + decay_steps

    def lr_lambda_wsd(current_step):
        current_step = int(current_step)

        if warmup_steps > 0 and current_step < warmup_steps:
            # Phase 1: Linear warmup from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + stable_steps:
            # Phase 2: Stable at peak LR
            return 1.0
        else:
            # Phase 3: Cosine decay from 1 to min_lr_ratio
            if decay_steps <= 0:
                return min_lr_ratio
            decay_progress = float(current_step - warmup_steps - stable_steps) / float(decay_steps)
            decay_progress = min(1.0, decay_progress)  # Clamp to [0, 1]
            # Cosine decay from 1 to min_lr_ratio
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_wsd)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

_VALID_MODES = ("step", "cosine", "constant", "wsd")


def create_scheduler(optimizer, config) -> lr_scheduler.LRScheduler:
    """Return the LR scheduler specified by ``config.TRAINING.LR_SCHEDULE_MODE``.

    Modes:
        ``step``     — MultiStepLR with ``MILESTONES`` and ``LR_GAMMA``.
        ``cosine``   — Linear warmup then cosine decay.
        ``constant`` — Linear warmup then constant LR.
        ``wsd``      — Warmup-Stable-Decay; warmup uses ``WARMUP_STEPS``,
                       decay uses ``WSD_DECAY_RATIO`` × total steps,
                       stable fills the remainder.
    """
    mode = config.TRAINING.LR_SCHEDULE_MODE
    total_steps = config.TRAINING.STEPS
    warmup_steps = config.TRAINING.WARMUP_STEPS

    if mode == "step":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.TRAINING.MILESTONES,
            gamma=config.TRAINING.LR_GAMMA,
        )
    elif mode == "cosine":
        return create_warmup_cosine_scheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
    elif mode == "constant":
        return create_warmup_constant_scheduler(
            optimizer, warmup_steps=warmup_steps
        )
    elif mode == "wsd":
        decay_steps = int(total_steps * config.TRAINING.WSD_DECAY_RATIO)
        stable_steps = total_steps - warmup_steps - decay_steps
        return create_wsd_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            min_lr_ratio=config.TRAINING.WSD_MIN_LR_RATIO,
        )
    else:
        raise ValueError(
            f"Invalid LR_SCHEDULE_MODE '{mode}'. Must be one of {_VALID_MODES}."
        )