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
