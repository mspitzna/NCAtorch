"""Trainer of the spatial reasoning tasks.

The host image trainer with two changes: the loss is chosen with
``EXTENSIONS.SPATIAL_REASONING.LOSS`` and also receives the condition (Sudoku clues, maze
walls and endpoints), and the optimizer is AdamW with
``EXTENSIONS.SPATIAL_REASONING.WEIGHT_DECAY``. ``TRAINING.LOSS_FN`` is not used.
"""

from torch import optim

from nca.training.trainers.image_gen_trainer import ImageGenTrainer
from nca.training.training_utils import create_scheduler

from .config import settings
from .losses import create_loss


class SpatialReasoningTrainer(ImageGenTrainer):
    USES_FRAMEWORK_LOSS = False
    MAX_LOG_SAMPLES = 8

    def _initialize_additional_components(self):
        self.loss_fn = create_loss(self.config)

    def _initialize_base_optimizers(self):
        weight_decay = settings(self.config).WEIGHT_DECAY
        training = self.config.TRAINING
        self.optimizer = optim.AdamW(
            self.ca_model.parameters(), lr=training.LEARNING_RATE, betas=training.OPTIMIZER_BETAS,
            weight_decay=weight_decay,
        )
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        print(f"Using AdamW (weight decay {weight_decay}), LR schedule {training.LR_SCHEDULE_MODE}")

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)
        prediction, final_state = self.forward(initial_state, cond, target, logging=logging)
        return prediction, final_state, self.loss_fn(prediction, target, cond)

    def add_img_logs(self, x0, x, target, cond=None):
        """Render only a few samples; Sudoku boards are drawn cell by cell."""
        n = self.MAX_LOG_SAMPLES
        self.logger.set_state_logs({k: v[:n] for k, v in self.logger.get_state_logs().items()})
        super().add_img_logs(x0[:n], x[:n], target[:n], cond[:n] if cond is not None else None)
