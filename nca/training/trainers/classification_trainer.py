import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_functions import ReconstructionLoss


class ClassificationTrainer(BaseTrainer):
    """Trainer for per-pixel or per-image classification tasks.

    The CA channels are split into state channels and class-logit channels.
    The last ``num_classes`` channels of the CA output are treated as logits
    and passed to the loss function instead of the raw pixel output.

    Supports MNIST (single-channel, 10 classes) and CIFAR-10 (RGB, 10 classes)
    out of the box. Selected automatically when
    ``DATASET.NAME`` is ``"mnist"`` or ``"cifar10"``.
    """

    def _initialize_additional_components(self):
        self.freeze_channels = 1 if self.config.DATASET.NAME == "mnist" else 3
        self.num_classes = 10
        # Initialize loss function and test if from type nn.Module
        if self.loss_fn is None:
            self.loss_fn = ReconstructionLoss()
        assert isinstance(
            self.loss_fn, torch.nn.Module
        ), "Loss function must be a subclass of torch.nn.Module"

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)
        prediction_image, final_state = self.forward(initial_state, cond, target, logging=logging, freeze_channels=self.freeze_channels)
        classified_x = self.classify(prediction_image).to(self.device)
        loss_dict = self.loss_fn(classified_x, target)
        return prediction_image, final_state, loss_dict

    def classify(self, x):
        """Classify the input x."""
        return x[:, -self.num_classes :]
