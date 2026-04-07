import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_functions import ReconstructionLoss


class ImageGenTrainer(BaseTrainer):
    """Trainer for unconditional and conditional image generation.

    Runs the CA forward pass and minimises a pixel-level reconstruction loss
    (default: MSE via ``ReconstructionLoss``).

    Configure via ``TRAINING.TRAINER_TYPE: "image_gen"`` or leave unset for
    automatic selection on non-classification datasets.
    """

    def _initialize_additional_components(self):
        if self.loss_fn is None:
            self.loss_fn = ReconstructionLoss()
        assert isinstance(self.loss_fn, torch.nn.Module), "Loss function must be a subclass of torch.nn.Module"

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)
        prediction_image, final_state = self.forward(initial_state, cond, target, logging=logging)
        loss_dict = self.loss_fn(prediction_image, target)
        return prediction_image, final_state, loss_dict