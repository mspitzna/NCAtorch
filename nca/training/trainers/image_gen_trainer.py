import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_factory import create_metric


class ImageGenTrainer(BaseTrainer):

    def _initialize_additional_components(self):
        # Initialize loss function and test if from type nn.Module
        if self.loss_fn is None:
            self.loss_fn = create_metric("mse")
        assert isinstance(self.loss_fn, torch.nn.Module), "Loss function must be a subclass of torch.nn.Module"

    def _run_train_step(self, initial_state, cond, target, logging=False):
        initial_state = initial_state.to(self.device, non_blocking=True)
        if cond is not None:
            cond = cond.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
            prediction_image, final_state_for_commit = self.forward(initial_state, cond, target, logging=logging)

            # Check if cond is inpainting mask
            if self.config.DATASET.COND_INPAINTING_MASK:
                prediction_image = prediction_image * cond + (1 - cond) * target

            # Compute loss - slice prediction to match target dims if needed
            pred_for_loss = prediction_image[:, :target.shape[1]]
            loss_dict = self.loss_fn(pred_for_loss, target)
            # Scale loss for accumulation
            total_loss = loss_dict["total_loss"] / self.accumulation_steps

        if torch.isnan(total_loss):
            raise ValueError("Loss is NaN. Halting training.")

        if self.config.TRAINING.MIXED_PRECISION:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        with torch.no_grad():
            self.logger.add_metrics(loss_dict)
        # Return final state (x or z) for pool commit, and prediction image
        return prediction_image.detach(), final_state_for_commit.detach()