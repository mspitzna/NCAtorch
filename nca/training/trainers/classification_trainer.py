import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_functions import ReconstructionLoss
from nca.utils.image_utils import to_grayscale, to_rgb, to_rgba
from nca.utils.visualization import save_image


class ClassificationTrainer(BaseTrainer):

    def _initialize_additional_components(self):
        self.freeze_channels = 0 if self.config.DATASET.NAME == "mnist" else 2
        self.num_classes = 10
        # Initialize loss function and test if from type nn.Module
        if self.loss_fn is None:
            self.loss_fn = ReconstructionLoss()
        assert isinstance(
            self.loss_fn, torch.nn.Module
        ), "Loss function must be a subclass of torch.nn.Module"

    def inference(self, x0, cond=None, iter_n=None):
        """
        Perform inference using the trained model.

        Args:
            x0 (torch.Tensor): Initial input tensor.
            cond (torch.Tensor, optional): Condition tensor, if applicable.
            iter_n (int, optional): Number of iterations to evolve. Defaults to trained settings.

        Returns:
            torch.Tensor: The output tensor after model evolution.
        """
        self.ca_model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            x0 = x0.to(self.device)
            if cond is not None:
                cond = cond.to(self.device)

            # Determine the number of iterations
            if iter_n is None:
                iter_n = self.get_iter_range()

            # Perform evolution (forward steps)
            output = self._evolve(
                x0, cond, iter_n, freeze_channels=self.freeze_channels, logging=False
            )

        return output

    def _run_train_step(self, initial_state, cond, target, logging=False):
        initial_state = initial_state.to(self.device, non_blocking=True)
        if cond is not None:
            cond = cond.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
            prediction_image, final_state_for_commit = self.forward(initial_state, cond, target, logging=logging, freeze_channels=self.freeze_channels)
            # Compute loss using the prediction image and target
            classified_x = self.classify(prediction_image).to(self.device)
            loss_dict = self.loss_fn(classified_x, target)
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

    def add_img_logs(self, x0, x, target, cond=None):
        if self.config.DATASET.NAME == "mnist":
            # Store for visualization
            self.logger.add_img_logs(
                self.dataloader.get_dataset().generate_colored_image(
                    to_grayscale(x0), self.classify(x0)
                ),
                self.dataloader.get_dataset().generate_colored_image(
                    to_grayscale(x), self.classify(x)
                ),
                self.dataloader.get_dataset().generate_colored_image(
                    to_grayscale(x0), target  # Use current seed image with target labels
                ),
            )
            for key, val in self.logger.get_state_logs().items():
                val = val.to(self.device)
                self.logger.add_state_log(key, self.dataloader.get_dataset().generate_colored_image(
                    to_grayscale(val), self.classify(val)
                ))
        else:

            self.logger.add_img_logs(
                self.dataloader.get_dataset().apply_per_pixel_coloring(
                    to_rgba(x0), target
                ),
                self.dataloader.get_dataset().apply_per_pixel_coloring(
                    to_rgba(x), self.classify(x)
                ),
                to_rgb(x0[:, :3]),
            )
            for key, val in self.logger.get_state_logs().items():
                val = val.to(self.device)
                self.logger.add_state_log(key, self.dataloader.get_dataset().apply_per_pixel_coloring(
                    to_rgba(val), self.classify(val)
                ))

    def save_to_image(self, x, path):
        tmp = x.clone().detach().cpu().unsqueeze(0)
        if self.config.DATASET.NAME == "mnist":
            img = self.dataloader.get_dataset().generate_colored_image(
                to_grayscale(tmp), self.classify(tmp)
            )
        else:
            img = self.dataloader.get_dataset().apply_per_pixel_coloring(
                to_rgba(tmp), self.classify(tmp)
            )
        save_image(img, path)

    def classify(self, x):
        """Classify the input x."""
        return x[:, -self.num_classes :]
