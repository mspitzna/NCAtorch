# trainers/trainer.py
import os
import torch
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import functools
from abc import ABC, abstractmethod
from tqdm import tqdm
from nca.utils.config import Config
from nca.utils.visualization import save_image
from nca.training.sample_pool import SamplePool, TimeseriesSamplePool
from nca.training.training_utils import export_model
from nca.training.logger import Logger
from nca.utils.image_utils import create_warmup_cosine_scheduler
from nca.core.models.latent_wrapper import LatentWrapper


class BaseTrainer(ABC):
    def __init__(
        self,
        ca_model,
        dataloader,
        config: Config,
        config_path,
        use_latent: bool = False,
        loss_fn=None,
    ):
        self.ca_model = ca_model
        self.config: Config = config
        self.log_folder = config_path
        self.device = self.config.DEVICE
        self.loss_fn = loss_fn
        self.use_latent = use_latent
        self.setup_seed(self.config.SEED)
        self.steps = self.config.TRAINING.STEPS
        self.log_interval = self.config.TRAINING.LOG_INTERVAL
        self.save_interval = self.config.TRAINING.SAVE_INTERVAL
        self.gradient_checkpointing = config.TRAINING.GRADIENT_CHECKPOINTING
        self.accumulation_steps = config.TRAINING.GRADIENT_ACCUMULATION_STEPS
        self.scaler = (
            torch.amp.GradScaler(self.device)
            if config.TRAINING.MIXED_PRECISION
            else None
        )

        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

        self.logger = Logger(config, config_path=config_path, model=self.ca_model)
        self.output_folder = self.logger.get_output_folder()

        self.intermediate_logging_steps = (
            self.config.TRAINING.INTERMEDIATE_LOGGING_STEPS
        )

        # Ensure CA model is initialized
        assert (
            self.ca_model is not None
        ), "CA model is not initialized. Please initialize the CA model before proceeding."

        if self.use_latent:
            # Ensure LatentWrapper is initialized ONLY if use_latent is True
            self.latent_wrapper = LatentWrapper(self.ca_model, self.config)
            print("Latent training enabled. Initializing LatentWrapper.")
        else:
            self.latent_wrapper = None
            print("Latent training disabled. Operating in pixel space.")

        # Initialize models, optimizers, and schedulers
        self._initialize_base_optimizers()

        # Initialize sample pool if enabled
        self.pool = self._initialize_sample_pool(config, self.device)

        self._initialize_additional_components()  # Hook for children

    def _initialize_base_optimizers(self):
        """Initialize models, optimizers, and schedulers with warmup + cosine decay for 'exp' mode."""
        self.optimizer = optim.Adam(
            self.ca_model.parameters(),
            lr=self.config.TRAINING.LEARNING_RATE,  # Base LR (peak LR after warmup)
            betas=self.config.TRAINING.OPTIMIZER_BETAS,  # Use config for betas
        )

        # Scheduler switch
        if self.config.TRAINING.LR_SCHEDULE_MODE == "step":
            # MultiStepLR typically steps per epoch. Warmup + cosine decay is harder here.
            print(
                "Warning: Warmup + cosine decay not implemented for 'step' mode. Using MultiStepLR."
            )
            self.lr_scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.TRAINING.MILESTONES,  # Assumed to be in epochs
                gamma=self.config.TRAINING.LR_GAMMA,
            )
        elif self.config.TRAINING.LR_SCHEDULE_MODE == "cosine":
            # Use LambdaLR with a custom function for warmup + cosine decay
            warmup_steps = self.config.TRAINING.WARMUP_STEPS
            total_steps = self.config.TRAINING.STEPS

            print(
                f"Using LambdaLR scheduler with linear warmup for {warmup_steps} steps "
                f"followed by cosine decay over remaining {total_steps - warmup_steps} steps."
            )

            self.lr_scheduler = create_warmup_cosine_scheduler(
                self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps
            )
        else:
            raise ValueError("Invalid LR_SCHEDULE_MODE. Must be 'step' or 'cosine'.")

    def _evolve(self, state_in, conds, iter_n, freeze_channels=None, logging=False):
        """Evolves the CA model on the given state (image x or latent z)."""
        state = state_in
        with torch.set_grad_enabled(self.ca_model.training):
            # Helper function (can stay the same)
            def evolve_step(step, current_state):
                new_state = self.ca_model(
                    current_state, conds, freeze_channels=freeze_channels
                )
                # Log state (x or z)
                if logging and (step in self.intermediate_logging_steps):
                    # Note: Logger might need adjustment if it assumes images for state logs
                    self.logger.add_state_log(step, new_state)
                return new_state

            # Loop or checkpointing (remains the same)
            if not self.gradient_checkpointing:
                for step in range(iter_n):
                    state = evolve_step(step, state)
            else:
                layers = [functools.partial(evolve_step, i) for i in range(iter_n)]
                state = torch.utils.checkpoint.checkpoint_sequential(
                    layers,
                    self.config.TRAINING.GRADIENT_CHECKPOINT_SEGMENTS,
                    state,
                    use_reentrant=False,
                )

        return state  # Returns final state (x or z)

    @abstractmethod
    def _initialize_additional_components(self):
        """Hook for child classes to add extra modules (like AE, Discriminator, etc.)."""
        pass

    def forward(self, initial_state, cond, target, logging=False, freeze_channels=None):
        # initial_state is either image x or latent z, prepared by train loop
        iter_n = self.get_iter_range()

        if self.use_latent:
            # Input state0 is latent z
            z0 = initial_state
            # Evolve in latent space using the agnostic _evolve
            z_final = self._evolve(
                z0, cond, iter_n, logging=logging, freeze_channels=freeze_channels
            )  # _evolve now just runs CA
            # Decode final latent state for loss calculation
            prediction_image = self.latent_wrapper.decode(z_final)
            final_state_for_commit = z_final  # Commit latent state
        else:
            # Input state0 is image x
            x0 = initial_state
            # Evolve in image space using the agnostic _evolve
            x_final = self._evolve(
                x0, cond, iter_n, logging=logging, freeze_channels=freeze_channels
            )  # _evolve just runs CA
            # Prediction is the final image state
            prediction_image = x_final
            final_state_for_commit = x_final  # Commit image state

        return prediction_image, final_state_for_commit

    @abstractmethod
    def _run_train_step(self, initial_state, cond, target, logging=False):
        """
        This method should be implemented by child classes to define the training step logic.
        It should handle the forward pass, loss computation, and backward pass.
        """
        pass

    def train(self):
        try:
            self.current_step = 0  # Initialize current step counter
            accumulated_steps = 0  # Track gradient accumulation
            for i in tqdm(range(self.steps), desc="Training", mininterval=1):
                try:
                    seed, cond, target = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.dataloader)
                    seed, cond, target = next(self.data_iter)

                # --- Prepare Initial State for this Step (state0: x or z) ---
                if self.use_latent:
                    with torch.no_grad():
                        # Encode only the raw seed data
                        state0_initial = self.latent_wrapper.encode(
                            seed.to(self.device)
                        )
                else:
                    state0_initial = seed.to(self.device)  # Use raw seed image

                # --- Apply Sample Pool (operates on state0_initial: x or z) ---
                if self.pool:
                    self.pool.step()  # Update pool scheduling (e.g., seed_ratio)
                    # sample_and_replace takes initial state (x or z) and image targets/conds
                    # returns modified state (x or z) and potentially modified image targets/conds
                    state0, cond, target = self.pool.sample_and_replace(
                        current_batch_data=state0_initial, cond=cond, true=target
                    )
                else:
                    state0 = state0_initial  # State to start evolution from (x or z)

                # --- Perform Training Step ---
                # Pass the prepared state0 (x or z), image cond, image target
                # train_step calls forward, which handles internal encode/decode if needed based on self.use_latent
                # train_step returns final_state (x or z) and prediction_image
                prediction_image, final_state = self._run_train_step(
                    state0,
                    None if self.config.COND_DIM == 0 else cond,
                    target,  # Use target (potentially modified by pool)
                    logging=((i + 1) % self.log_interval == 0),
                )

                if (i + 1) % self.log_interval == 0:
                    if self.use_latent:
                        self.add_img_logs(
                            self.latent_wrapper.decode(state0), prediction_image, target
                        )
                    else:
                        self.add_img_logs(state0, prediction_image, target)

                # Update gradient accumulation counter
                accumulated_steps += 1

                # --- Optimizer Step / LR Scheduling / Grad Accumulation ---
                if accumulated_steps % self.accumulation_steps == 0:
                    # Gradient clipping and optimizer step
                    optimizer_step_ran = True
                    if self.config.TRAINING.MIXED_PRECISION and self.scaler is not None:
                        scale_before = self.scaler.get_scale()
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.ca_model.parameters(),
                            max_norm=self.config.TRAINING.GRADIENT_CLIPPING_NORM,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        # GradScaler lowers the scale when it skips the optimizer step (overflow).
                        optimizer_step_ran = scale_after >= scale_before
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.ca_model.parameters(),
                            max_norm=self.config.TRAINING.GRADIENT_CLIPPING_NORM,
                        )
                        self.optimizer.step()
                        optimizer_step_ran = True

                    self.optimizer.zero_grad()
                    accumulated_steps = 0  # Reset accumulation counter
                    if self.lr_scheduler is not None and optimizer_step_ran:
                        self.lr_scheduler.step()  # Step scheduler after optimizer
                        self.logger.add_metric(
                            "lr", self.optimizer.param_groups[0]["lr"]
                        )

                # --- Commit to Pool ---
                if self.pool:
                    # Commit the final state (x or z) from this step
                    # Ensure state is in the same space as initial state for consistency
                    # final_state is already in the correct space (pixel or latent) matching pool expectations
                    self.pool.commit(
                        final_state, cond, target
                    )  # Use final_state (x or z)

                # Logging and visualization
                if (i + 1) % self.log_interval == 0:
                    self.commit_logs(i, images=True, silent=True)
                else:
                    self.commit_logs(i, silent=True)

                # Save model
                if (i + 1) % self.save_interval == 0:
                    self.save_model(i + 1)

            # Save final model
            self.save_model("final")
            return 0  # Success
        except KeyboardInterrupt:
            print("Training interrupted by user. Saving current state...")
            self.save_model(f"interrupted_step_{i}")
            return -1  # Failure
        except Exception as e:
            print(f"Training failed with error: {e}")

            # print exact issue
            print("Exact issue:", e)
            print("Stack trace:", e.__traceback__)
            import traceback

            traceback.print_exc()

            # Save model in case of unexpected failure
            try:
                self.save_model("error_state")
            except:
                pass
            return -1  # Failure

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

            if self.use_latent:
                x0 = self.latent_wrapper.encode(x0)
            # Perform evolution (forward steps)
            output = self._evolve(x0, cond, iter_n, logging=False)

            if self.use_latent:
                output = self.latent_wrapper.decode(output)

        return output

    def setup_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"Setting random seed to {seed}")

    def _initialize_sample_pool(self, config: Config, device):
        """Initialize the sample pool if pooling is enabled."""
        if config.PATTERN_POOL.ENABLED:
            pool = (
                SamplePool(
                    pool_size=config.PATTERN_POOL.POOL_SIZE,
                    delay=config.PATTERN_POOL.POOL_DELAY,
                    damage_ratio=config.PATTERN_POOL.POOL_DMG_RATIO,
                    mutation_ratio=config.PATTERN_POOL.POOL_MUTATION_RATIO,
                    device=device,
                    replace_after_layer=(
                        1 if config.DATASET.NAME == "mnist" else None
                    ),  # keep first layer and just replace rest of channels
                )
                if not config.PATTERN_POOL.TIMESERIES_POOL
                else TimeseriesSamplePool(
                    pool_size=config.PATTERN_POOL.POOL_SIZE,
                    delay=config.PATTERN_POOL.POOL_DELAY,
                    damage_ratio=config.PATTERN_POOL.POOL_DMG_RATIO,
                    device=device,
                    replace_after_layer=(
                        1 if config.DATASET.NAME == "mnist" else None
                    ),  # keep first layer and just replace rest of channels
                )
            )
            pool.enable_seed_scheduling(
                total_steps=config.TRAINING.STEPS,
                start_ratio=config.PATTERN_POOL.POOL_START_RATIO,
                end_ratio=config.PATTERN_POOL.POOL_END_RATIO,
            )

            if config.PATTERN_POOL.POOL_DMG_DELAY is not None:
                pool.enable_damage_delay(config.PATTERN_POOL.POOL_DMG_DELAY)

            return pool
        return None

    def save_model(self, iteration):
        """Save the model at the given iteration."""
        self.ca_model.eval()
        export_model(
            self.ca_model, os.path.join(self.output_folder, f"ca_{iteration}.pt")
        )
        self.ca_model.train()

    def get_iter_range(self):
        """Get the range of iterations."""
        if self.config.TRAINING.ITER_N_MIN == self.config.TRAINING.ITER_N_MAX:
            # Fixed iteration number
            iter_n = self.config.TRAINING.ITER_N_MIN
        else:
            iter_n = torch.randint(
                self.config.TRAINING.ITER_N_MIN, self.config.TRAINING.ITER_N_MAX, (1,)
            ).item()

        return iter_n

    def add_img_logs(self, x0, x, target):
        self.logger.add_img_logs(x0, x, target)

    def commit_logs(self, step, images=False, silent=False):
        if images:
            if self.use_latent:
                for key, val in self.logger.get_state_logs().items():
                    # val is already on correct device, no need to move it
                    decoded_val = self.latent_wrapper.decode(val).detach().cpu()
                    self.logger.add_state_log(key, decoded_val)
            self.logger.log_images(step + 1)
        self.logger.log_metrics(step + 1, silent=silent)
        self.logger.reset_metrics()

    def save_to_image(self, x, path):
        save_image(x, path)
