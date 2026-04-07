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
from nca.training.training_utils import (
    create_warmup_constant_scheduler,
    create_warmup_cosine_scheduler,
    export_model,
)
from nca.training.logger import Logger
from nca.core.models.latent_wrapper import LatentWrapper


class BaseTrainer(ABC):
    """Abstract base class for all NCA trainers.

    Handles the shared training infrastructure so subclasses only need to
    implement two methods:

    - ``_initialize_additional_components()`` — set up extra modules, loss
      functions, or optimizers that are specific to the trainer.
    - ``_compute_losses(initial_state, cond, target, logging)`` — run the
      forward pass and return ``(prediction_image, final_state, loss_dict)``.
      ``BaseTrainer`` wraps this call in AMP autocast and takes care of
      accumulation scaling, NaN detection, backward, and metric logging.

    For trainers that require multiple optimizers (e.g. GANs), override
    ``_run_train_step`` directly instead of ``_compute_losses``.

    Optional hooks: ``_on_step_end(step)`` and ``_on_train_end()``.

    Attributes:
        ca_model: The cellular automaton model being trained.
        config (Config): Full experiment configuration.
        device (str): Target compute device (e.g. ``"cuda"``).
        loss_fn: Primary loss module; may be ``None`` until
            ``_initialize_additional_components`` sets it.
        use_latent (bool): Whether training runs in latent space via
            ``self.latent_wrapper``.
        scaler: AMP ``GradScaler`` or ``None`` when mixed precision is off.
        pool: Sample pool for persistent-state training, or ``None``.
        logger (Logger): Handles metric and image logging to disk / W&B.
        accumulation_steps (int): Gradient accumulation window.
    """

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
        self._interval_metric_buffer = {}

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

    @abstractmethod
    def _initialize_additional_components(self):
        """Hook for child classes to add extra modules (like AE, Discriminator, etc.)."""
        pass

    def _on_step_end(self, step: int):
        """Hook called at the end of each training step. Override in subclasses."""
        pass

    def _on_train_end(self):
        """Hook called at the end of training. Override in subclasses."""
        pass

    def _to_device(self, *tensors):
        """Move tensors to the training device, passing None through unchanged."""
        return [t.to(self.device, non_blocking=True) if t is not None else None for t in tensors]

    def _compute_losses(self, initial_state, cond, target, logging=False):
        """Forward pass and loss computation.

        Override this method to define your training logic. ``BaseTrainer``
        calls it from ``_run_train_step`` inside an ``autocast`` context and
        handles accumulation scaling, AMP, NaN checking, and the backward pass
        automatically.

        Use ``self._to_device()`` to move tensors and ``self.forward()`` to
        run the CA. Return a ``loss_dict`` with at least a ``"total_loss"`` key.

        Args:
            initial_state: Seed state tensor (pixel or latent space).
            cond: Condition tensor or ``None``.
            target: Target image tensor.
            logging: Whether this step should log intermediate states.

        Returns:
            Tuple ``(prediction_image, final_state, loss_dict)``.

        Raises:
            NotImplementedError: If neither this method nor ``_run_train_step``
                is overridden by the subclass.
        """
        raise NotImplementedError(
            "Implement _compute_losses(initial_state, cond, target, logging) "
            "or override _run_train_step() for custom multi-loss setups (e.g. GAN)."
        )

    def _run_train_step(self, initial_state, cond, target, logging=False):
        """Default training step.

        Calls ``_compute_losses``, then handles accumulation scaling, AMP,
        NaN checking, backward, and metric logging. Override this method
        directly only when you need full control (e.g. multi-optimizer GAN
        training with separate scalers).
        """
        with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
            prediction_image, final_state, loss_dict = self._compute_losses(
                initial_state, cond, target, logging=logging
            )
        total_loss = loss_dict["total_loss"] / self.accumulation_steps

        if torch.isnan(total_loss):
            raise ValueError("Loss is NaN. Halting training.")

        if self.config.TRAINING.MIXED_PRECISION:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        with torch.no_grad():
            self.logger.add_metrics(loss_dict)
        return prediction_image.detach(), final_state.detach()


    def _initialize_base_optimizers(self):
        """Initialize models, optimizers, and schedulers."""
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
        elif self.config.TRAINING.LR_SCHEDULE_MODE == "constant":
            warmup_steps = self.config.TRAINING.WARMUP_STEPS

            print(
                f"Using LambdaLR scheduler with linear warmup for {warmup_steps} steps "
                "followed by constant learning rate."
            )

            self.lr_scheduler = create_warmup_constant_scheduler(
                self.optimizer, warmup_steps=warmup_steps
            )
        else:
            raise ValueError("Invalid LR_SCHEDULE_MODE. Must be 'step', 'cosine', or 'constant'.")

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
                            self.latent_wrapper.decode(state0), prediction_image, target, cond
                        )
                    else:
                        self.add_img_logs(state0, prediction_image, target, cond)

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

                # Hook for subclass-specific per-step logic
                self._on_step_end(i)

                # Logging and visualization
                if (i + 1) % self.log_interval == 0:
                    self.commit_logs(i, images=True, silent=self.logger.use_wandb)
                else:
                    self.commit_logs(i, silent=True)

                # Save model
                if (i + 1) % self.save_interval == 0:
                    self.save_model(i + 1)

            # Hook for subclass-specific end-of-training logic
            self._on_train_end()

            # Final logging
            self.commit_logs(i, silent=True)

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

    def add_img_logs(self, x0, x, target, cond=None):
        dataset = self.dataloader.get_dataset()
        if hasattr(dataset, 'batch_to_rgb'):
            if not self.use_latent:
                for key, val in self.logger.get_state_logs().items():
                    _, val_vis, _ = dataset.batch_to_rgb(x0, val, target, cond)
                    self.logger.add_state_log(key, val_vis)
            x0_vis, x_vis, target_vis = dataset.batch_to_rgb(x0, x, target, cond)
        else:
            x0_vis, x_vis, target_vis = x0, x, target
        self.logger.add_img_logs(x0_vis, x_vis, target_vis)

    def commit_logs(self, step, images=False, silent=False):
        metrics_snapshot = self.logger.peek_metrics()
        if metrics_snapshot:
            for key, value in metrics_snapshot.items():
                self._interval_metric_buffer.setdefault(key, []).append(value)
            if not images and not silent:
                print(f"Step {step + 1}: {metrics_snapshot}", flush=True)

        # Always forward metrics to logger outputs (e.g., wandb) but suppress console prints here.
        self.logger.log_metrics(step + 1)

        if images:
            if self.use_latent:
                for key, val in self.logger.get_state_logs().items():
                    # val is already on correct device, no need to move it
                    decoded_val = self.latent_wrapper.decode(val).detach().cpu()
                    self.logger.add_state_log(key, decoded_val)
            self.logger.log_images(step + 1)
            if not silent and self._interval_metric_buffer:
                averaged_metrics = {
                    key: float(np.mean(values))
                    for key, values in self._interval_metric_buffer.items()
                }
                print(f"Step {step + 1}: {averaged_metrics}", flush=True)
            # Reset the running buffer regardless of print path to avoid stale accumulation.
            self._interval_metric_buffer.clear()

        self.logger.reset_metrics()