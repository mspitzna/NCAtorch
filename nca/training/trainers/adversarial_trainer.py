import torch
import torch.nn as nn
import wandb
from nca.training.trainers.base_trainer import BaseTrainer # Assuming this is your original BaseTrainer file
from nca.core.models.critic import Critic
from nca.training.training_utils import create_warmup_cosine_scheduler
from nca.core.losses.loss_factory import create_loss_fn
from nca.core.losses.loss_functions import LPIPSLoss

class AdversarialTrainer(BaseTrainer):
    """Adversarial trainer implementing WGAN-GP for high-quality image generation.

    Maintains a separate ``Critic`` network alongside the CA generator and
    alternates their updates according to the ``D_N_CRITIC`` schedule.  The
    total generator loss is a weighted sum of the registered loss selected
    by ``TRAINING.LOSS_FN``, the adversarial term,
    and an optional additional LPIPS term weighted by ``LPIPS_WEIGHT``.

    Because two optimizers and two scalers are involved, this trainer overrides
    ``_run_train_step`` directly rather than implementing ``_compute_losses``.

    Selected automatically when ``ADVERSARIAL.ENABLED`` is ``true``.
    Configure via ``TRAINING.TRAINER_TYPE: "adversarial"`` to force it.

    Key config fields:
        ``ADVERSARIAL.ADV_WEIGHT`` — weight of the adversarial loss term.
        ``ADVERSARIAL.RECON_WEIGHT`` — weight of the reconstruction loss term.
        ``ADVERSARIAL.D_N_CRITIC`` — batch interval between generator backward passes.
        Generator gradients accumulate across these passes; the critic updates each batch.
        ``ADVERSARIAL.D_START_TRAINING`` — step at which adversarial training begins.
        ``ADVERSARIAL.D_GP_WEIGHT`` — gradient-penalty coefficient.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_step = 0  # Initialize current step counter
    
    def _initialize_additional_components(self):
        """Initializes the Critic model and its dedicated optimizer/scheduler."""
        # Calculate input channels for critic based on flags
        critic_image_channels = self.config.ADVERSARIAL.D_IN_CHANNELS
        if self.config.ADVERSARIAL.SEED_TO_CRITIC:
            critic_image_channels += self.config.MODEL.CHANNEL_N
            
        self.critic = Critic(
            image_channels=critic_image_channels,
            condition_channels=self.config.COND_DIM,
            features=self.config.ADVERSARIAL.D_FEATURES,
            use_norm=False 
        ).to(self.device)
        print(f"Generator ('ca_model') params: {sum(p.numel() for p in self.ca_model.parameters() if p.requires_grad):,}")
        print(f"Critic params: {sum(p.numel() for p in self.critic.parameters() if p.requires_grad):,}")
        
        # Critic's dedicated optimizer and scheduler
        self.d_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.ADVERSARIAL.D_LEARNING_RATE, betas=self.config.TRAINING.OPTIMIZER_BETAS)
        # The critic's clock starts when discriminator training begins.
        critic_steps = max(1, self.config.TRAINING.STEPS - self.config.ADVERSARIAL.D_START_TRAINING)
        self.d_scheduler = create_warmup_cosine_scheduler(
            self.d_optimizer,
            self.config.ADVERSARIAL.D_WARMUP_STEPS,
            critic_steps,
            min_lr_ratio=self.config.ADVERSARIAL.D_GAMMA,
        )
        
        # Load hyperparameters from config
        self.adv_weight = self.config.ADVERSARIAL.ADV_WEIGHT
        self.recon_weight = self.config.ADVERSARIAL.RECON_WEIGHT
        self.lpips_weight = self.config.ADVERSARIAL.LPIPS_WEIGHT
        self.gp_weight = self.config.ADVERSARIAL.D_GP_WEIGHT
        self.d_start_training = self.config.ADVERSARIAL.D_START_TRAINING
        self.seed_to_critic = self.config.ADVERSARIAL.SEED_TO_CRITIC

        # Separate GradScaler for the critic (recommended for GANs)
        self.d_scaler = (
            torch.amp.GradScaler(self.device)
            if self.config.TRAINING.MIXED_PRECISION
            else None
        )

        # Initialize reconstruction loss based on config
        recon_loss_type = self.config.TRAINING.LOSS_FN
        self.recon_loss = create_loss_fn(self.config)
        self.lpips_loss = None
        if self.lpips_weight > 0:
            self.lpips_loss = (
                self.recon_loss if recon_loss_type == "lpips"
                else LPIPSLoss(device=self.device, net=self.config.TRAINING.LPIPS_NET)
            )
            self.lpips_loss.eval().requires_grad_(False)
        print(f"Using reconstruction loss: {recon_loss_type}")

        if self.config.LOGGING.WANDB:
            wandb.watch(self.critic, log="all", log_freq=self.config.LOGGING.LOG_INTERVAL)

    def _run_train_step(self, initial_state, cond, target, logging=False):
        """
        This method overrides the BaseTrainer's train_step. It implements the
        alternating training schedule for the WGAN-GP.
        """
        # --- Prepare inputs ---
        initial_state = initial_state.to(self.device, non_blocking=True)
        target_img = target.to(self.device, non_blocking=True)
        condition = cond.to(self.device, non_blocking=True) if cond is not None else None
        
        if condition is not None and condition.dim() == 2:
            condition = condition.unsqueeze(-1).unsqueeze(-1)
            condition = condition.expand(-1, -1, target_img.shape[2], target_img.shape[3])
        
        # BaseTrainer owns the generator's gradient group and clears it after updates.
        
        # --- 1. TRAIN THE CRITIC (on every step after start_training) ---
        if self.current_step >= self.d_start_training and self.adv_weight > 0:
            self.critic.train()
            self.d_optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
                # Generate a fake image but without tracking gradients for the generator
                # This is more efficient as we don't build a computation graph for the generator here
                with torch.no_grad():
                    prediction_image, _ = self.forward(initial_state, condition, target_img)
                
                fake_images = prediction_image.detach()

                real_input = self._prepare_critic_input(target_img, initial_state, condition)
                fake_input = self._prepare_critic_input(fake_images, initial_state, condition)

                real_logits = self.critic(real_input)
                fake_logits = self.critic(fake_input)

                # Calculate WGAN-GP loss
                loss_fake = torch.mean(fake_logits)
                loss_real = -torch.mean(real_logits)
                gp = self.gradient_penalty(initial_state, target_img[:, :self.config.ADVERSARIAL.D_IN_CHANNELS], fake_images[:, :self.config.ADVERSARIAL.D_IN_CHANNELS], condition)
                d_loss = loss_fake + loss_real + self.gp_weight * gp
            
            # Backward pass and optimizer step for the Critic
            d_optimizer_step_ran = True
            if self.config.TRAINING.MIXED_PRECISION:
                scale_before = self.d_scaler.get_scale()
                self.d_scaler.scale(d_loss).backward()
                self.d_scaler.unscale_(self.d_optimizer)
                self._clip_gradients(self.critic.parameters())
                self.d_scaler.step(self.d_optimizer)
                self.d_scaler.update()
                scale_after = self.d_scaler.get_scale()
                # Skip LR scheduling if the optimizer step was dropped because of overflow.
                d_optimizer_step_ran = scale_after >= scale_before
            else:
                d_loss.backward()
                self._clip_gradients(self.critic.parameters())
                self.d_optimizer.step()
                d_optimizer_step_ran = True
            
            if self.d_scheduler and d_optimizer_step_ran:
                self.d_scheduler.step()
            
            # Log critic metrics
            self.logger.add_metric("d_loss", d_loss.item())
            self.logger.add_metric("loss_real_mean", -loss_real.item())
            self.logger.add_metric("loss_fake_mean", loss_fake.item())
            self.logger.add_metric("wasserstein_distance", -loss_real.item() - loss_fake.item())
            self.logger.add_metric("gradient_penalty", gp.item())

        # --- 2. TRAIN THE GENERATOR (conditionally, every n_critic steps) ---
        self.ca_model.train()
        
        # Check if it's the generator's turn to be updated
        is_generator_turn = self._should_train_generator(self.current_step)

        # Generator batches contribute to the gradient group owned by BaseTrainer.
        if is_generator_turn:
            with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
                # We need to run the forward pass with gradient tracking to update the generator
                prediction_image, final_state = self.forward(initial_state, condition, target_img, logging=logging)
                total_g_loss = torch.tensor(0.0, device=self.device)

                # Add reconstruction loss
                recon_loss = self.recon_loss(prediction_image, target_img)["total_loss"]
                self.logger.add_metric("recon_loss", recon_loss.item())
                if self.recon_weight > 0:
                    total_g_loss += self.recon_weight * recon_loss / self.accumulation_steps

                # Perceptual supervision uses full-resolution images, including warmup.
                if self.lpips_loss is not None:
                    lpips_loss = (
                        recon_loss if self.lpips_loss is self.recon_loss
                        else self.lpips_loss(prediction_image, target_img)["total_loss"]
                    )
                    self.logger.add_metric("lpips_loss", lpips_loss.item())
                    total_g_loss += self.lpips_weight * lpips_loss / self.accumulation_steps

                # Add adversarial loss only when it's the generator's turn during adversarial phase
                if self.current_step >= self.d_start_training and self.adv_weight > 0:
                    fake_input_for_g = self._prepare_critic_input(prediction_image, initial_state, condition)
                    fake_logits_for_g = self.critic(fake_input_for_g)
                    g_loss = -torch.mean(fake_logits_for_g)
                    self.logger.add_metric("g_loss", g_loss.item())
                    total_g_loss += (self.adv_weight * g_loss) / self.accumulation_steps

            # Perform backward pass. This populates gradients for the main optimizer.
            if self.config.TRAINING.MIXED_PRECISION:
                self.scaler.scale(total_g_loss).backward()
            else:
                total_g_loss.backward()
            
            self.logger.add_metric("g_total_loss", total_g_loss.item() * self.accumulation_steps)

        else:
            # If it's not the generator's turn, just run a forward pass without gradients
            # to get the output for logging and returning.
            with torch.no_grad():
                prediction_image, final_state = self.forward(initial_state, condition, target_img, logging=logging)
                # Still log the reconstruction loss for a smooth graph
                recon_loss = self.recon_loss(prediction_image, target_img)["total_loss"]
                self.logger.add_metric("recon_loss", recon_loss.item())
        
        # --- Finalization ---
        final_state_for_commit = final_state.detach()
        prediction_image_for_log = prediction_image.detach()
        
        self.current_step += 1
        
        return prediction_image_for_log, final_state_for_commit

    def _should_train_generator(self, step: int) -> bool:
        adv = self.config.ADVERSARIAL
        return (
            adv.ADV_WEIGHT == 0
            or step < adv.D_START_TRAINING
            or step % adv.D_N_CRITIC == 0
        )

    def _prepare_critic_input(self, images, initial_state, condition=None):
        """Resize images, conditions and optional seeds to the critic's resolution."""
        factor = self.config.ADVERSARIAL.D_DOWNSCALE_FACTOR
        size = (images.shape[-2] // factor, images.shape[-1] // factor)
        if min(size) < 1:
            raise ValueError("D_DOWNSCALE_FACTOR exceeds the image dimensions.")

        def resize(tensor):
            if tensor.shape[-2:] == size:
                return tensor
            return nn.functional.interpolate(tensor, size=size, mode="area")

        parts = [resize(images[:, :self.config.ADVERSARIAL.D_IN_CHANNELS])]
        if condition is not None:
            if condition.ndim == 2:
                condition = condition[:, :, None, None].expand(-1, -1, *size)
            parts.append(resize(condition))
        if self.seed_to_critic:
            parts.append(resize(initial_state))
        return torch.cat(parts, dim=1)

    def gradient_penalty(self, initial_state, real_images, fake_images, condition=None):
        """Calculates the gradient penalty for WGAN-GP."""
        batch_size, c, h, w = real_images.shape
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device).repeat(1, c, h, w)
        interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
        
        interpolated_input = self._prepare_critic_input(interpolated_images, initial_state, condition)

        interpolated_input.requires_grad_(True)
        
        critic_interpolated = self.critic(interpolated_input)
        grad_outputs = torch.ones_like(critic_interpolated, device=self.device)
        
        gradients = torch.autograd.grad(
            outputs=critic_interpolated, 
            inputs=interpolated_input, 
            grad_outputs=grad_outputs, 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
