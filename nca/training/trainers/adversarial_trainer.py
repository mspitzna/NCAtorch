import torch
import torch.nn as nn
import wandb
from nca.training.trainers.base_trainer import BaseTrainer # Assuming this is your original BaseTrainer file
from nca.core.models.critic import Critic
from nca.training.training_utils import create_warmup_cosine_scheduler
from nca.core.losses.loss_functions import LPIPSLoss, ReconstructionLoss, L1Loss

class AdversarialTrainer(BaseTrainer):
    """
    Implements a stable and conventional adversarial training logic (WGAN-GP).
    This trainer updates the critic on each step and the generator every 'n_critic' steps
    to ensure the critic is robust before the generator is updated.
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
        self.d_scheduler = create_warmup_cosine_scheduler(self.d_optimizer, self.config.ADVERSARIAL.D_WARMUP_STEPS, self.config.TRAINING.STEPS)
        
        # Load hyperparameters from config
        self.adv_weight = self.config.ADVERSARIAL.ADV_WEIGHT
        self.recon_weight = self.config.ADVERSARIAL.RECON_WEIGHT
        self.gp_weight = self.config.ADVERSARIAL.D_GP_WEIGHT
        self.n_critic = self.config.ADVERSARIAL.D_N_CRITIC
        self.d_start_training = self.config.ADVERSARIAL.D_START_TRAINING
        self.seed_to_critic = self.config.ADVERSARIAL.SEED_TO_CRITIC

        # This counter tracks steps to decide when to update the generator
        self.step_counter = 0

        # Separate GradScaler for the critic (recommended for GANs)
        self.d_scaler = (
            torch.amp.GradScaler(self.device)
            if self.config.TRAINING.MIXED_PRECISION
            else None
        )

        # Initialize reconstruction loss based on config
        recon_loss_type = self.config.TRAINING.LOSS_FN
        if recon_loss_type == "lpips":
            self.recon_loss = LPIPSLoss(device=self.device)
        elif recon_loss_type == "l1":
            self.recon_loss = L1Loss(overflow_loss=self.config.TRAINING.OVERFLOW_LOSS)
        elif recon_loss_type == "mse":
            self.recon_loss = ReconstructionLoss(overflow_loss=self.config.TRAINING.OVERFLOW_LOSS)
        else:
            raise ValueError(f"Unknown RECON_LOSS_TYPE: {recon_loss_type}")
        print(f"Using reconstruction loss: {recon_loss_type}")

        if self.config.WANDB:
            wandb.watch(self.critic, log="all", log_freq=self.config.TRAINING.LOG_INTERVAL)

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
        
        # Always zero out the generator's gradients. The BaseTrainer will call step() later.
        # Gradients will only be present if it's the generator's turn to be trained.
        self.optimizer.zero_grad()
        
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

                # Handle inpainting masks if necessary
                if self.config.DATASET.COND_INPAINTING_MASK:
                    fake_images = fake_images * condition + (1 - condition) * target_img

                # Construct critic inputs based on flags
                real_parts = [target_img[:, :self.config.ADVERSARIAL.D_IN_CHANNELS]]
                fake_parts = [fake_images[:, :self.config.ADVERSARIAL.D_IN_CHANNELS]]
                
                if condition is not None:
                    real_parts.append(condition)
                    fake_parts.append(condition)
                
                if self.seed_to_critic:
                    real_parts.append(initial_state)
                    fake_parts.append(initial_state)
                
                real_input = torch.cat(real_parts, dim=1)
                fake_input = torch.cat(fake_parts, dim=1)

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
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config.TRAINING.GRADIENT_CLIPPING_NORM)
                self.d_scaler.step(self.d_optimizer)
                self.d_scaler.update()
                scale_after = self.d_scaler.get_scale()
                # Skip LR scheduling if the optimizer step was dropped because of overflow.
                d_optimizer_step_ran = scale_after >= scale_before
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config.TRAINING.GRADIENT_CLIPPING_NORM)
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
        is_generator_turn = (self.step_counter % self.n_critic == 0)

        # Always calculate reconstruction loss for logging and for generator updates.
        # Generator update only happens if adversarial training is active.
        if (self.current_step < self.d_start_training) or is_generator_turn:
            with torch.amp.autocast(device_type=self.device, enabled=self.config.TRAINING.MIXED_PRECISION):
                # We need to run the forward pass with gradient tracking to update the generator
                prediction_image, final_state = self.forward(initial_state, condition, target_img, logging=logging)
                total_g_loss = torch.tensor(0.0, device=self.device)

                # Add reconstruction loss
                recon_loss = self.recon_loss(prediction_image, target_img)["total_loss"]
                self.logger.add_metric("recon_loss", recon_loss.item())
                if self.recon_weight > 0:
                    total_g_loss += self.recon_weight * recon_loss / self.accumulation_steps

                # Add adversarial loss only when it's the generator's turn during adversarial phase
                if self.current_step >= self.d_start_training and self.adv_weight > 0:
                    fake_parts_for_g = [prediction_image[:, :self.config.ADVERSARIAL.D_IN_CHANNELS]]
                    if condition is not None:
                        fake_parts_for_g.append(condition)
                    if self.seed_to_critic:
                        fake_parts_for_g.append(initial_state)
                    
                    fake_input_for_g = torch.cat(fake_parts_for_g, dim=1)
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
        final_state_for_commit = final_state
        prediction_image_for_log = prediction_image.detach()
        
        self.step_counter += 1
        self.current_step += 1
        
        return prediction_image_for_log, final_state_for_commit

    def gradient_penalty(self, initial_state, real_images, fake_images, condition=None):
        """Calculates the gradient penalty for WGAN-GP."""
        batch_size, c, h, w = real_images.shape
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device).repeat(1, c, h, w)
        interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
        
        interpolated_parts = [interpolated_images]
        
        if condition is not None:
            # Ensure condition is resized to match interpolated_images if needed
            if condition.shape[2:] != (h, w):
                condition_resized = nn.functional.interpolate(condition, size=(h, w), mode='bilinear', align_corners=False)
            else:
                condition_resized = condition
            interpolated_parts.append(condition_resized)
        
        if self.seed_to_critic:
            interpolated_parts.append(initial_state)
        
        interpolated_input = torch.cat(interpolated_parts, dim=1)

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
