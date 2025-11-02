import os
import torch
import torch.nn as nn
from nca.core.models.model_factory import get_latent_encoder
from nca.utils.config import Config


class LatentWrapper(nn.Module):
    """Wrapper that adds encode/decode capabilities to any model"""
    def __init__(self, base_model, config: Config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = self.config.DEVICE
        self.encoder_decoder = self._load_encoder_decoder()
        
    def _load_encoder_decoder(self):
        """Load encoder/decoder from checkpoint"""
        
        ae, reconstruction_criterion, vae_kl_beta = get_latent_encoder(self.config, self.device)

        # Load AE weights
        folder_name = self.config.FOLDER_NAME
        if folder_name != "??testing":
            ae_checkpoint = self.config.LATENT_TRAINING.AE_CHECKPOINT
            default_ae_path = os.path.join(folder_name, "ae_checkpoints", "ae.pt" if self.config.LATENT_TRAINING.ENCODER_TYPE == "AE" else "vae.pt")
            assert os.path.exists(default_ae_path) or (ae_checkpoint is not None), "AutoEncoder weights not found"

            if ae_checkpoint is not None:
                ae.load_state_dict(torch.load(ae_checkpoint, weights_only=True))
            else:
                ae.load_state_dict(torch.load(default_ae_path, weights_only=True))
                print(f"Loaded AutoEncoder weights from {default_ae_path}")

        ae.eval()
        ae.to(self.device)
        return ae# , reconstruction_criterion, vae_kl_beta
    
        
    def encode(self, x):
        """Encode from pixel to latent space"""
        return self.encoder_decoder.encode(x)[0]
        
    def decode(self, z):
        """Decode from latent to pixel space"""
        return self.encoder_decoder.decode(z)
        
    def evolve_in_pixel_space(self, x, cond=None, freeze_channels=None, fire_rate=None, step_size=1.0, return_residuals=False):
        """Evolve directly in pixel space"""
        return self.base_model(x, cond, fire_rate, step_size, freeze_channels, return_residuals)
        
    def evolve_in_latent_space(self, x, cond=None, freeze_channels=None, fire_rate=None, step_size=1.0, return_residuals=False):
        """Encode, evolve in latent space, then decode"""
        print(f"Input shape: {x.shape}, cond shape: {cond.shape if cond is not None else 'None'}")
        latent_x = self.encode(x[:, :self.config.LATENT_TRAINING.LATENT_AE_IN_CHANNEL])
        evolved_latent = self.base_model(latent_x, cond, fire_rate, step_size, freeze_channels, return_residuals)
        
        # Handle the case where return_residuals is True
        if return_residuals:
            evolved_state, residuals = evolved_latent
            return self.decode(evolved_state), residuals
        
        return self.decode(evolved_latent)
        
    def forward(self, x, cond=None, fire_rate=None, step_size=1.0, freeze_channels=None, return_residuals=False):
        """Evolve in latent space"""
        return self.evolve_in_latent_space(x, cond, freeze_channels, fire_rate, step_size, return_residuals)