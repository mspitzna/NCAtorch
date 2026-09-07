import torch.nn as nn
from nca.core.models.latent_encoder_factory import load_latent_encoder
from nca.utils.config import Config


class LatentWrapper(nn.Module):
    """Wraps a ``CAModel`` to operate in the latent space of a pre-trained encoder.

    The encoder is loaded from ``<FOLDER_NAME>/ae_checkpoints/<type>.pt`` (or
    the explicit ``LATENT_TRAINING.AE_CHECKPOINT`` path) and kept frozen during
    CA training. The forward pass is:

        pixel input → encode → CA update (latent) → decode → pixel output

    Args:
        base_model: A ``CAModel`` instance sized for the latent space.
        config: Full ``Config`` object; ``LATENT_TRAINING`` and ``FOLDER_NAME``
            fields are used to locate and load the encoder checkpoint.
    """
    def __init__(self, base_model, config: Config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = self.config.DEVICE
        self.encoder_decoder = self._load_encoder_decoder()
        
    def _load_encoder_decoder(self):
        """Load frozen weights without constructing an encoder training loss."""
        return load_latent_encoder(self.config, self.device)
    
        
    def encode(self, x):
        """Encode from pixel to latent space"""
        result = self.encoder_decoder.encode(
            x[:, :self.config.LATENT_TRAINING.LATENT_AE_IN_CHANNEL]
        )
        # AE.encode() returns a plain tensor
        # VAE.encode() returns (mu, logvar) — use mu for deterministic encoding
        return result[0] if isinstance(result, tuple) else result
        
    def decode(self, z):
        """Decode from latent to pixel space"""
        return self.encoder_decoder.decode(z)

    def evolve_in_pixel_space(self, x, cond=None, freeze_channels=None, step_size=1.0):
        """Evolve directly in pixel space. Returns ``(state, dx)``."""
        return self.base_model(x=x, cond=cond, step_size=step_size, freeze_channels=freeze_channels)

    def evolve_in_latent_space(self, x, cond=None, freeze_channels=None, step_size=1.0):
        """Encode, evolve in latent space, then decode. Returns ``(state, dx)``."""
        latent_x = self.encode(x)
        evolved_latent, residuals = self.base_model(
            x=latent_x, cond=cond, step_size=step_size, freeze_channels=freeze_channels
        )
        return self.decode(evolved_latent), residuals

    def forward(self, x, cond=None, step_size=1.0, freeze_channels=None):
        """Evolve in latent space. Returns ``(state, dx)``."""
        return self.evolve_in_latent_space(x, cond, freeze_channels, step_size)
