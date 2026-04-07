import torch
import torch.nn as nn
import torch.nn.functional as F

from nca.core.models.auto_encoder.vae import ResBlock, UpsampleConvBlock


class VectorQuantizer(nn.Module):
    """
    Discretises a continuous encoder output into the nearest codebook entry.

    Gradients pass back to the encoder via the straight-through estimator:
    the decoder sees z_q but receives gradients as if it had seen z_e.

    Losses returned from forward():
      codebook_loss    — moves codebook entries toward encoder output
      commitment_loss  — moves encoder output toward codebook entries
      vq_loss          — codebook_loss + commitment_cost * commitment_loss
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: [B, C, H, W] continuous encoder output

        Returns:
            z_q_st:  [B, C, H, W] quantized tensor (straight-through)
            vq_loss: scalar training loss
            perplexity: codebook usage metric (higher = more codes used)
        """
        B, C, H, W = z_e.shape

        # [B, H, W, C] → [B*H*W, C]
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Squared Euclidean distance to every codebook entry
        distances = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        encoding_indices = distances.argmin(1)                      # [B*H*W]
        z_q_flat = self.embedding(encoding_indices)                  # [B*H*W, C]
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        codebook_loss   = F.mse_loss(z_q,          z_e.detach())
        commitment_loss = F.mse_loss(z_e,          z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: copy z_q values but pass z_e gradients
        z_q_st = z_e + (z_q - z_e).detach()

        # Perplexity — measures how many codebook entries are actually used
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs  = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, perplexity


class VQVAE(nn.Module):
    """
    Vector-Quantized VAE (van den Oord et al., 2017).

    encode(x)  → quantized latent tensor   [B, latent_channels, H', W']
    decode(z)  → reconstructed image       [B, out_channels, H, W]
    forward(x) → (reconstruction, vq_loss, perplexity)

    The vq_loss must be added to the reconstruction loss during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        base_channels: int = 64,
        num_downsamples: int = 3,
        norm_groups: int = 32,
        activation_fn: nn.Module = None,
        final_activation: nn.Module = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        if activation_fn is None:
            activation_fn = nn.LeakyReLU(0.2, inplace=True)

        # --- Encoder ---
        encoder_blocks = [nn.Conv2d(in_channels, base_channels, 3, 1, 1)]
        ch = base_channels
        for _ in range(num_downsamples):
            out_ch = ch * 2
            encoder_blocks.append(ResBlock(ch, out_ch, norm_groups, activation_fn, downsample=True))
            ch = out_ch
        # Project to latent_channels
        encoder_blocks.append(nn.Conv2d(ch, latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_blocks)

        # --- Vector Quantizer ---
        self.quantizer = VectorQuantizer(num_embeddings, latent_channels, commitment_cost)

        # --- Decoder ---
        decoder_blocks = [
            nn.Conv2d(latent_channels, ch, 3, 1, 1),
            activation_fn,
        ]
        for _ in range(num_downsamples):
            out_ch = ch // 2
            decoder_blocks.append(UpsampleConvBlock(ch, out_ch, norm_groups, activation_fn))
            ch = out_ch
        decoder_blocks.append(nn.Conv2d(base_channels, out_channels, 3, 1, 1))
        if final_activation is not None:
            decoder_blocks.append(final_activation)
        self.decoder = nn.Sequential(*decoder_blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the quantized latent — plain tensor, compatible with LatentWrapper."""
        z_e = self.encoder(x)
        z_q, _, _ = self.quantizer(z_e)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity = self.quantizer(z_e)
        reconstruction = self.decoder(z_q)
        return reconstruction, vq_loss, perplexity

    def get_input_channels(self) -> int:
        return self.in_channels
