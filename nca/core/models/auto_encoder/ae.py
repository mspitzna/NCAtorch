import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=64, compression_level=4):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            latent_channels (int): Number of channels in the latent representation.
            compression_level (int): Number of encoder layers to control the compression.
        """
        super(AutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.compression_level = compression_level

        # Build Encoder
        encoder_layers = []
        current_channels = in_channels
        for i in range(compression_level - 1):
            encoder_layers.append(
                nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1)
            )
            encoder_layers.append(nn.ReLU())
            current_channels *= 2
        
        # Latent Layer
        encoder_layers.append(
            nn.Conv2d(current_channels, latent_channels, kernel_size=3, stride=2, padding=1)
        )
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder
        decoder_layers = []
        current_channels = latent_channels
        for i in range(compression_level - 1):
            decoder_layers.append(
                nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            decoder_layers.append(nn.ReLU())
            current_channels //= 2
        
        # Output Layer
        decoder_layers.append(
            nn.ConvTranspose2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # Encoder output and skip connection
        skip_connection = self.encoder(x)
        latent = skip_connection
        
        # Decoder using latent and skip connection
        reconstructed = self.decoder(latent)
        return reconstructed, skip_connection, latent
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def get_input_channels(self):
        return self.in_channels
