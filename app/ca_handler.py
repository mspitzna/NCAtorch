import numpy as np
import torch
import os

from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.utils.config import load_config, Config
from nca.core.models.model_factory import get_latent_encoder


class CaHandler:
    def __init__(self):
        self.config = None
        self.dataloader = None
        self.ca_model = None

    def load_model(self, log_path: str) -> Config:
        self.config = load_config(os.path.join(log_path, "config.yaml"))

        # Set random seed for deterministic inference
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)

        # Use GPU if available for better performance
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        self.config = self.config.model_copy(
            update={
                "DEVICE": device,
            }
        )

        self.dataloader, cond_dim, im_height, im_width = create_dataset(self.config, train=False)
        self.ca_model = create_model(self.config, cond_dim, im_height, im_width)
        self.config = self.config.model_copy(
            update={"COND_DIM": cond_dim, "IM_HEIGHT": im_height, "IM_WIDTH": im_width}
        )
        checkpoint_path = os.path.join(log_path, "ca_final.pt")
        device = self.config.DEVICE
        self.ca_model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=device))
        self.ca_model.to(device)
        self.ca_model.eval()

        if self.config.LATENT_TRAINING.ENABLED:
            self.ae, reconstruction_criterion, vae_kl_beta = get_latent_encoder(self.config, "cpu")

            # Load AE weights
            print(f"Loading AutoEncoder weights from {log_path}")

            default_ae_path = os.path.join(log_path, "ae_checkpoints", "ae.pt" if self.config.LATENT_TRAINING.ENCODER_TYPE == "AE" else "vae.pt")
            self.ae.load_state_dict(torch.load(default_ae_path, weights_only=True))
            print(f"Loaded AutoEncoder weights from {default_ae_path}")
            self.ae.eval()
            self.ae.to(self.config.DEVICE)

        return self.config

    def get_initial_data(self):
        data_iter = iter(self.dataloader)
        data = next(data_iter)
        IMG_SIZE = data[0].shape[-1]
        sample = data[0][0:1]
        cond = data[1][0:1] if self.config.COND_DIM > 0 else None
        COLOR_DIM = (
            data[2].shape[1]
            if self.config.MODEL.LIVING_MASK is True
            else data[2].shape[1]
        )  # TODO Check this
        x_tensor = sample.clone().detach()
        current_input_image = (
            sample[0, :3].squeeze(0).numpy().transpose(1, 2, 0) * 255
        ).astype(np.uint8)
        return (
            IMG_SIZE,
            COLOR_DIM,
            self.config.COND_DIM,
            x_tensor,
            current_input_image,
            cond,
        )

    def get_condition_tensor(self, idx):
        return self.dataloader.get_dataset().get_condition_tensor(idx).unsqueeze(0)

    def get_dataset(self):
        return self.dataloader.get_dataset()

    def get_dataset_name(self):
        return self.config.DATASET.NAME

    def forward(self, x_tensor, cond):
        with torch.no_grad():
            device = self.config.DEVICE
            x_tensor = x_tensor.to(device)
            if cond is not None:
                cond = cond.to(device)
            if self.config.LATENT_TRAINING.ENABLED:
                if x_tensor.shape[1] == self.config.MODEL.CHANNEL_N:
                    x_latent = self.ae.encode(x_tensor[:, :4])[0]
                else:
                    x_latent = x_tensor
                x_latent = self.ca_model(x_latent, cond)
                x_tensor = self.ae.decode(x_latent)
                return x_tensor, x_latent
            elif self.config.DATASET.NAME == "mnist":
                x_tensor = self.ca_model(x_tensor, cond, freeze_channels=0)
            else:
                x_tensor = self.ca_model(x_tensor, cond)
        return x_tensor, x_tensor

    def get_ui_config(self):
        # Convert Pydantic config to a nicely formatted dict
        config_dict = self._format_config_for_ui(self.config)

        if self.config.DATASET.NAME == "emoji":
            return {
                **config_dict,
                "emojis": self.config.DATASET.EMOJIS,
            }
        elif self.config.DATASET.NAME == "ot":
            return config_dict
        else:
            return config_dict

    def _format_config_for_ui(self, config):
        """Format config object into a clean nested dictionary for UI display, sorted by complexity."""
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()

        # Separate simple values from nested structures
        simple_values = {}
        nested_structures = {}

        for key, value in config_dict.items():
            if value is None:
                continue
            elif isinstance(value, (dict, list)):
                # Nested structures (dicts and lists)
                nested_structures[key] = value
            else:
                # Simple values (strings, numbers, bools)
                simple_values[key] = value

        # Return with simple values first, then nested structures
        return {**simple_values, **nested_structures}
