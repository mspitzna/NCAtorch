import numpy as np
import torch
import os

from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import create_dataset
from nca.utils.config import load_config, Config
from nca.core.models.latent_encoder_factory import create_latent_encoder, get_checkpoint_filename


class CaHandler:
    def __init__(self, device_preference=None):
        self.config = None
        self.dataloader = None
        self.ca_model = None
        self.device_preference = device_preference

    def load_model(self, log_path: str) -> Config:
        self.config = load_config(os.path.join(log_path, "config.yaml"))

        # Use device preference if set, otherwise auto-detect
        if self.device_preference:
            device = self.device_preference
            print(f"Using device (from preference): {device}")
        else:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device (auto-detected): {device}")

        self.config = self.config.model_copy(
            update={
                "DEVICE": device,
                "SEED": -1
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
            self.ae, _, _ = create_latent_encoder(self.config, "cpu", inference_only=True)

            checkpoint_file = get_checkpoint_filename(self.config.LATENT_TRAINING.ENCODER_TYPE)
            default_ae_path = os.path.join(log_path, "ae_checkpoints", checkpoint_file)
            self.ae.load_state_dict(torch.load(default_ae_path, weights_only=True, map_location=device))
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
        device = self.config.DEVICE
        x_tensor = sample.clone().detach().to(device)
        current_input_image = self.get_dataset().state_to_img(sample.cpu(), COLOR_DIM)
        cond = cond.to(device) if cond is not None else None
        return (
            IMG_SIZE,
            COLOR_DIM,
            self.config.COND_DIM,
            x_tensor,
            current_input_image,
            cond,
        )

    def get_condition_tensor(self, idx):
        return self.dataloader.get_dataset().get_condition_tensor(idx).unsqueeze(0).to(self.config.DEVICE)

    def get_dataset(self):
        return self.dataloader.get_dataset()

    def get_dataset_name(self):
        return self.config.DATASET.NAME

    def forward(self, x_tensor, cond, n_steps=1):
        with torch.no_grad():
            device = self.config.DEVICE
            x_tensor = x_tensor.to(device)
            if cond is not None:
                cond = cond.to(device)
            if self.config.LATENT_TRAINING.ENABLED:
                if x_tensor.shape[1] == self.config.MODEL.CHANNEL_N:
                    enc = self.ae.encode(x_tensor[:, :4])
                    x_latent = enc[0] if isinstance(enc, tuple) else enc
                else:
                    x_latent = x_tensor
                for _ in range(n_steps):
                    x_latent = self.ca_model(x_latent, cond)
                x_tensor = self.ae.decode(x_latent)
                return x_tensor, x_latent
            elif self.config.DATASET.NAME == "mnist":
                for _ in range(n_steps):
                    x_tensor = self.ca_model(x_tensor, cond, freeze_channels=1)
            else:
                for _ in range(n_steps):
                    x_tensor = self.ca_model(x_tensor, cond)
        return x_tensor, x_tensor

    def get_ui_config(self):
        # Convert Pydantic config to a nicely formatted dict
        config_dict = self._format_config_for_ui(self.config)

        try:
            model_cfg = self.config.MODEL
            perceptions = []

            # Safely iterate through perceptions
            if hasattr(model_cfg, 'PERCEPTIONS') and model_cfg.PERCEPTIONS:
                for idx, percep in enumerate(model_cfg.PERCEPTIONS):
                    perception_entry = {
                        "label": f"{idx + 1}. {getattr(percep, 'MODE', 'unknown')}",
                        "kernel": getattr(percep, "KERNEL_SIZE", None),
                        "dilation": getattr(percep, "DILATION", 1),
                        "out_channel": getattr(percep, "OUT_CHANNEL", None),
                    }
                    perceptions.append(perception_entry)

            hidden_channels = getattr(model_cfg, "HIDDEN_CHANNELS", [])
            hidden_channels_display = ", ".join(str(v) for v in hidden_channels) if hidden_channels else "—"

            model_overview = {
                "name": getattr(model_cfg, "NAME", "Unknown"),
                "channel_n": getattr(model_cfg, "CHANNEL_N", None),
                "channel_out": getattr(model_cfg, "CHANNEL_OUT", None),
                "hidden_channels": hidden_channels_display,
                "fire_rate": getattr(model_cfg, "FIRE_RATE", None),
                "use_positional_embeddings": getattr(model_cfg, "USE_POSITIONAL_EMBEDDINGS", False),
                "perceptions": perceptions,
            }

            # Add MODEL_OVERVIEW to config_dict
            config_dict["MODEL_OVERVIEW"] = model_overview

            print(f"DEBUG: Created MODEL_OVERVIEW with {len(perceptions)} perceptions for dataset {self.config.DATASET.NAME}")
            print(f"DEBUG: Perceptions: {perceptions}")

        except Exception as e:
            print(f"ERROR creating MODEL_OVERVIEW: {e}")
            import traceback
            traceback.print_exc()

        if self.config.DATASET.NAME == "emoji":
            result = {
                **config_dict,
                "emojis": self.config.DATASET.EMOJIS,
            }
            print(f"DEBUG: Emoji dataset - MODEL_OVERVIEW in result: {'MODEL_OVERVIEW' in result}")
            return result
        elif self.config.DATASET.NAME == "ot":
            print(f"DEBUG: OT dataset - MODEL_OVERVIEW in config_dict: {'MODEL_OVERVIEW' in config_dict}")
            return config_dict
        else:
            print(f"DEBUG: Other dataset ({self.config.DATASET.NAME}) - MODEL_OVERVIEW in config_dict: {'MODEL_OVERVIEW' in config_dict}")
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
