import os
import shutil
from datetime import datetime
import numpy as np
import wandb
import torch
from nca.utils.visualization import visualize_batch
from nca.utils.config import Config

class Logger:
    _instance = None

    def __init__(self, config: Config, config_path: str = None, model=None):
        self.config = config
        self.use_wandb = config.WANDB
        self.metrics = {}  # Store aggregated metrics
        self.output_folder = self._create_output_folder(config_path)

        self.x0_log = None
        self.x_log = None
        self.target_log = None
        self.x_state_logs = {}
        
        if self.use_wandb:
            # If using pydantic, you might use config.model_dump(), else use __dict__
            flat_config = config.model_dump() if hasattr(config, "model_dump") else config.__dict__
            wandb.init(project=config.PROJECT_NAME, name=config.TRAIN_NAME, config=flat_config)

            if model is not None:
                wandb.watch(model, log="all", log_freq=self.config.TRAINING.LOG_INTERVAL)
        
        Logger._instance = self

    @staticmethod
    def get_instance():
        if Logger._instance is None:
            raise ValueError("Logger instance has not been initialized.")
        return Logger._instance

    def _create_output_folder(self, config_path):
        if self.config.FOLDER_NAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = os.path.join("train_log", f"{self.config.TRAIN_NAME}_{timestamp}")
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(os.path.join(output_folder, "imgs"), exist_ok=True)
            shutil.copy(
                config_path,
                os.path.join(output_folder, "config.yaml"),
                follow_symlinks=True,
            )
            return output_folder
        else:
            os.makedirs(os.path.join(self.config.FOLDER_NAME, "imgs"), exist_ok=True)
            return self.config.FOLDER_NAME

    def log_metrics(self, step):
        """
        Log a dictionary of metrics for a given step. Metrics are stored locally
        and sent to wandb (if enabled).
        """
        agg_metrics = self._get_aggregated_metrics()
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({**agg_metrics, "Iteration": step}, step=step)

    def peek_metrics(self):
        """
        Return the aggregated metrics without mutating internal buffers.
        """
        return self._get_aggregated_metrics()

    def log_images(self, step):
        """
        Visualize a batch and log/save the resulting image grid.
        """
        grid = visualize_batch(
            self.x0_log,
            self.x_log,
            self.target_log,
            self.x_state_logs,
            step,
            plot=False,
            save_path=os.path.join(self.output_folder, "imgs"),
            grayscale=(True if self.config.DATASET.NAME == "moving_mnist" else False)
        )
        save_path = os.path.join(self.output_folder, "imgs", f"sample_{step}.png")
        
        if self.use_wandb:
            wandb.log({"Images": [wandb.Image(grid, caption=f"Step {step} - {self.config.TRAIN_NAME}")]}, step=step)
        else:
            print(f"Images saved at {save_path}", flush=True)

    def add_metric(self, key, value):
        """
        Add a single metric value. Useful for nested modules that want to log additional metrics.
        """
        self.metrics.setdefault(key, []).append(value)

    def add_metrics(self, metrics):
        """
        Add multiple metrics at once.
        """
        for key, value in metrics.items():
            self.add_metric(key, value)

    def add_img_logs(self, x0, x, target):
        self.x0_log = x0
        self.x_log = x
        self.target_log = target

    def add_state_log(self, step, x_state):
        self.x_state_logs[step] = x_state

    def get_state_logs(self):
        return self.x_state_logs
    
    def set_state_logs(self, x_state_logs):
        self.x_state_logs = x_state_logs

    def _get_aggregated_metrics(self):
        aggregated = {}
        for k, values in self.metrics.items():
            float_values = []
            for v in values:
                if torch.is_tensor(v):
                    # Convert GPU tensor to a float value
                    float_values.append(v.cpu().detach().item())
                else:
                    float_values.append(v)
            
            # If there's only one value, no need to calculate mean
            if len(float_values) == 1:
                aggregated[k] = float_values[0]
            else:
                aggregated[k] = np.mean(float_values)
        return aggregated


    def reset_metrics(self):
        """
        Clear all logged metrics.
        """
        self.metrics = {}

    def get_output_folder(self):
        return self.output_folder
