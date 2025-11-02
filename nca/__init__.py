"""NCA-Torch: Neural Cellular Automata Framework"""

__version__ = "0.1.0"

# Core models
from nca.core.models.ca.ca_model import CAModel
from nca.core.models.latent_wrapper import LatentWrapper

# Training
from nca.training.sample_pool import SamplePool, TimeseriesSamplePool
from nca.training.trainers.base_trainer import BaseTrainer

__all__ = [
    "CAModel",
    "LatentWrapper",
    "SamplePool",
    "TimeseriesSamplePool",
    "BaseTrainer",
]
