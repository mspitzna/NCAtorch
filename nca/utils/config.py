from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PerceptionConfig(BaseModel):
    MODE: str = "conv"
    KERNEL_SIZE: int = 3
    DILATION: int = 1
    OUT_CHANNEL: int = 80

    @field_validator("MODE")
    @classmethod
    def check_mode(cls, value):
        valid_modes = ["conv", "attention", "sobel", "deformable_conv", "residual_conv", "mh_attention"]
        if value not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}.')
        return value

    @field_validator("OUT_CHANNEL")
    @classmethod
    def check_out_channel(cls, value):
        if value <= 0:
            raise ValueError("OUT_CHANNEL must be a positive integer.")
        return value


class ModelConfig(BaseModel):
    NAME: str = "MLP"
    HIDDEN_CHANNELS: List[int] = [64]
    CHANNEL_N: int = 16
    CHANNEL_OUT: Optional[int] = None
    USE_POSITIONAL_EMBEDDINGS: bool = False
    LIVING_MASK: bool = False
    LIVING_MASK_INDEX: int = 3
    NOISE_INJECTION: float = 0.0
    FINAL_ACTIVATION: bool = False
    CLAMP_OUTPUT: bool = False
    FIRE_RATE: float = 0.5

    RESNET_BLOCKS: int = 2

    PERCEPTIONS: List[PerceptionConfig] = Field(default_factory=lambda: [PerceptionConfig()])

    @field_validator("CHANNEL_N")
    @classmethod
    def check_channel_n(cls, value):
        if value <= 0:
            raise ValueError("CHANNEL_N must be a positive integer.")
        return value

    @field_validator("NOISE_INJECTION")
    @classmethod
    def check_noise(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("NOISE_INJECTION must be between 0 and 1.")
        return value

    @field_validator("NAME")
    @classmethod
    def check_model_name(cls, value):
        valid_models = ["MLP", "ResNet"]
        if value not in valid_models:
            raise ValueError(f"MODEL.NAME must be one of {valid_models}.")
        return value

    @model_validator(mode='after')
    def set_channel_out(self):
        if self.CHANNEL_OUT is None:  # Not explicitly set, use CHANNEL_N
            self.CHANNEL_OUT = self.CHANNEL_N
        return self


class TrainingConfig(BaseModel):
    BATCH_SIZE: int = 12
    STEPS: int = 10000
    LOSS_FN: str = "mse"
    OVERFLOW_LOSS: bool = False
    LEARNING_RATE: float = 0.002
    WARMUP_STEPS: int = 2000
    LR_SCHEDULE_MODE : str = "step"
    MILESTONES: List[int] = [2000, 8000]
    LR_GAMMA: float = 0.1
    OPTIMIZER_BETAS: List[float] = [0.9, 0.999]
    LOG_INTERVAL: int = 100
    SAVE_INTERVAL: int = 10000
    ITER_N_MIN: int = 32
    ITER_N_MAX: int = 64
    INTERMEDIATE_LOGGING_STEPS: List[int] = [5, 15, 25]
    GRADIENT_CLIPPING_NORM: float = 1.0
    GRADIENT_CHECKPOINTING: bool = False
    GRADIENT_CHECKPOINT_SEGMENTS: int = 16
    GRADIENT_ACCUMULATION_STEPS: int = 1
    MIXED_PRECISION: bool = False

    @field_validator("LOSS_FN")
    @classmethod
    def check_loss_fn(cls, value):
        valid_loss_fns = ["mse", "l2", "vggstyle", "p_ce", "i_ce", "overflow", "seed_preserving_mse", "seed_preserving_l1", "lpips", "l1"]
        if value not in valid_loss_fns:
            raise ValueError(f'LOSS_FN must be one of {valid_loss_fns}.')
        return value

    @field_validator("LR_SCHEDULE_MODE")
    @classmethod
    def check_lr_schedule_mode(cls, value):
        if value not in ["step", "cosine"]:
            raise ValueError('LR_SCHEDULE_MODE  must be "step" or "cosine".')
        return value

    @field_validator("BATCH_SIZE")
    @classmethod
    def check_batch_size(cls, value):
        if value <= 0:
            raise ValueError("BATCH_SIZE must be greater than zero.")
        return value

    @field_validator("LEARNING_RATE")
    @classmethod
    def check_learning_rate(cls, value):
        if value <= 0 or value > 1:
            raise ValueError("LEARNING_RATE must be between 0 and 1.")
        return value

    @field_validator("INTERMEDIATE_LOGGING_STEPS")
    @classmethod
    def check_logging_steps(cls, value, values):
        """Ensure all intermediate logging steps are lower than ITER_N_MIN"""
        iter_n_min = values.data.get("ITER_N_MIN") 
        if iter_n_min is not None:
            for step in value:
                if step >= iter_n_min:
                    raise ValueError(
                        f"INTERMEDIATE_LOGGING_STEPS contains {step}, but it must be less than ITER_N_MIN ({iter_n_min})."
                    )
        return value
    
    @field_validator("GRADIENT_CHECKPOINT_SEGMENTS")
    @classmethod
    def check_checkpoint_segments(cls, value):
        if value <= 0:
            raise ValueError("GRADIENT_CHECKPOINT_SEGMENTS must be positive.")
        return value


class DatasetConfig(BaseModel):
    NAME: str = "emoji"
    DATAROOT: Path = None
    DATASET_SAMPLE_PATH: Path = None
    DROP_LAST_BATCH: bool = True
    TARGET_SIZE: int = 64
    TARGET_PADDING: int = 0
    EMOJIS: List[str] = [] #["🙂", "🌈", "🦅", "🐧", "🌻", "🍕"]
    HISTORY_N: int = 1
    REVERSE_HISTORY_SEED: bool = False
    NUM_WORKERS: int = 0
    COND_INPAINTING_MASK: bool = False
    INVERTIBLE: bool = False
    SEED_SIZE: int = 1  # Size of the cross pattern for GrowingMNISTDataset
    ENABLE_ROTATION: bool = False  # Enable rotation transformations in GrowingMNISTDataset
    ENABLE_ZOOM: bool = False  # Enable zoom transformations in GrowingMNISTDataset
    Z_LATENT_NOISE_CHANNEL: bool = False  # Add latent noise channel as last dimension to target and seed
    
    @field_validator("DATASET_SAMPLE_PATH")
    @classmethod
    def check_path_exists(cls, value: Path):
        if not value.exists():
            raise ValueError(f"Dataset sample path does not exist: {value}")
        return value

    @field_validator("TARGET_SIZE")
    @classmethod
    def check_target_size(cls, value):
        if value <= 0:
            raise ValueError("TARGET_SIZE must be greater than zero.")
        return value

class CFGConfig(BaseModel):
    ENABLED: bool = False
    DROPOUT_PROB: float = 0.1
    NULL_CONDITION_TYPE: str = "zeros"
    GOAL_CHANNELS: bool = False
    PRESERVE_CHANNELS: List[int] = Field(default_factory=list)  # Channels to NOT zero out during CFG



class SamplePoolConfig(BaseModel):
    ENABLED: bool = False
    TIMESERIES_POOL: bool = False
    POOL_SIZE: int = 1024
    POOL_DELAY: int = 1000
    POOL_START_RATIO: float = 0.5
    POOL_END_RATIO: float = 0.5
    POOL_DMG_RATIO: float = 0.0
    POOL_DMG_DELAY: int = None
    POOL_MUTATION_RATIO: float = 0.0

    @field_validator("POOL_START_RATIO", "POOL_END_RATIO", "POOL_DMG_RATIO")
    @classmethod
    def check_ratio(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Ratios must be between 0 and 1.")
        return value

    @field_validator("POOL_SIZE", "POOL_DELAY", "POOL_DMG_DELAY")
    @classmethod
    def check_positive(cls, value):
        if value <= 0:
            raise ValueError("Values must be positive.")
        return value


class LatentConfig(BaseModel):
    ENABLED: bool = False
    ENCODER_TYPE: str = "AE"
    LATENT_AE_STEPS: int = 10000
    LATENT_AE_WARMUP_STEPS: int = 2000
    LATENT_AE_LR: float = 0.001
    LATENT_AE_IN_CHANNEL: int = 4
    LATENT_AE_OUT_CHANNEL: int = 4
    LATENT_AE_CHANNEL: int = 64
    LATENT_AE_COMPRESSION: int = 3
    LATENT_AE_LOG_INTERVAL: int = 2500
    LATENT_AE_SAVE_INTERVAL: int = 5000
    APPLY_DAMAGE: bool = False
    AE_CHECKPOINT: Path = None
    VAE_KL_BETA: float = 1.0
    VAE_BASE_CHANNELS: int = 64
    VAE_NUM_DOWNSAMPLES: int = 5
    VAE_NORM_GROUPS: int = 32
    VAE_KL_WARMUP_STEPS: int = 0
    VAE_BATCH_SIZE: int = 18
    VAE_RECON_LOSS_TYPE: str = "l1"  # Type of reconstruction loss for VAE: "l1" or "mse"
    VAE_RECON_LOSS_WEIGHT: float = 1.0  # Weight for reconstruction loss in VAE
    VAE_VGG_LOSS_WEIGHT: float = 1.0  # Weight for VGG loss in VAE
    @field_validator("ENCODER_TYPE")
    @classmethod
    def check_encoder_type(cls, value):
        if value not in ["AE", "VAE"]:
            raise ValueError("ENCODER_TYPE must be one of ['AE', 'VAE'].")
        return value


class AdversarialConfig(BaseModel):
    ENABLED: bool = False  # Set to False to disable GAN training
    D_IN_CHANNELS: int = 4  # Number of input channels for the discriminator
    D_FEATURES: List[int] = [64, 128, 256, 512]  # Features for the discriminator
    D_LEARNING_RATE: float = 0.001  # Learning rate for the discriminator
    D_START_TRAINING: int = 0  # Steps at which the discriminator starts training
    D_WARMUP_STEPS: int = 0
    D_GAMMA: float = 0.1  # Learning rate decay factor for the discriminator
    D_N_CRITIC: int = 1  # Number of critic updates per generator update
    D_GP_WEIGHT: float = 10.0  # Weight for the gradient penalty in WGAN-GP
    D_DOWNSCALE_FACTOR: int = 1  # Downscale factor for the discriminator input
    LPIPS_WEIGHT: float = 0.0  # Weight for LPIPS loss in the generator's total loss
    ADV_WEIGHT: float = 1.0  # Weight for adversarial loss in the generator's total loss
    RECON_WEIGHT: float = 1.0  # Weight for reconstruction loss in the generator's total loss
    SEED_TO_CRITIC: bool = False  # Whether to use seed information in the critic

class Config(BaseModel):
    PROJECT_NAME: str = "growing_ca"
    FOLDER_NAME: str = None
    TRAIN_NAME: str = "TEST"
    SEED: int = -1
    DEVICE: str = "cuda"
    WANDB: bool = Field(default=False, description="Enable Weights & Biases logging")
    DEBUG: bool = False

    MODEL: ModelConfig = Field(default_factory=ModelConfig)
    TRAINING: TrainingConfig = Field(default_factory=TrainingConfig)
    DATASET: DatasetConfig = Field(default_factory=DatasetConfig)
    CFG: CFGConfig = Field(default_factory=CFGConfig)
    PATTERN_POOL: SamplePoolConfig = Field(default_factory=SamplePoolConfig)
    LATENT_TRAINING: LatentConfig = Field(default_factory=LatentConfig)
    ADVERSARIAL: AdversarialConfig = Field(default_factory=AdversarialConfig)


    COND_DIM: Optional[int] = None
    IM_HEIGHT: Optional[int] = None
    IM_WIDTH: Optional[int] = None
    
    def model_post_init(self, __context) -> None:
        """Perform cross-field validation after model initialization."""
        # Validate latent training configuration
        if self.LATENT_TRAINING.ENABLED:
            if self.LATENT_TRAINING.ENCODER_TYPE not in ["AE", "VAE"]:
                raise ValueError("When LATENT_TRAINING.ENABLED=True, ENCODER_TYPE must be 'AE' or 'VAE'")
            if self.LATENT_TRAINING.LATENT_AE_COMPRESSION < 1:
                raise ValueError("LATENT_AE_COMPRESSION must be >= 1")
        
        # Validate adversarial training configuration
        if self.ADVERSARIAL.ENABLED:
            if self.ADVERSARIAL.D_LEARNING_RATE <= 0:
                raise ValueError("When ADVERSARIAL.ENABLED=True, D_LEARNING_RATE must be > 0")
            if len(self.ADVERSARIAL.D_FEATURES) == 0:
                raise ValueError("When ADVERSARIAL.ENABLED=True, D_FEATURES cannot be empty")
        
        # Validate dataset-specific requirements
        if self.DATASET.NAME in ["emoji"] and len(self.DATASET.EMOJIS) == 0:
            raise ValueError(f"Dataset '{self.DATASET.NAME}' requires EMOJIS list to be non-empty")
        
        if self.DATASET.NAME in ["e2h", "celeba"] and self.DATASET.DATAROOT is None:
            raise ValueError(f"Dataset '{self.DATASET.NAME}' requires DATAROOT to be specified")
        
        # Validate pattern pool configuration
        if self.PATTERN_POOL.ENABLED and self.PATTERN_POOL.POOL_SIZE <= 0:
            raise ValueError("When PATTERN_POOL.ENABLED=True, POOL_SIZE must be > 0")
            
        # Validate training configuration consistency
        if self.TRAINING.ITER_N_MIN > self.TRAINING.ITER_N_MAX:
            raise ValueError("ITER_N_MIN cannot be greater than ITER_N_MAX")
            
        if any(step >= self.TRAINING.ITER_N_MIN for step in self.TRAINING.INTERMEDIATE_LOGGING_STEPS):
            raise ValueError("All INTERMEDIATE_LOGGING_STEPS must be < ITER_N_MIN")

    def set_cond_dim(self, cond_dim: int):
        self.COND_DIM = cond_dim

    def set_im_height(self, im_height: int):
        self.IM_HEIGHT = im_height

    def set_im_width(self, im_width: int):
        self.IM_WIDTH = im_width


def load_config(config_path: str) -> Config:
    """Load YAML config and parse it into a Pydantic model."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    config = Config(**raw_config)
    return config
