from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PerceptionConfig(BaseModel):
    """Configuration for one perception branch.

    Multiple entries under ``MODEL.PERCEPTIONS`` run in parallel; their outputs
    are concatenated before the update model. Valid ``MODE`` values are the keys
    of ``PERCEPTION_REGISTRY`` in ``perception_factory.py``.

    Attributes:
        MODE: Neighbourhood operator.
        KERNEL_SIZE: Spatial kernel size for convolution-based perceptions.
        DILATION: Dilation factor for ``conv`` perception.
        OUT_CHANNEL: Number of output channels (filters) from this branch.
        NUM_HEADS: Attention heads for ``attention`` / ``mh_attention``.
        EMBED_DIM: Projection dimension for ``mh_attention``.
        USE_REL_POS_BIAS: Add relative positional bias in attention layers.
        USE_LAYER_NORM: Apply layer normalisation in ``mh_attention``.
        INCLUDE_FFN: Include feed-forward sub-layer in ``mh_attention``.
    """
    MODE: str = "conv"
    KERNEL_SIZE: int = 3
    DILATION: int = 1
    OUT_CHANNEL: int = 80
    # attention / mh_attention
    NUM_HEADS: int = 4
    EMBED_DIM: int = 128
    USE_REL_POS_BIAS: bool = True
    # mh_attention only
    USE_LAYER_NORM: bool = True
    INCLUDE_FFN: bool = True

    @field_validator("MODE")
    @classmethod
    def check_mode(cls, value):
        from nca.core.models.perception_factory import PERCEPTION_REGISTRY
        if value not in PERCEPTION_REGISTRY:
            raise ValueError(f"MODE must be one of {sorted(PERCEPTION_REGISTRY)}.")
        return value

    @field_validator("OUT_CHANNEL")
    @classmethod
    def check_out_channel(cls, value):
        if value <= 0:
            raise ValueError("OUT_CHANNEL must be a positive integer.")
        return value


class ModelConfig(BaseModel):
    """Configuration for the CA model architecture.

    Attributes:
        NAME: Update model architecture — ``MLP`` or ``ResNet``.
            Valid values are the keys of ``UPDATE_MODEL_REGISTRY``.
        HIDDEN_CHANNELS: Hidden layer sizes in the update model.
        CHANNEL_N: Number of CA state channels per cell.
        CHANNEL_OUT: Output channels after the update step; defaults to
            ``CHANNEL_N`` if not set.
        USE_POSITIONAL_EMBEDDINGS: Append learnable (x, y) coordinate channels
            to the state before perception.
        LIVING_MASK: Zero out updates for cells below the alive threshold.
        LIVING_MASK_INDEX: Channel index used to determine cell liveness.
        NOISE_INJECTION: Std of Gaussian noise added to ``dx`` each step.
        FINAL_ACTIVATION: Apply Tanh to the update model output.
        CLAMP_OUTPUT: Clamp state values to ``[-1, 1]`` after each step.
        FIRE_RATE: Fraction of cells updated per step (stochastic dropout).
        RESNET_BLOCKS: Number of residual blocks (``ResNet`` only).
        PERCEPTIONS: List of perception branch configs; outputs are concatenated.
    """
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
        from nca.core.models.update_model_factory import UPDATE_MODEL_REGISTRY
        if value not in UPDATE_MODEL_REGISTRY:
            raise ValueError(f"MODEL.NAME must be one of {sorted(UPDATE_MODEL_REGISTRY)}.")
        return value

    @model_validator(mode='after')
    def set_channel_out(self):
        if self.CHANNEL_OUT is None:  # Not explicitly set, use CHANNEL_N
            self.CHANNEL_OUT = self.CHANNEL_N
        return self


class TrainingConfig(BaseModel):
    """Hyperparameters for the main CA training loop.

    Attributes:
        BATCH_SIZE: Number of grids per optimisation step.
        STEPS: Total number of training steps.
        LOSS_FN: Reconstruction loss key from ``LOSS_FN_REGISTRY`` —
            ``mse``, ``l1``, ``lpips``, ``vggstyle``, ``p_ce``, ``i_ce``, ``overflow``.
        OVERFLOW_LOSS: Add an overflow penalty to the loss.
        OVERFLOW_WEIGHT: Weight applied to the overflow penalty term.
        LEARNING_RATE: Initial learning rate.
        WARMUP_STEPS: Linear LR warm-up duration.
        LR_SCHEDULE_MODE: LR schedule — ``step``, ``cosine``, or ``constant``.
        ITER_N_MIN: Minimum CA rollout steps per batch.
        ITER_N_MAX: Maximum CA rollout steps per batch (sampled uniformly).
        GRADIENT_CLIPPING_NORM: Max gradient norm; set to ``0`` to disable.
        MIXED_PRECISION: Enable automatic mixed precision (AMP).
        LPIPS_NET: Backbone for LPIPS loss — ``alex``, ``vgg``, or ``squeeze``.
        VGG_PROJ_N: Number of random projections in the VGG style loss.
        TRAINER_TYPE: Trainer to use.
    """
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
    OVERFLOW_WEIGHT: float = 1.0
    LPIPS_NET: str = "alex"
    VGG_PROJ_N: int = 32
    TRAINER_TYPE: Optional[str] = None
    # WSD schedule — stable phase fills the gap between warmup and decay
    WSD_DECAY_RATIO: float = 0.1
    WSD_MIN_LR_RATIO: float = 0.0

    @field_validator("TRAINER_TYPE")
    @classmethod
    def check_trainer_type(cls, value):
        if value is None:
            return value
        from nca.training.trainer_factory import TRAINER_REGISTRY
        if value not in TRAINER_REGISTRY:
            raise ValueError(
                f"TRAINER_TYPE must be one of {sorted(TRAINER_REGISTRY)} or null for auto-selection."
            )
        return value

    @field_validator("LOSS_FN")
    @classmethod
    def check_loss_fn(cls, value):
        from nca.core.losses.loss_factory import LOSS_FN_REGISTRY
        if value not in LOSS_FN_REGISTRY:
            raise ValueError(f'LOSS_FN must be one of {sorted(LOSS_FN_REGISTRY)}.')
        return value

    @field_validator("LR_SCHEDULE_MODE")
    @classmethod
    def check_lr_schedule_mode(cls, value):
        if value not in ["step", "cosine", "constant", "wsd"]:
            raise ValueError('LR_SCHEDULE_MODE must be "step", "cosine", "constant", or "wsd".')
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
    """Configuration for latent-space NCA training.

    When ``ENABLED=True`` the CA operates in the compressed latent space of a
    pre-trained encoder rather than directly on pixels, enabling high-resolution
    generation at a fraction of the compute cost.

    Attributes:
        ENABLED: Activate latent-space mode.
        ENCODER_TYPE: Encoder architecture — ``AE``, ``VAE``, or ``VQVAE``.
            Valid values are the keys of ``LATENT_ENCODER_REGISTRY``.
        LATENT_AE_IN_CHANNEL: Input channels to the encoder (e.g. 4 for RGBA).
        LATENT_AE_OUT_CHANNEL: Output channels from the decoder.
        LATENT_AE_CHANNEL: Latent bottleneck channels (CA state size in latent mode).
        LATENT_AE_COMPRESSION: Spatial downsampling factor as 2^N.
        AE_CHECKPOINT: Explicit path to a pre-trained encoder checkpoint;
            if ``None`` the default path inside ``FOLDER_NAME`` is used.
        VAE_KL_BETA: Weight of the KL divergence term in the VAE loss.
        VAE_BASE_CHANNELS: Base feature channels in VAE encoder/decoder.
        VAE_NUM_DOWNSAMPLES: Number of stride-2 downsampling stages.
        VAE_NORM_GROUPS: Group normalisation groups in VAE conv layers.
        VAE_RECON_LOSS_TYPE: Pixel reconstruction loss — ``l1`` or ``mse``.
        VAE_RECON_LOSS_WEIGHT: Weight for the pixel reconstruction term.
        VAE_VGG_LOSS_WEIGHT: Weight for the VGG perceptual loss term.
        VQVAE_NUM_EMBEDDINGS: Codebook size for VQVAE.
        VQVAE_COMMITMENT_COST: Commitment loss weight (β) for VQVAE.
    """
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
    VQVAE_NUM_EMBEDDINGS: int = 512
    VQVAE_COMMITMENT_COST: float = 0.25

    @field_validator("ENCODER_TYPE")
    @classmethod
    def check_encoder_type(cls, value):
        from nca.core.models.latent_encoder_factory import LATENT_ENCODER_REGISTRY
        if value not in LATENT_ENCODER_REGISTRY:
            raise ValueError(f"ENCODER_TYPE must be one of {sorted(LATENT_ENCODER_REGISTRY)}.")
        return value


class AdversarialConfig(BaseModel):
    """Configuration for optional GAN (adversarial) training.

    When ``ENABLED=True`` a patch discriminator is trained alongside the CA
    generator using a WGAN-GP objective. The generator loss is a weighted sum
    of the reconstruction loss, the adversarial loss, and an optional LPIPS
    perceptual term. Generator and discriminator use separate optimisers and
    separate ``GradScaler`` instances for mixed-precision training.

    Attributes:
        ENABLED: Activate adversarial training.
        D_IN_CHANNELS: Input channels to the discriminator (must match the
            generator output channels).
        D_FEATURES: Channel progression in the discriminator
            (e.g. ``[64, 128, 256, 512]``).
        D_LEARNING_RATE: Discriminator learning rate.
        D_START_TRAINING: Step at which discriminator training begins; allows
            the generator to warm up before the critic is introduced.
        D_WARMUP_STEPS: Linear LR warm-up steps for the discriminator.
        D_GAMMA: LR decay factor for the discriminator scheduler.
        D_N_CRITIC: Discriminator updates per generator update.
        D_GP_WEIGHT: Gradient penalty coefficient λ in the WGAN-GP loss.
        D_DOWNSCALE_FACTOR: Spatially downscale inputs to the discriminator
            by this factor before the forward pass.
        LPIPS_WEIGHT: Weight for the LPIPS perceptual term in the generator loss.
        ADV_WEIGHT: Weight for the adversarial term in the generator loss.
        RECON_WEIGHT: Weight for the reconstruction term in the generator loss.
        SEED_TO_CRITIC: Pass the seed image as an additional channel to the
            discriminator (conditional GAN setup).
    """
    ENABLED: bool = False
    D_IN_CHANNELS: int = 4
    D_FEATURES: List[int] = [64, 128, 256, 512]
    D_LEARNING_RATE: float = 0.001
    D_START_TRAINING: int = 0
    D_WARMUP_STEPS: int = 0
    D_GAMMA: float = 0.1
    D_N_CRITIC: int = 1
    D_GP_WEIGHT: float = 10.0
    D_DOWNSCALE_FACTOR: int = 1
    LPIPS_WEIGHT: float = 0.0
    ADV_WEIGHT: float = 1.0
    RECON_WEIGHT: float = 1.0
    SEED_TO_CRITIC: bool = False

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
