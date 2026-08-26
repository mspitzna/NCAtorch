from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PerceptionConfig(StrictModel):
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


class ModelConfig(StrictModel):
    """Configuration for the CA model architecture.

    Attributes:
        ARCHITECTURE: CA step architecture — ``residual`` (state + step_size *
            dx, the default). Valid values are the keys of
            ``MODEL_REGISTRY``.
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
        NOISE_INJECTION: Std of Gaussian noise added directly to the state
            before each step's perception (see ``NoiseInjection``). Applied on
            every step, including the seed; never leaks into the final
            output since the rollout simply stops after the last step.
        FINAL_ACTIVATION: Apply Tanh to the update model output.
        CLAMP_OUTPUT: Clamp state values to ``[-1, 1]`` after each step.
        FIRE_RATE: Fraction of cells updated per step (stochastic dropout).
        RESNET_BLOCKS: Number of residual blocks (``ResNet`` only).
        PERCEPTIONS: List of perception branch configs; outputs are concatenated.
    """
    ARCHITECTURE: str = "residual"
    NAME: str = "MLP"
    HIDDEN_CHANNELS: list[int] = [64]
    CHANNEL_N: int = 16
    CHANNEL_OUT: int | None = None
    USE_POSITIONAL_EMBEDDINGS: bool = False
    LIVING_MASK: bool = False
    LIVING_MASK_INDEX: int = 3
    NOISE_INJECTION: float = 0.0
    FINAL_ACTIVATION: bool = False
    CLAMP_OUTPUT: bool = False
    CLAMP_OUTPUT_MIN: float = -1.0
    CLAMP_OUTPUT_MAX: float = 1.0
    FIRE_RATE: float = 0.5

    RESNET_BLOCKS: int = 2

    PERCEPTIONS: list[PerceptionConfig] = Field(
        default_factory=lambda: [PerceptionConfig()]
    )

    @field_validator("ARCHITECTURE")
    @classmethod
    def check_architecture(cls, value):
        from nca.core.models.model_factory import MODEL_REGISTRY
        if value not in MODEL_REGISTRY:
            raise ValueError(f"MODEL.ARCHITECTURE must be one of {sorted(MODEL_REGISTRY)}.")
        return value

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
            raise ValueError(
                f"MODEL.NAME must be one of {sorted(UPDATE_MODEL_REGISTRY)}."
            )
        return value

    @model_validator(mode="after")
    def set_channel_out(self):
        if self.CHANNEL_OUT is None:  # Not explicitly set, use CHANNEL_N
            self.CHANNEL_OUT = self.CHANNEL_N
        return self

    @model_validator(mode="after")
    def check_living_mask_index(self):
        if self.LIVING_MASK and not (0 <= self.LIVING_MASK_INDEX < self.CHANNEL_N):
            raise ValueError(
                f"LIVING_MASK_INDEX ({self.LIVING_MASK_INDEX}) must be in [0, CHANNEL_N) "
                f"(0 to {self.CHANNEL_N - 1} inclusive)."
            )
        return self


class TrainingConfig(StrictModel):
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
        EVOLVE_MODE: Rollout strategy key from ``EVOLVER_REGISTRY``.
    """

    BATCH_SIZE: int = 12
    STEPS: int = 10000
    LOSS_FN: str = "mse"
    OVERFLOW_LOSS: bool = False
    LEARNING_RATE: float = 0.002
    WARMUP_STEPS: int = 2000
    LR_SCHEDULE_MODE: str = "step"
    MILESTONES: list[int] = [2000, 8000]
    LR_GAMMA: float = 0.1
    OPTIMIZER_BETAS: list[float] = [0.9, 0.999]
    ITER_N_MIN: int = 32
    ITER_N_MAX: int = 64
    GRADIENT_CLIPPING_NORM: float = 1.0
    GRADIENT_CHECKPOINTING: bool = False
    GRADIENT_CHECKPOINT_SEGMENTS: int = 16
    GRADIENT_ACCUMULATION_STEPS: int = 1
    MIXED_PRECISION: bool = False
    OVERFLOW_WEIGHT: float = 1.0
    LPIPS_NET: str = "alex"
    VGG_PROJ_N: int = 32
    TRAINER_TYPE: str | None = None
    EVOLVE_MODE: str = "base"
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

    @field_validator("EVOLVE_MODE")
    @classmethod
    def check_evolve_mode(cls, value):
        from nca.training.evolve_factory import EVOLVER_REGISTRY

        if value not in EVOLVER_REGISTRY:
            raise ValueError(f"EVOLVE_MODE must be one of {sorted(EVOLVER_REGISTRY)}.")
        return value

    @field_validator("LOSS_FN")
    @classmethod
    def check_loss_fn(cls, value):
        from nca.core.losses.loss_factory import LOSS_FN_REGISTRY

        if value not in LOSS_FN_REGISTRY:
            raise ValueError(f"LOSS_FN must be one of {sorted(LOSS_FN_REGISTRY)}.")
        return value

    @field_validator("LR_SCHEDULE_MODE")
    @classmethod
    def check_lr_schedule_mode(cls, value):
        if value not in ["step", "cosine", "constant", "wsd"]:
            raise ValueError(
                'LR_SCHEDULE_MODE must be "step", "cosine", "constant", or "wsd".'
            )
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

    @field_validator("GRADIENT_CHECKPOINT_SEGMENTS")
    @classmethod
    def check_checkpoint_segments(cls, value):
        if value <= 0:
            raise ValueError("GRADIENT_CHECKPOINT_SEGMENTS must be positive.")
        return value


class DatasetConfig(StrictModel):
    NAME: str = "emoji"
    DATAROOT: Path = None
    DATASET_SAMPLE_PATH: Path = None
    DROP_LAST_BATCH: bool = True
    TARGET_SIZE: int = 64
    TARGET_PADDING: int = 0
    EMOJIS: list[str] = []  # ["🙂", "🌈", "🦅", "🐧", "🌻", "🍕"]
    HISTORY_N: int = 1
    REVERSE_HISTORY_SEED: bool = False
    NUM_WORKERS: int = 0
    SEED_SIZE: int = 1  # Size of the cross pattern for GrowingMNISTDataset
    ENABLE_ROTATION: bool = (
        False  # Enable rotation transformations in GrowingMNISTDataset
    )
    ENABLE_ZOOM: bool = False  # Enable zoom transformations in GrowingMNISTDataset
    Z_LATENT_NOISE_CHANNEL: bool = (
        False  # Add latent noise channel as last dimension to target and seed
    )

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


class CFGConfig(StrictModel):
    ENABLED: bool = False
    DROPOUT_PROB: float = 0.1
    NULL_CONDITION_TYPE: str = "zeros"
    GOAL_CHANNELS: bool = False
    PRESERVE_CHANNELS: list[int] = Field(
        default_factory=list
    )  # Channels to NOT zero out during CFG


class SamplePoolConfig(StrictModel):
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


class LatentConfig(StrictModel):
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
    VAE_RECON_LOSS_TYPE: str = (
        "l1"  # Type of reconstruction loss for VAE: "l1" or "mse"
    )
    VAE_RECON_LOSS_WEIGHT: float = 1.0  # Weight for reconstruction loss in VAE
    VAE_VGG_LOSS_WEIGHT: float = 1.0  # Weight for VGG loss in VAE
    VQVAE_NUM_EMBEDDINGS: int = 512
    VQVAE_COMMITMENT_COST: float = 0.25

    @field_validator("ENCODER_TYPE")
    @classmethod
    def check_encoder_type(cls, value):
        from nca.core.models.latent_encoder_factory import LATENT_ENCODER_REGISTRY

        if value not in LATENT_ENCODER_REGISTRY:
            raise ValueError(
                f"ENCODER_TYPE must be one of {sorted(LATENT_ENCODER_REGISTRY)}."
            )
        return value


class TorchCompileConfig(StrictModel):
    """Configuration for ``torch.compile`` model compilation.

    Attributes:
        ENABLED: Compile the CA model with ``torch.compile``.
        MODE: Compilation mode — ``default`` or
            ``max-autotune-no-cudagraphs`` (slower compile, best kernel
            selection).  Modes that use CUDA graphs (``reduce-overhead``,
            ``max-autotune``) are excluded because CUDA graphs reuse GPU
            memory buffers across replays, which corrupts the autograd
            intermediates needed by the iterative NCA forward loop.
        DEBUG: Enable ``torch._inductor`` debug output.
    """

    ENABLED: bool = False
    MODE: str = "default"
    DEBUG: bool = False

    @field_validator("MODE")
    @classmethod
    def check_mode(cls, value):
        allowed = {"default", "max-autotune-no-cudagraphs"}
        if value not in allowed:
            raise ValueError(f"TORCH_COMPILE.MODE must be one of {sorted(allowed)}.")
        return value


class ReproducibilityConfig(StrictModel):
    """Deterministic-algorithm settings for bit-exact reproducible runs.

    Unlike ``SEED`` (which fixes random-number generation), these control
    whether PyTorch/CUDA are allowed to pick non-deterministic kernels.
    Each field is applied unconditionally; their defaults already match the
    framework's non-deterministic behavior, so setting them is a no-op until
    the user opts into determinism.

    Attributes:
        USE_DETERMINISTIC_ALGORITHMS: Call ``torch.use_deterministic_algorithms``
            with this value. ``True`` makes PyTorch raise if a
            non-deterministic CUDA kernel would be used.
        CUDNN_BENCHMARK: Set ``torch.backends.cudnn.benchmark`` to this value.
        CUBLAS_WORKSPACE_CONFIG: If not ``None``, set the ``CUBLAS_WORKSPACE_CONFIG``
            env var (required by NVIDIA for deterministic cuBLAS, e.g.
            ``":4096:8"``). ``None`` leaves the environment untouched.
    """

    USE_DETERMINISTIC_ALGORITHMS: bool = False
    CUDNN_BENCHMARK: bool = False
    CUBLAS_WORKSPACE_CONFIG: str | None = None


class AdversarialConfig(StrictModel):
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
    D_FEATURES: list[int] = [64, 128, 256, 512]
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


class ObserverConfig(StrictModel):
    """One diagnostic logging observer, instantiated via the observer registry.

    Observers hook into the CA rollout on logging steps, collect data, and log
    it themselves (to W&B and/or console) during the logging phase. New observer
    types are added by implementing ``LoggingObserver`` and registering them in
    ``LOGGING_OBSERVER_REGISTRY`` — no change to this schema is required.

    Attributes:
        TYPE: Registry key selecting the observer implementation.
        PARAMS: Keyword arguments forwarded to the observer's constructor.
    """

    TYPE: str
    PARAMS: dict = Field(default_factory=dict)

    @field_validator("TYPE")
    @classmethod
    def check_type(cls, value):
        from nca.training.observers import LOGGING_OBSERVER_REGISTRY

        if value not in LOGGING_OBSERVER_REGISTRY:
            raise ValueError(
                f"LOGGING.OBSERVERS.TYPE must be one of "
                f"{sorted(LOGGING_OBSERVER_REGISTRY)}."
            )
        return value


class LoggingConfig(StrictModel):
    """All logging, run-identity and output configuration.

    Consolidates every knob that controls *how/where* a run reports itself —
    W&B, run naming, the output folder, and the logging/checkpoint intervals —
    so logging concerns live in one place and the framework can be extended
    (e.g. with diagnostic step observers) without scattering new flags across
    ``Config`` and ``TrainingConfig``.

    Attributes:
        WANDB: Enable Weights & Biases logging.
        PROJECT_NAME: W&B project / top-level output folder name.
        TRAIN_NAME: Run name (sub-folder under ``PROJECT_NAME``).
        FOLDER_NAME: Explicit output/checkpoint folder; ``None`` = auto from
            ``TRAIN_NAME`` + timestamp.
        DEBUG: Verbose debug output.
        LOG_INTERVAL: Log metrics/images every N training steps.
        SAVE_INTERVAL: Save a checkpoint every N training steps.
        INTERMEDIATE_LOGGING_STEPS: CA rollout steps at which intermediate
            states are captured for image logging (all must be < ITER_N_MIN).
    """

    WANDB: bool = Field(default=False, description="Enable Weights & Biases logging")
    PROJECT_NAME: str = "growing_ca"
    TRAIN_NAME: str = "TEST"
    FOLDER_NAME: str | None = None
    DEBUG: bool = False
    LOG_INTERVAL: int = 100
    SAVE_INTERVAL: int = 10000
    INTERMEDIATE_LOGGING_STEPS: list[int] = [5, 15, 25]
    OBSERVERS: list[ObserverConfig] = Field(default_factory=list)


class Config(StrictModel):
    SEED: int = -1
    DEVICE: str = "cuda"

    LOGGING: LoggingConfig = Field(default_factory=LoggingConfig)
    MODEL: ModelConfig = Field(default_factory=ModelConfig)
    TRAINING: TrainingConfig = Field(default_factory=TrainingConfig)
    DATASET: DatasetConfig = Field(default_factory=DatasetConfig)
    CFG: CFGConfig = Field(default_factory=CFGConfig)
    PATTERN_POOL: SamplePoolConfig = Field(default_factory=SamplePoolConfig)
    LATENT_TRAINING: LatentConfig = Field(default_factory=LatentConfig)
    ADVERSARIAL: AdversarialConfig = Field(default_factory=AdversarialConfig)
    TORCH_COMPILE: TorchCompileConfig = Field(default_factory=TorchCompileConfig)
    REPRODUCIBILITY: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )

    COND_DIM: int | None = None
    IM_HEIGHT: int | None = None
    IM_WIDTH: int | None = None

    def model_post_init(self, __context) -> None:
        """Perform cross-field validation after model initialization."""
        # Validate latent training configuration
        if self.LATENT_TRAINING.ENABLED:
            if self.LATENT_TRAINING.LATENT_AE_COMPRESSION < 1:
                raise ValueError("LATENT_AE_COMPRESSION must be >= 1")

        # Validate adversarial training configuration
        if self.ADVERSARIAL.ENABLED:
            if self.ADVERSARIAL.D_LEARNING_RATE <= 0:
                raise ValueError(
                    "When ADVERSARIAL.ENABLED=True, D_LEARNING_RATE must be > 0"
                )
            if len(self.ADVERSARIAL.D_FEATURES) == 0:
                raise ValueError(
                    "When ADVERSARIAL.ENABLED=True, D_FEATURES cannot be empty"
                )

        # Validate dataset-specific requirements
        if self.DATASET.NAME in ["emoji"] and len(self.DATASET.EMOJIS) == 0:
            raise ValueError(
                f"Dataset '{self.DATASET.NAME}' requires EMOJIS list to be non-empty"
            )

        if self.DATASET.NAME in ["e2h", "celeba"] and self.DATASET.DATAROOT is None:
            raise ValueError(
                f"Dataset '{self.DATASET.NAME}' requires DATAROOT to be specified"
            )

        # Validate pattern pool configuration
        if self.PATTERN_POOL.ENABLED and self.PATTERN_POOL.POOL_SIZE <= 0:
            raise ValueError("When PATTERN_POOL.ENABLED=True, POOL_SIZE must be > 0")

        # Validate training configuration consistency
        if self.TRAINING.ITER_N_MIN > self.TRAINING.ITER_N_MAX:
            raise ValueError("ITER_N_MIN cannot be greater than ITER_N_MAX")

        if any(
            step >= self.TRAINING.ITER_N_MIN
            for step in self.LOGGING.INTERMEDIATE_LOGGING_STEPS
        ):
            raise ValueError(
                "All LOGGING.INTERMEDIATE_LOGGING_STEPS must be < TRAINING.ITER_N_MIN"
            )

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
