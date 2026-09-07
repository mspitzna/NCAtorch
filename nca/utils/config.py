from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields and non-finite floats; validate defaults like user input."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


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
    KERNEL_SIZE: int = Field(default=3, gt=0)
    DILATION: int = Field(default=1, gt=0)
    OUT_CHANNEL: int = Field(default=80, gt=0)
    # attention / mh_attention
    NUM_HEADS: int = Field(default=4, gt=0)
    EMBED_DIM: int = Field(default=128, gt=0)
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

    @model_validator(mode="after")
    def check_attention_dimensions(self):
        if self.MODE in {"attention", "mh_attention"}:
            if self.KERNEL_SIZE % 2 == 0:
                raise ValueError("Attention KERNEL_SIZE must be odd.")
            dimension = self.OUT_CHANNEL if self.MODE == "attention" else self.EMBED_DIM
            if dimension % self.NUM_HEADS:
                raise ValueError("Attention projection dimension must be divisible by NUM_HEADS.")
        return self


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
    HIDDEN_CHANNELS: list[Annotated[int, Field(gt=0)]] = Field(default_factory=lambda: [64])
    CHANNEL_N: int = Field(default=16, gt=0)
    CHANNEL_OUT: int | None = Field(default=None, gt=0)
    _channel_out_auto: bool = PrivateAttr(default=False)
    USE_POSITIONAL_EMBEDDINGS: bool = False
    LIVING_MASK: bool = False
    LIVING_MASK_INDEX: int = Field(default=3, ge=0)
    NOISE_INJECTION: float = Field(default=0.0, ge=0, le=1)
    FINAL_ACTIVATION: bool = False
    CLAMP_OUTPUT: bool = False
    CLAMP_OUTPUT_MIN: float = -1.0
    CLAMP_OUTPUT_MAX: float = 1.0
    FIRE_RATE: float = Field(default=0.5, ge=0, le=1)

    RESNET_BLOCKS: int = Field(default=2, ge=0)

    PERCEPTIONS: list[PerceptionConfig] = Field(
        default_factory=lambda: [PerceptionConfig()], min_length=1
    )

    @field_validator("ARCHITECTURE")
    @classmethod
    def check_architecture(cls, value):
        from nca.core.models.model_factory import MODEL_REGISTRY
        if value not in MODEL_REGISTRY:
            raise ValueError(f"MODEL.ARCHITECTURE must be one of {sorted(MODEL_REGISTRY)}.")
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
            self._channel_out_auto = True
            self.CHANNEL_OUT = self.CHANNEL_N
        return self

    @property
    def channel_out_is_auto(self) -> bool:
        """Whether CHANNEL_OUT was derived from CHANNEL_N when parsing this config."""
        return self._channel_out_auto

    @field_serializer("CHANNEL_OUT")
    def serialize_channel_out(self, value):
        # Preserve automatic sizing across saved configs and subsequent overrides.
        return None if self._channel_out_auto else value

    @model_validator(mode="after")
    def check_living_mask_index(self):
        if self.LIVING_MASK and not (0 <= self.LIVING_MASK_INDEX < self.CHANNEL_N):
            raise ValueError(
                f"LIVING_MASK_INDEX ({self.LIVING_MASK_INDEX}) must be in [0, CHANNEL_N) "
                f"(0 to {self.CHANNEL_N - 1} inclusive)."
            )
        return self

    @model_validator(mode="after")
    def check_architecture_settings(self):
        if self.NAME == "ResNet" and not self.HIDDEN_CHANNELS:
            raise ValueError("ResNet requires at least one HIDDEN_CHANNELS entry.")
        if self.CLAMP_OUTPUT_MIN > self.CLAMP_OUTPUT_MAX:
            raise ValueError("CLAMP_OUTPUT_MIN cannot exceed CLAMP_OUTPUT_MAX.")
        return self


class TrainingConfig(StrictModel):
    """Hyperparameters for the main CA training loop.

    Attributes:
        BATCH_SIZE: Number of grids per training batch.
        STEPS: Total training batches, including critic-only batches.
        LOSS_FN: Reconstruction loss key from ``LOSS_FN_REGISTRY`` —
            ``mse``, ``l1``, ``lpips``, ``vgg``, ``p_ce``, ``i_ce``, ``overflow``.
        OVERFLOW_LOSS: Add an overflow penalty to the loss.
        OVERFLOW_WEIGHT: Weight applied to the overflow penalty term.
        LEARNING_RATE: Initial learning rate.
        WARMUP_STEPS: Linear LR warm-up in training batches (cosine/constant/wsd).
            Scheduler values are refreshed after successful optimizer updates.
        GRADIENT_ACCUMULATION_STEPS: Generator-backward batches averaged per
            optimizer update. A final partial group is averaged and applied.
        LR_SCHEDULE_MODE: LR schedule — ``step``, ``cosine``, or ``constant``.
        ITER_N_MIN: Minimum CA rollout steps per batch.
        ITER_N_MAX: Inclusive maximum CA rollout steps per batch (sampled uniformly).
        GRADIENT_CLIPPING_NORM: Max gradient norm; set to ``0`` to disable.
        MIXED_PRECISION: Enable automatic mixed precision (AMP).
        LPIPS_NET: Backbone for LPIPS loss — ``alex``, ``vgg``, or ``squeeze``.
        VGG_PROJ_N: Number of random projections in the VGG style loss.
        TRAINER_TYPE: Trainer to use.
        EVOLVE_MODE: Rollout strategy key from ``EVOLVER_REGISTRY``.
    """

    BATCH_SIZE: int = Field(default=12, gt=0)
    STEPS: int = Field(default=10000, gt=0)
    LOSS_FN: str = "mse"
    OVERFLOW_LOSS: bool = False
    LEARNING_RATE: float = Field(default=0.002, gt=0, le=1)
    WARMUP_STEPS: int = Field(default=2000, ge=0)
    LR_SCHEDULE_MODE: Literal["step", "cosine", "constant", "wsd"] = "step"
    MILESTONES: list[Annotated[int, Field(ge=0)]] = Field(default_factory=lambda: [2000, 8000])
    LR_GAMMA: float = Field(default=0.1, ge=0)
    OPTIMIZER_BETAS: list[Annotated[float, Field(ge=0, lt=1)]] = Field(
        default_factory=lambda: [0.9, 0.999], min_length=2, max_length=2
    )
    ITER_N_MIN: int = Field(default=32, gt=0)
    ITER_N_MAX: int = Field(default=64, gt=0)
    GRADIENT_CLIPPING_NORM: float = Field(default=1.0, ge=0)
    GRADIENT_CHECKPOINTING: bool = False
    GRADIENT_CHECKPOINT_SEGMENTS: int = Field(default=16, gt=0)
    GRADIENT_ACCUMULATION_STEPS: int = Field(default=1, gt=0)
    MIXED_PRECISION: bool = False
    OVERFLOW_WEIGHT: float = Field(default=1.0, ge=0)
    LPIPS_NET: Literal["alex", "vgg", "squeeze"] = "alex"
    VGG_PROJ_N: int = Field(default=32, gt=0)
    TRAINER_TYPE: str | None = None
    EVOLVE_MODE: str = "base"
    # WSD schedule — stable phase fills the gap between warmup and decay
    WSD_DECAY_RATIO: float = Field(default=0.1, ge=0, le=1)
    WSD_MIN_LR_RATIO: float = Field(default=0.0, ge=0, le=1)

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

    @model_validator(mode="after")
    def check_rollout_lengths(self):
        if self.ITER_N_MIN > self.ITER_N_MAX:
            raise ValueError("ITER_N_MIN cannot be greater than ITER_N_MAX")
        if self.GRADIENT_CHECKPOINTING and self.GRADIENT_CHECKPOINT_SEGMENTS > self.ITER_N_MIN:
            raise ValueError("GRADIENT_CHECKPOINT_SEGMENTS cannot exceed ITER_N_MIN when checkpointing is enabled.")
        return self

    @model_validator(mode="after")
    def check_wsd_phase_lengths(self):
        if self.LR_SCHEDULE_MODE == "wsd":
            decay_steps = int(self.STEPS * self.WSD_DECAY_RATIO)
            if self.WARMUP_STEPS + decay_steps > self.STEPS:
                raise ValueError("WSD warmup and decay phases cannot exceed TRAINING.STEPS.")
        return self


class DatasetConfig(StrictModel):
    NAME: str = "emoji"
    DATAROOT: Path | None = None
    DATASET_SAMPLE_PATH: Path | None = None
    DROP_LAST_BATCH: bool = True
    TARGET_SIZE: int = Field(default=64, gt=0)
    TARGET_PADDING: int = Field(default=0, ge=0)
    EMOJIS: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    HISTORY_N: int = Field(default=1, gt=0)
    REVERSE_HISTORY_SEED: bool = False
    NUM_WORKERS: int = Field(default=0, ge=0)
    SEED_SIZE: int = Field(default=1, gt=0)  # Cross size for GrowingMNISTDataset
    ENABLE_ROTATION: bool = (
        False  # Enable rotation transformations in GrowingMNISTDataset
    )
    ENABLE_ZOOM: bool = False  # Enable zoom transformations in GrowingMNISTDataset
    Z_LATENT_NOISE_CHANNEL: bool = (
        False  # Add latent noise channel as last dimension to target and seed
    )

    @field_validator("DATASET_SAMPLE_PATH")
    @classmethod
    def check_path_exists(cls, value: Path | None):
        if value is not None and not value.exists():
            raise ValueError(f"Dataset sample path does not exist: {value}")
        return value

    @field_validator("NAME")
    @classmethod
    def check_dataset_name(cls, value):
        from nca.data.dataset_factory import DATASET_REGISTRY

        if value not in DATASET_REGISTRY:
            raise ValueError(f"DATASET.NAME must be one of {sorted(DATASET_REGISTRY)}.")
        return value


class CFGConfig(StrictModel):
    ENABLED: bool = False
    DROPOUT_PROB: float = Field(default=0.1, ge=0, le=1)
    NULL_CONDITION_TYPE: Literal["zeros", "learned"] = "zeros"
    GOAL_CHANNELS: bool = False
    PRESERVE_CHANNELS: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list
    )  # Channels to NOT zero out during CFG


class SamplePoolConfig(StrictModel):
    """Persistent pool settings.

    POOL_START_RATIO and POOL_END_RATIO are the initial and final fractions
    of each batch sampled from the pool, scheduled linearly over TRAINING.STEPS.
    POOL_DELAY gates sampling until that training step. For timeseries pools,
    replacement additionally requires a matching previous frame.
    POOL_DMG_RATIO is the fraction of successfully reused samples to damage,
    rounded down to a whole number of samples. These ratios do not limit commits.
    """

    ENABLED: bool = False
    TIMESERIES_POOL: bool = False
    POOL_SIZE: int = Field(default=1024, gt=0)
    POOL_DELAY: int = Field(default=1000, ge=0)
    POOL_START_RATIO: float = Field(default=0.5, ge=0, le=1)
    POOL_END_RATIO: float = Field(default=0.5, ge=0, le=1)
    POOL_DMG_RATIO: float = Field(default=0.0, ge=0, le=1)
    POOL_DMG_DELAY: int | None = Field(default=None, gt=0)  # None means immediate.
    POOL_MUTATION_RATIO: float = Field(default=0.0, ge=0, le=1)


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
        LATENT_AE_COMPRESSION: AE spatial downsampling factor as 2^N.
        AE_CHECKPOINT: Explicit path to a pre-trained encoder checkpoint;
            CA training/inference use the default path inside ``FOLDER_NAME``
            if ``None``. Encoder training starts fresh if ``None``.
        VAE_KL_BETA: Weight of the KL divergence term in the VAE loss.
        VAE_BASE_CHANNELS: Base feature channels in VAE encoder/decoder.
        VAE_NUM_DOWNSAMPLES: Number of stride-2 stages for VAE and VQVAE.
        VAE_NORM_GROUPS: Group normalisation groups in VAE conv layers.
        VAE_RECON_LOSS_TYPE: Pixel reconstruction loss — ``l1`` or ``mse``.
        VAE_RECON_LOSS_WEIGHT: Weight for the pixel reconstruction term.
        VAE_VGG_LOSS_WEIGHT: Weight for the VGG perceptual loss term.
        VQVAE_NUM_EMBEDDINGS: Codebook size for VQVAE.
        VQVAE_COMMITMENT_COST: Commitment loss weight (β) for VQVAE.
    """

    ENABLED: bool = False
    ENCODER_TYPE: str = "AE"
    LATENT_AE_STEPS: int = Field(default=10000, gt=0)
    LATENT_AE_WARMUP_STEPS: int = Field(default=2000, ge=0)
    LATENT_AE_LR: float = Field(default=0.001, gt=0)
    LATENT_AE_IN_CHANNEL: int = Field(default=4, gt=0)
    LATENT_AE_OUT_CHANNEL: int = Field(default=4, gt=0)
    LATENT_AE_CHANNEL: int = Field(default=64, gt=0)
    LATENT_AE_COMPRESSION: int = Field(default=3, ge=1)
    LATENT_AE_LOG_INTERVAL: int = Field(default=2500, gt=0)
    LATENT_AE_SAVE_INTERVAL: int = Field(default=5000, gt=0)
    APPLY_DAMAGE: bool = False
    AE_CHECKPOINT: Path | None = None
    VAE_KL_BETA: float = Field(default=1.0, ge=0)
    VAE_BASE_CHANNELS: int = Field(default=64, gt=0)
    VAE_NUM_DOWNSAMPLES: int = Field(default=5, ge=0)
    VAE_NORM_GROUPS: int = Field(default=32, gt=0)
    VAE_KL_WARMUP_STEPS: int = Field(default=0, ge=0)
    VAE_BATCH_SIZE: int = Field(default=18, gt=0)
    VAE_RECON_LOSS_TYPE: Literal["l1", "mse"] = "l1"
    VAE_RECON_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VAE_VGG_LOSS_WEIGHT: float = Field(default=1.0, ge=0)
    VQVAE_NUM_EMBEDDINGS: int = Field(default=512, gt=0)
    VQVAE_COMMITMENT_COST: float = Field(default=0.25, ge=0)

    @model_validator(mode="after")
    def check_encoder_dimensions(self):
        if self.ENCODER_TYPE == "AE":
            if self.LATENT_AE_CHANNEL < 2 ** (self.LATENT_AE_COMPRESSION - 1):
                raise ValueError(
                    "LATENT_AE_CHANNEL must be at least 2^(LATENT_AE_COMPRESSION - 1) "
                    "to keep AE decoder channels positive."
                )
        elif self.ENCODER_TYPE in {"VAE", "VQVAE"}:
            if self.VAE_NUM_DOWNSAMPLES > 0 and self.VAE_BASE_CHANNELS % self.VAE_NORM_GROUPS:
                raise ValueError("VAE_BASE_CHANNELS must be divisible by VAE_NORM_GROUPS.")
        return self

    def get_latent_shape(self, height: int, width: int) -> tuple[int, int]:
        """Validate actual image dimensions and return the encoder's spatial shape."""
        stages = (
            self.LATENT_AE_COMPRESSION
            if self.ENCODER_TYPE == "AE"
            else self.VAE_NUM_DOWNSAMPLES
        )
        factor = 2 ** stages
        if height < factor or width < factor or height % factor or width % factor:
            raise ValueError(
                f"{self.ENCODER_TYPE} image height and width must be positive multiples of {factor}; "
                f"got {height}x{width}. The decoder upsamples by {factor}."
            )
        return height // factor, width // factor

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
    MODE: Literal["default", "max-autotune-no-cudagraphs"] = "default"
    DEBUG: bool = False


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
        D_WARMUP_STEPS: Linear LR warm-up duration in discriminator updates.
        D_GAMMA: Final discriminator LR divided by D_LEARNING_RATE (0 to 1).
            After warmup, cosine decay spans the remaining updates following
            D_START_TRAINING; skipped AMP updates do not advance the schedule.
        D_N_CRITIC: Batch interval between generator backward passes during
            adversarial training. The critic updates each batch; generator
            updates average GRADIENT_ACCUMULATION_STEPS backward passes.
        D_GP_WEIGHT: Gradient penalty coefficient λ in the WGAN-GP loss.
        D_DOWNSCALE_FACTOR: Spatially downscale inputs to the discriminator
            by this positive integer factor using area resizing. Applies to
            real, fake and gradient-penalty inputs, including conditions/seeds.
        LPIPS_WEIGHT: Weight for an additional full-resolution LPIPS term in
            the generator loss, including generator warmup; 0 disables it.
            Uses TRAINING.LPIPS_NET. Adds to RECON_WEIGHT if LOSS_FN is lpips.
        ADV_WEIGHT: Weight for the adversarial term in the generator loss.
        RECON_WEIGHT: Weight for the reconstruction term in the generator loss.
        SEED_TO_CRITIC: Pass the seed image as an additional channel to the
            discriminator (conditional GAN setup).
    """

    ENABLED: bool = False
    D_IN_CHANNELS: int = Field(default=4, gt=0)
    D_FEATURES: list[Annotated[int, Field(gt=0)]] = Field(
        default_factory=lambda: [64, 128, 256, 512], min_length=1
    )
    D_LEARNING_RATE: float = Field(default=0.001, gt=0)
    D_START_TRAINING: int = Field(default=0, ge=0)
    D_WARMUP_STEPS: int = Field(default=0, ge=0)
    D_GAMMA: float = Field(default=0.1, ge=0, le=1)
    D_N_CRITIC: int = Field(default=1, gt=0)
    D_GP_WEIGHT: float = Field(default=10.0, ge=0)
    D_DOWNSCALE_FACTOR: int = Field(default=1, ge=1)
    LPIPS_WEIGHT: float = Field(default=0.0, ge=0)
    ADV_WEIGHT: float = Field(default=1.0, ge=0)
    RECON_WEIGHT: float = Field(default=1.0, ge=0)
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
    LOG_INTERVAL: int = Field(default=100, gt=0)
    SAVE_INTERVAL: int = Field(default=10000, gt=0)
    INTERMEDIATE_LOGGING_STEPS: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=lambda: [5, 15, 25]
    )
    OBSERVERS: list[ObserverConfig] = Field(default_factory=list)


class Config(StrictModel):
    SEED: int = Field(default=-1, ge=-1, le=2**32 - 1)
    DEVICE: str = Field(default="cuda", min_length=1)

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

    COND_DIM: int | None = Field(default=None, ge=0)
    IM_HEIGHT: int | None = Field(default=None, gt=0)
    IM_WIDTH: int | None = Field(default=None, gt=0)

    def model_post_init(self, __context) -> None:
        """Perform cross-field validation after model initialization."""
        # Validate dataset-specific requirements
        if self.DATASET.NAME in ["emoji"] and len(self.DATASET.EMOJIS) == 0:
            raise ValueError(
                f"Dataset '{self.DATASET.NAME}' requires EMOJIS list to be non-empty"
            )

        if self.DATASET.NAME in ["e2h", "celeba"] and self.DATASET.DATAROOT is None:
            raise ValueError(
                f"Dataset '{self.DATASET.NAME}' requires DATAROOT to be specified"
            )

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
