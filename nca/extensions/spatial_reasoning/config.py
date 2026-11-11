"""The ``EXTENSIONS.SPATIAL_REASONING`` config section and its cross-field validation.

All extension settings live under EXTENSIONS.SPATIAL_REASONING, grouped per task,
so the host ``TRAINING``/``DATASET`` schemas stay untouched. Independent of the
host ``Config`` and of model/trainer modules to avoid circular imports.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Losses of the spatial_reasoning trainer and the dataset whose targets and conditions they expect.
LOSS_DATASETS = {"sudoku_constraint": "sudoku", "maze_path": "maze", "color_balance": "color_balance"}
DEFORMABLE_MODES = ("anca_deform", "anca_deform_v1")


class Section(BaseModel):
    """Same strictness as the host schema: unknown keys and non-finite floats are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class SudokuSettings(Section):
    """Array Sudoku data and loss.

    DIFFICULTY: Fraction of removed cells, a value or a ``[min, max]`` range.
    TRAIN_SAMPLES: Epoch length of the on-the-fly training set.
    *_WEIGHT: Row/column/box constraint, given-cell cross-entropy and one-hot
        tightening (squared probabilities must also sum to one per unit).
    SOFTMAX_TEMPERATURE: Applied to the digit logits before the softmax.
    """

    DIFFICULTY: float | list[float] = Field(default_factory=lambda: [0.01, 0.81])
    TRAIN_SAMPLES: int = Field(default=1_000_000, gt=0)
    CONSTRAINT_WEIGHT: float = Field(default=1.0, ge=0)
    GIVEN_WEIGHT: float = Field(default=1.0, ge=0)
    ONEHOT_WEIGHT: float = Field(default=0.1, ge=0)
    SOFTMAX_TEMPERATURE: float = Field(default=0.25, gt=0)

    @field_validator("DIFFICULTY")
    @classmethod
    def check_difficulty(cls, value):
        bounds = value if isinstance(value, list) else [value, value]
        if len(bounds) != 2 or not 0.0 <= bounds[0] <= bounds[1] <= 1.0:
            raise ValueError("DIFFICULTY must be in [0, 1] or a [min, max] range within it.")
        return value


class MazeSettings(Section):
    """Maze benchmark and the BCE + Dice + path-length loss on non-wall cells."""

    HF_DATASET_ID: str = "sapientinc/maze-30x30-hard-1k"
    BCE_WEIGHT: float = Field(default=1.0, ge=0)
    DICE_WEIGHT: float = Field(default=1.0, ge=0)
    LENGTH_WEIGHT: float = Field(default=0.05, ge=0)


class ColorBalanceSettings(Section):
    """Color balance data and loss; the image size is ``DATASET.TARGET_SIZE``.

    BIAS_MIN/MAX: Range of the seed's white fraction.
    WHITE_RATIO: Prescribed white fraction of the output.
    """

    BIAS_MIN: float = Field(default=0.1, ge=0, le=1)
    BIAS_MAX: float = Field(default=0.9, ge=0, le=1)
    WHITE_RATIO: float = Field(default=0.5, ge=0, le=1)
    COUNT_WEIGHT: float = Field(default=1.0, ge=0)
    SATURATION_WEIGHT: float = Field(default=0.01, ge=0)


class CodecSettings(Section):
    """MNIST Sudoku codec training (``scripts/train_codec.py``).

    MODE: Train the ``encoder`` (masked cross-entropy on clue cells) or the
        ``decoder`` (conditional VAE with pixel, feature, classifier, TV, ink
        and binary terms). The two are trained separately.
    CELL_SIZE: Pixel size of one MNIST Sudoku cell.
    CLASSIFIER_PATH: Frozen digit classifier for the decoder losses and for scoring.
    SOLUTION_RATIO: Fraction of decoder samples rendered from the full solution.
    """

    MODE: Literal["encoder", "decoder"] = "decoder"
    CELL_SIZE: int = Field(default=28, gt=0)
    CLASSIFIER_PATH: Path = Path("checkpoints/mnist_cnn.pt")
    BATCH_SIZE: int = Field(default=32, gt=0)
    STEPS: int = Field(default=100_000, gt=0)
    LEARNING_RATE: float = Field(default=1e-3, gt=0)
    WEIGHT_DECAY: float = Field(default=1e-5, ge=0)
    WARMUP_STEPS: int = Field(default=1000, ge=0)
    LOG_INTERVAL: int = Field(default=1000, gt=0)
    SAVE_INTERVAL: int = Field(default=10_000, gt=0)
    SOLUTION_RATIO: float = Field(default=0.5, ge=0, le=1)
    ENCODER_BASE_CHANNELS: int = Field(default=32, gt=0)
    ENCODER_HIDDEN_DIM: int = Field(default=128, gt=0)
    DECODER_BASE_CHANNELS: int = Field(default=64, gt=0)
    DECODER_HIDDEN_DIM: int = Field(default=128, gt=0)
    DECODER_STYLE_DIM: int = Field(default=8, ge=0)
    KL_WEIGHT: float = Field(default=0.001, ge=0)
    PIXEL_L1_WEIGHT: float = Field(default=1.0, ge=0)
    FEATURE_L1_WEIGHT: float = Field(default=0.5, ge=0)
    CLASSIFIER_CE_WEIGHT: float = Field(default=1.0, ge=0)
    TV_WEIGHT: float = Field(default=0.01, ge=0)
    INK_WEIGHT: float = Field(default=0.0025, ge=0)
    BINARY_WEIGHT: float = Field(default=0.01, ge=0)


class SpatialReasoningConfig(Section):
    """``EXTENSIONS.SPATIAL_REASONING`` section; only the selected task's subsection is used.

    LOSS: Training objective of the ``spatial_reasoning`` trainer; it replaces
        ``TRAINING.LOSS_FN`` because these losses also need the condition.
    WEIGHT_DECAY: AdamW weight decay of the ``spatial_reasoning`` trainer
        (0 makes AdamW identical to Adam).
    CODEC: Set only in codec configs.
    """

    LOSS: Literal["sudoku_constraint", "maze_path", "color_balance"] | None = None
    WEIGHT_DECAY: float = Field(default=0.0, ge=0)
    SUDOKU: SudokuSettings = Field(default_factory=SudokuSettings)
    MAZE: MazeSettings = Field(default_factory=MazeSettings)
    COLOR_BALANCE: ColorBalanceSettings = Field(default_factory=ColorBalanceSettings)
    CODEC: CodecSettings | None = None


def settings(config) -> SpatialReasoningConfig:
    """The ``EXTENSIONS.SPATIAL_REASONING`` section, or its defaults when the YAML omits it."""
    return config.EXTENSIONS.SPATIAL_REASONING or SpatialReasoningConfig()


def validate_spatial_reasoning_config(config):
    """Check task, trainer, dataset and perception combinations."""
    for perception in config.MODEL.PERCEPTIONS:
        if perception.MODE in DEFORMABLE_MODES and perception.KERNEL_SIZE % 2 == 0:
            raise ValueError(f"{perception.MODE} requires an odd KERNEL_SIZE.")

    section = settings(config)
    if section.CODEC is not None:
        if config.DATASET.NAME != "sudoku":
            raise ValueError("Codec configs require DATASET.NAME=sudoku.")
        return

    dataset, loss = config.DATASET.NAME, section.LOSS
    trainer = config.TRAINING.TRAINER_TYPE
    if dataset in LOSS_DATASETS.values():
        if trainer not in (None, "spatial_reasoning"):
            raise ValueError(f"Dataset '{dataset}' requires TRAINER_TYPE='spatial_reasoning'.")
        trainer = "spatial_reasoning"  # Match the factory's automatic selection.
    if trainer == "spatial_reasoning":
        if loss is None:
            raise ValueError("TRAINER_TYPE 'spatial_reasoning' requires EXTENSIONS.SPATIAL_REASONING.LOSS.")
        if dataset != LOSS_DATASETS[loss]:
            raise ValueError(f"EXTENSIONS.SPATIAL_REASONING.LOSS '{loss}' requires DATASET.NAME '{LOSS_DATASETS[loss]}'.")
        if config.LATENT_TRAINING.ENABLED or config.ADVERSARIAL.ENABLED or config.CFG.ENABLED:
            raise ValueError("The spatial_reasoning trainer does not support latent, adversarial or CFG training.")
    if dataset == "sudoku" and config.MODEL.CHANNEL_N < 9:
        raise ValueError("Sudoku requires MODEL.CHANNEL_N >= 9 (one channel per digit).")
    if dataset == "color_balance" and section.COLOR_BALANCE.BIAS_MIN > section.COLOR_BALANCE.BIAS_MAX:
        raise ValueError("COLOR_BALANCE.BIAS_MIN cannot exceed BIAS_MAX.")
