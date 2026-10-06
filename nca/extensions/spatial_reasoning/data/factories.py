"""``DATASET_REGISTRY`` constructors: ``(config, train) -> (dataset, cond_dim, height, width)``."""

from pathlib import Path

from ..config import settings


def dataroot(config) -> Path:
    return Path(config.DATASET.DATAROOT or "datasets")


def create_sudoku(config, train):
    from .sudoku import GRID_SIZE, SudokuDataset

    sudoku = settings(config).SUDOKU
    dataset = SudokuDataset(
        channel_n=config.MODEL.CHANNEL_N,
        difficulty=sudoku.DIFFICULTY,
        total_samples=sudoku.TRAIN_SAMPLES if train else 1000,
        train=train,
        seed=config.SEED,
        dataroot=dataroot(config),
    )
    return dataset, 1, GRID_SIZE, GRID_SIZE


def create_maze(config, train):
    from .maze import MazeDataset

    dataset = MazeDataset(
        channel_n=config.MODEL.CHANNEL_N,
        train=train,
        hf_dataset_id=settings(config).MAZE.HF_DATASET_ID,
        dataroot=dataroot(config),
    )
    return dataset, 2, dataset.grid_size, dataset.grid_size


def create_color_balance(config, train):
    from .color_balance import ColorBalanceDataset

    balance, size = settings(config).COLOR_BALANCE, config.DATASET.TARGET_SIZE
    dataset = ColorBalanceDataset(
        channel_n=config.MODEL.CHANNEL_N,
        size=size,
        train=train,
        bias_min=balance.BIAS_MIN,
        bias_max=balance.BIAS_MAX,
        white_ratio=balance.WHITE_RATIO,
    )
    return dataset, 0, size, size
