"""Explicit integration hooks; registries are owned by the host framework."""


def register_perceptions(registry):
    from .perceptions import AncaDeformablePerception, ColumnConvPerception, RowConvPerception

    registry["anca_deform"] = lambda in_ch, cfg, dev: AncaDeformablePerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE, version="v2",
    )
    registry["anca_deform_v1"] = lambda in_ch, cfg, dev: AncaDeformablePerception(
        in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE, version="v1",
    )
    registry["row_conv"] = lambda in_ch, cfg, dev: RowConvPerception(in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE)
    registry["column_conv"] = lambda in_ch, cfg, dev: ColumnConvPerception(in_ch, cfg.OUT_CHANNEL, cfg.KERNEL_SIZE)


def register_trainers(registry, required_trainers):
    from .config import LOSS_DATASETS
    from .trainer import SpatialReasoningTrainer

    registry["spatial_reasoning"] = SpatialReasoningTrainer
    required_trainers.update({dataset: "spatial_reasoning" for dataset in LOSS_DATASETS.values()})


def register_datasets(registry):
    from .data.factories import create_color_balance, create_maze, create_sudoku

    registry["sudoku"] = create_sudoku
    registry["maze"] = create_maze
    registry["color_balance"] = create_color_balance
