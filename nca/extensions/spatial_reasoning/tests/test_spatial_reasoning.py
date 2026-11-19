"""Spatial reasoning extension: configs, perceptions, losses, metrics and training.

Runs offline: Sudoku and maze training use synthetic stand-ins for the
Hugging Face data; color balance uses its real (synthetic) dataset.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from pydantic import ValidationError

from nca.core.models.model_factory import create_model
from nca.data.dataset_factory import DATASET_REGISTRY, create_dataset
from nca.extensions.spatial_reasoning.data.color_balance import ColorBalanceDataset
from nca.extensions.spatial_reasoning.trainer import SpatialReasoningTrainer
from nca.extensions.spatial_reasoning.data.maze import generate_prims_mazes, shortest_path, prims_maze
from nca.extensions.spatial_reasoning.data.sudoku import board_to_onehot
from nca.extensions.spatial_reasoning.evaluation import load_checkpoint, maze_metrics, rollout, sudoku_metrics
from nca.extensions.spatial_reasoning.losses import ColorBalanceLoss, MazeDiceBCELoss, SudokuConstraintLoss
from nca.extensions.spatial_reasoning.perceptions import TorusDeformableConv2d
from nca.training.trainer_factory import TRAINER_REGISTRY, create_trainer
from nca.utils.config import Config, load_config
from scripts import train_ca

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# A valid Sudoku solution (digits 1..9).
SOLUTION = torch.tensor([
    [5, 3, 4, 6, 7, 8, 9, 1, 2], [6, 7, 2, 1, 9, 5, 3, 4, 8], [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3], [4, 2, 6, 8, 5, 3, 7, 9, 1], [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4], [2, 8, 7, 4, 1, 9, 6, 3, 5], [3, 4, 5, 2, 8, 6, 1, 7, 9],
])


def config_dict(name, **sections):
    raw = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    for key, value in sections.items():
        if key == "EXTENSIONS":
            for name, settings in value.items():
                raw[key].setdefault(name, {}).update(settings)
        else:
            raw[key] = {**raw.get(key, {}), **value}
    return raw


# --- configuration --------------------------------------------------------------

@pytest.mark.parametrize("name, loss, cond_dim, size, params", [
    ("sudoku", "sudoku_constraint", 1, 9, 165_615),
    ("maze", "maze_path", 2, 30, 256_995),
    ("color_balance", "color_balance", 0, 32, 152_897),
])
def test_example_configs_resolve_to_paper_models(name, loss, cond_dim, size, params):
    config = load_config(str(CONFIGS / f"{name}.yaml")).model_copy(update={"DEVICE": "cpu"})
    assert TRAINER_REGISTRY[config.TRAINING.TRAINER_TYPE] is SpatialReasoningTrainer
    assert config.EXTENSIONS.SPATIAL_REASONING.LOSS == loss
    assert DATASET_REGISTRY[name].__module__.startswith("nca.extensions.spatial_reasoning")
    model = create_model(config, cond_dim, size, size)
    assert sum(p.numel() for p in model.parameters()) == params


def test_codec_config_and_override_round_trip():
    config = load_config(str(CONFIGS / "codec.yaml"))
    assert config.EXTENSIONS.SPATIAL_REASONING.CODEC.MODE == "decoder"
    updated = train_ca.apply_overrides(config, {"EXTENSIONS.SPATIAL_REASONING.CODEC.MODE": "encoder"})
    assert updated.EXTENSIONS.SPATIAL_REASONING.CODEC.MODE == "encoder"
    assert Config.model_validate_json(updated.model_dump_json()) == updated


@pytest.mark.parametrize("sections, message", [
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"LOSS": None}}}, "requires EXTENSIONS.SPATIAL_REASONING.LOSS"),
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"LOSS": "maze_path"}}}, "requires DATASET.NAME 'maze'"),
    ({"LATENT_TRAINING": {"ENABLED": True}}, "does not support"),
    ({"MODEL": {"CHANNEL_N": 8}}, "CHANNEL_N"),
    ({"MODEL": {"PERCEPTIONS": [{"MODE": "anca_deform", "KERNEL_SIZE": 4}]}}, "odd KERNEL_SIZE"),
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"SUDOKU": {"UNKNOWN": 1}}}}, "Extra inputs"),
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"SUDOKU": {"DIFFICULTY": [0.9, 0.1]}}}}, "DIFFICULTY"),
])
def test_invalid_combinations_are_rejected(sections, message):
    with pytest.raises(ValidationError, match=message):
        Config(**config_dict("sudoku", **sections))


def test_host_configs_without_the_section_still_load():
    config = Config(DATASET={"EMOJIS": ["x"]})
    assert config.EXTENSIONS.SPATIAL_REASONING is None
    assert config.model_dump(exclude_none=True)["EXTENSIONS"] == {}


# --- perceptions ----------------------------------------------------------------

def test_deformable_conv_starts_as_circular_convolution():
    torch.manual_seed(0)
    layer = TorusDeformableConv2d(4, 6, kernel_size=3, padding=1)
    x = torch.randn(2, 4, 9, 9)
    expected = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), layer.regular_conv.weight)
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_deformable_sampling_wraps_around_the_torus():
    torch.manual_seed(0)
    layer = TorusDeformableConv2d(3, 5, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        layer.offset_conv.bias.fill_(2.0)  # every tap moves two cells, beyond the 1-cell padding
        shifted = layer(x)
        layer.offset_conv.bias.zero_()
        rolled = layer(torch.roll(x, shifts=(-2, -2), dims=(-2, -1)))
    assert torch.allclose(shifted, rolled, atol=1e-5)


def test_dcn_v1_has_no_modulation():
    assert not hasattr(TorusDeformableConv2d(2, 2, version="v1"), "modulator_conv")
    assert hasattr(TorusDeformableConv2d(2, 2, version="v2"), "modulator_conv")


# --- losses and metrics ---------------------------------------------------------

def test_sudoku_loss_vanishes_on_valid_solutions_and_penalizes_uniform_hedging():
    target = board_to_onehot(SOLUTION).unsqueeze(0)
    cond = torch.zeros(1, 1, 9, 9)
    cond[..., :3, :] = 1.0
    loss = SudokuConstraintLoss()
    solved = loss(target * 20.0, target, cond)
    assert solved["constraint_loss"] < 1e-6 and solved["onehot_loss"] < 1e-6 and solved["given_loss"] < 1e-6

    uniform = torch.zeros(1, 9, 9, 9, requires_grad=True)
    hedged = loss(uniform, target, cond)
    assert hedged["constraint_loss"] < 1e-6 and hedged["onehot_loss"] > 0.1
    hedged["total_loss"].backward()
    assert torch.isfinite(uniform.grad).all() and uniform.grad.abs().sum() > 0


def test_sudoku_metrics_judge_validity_not_the_reference_solution():
    target = (SOLUTION - 1).unsqueeze(0)
    given = torch.zeros(1, 9, 9)
    relabeled = torch.roll(torch.arange(9), 1)[target]  # another valid Sudoku, every cell differs
    m = sudoku_metrics(relabeled, target, given)
    assert m["valid"].item() == 1 and m["acp_valid"].item() == 100
    assert m["solved"].item() == 0 and m["acp"].item() == 0

    broken = target.clone()
    broken[0, 0, 0] = broken[0, 0, 1]
    m = sudoku_metrics(broken, target, given)
    assert m["valid"].item() == 0 and m["acp_valid"].item() < 100


def test_maze_loss_and_metrics_on_a_generated_maze():
    seeds, conds, targets = generate_prims_mazes(11, 2, channel_n=4, seed=1)
    assert torch.equal(generate_prims_mazes(11, 2, channel_n=4, seed=1)[2], targets)
    perfect = maze_metrics(targets, conds, targets)
    assert perfect["valid_path"].tolist() == [1.0, 1.0] and perfect["valid_optimal"].tolist() == [1.0, 1.0]
    empty = maze_metrics(torch.zeros_like(targets), conds, targets)
    assert empty["valid_path"].sum() == 0

    losses = MazeDiceBCELoss()(targets, targets, conds)
    assert torch.isfinite(losses["total_loss"]) and losses["length_loss"] < 1e-4 and losses["dice_loss"] < 1e-4


def test_prims_maze_has_a_unique_corridor_path():
    import numpy as np

    grid = prims_maze(9, np.random.default_rng(0))
    path = shortest_path(grid, (1, 1), (7, 7))
    assert path[0] == (1, 1) and path[-1] == (7, 7)
    assert all(grid[cell] != "#" for cell in path)


def test_color_balance_loss_targets_the_prescribed_ratio():
    balanced = torch.tensor([1.0, -1.0]).repeat(32).view(1, 1, 8, 8)
    result = ColorBalanceLoss()(balanced)
    assert result["count_loss"] < 1e-8 and result["saturation_loss"] < 1e-8 and result["accuracy"] == 1


# --- training -------------------------------------------------------------------

class FakeTask(torch.utils.data.Dataset):
    """Random tensors with the shapes of a task; lets training run offline."""

    def __init__(self, channel_n, shapes, n=16, **_):
        self.channel_n, self.shapes, self.n = channel_n, shapes, n
        self.grid_size = shapes[0][-1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seed_shape, cond_shape, target_shape = self.shapes
        seed = torch.randn(self.channel_n, *seed_shape) * 0.1
        cond = (torch.rand(*cond_shape) > 0.5).float()
        target = F.one_hot(torch.randint(0, target_shape[0], target_shape[1:]), target_shape[0]).permute(2, 0, 1).float()
        return seed, cond, target[: target_shape[0]]


def run_training(monkeypatch, tmp_path, name, *overrides):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WANDB_SWEEP", raising=False)
    args = [
        "--config", str(CONFIGS / f"{name}.yaml"), "--device", "cpu",
        "-o", "TRAINING.STEPS=4", "-o", "TRAINING.WARMUP_STEPS=1", "-o", "TRAINING.BATCH_SIZE=4",
        "-o", "TRAINING.MIXED_PRECISION=false", "-o", "LOGGING.LOG_INTERVAL=2", "-o", "LOGGING.SAVE_INTERVAL=4",
        "-o", "LOGGING.INTERMEDIATE_LOGGING_STEPS=[1]", "-o", "TRAINING.ITER_N_MIN=2", "-o", "TRAINING.ITER_N_MAX=3",
        "-o", "DATASET.NUM_WORKERS=0", "-o", "PATTERN_POOL.POOL_SIZE=8", *overrides,
    ]
    monkeypatch.setattr("sys.argv", ["train_ca.py", *args])
    with pytest.raises(SystemExit) as exit_info:
        train_ca.main()
    assert exit_info.value.code == 0
    (run,) = Path("train_log").iterdir()
    return run


def assert_checkpoint_reloads(run, cond_dim, size, cond):
    config = load_config(str(run / "config.yaml")).model_copy(update={"DEVICE": "cpu"})
    torch.manual_seed(0)
    a = load_checkpoint(create_model(config, cond_dim, size, size), run / "ca_final.pt", "cpu")
    b = load_checkpoint(create_model(config, cond_dim, size, size), run / "ca_final.pt", "cpu")
    x = torch.randn(2, config.MODEL.CHANNEL_N, size, size)
    torch.manual_seed(1)
    out_a = rollout(a, x, cond, 3)
    torch.manual_seed(1)
    out_b = rollout(b, x, cond, 3)
    assert torch.isfinite(out_a).all() and torch.equal(out_a, out_b)


def test_color_balance_trains_and_checkpoint_reloads(monkeypatch, tmp_path):
    def reject_host_loss(_):
        pytest.fail("Spatial training must not construct the unused framework loss")

    monkeypatch.setattr("nca.training.trainer_factory.create_loss_fn", reject_host_loss)
    run = run_training(
        monkeypatch, tmp_path, "color_balance", "-o", "DATASET.TARGET_SIZE=8",
        "-o", "TRAINING.TRAINER_TYPE=null", "-o", "TRAINING.LOSS_FN=lpips",
    )
    assert_checkpoint_reloads(run, 0, 8, None)


def test_sudoku_trains_with_the_constraint_loss(monkeypatch, tmp_path):
    fake = FakeTask(24, ((9, 9), (1, 9, 9), (9, 9, 9)))
    monkeypatch.setitem(DATASET_REGISTRY, "sudoku", lambda config, train: (fake, 1, 9, 9))
    run = run_training(monkeypatch, tmp_path, "sudoku")
    assert_checkpoint_reloads(run, 1, 9, torch.ones(2, 1, 9, 9))


def test_maze_trains_with_the_path_loss(monkeypatch, tmp_path):
    fake = FakeTask(24, ((12, 12), (2, 12, 12), (1, 12, 12)))
    monkeypatch.setitem(DATASET_REGISTRY, "maze", lambda config, train: (fake, 2, 12, 12))
    run = run_training(monkeypatch, tmp_path, "maze")
    assert_checkpoint_reloads(run, 2, 12, torch.zeros(2, 2, 12, 12))


# --- integration regressions ----------------------------------------------------

@pytest.mark.parametrize("name", ["sudoku", "maze", "color_balance"])
@pytest.mark.parametrize("trainer", ["image_gen", "classification", "adversarial"])
def test_spatial_datasets_reject_wrong_trainers(name, trainer):
    with pytest.raises(ValidationError, match="requires TRAINER_TYPE='spatial_reasoning'"):
        Config(**config_dict(name, TRAINING={"TRAINER_TYPE": trainer}))


@pytest.mark.parametrize("name", ["sudoku", "maze", "color_balance"])
def test_spatial_auto_selection_uses_own_loss(monkeypatch, name):
    config = Config(**config_dict(name, TRAINING={"TRAINER_TYPE": None, "LOSS_FN": "lpips"}))
    received = {}

    def capture_init(self, *args, **kwargs):
        received.update(kwargs)

    def reject_host_loss(_):
        pytest.fail("The unused framework loss must not be constructed")

    monkeypatch.setattr(SpatialReasoningTrainer, "__init__", capture_init)
    monkeypatch.setattr("nca.training.trainer_factory.create_loss_fn", reject_host_loss)
    trainer = create_trainer(config, None, None, "unused")
    assert isinstance(trainer, SpatialReasoningTrainer)
    assert "loss_fn" not in received


@pytest.mark.parametrize("sections, message", [
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"LOSS": None}}}, "requires EXTENSIONS.SPATIAL_REASONING.LOSS"),
    ({"EXTENSIONS": {"SPATIAL_REASONING": {"LOSS": "maze_path"}}}, "requires DATASET.NAME 'maze'"),
    ({"LATENT_TRAINING": {"ENABLED": True}}, "does not support"),
    ({"ADVERSARIAL": {"ENABLED": True}}, "does not support"),
    ({"CFG": {"ENABLED": True}}, "does not support"),
])
def test_auto_selection_preserves_spatial_validation(sections, message):
    with pytest.raises(ValidationError, match=message):
        Config(**config_dict("sudoku", TRAINING={"TRAINER_TYPE": None}, **sections))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_color_balance_dataloader_respects_framework_seed(num_workers, monkeypatch):
    # Forking after multithreaded torch tests can deadlock; exercise real workers
    # with spawn so this regression is independent of the suite's execution order.
    if num_workers:
        from functools import partial
        monkeypatch.setattr(
            "nca.data.dataset_factory.DataLoader",
            partial(torch.utils.data.DataLoader, multiprocessing_context="spawn", timeout=20),
        )

    def batches(seed):
        config = Config(**config_dict(
            "color_balance", DATASET={"TARGET_SIZE": 8, "NUM_WORKERS": num_workers},
            TRAINING={"BATCH_SIZE": 2, "STEPS": 2, "WARMUP_STEPS": 0},
        ))
        config.SEED = seed
        train_ca.setup_seed(seed)
        loader, *_ = create_dataset(config)
        return list(loader)

    a, b, c = batches(42), batches(42), batches(43)
    assert len(a) == 2
    assert all(torch.equal(x, y) for ab, bb in zip(a, b) for x, y in zip(ab, bb))
    assert not torch.equal(a[0][0], c[0][0])
    assert not torch.equal(a[0][0], a[1][0])


def test_color_balance_test_samples_are_independent_of_global_seed():
    dataset = ColorBalanceDataset(4, size=8, train=False)
    train_ca.setup_seed(42)
    a = dataset[7]
    train_ca.setup_seed(43)
    b = dataset[7]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("trainer_key, expects_host_loss", [("image_gen", True), ("adversarial", False)])
def test_host_trainers_preserve_loss_ownership(monkeypatch, trainer_key, expects_host_loss):
    config = Config(DATASET={"EMOJIS": ["x"]}, TRAINING={"TRAINER_TYPE": trainer_key})
    received, calls = {}, []
    loss = torch.nn.MSELoss()

    def capture_init(self, *args, **kwargs):
        received.update(kwargs)

    def make_loss(cfg):
        calls.append(cfg)
        return loss

    monkeypatch.setattr(TRAINER_REGISTRY[trainer_key], "__init__", capture_init)
    monkeypatch.setattr("nca.training.trainer_factory.create_loss_fn", make_loss)
    create_trainer(config, None, None, "unused")
    assert len(calls) == int(expects_host_loss)
    if expects_host_loss:
        assert received["loss_fn"] is loss
    else:
        assert "loss_fn" not in received
