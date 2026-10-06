# Spatial reasoning with Adaptive NCAs

Train and evaluate Neural Cellular Automata on 2D spatial reasoning puzzles,
as in *2D Spatial Reasoning with Adaptive Neural Cellular Automata*:

| Task | What the NCA does |
| --- | --- |
| **Sudoku** | Fills in a 9x9 Sudoku from its clues. |
| **MNIST Sudoku** | Solves the same puzzles drawn as handwritten digits. Uses a Sudoku model plus a separately trained image encoder/decoder. |
| **Maze** | Marks the shortest path between start and goal in a 30x30 maze. |
| **Color balance** | Turns a random black/white image into one with exactly 50% white pixels. |

The proposed model, the *adaptive NCA* (aNCA), lets every cell learn where to
look: its perception uses deformable convolutions whose sampling positions
change with the cell's state. The grid is treated as a torus, like the surface
of a donut, so a cell at the left border can look at cells at the right
border. Fixed-kernel NCAs are included as baselines.

All commands run from the repository root.

## Framework integration

The host factories call explicit hooks from [registration.py](registration.py)
to add the following components to their registries. The host config schema
imports the extension settings and calls its validator directly.

| Component | Registered selectors |
| --- | --- |
| Perceptions | `anca_deform`, `anca_deform_v1`, `row_conv`, `column_conv` |
| Datasets | `sudoku`, `maze`, `color_balance` |
| Trainer | `spatial_reasoning` |
| Settings | `EXTENSIONS.SPATIAL_REASONING`, validated by [config.py](config.py) |

The three datasets automatically select the spatial trainer when `TRAINER_TYPE`
is omitted or null; explicitly choosing an incompatible trainer is rejected.
The trainer reuses the host rollout, sample pool, logging and checkpointing,
and supplies its own task loss and AdamW optimizer. Its
`USES_FRAMEWORK_LOSS = False` setting prevents construction of an unused host
loss from `TRAINING.LOSS_FN`.

See the general [extension guide](../../../docs/extensions_guide.md) for the
registration hooks and instructions for adding an independent extension.

## Quick start

```bash
uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/sudoku.yaml
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_sudoku --checkpoint train_log/<run>/ca_final.pt
```

Training writes to `train_log/<TRAIN_NAME>_<timestamp>/`. That folder holds
the checkpoints (`ca_final.pt`), the exact config used and sample images.
Source datasets are downloaded on first use through their data providers.
`DATASET.DATAROOT` controls the extension's local caches, defaulting to
`datasets/`; Hugging Face source downloads use its own cache.

## Training

| Task | Command |
| --- | --- |
| Sudoku | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/sudoku.yaml` |
| Maze | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/maze.yaml` |
| Color balance | `uv run python scripts/train_ca.py --config nca/extensions/spatial_reasoning/configs/color_balance.yaml` |

During training, the console (and W&B, if `LOGGING.WANDB: true`) shows the
loss terms every `LOG_INTERVAL` steps. To measure how well a model solves
the task, use the evaluation scripts below.

**Repeated runs:** change the seed, e.g. `-o SEED=43`. Color-balance training
data follows this seed, including when using DataLoader workers; repeat runs
with the same worker count to reproduce the data sequence. The paper reports four
Sudoku runs, five maze runs and three color-balance runs.

**Other models:** every config trains the aNCA by default. The baselines of
the paper's tables are listed in the config as commented `PERCEPTIONS` blocks:
3x3/5x5/7x7/9x9 convolution, row/column/box (R/C/B), dilated convolution and
DCNv1. Replace the active block with one of them.

**Any setting** can be overridden on the command line, e.g.
`-o TRAINING.STEPS=50000` or `-o EXTENSIONS.SPATIAL_REASONING.SUDOKU.DIFFICULTY=[0.5,0.8]`.

For CPU training with the Sudoku or maze example, also override
`-o TRAINING.MIXED_PRECISION=false` when using `--device cpu`: the deformable
perception currently does not support CPU autocast.

### MNIST Sudoku

The Sudoku model itself is not retrained. Instead, an encoder reads the
handwritten clues into a Sudoku board, the Sudoku model solves it and a
decoder draws the result as handwriting. Train the three helper networks once:

```bash
M=nca.extensions.spatial_reasoning.scripts
C=nca/extensions/spatial_reasoning/configs/codec.yaml
# 1. digit classifier, used to train the decoder and to read the final image
uv run python -m $M.train_digit_classifier --out checkpoints/mnist_cnn.pt
# 2. encoder and 3. decoder
uv run python -m $M.train_codec --config $C -o EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=encoder
uv run python -m $M.train_codec --config $C -o EXTENSIONS.SPATIAL_REASONING.CODEC.MODE=decoder
```

The encoder and decoder end up in `train_log/sudoku_codec_<mode>_<timestamp>/`.

## Evaluation (paper tables)

Each script prints a table. Without `--config`, the `config.yaml` next to the
checkpoint is used.

**Sudoku (Tab. 1):** solve rate per difficulty, 1000 puzzles each.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_sudoku --checkpoint train_log/<sudoku run>/ca_final.pt
```

**MNIST Sudoku (Tab. 2):** the Sudoku model together with the encoder and decoder.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_mnist_sudoku \
    --checkpoint train_log/<sudoku run>/ca_final.pt \
    --codec-config nca/extensions/spatial_reasoning/configs/codec.yaml \
    --encoder train_log/<encoder run>/encoder_final.pt \
    --decoder train_log/<decoder run>/decoder_final.pt
```

**Maze (Tab. 3 and 9):** the 1000 test mazes, plus 1000 new mazes of a
different style (Prim's algorithm) that the model has never seen.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_maze --checkpoint train_log/<maze run>/ca_final.pt
```

**Color balance (Tab. 4):** 1000 random start images.
```bash
uv run python -m nca.extensions.spatial_reasoning.scripts.eval_color_balance --checkpoint train_log/<color balance run>/ca_final.pt
```

Common options: `--n-steps` (rollout length; the defaults are the paper's
1500 / 100 / 200 steps), `--batch-size`, `--device`.

### What the numbers mean

| Task | Metric | Meaning |
| --- | --- | --- |
| Sudoku | **ASR** | % of valid completed boards after restoring the original clues. Puzzles can have several solutions, and any valid completion counts. |
| Sudoku | **ACP** | % of filled-in cells that break no row, column or box rule. |
| Sudoku | *ASR(exact)*, *ACP(exact)* | The same, but compared against the one stored solution. |
| Sudoku | Buckets | **Easy**, **Medium**, **Hard**, and **Full** use the removal-fraction ranges in `scripts/eval_sudoku.py`; removed-cell counts are rounded down. |
| Maze | **Valid Path** | % of mazes where the marked path connects start and goal without crossing walls. |
| Maze | **Valid Optimal** | % where that path is also as short as possible. |
| Color balance | **Imbalance** | Average number of pixels away from an exact 50/50 split. |
| Color balance | **Accuracy** | % of images with exactly 50/50. |

For Sudoku, evaluation takes predictions only at originally empty cells and
restores the supplied clues before checking validity. It therefore measures
completion of the missing cells, without separately measuring whether the raw
model output preserved every clue. This postprocessing does not supply the
missing solution digits to the model.

## Configuration reference

Task settings live under `EXTENSIONS.SPATIAL_REASONING`. Model, learning-rate
and rollout settings remain in the usual host sections (`MODEL`, `TRAINING`,
`PATTERN_POOL`). For example, the task selection in a Sudoku config is:

```yaml
DATASET:
  NAME: sudoku
TRAINING:
  TRAINER_TYPE: spatial_reasoning  # Optional: inferred from the dataset.
EXTENSIONS:
  SPATIAL_REASONING:
    LOSS: sudoku_constraint
    WEIGHT_DECAY: 0.0
    SUDOKU:
      DIFFICULTY: [0.01, 0.81]
```

This is a fragment; use [configs/sudoku.yaml](configs/sudoku.yaml) for a complete
model and training configuration. The extension section supplies settings;
`DATASET.NAME` and the model selectors determine which components are used.

The spatial trainer requires `LOSS` to match the selected dataset. The codec
uses the separate `CODEC` settings and its own training entry point.
Unknown settings and unsupported task combinations are validated when the
configuration is loaded. All settings below are relative to
`EXTENSIONS.SPATIAL_REASONING`.

| Setting | Meaning |
| --- | --- |
| `LOSS` | Training objective: `sudoku_constraint`, `maze_path` or `color_balance` (must match `DATASET.NAME`) |
| `WEIGHT_DECAY` | AdamW weight decay |
| `SUDOKU.DIFFICULTY` | Fraction of cells removed from training puzzles, a value or a `[min, max]` range |
| `SUDOKU.*_WEIGHT`, `SOFTMAX_TEMPERATURE` | Loss terms: Sudoku rules, given clues, decisiveness |
| `MAZE.HF_DATASET_ID` | Maze dataset on Hugging Face |
| `MAZE.*_WEIGHT` | Loss terms: path overlap, binary decisions, path length |
| `COLOR_BALANCE.BIAS_MIN/MAX` | Range of the white share of start images |
| `COLOR_BALANCE.WHITE_RATIO` | Target white share (0.5 = 50/50) |
| `COLOR_BALANCE.*_WEIGHT` | Loss terms: global balance, pure black/white pixels |
| `CODEC.*` | Encoder/decoder training (only in `codec.yaml`) |

Set `DATASET.DATAROOT` to relocate parsed maze data and generated Sudoku test
puzzles from `datasets/`. The image size for color balance is
`DATASET.TARGET_SIZE`.

### Accessing settings from Python

Python code should access the validated settings through the extension helper:

```python
from nca.extensions.spatial_reasoning.config import settings

task_config = settings(config)
loss_name = task_config.LOSS
```

For an explicitly supplied section, `config.EXTENSIONS.SPATIAL_REASONING`
returns the same typed settings. `settings(config)` returns default settings when the
section is omitted; those defaults do not select a task loss.
