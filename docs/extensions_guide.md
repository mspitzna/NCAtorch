# NCAtorch Extensions

Extensions add models, rollout strategies, trainers, and datasets to NCAtorch.
Each extension keeps its implementation and example configurations under
`nca/extensions/<name>/` and connects to the framework through explicit factory
registration and configuration schema changes.

Extensions live inside the Python package. There is no automatic discovery;
creating a directory or importing its `__init__.py` does not register components.
This guide uses `my_extension` as a placeholder. Implement the referenced classes
and functions before using the snippets.

## Integration points

Add only the components your feature needs. A new training objective can reuse
the existing model and evolver; a new rollout strategy can reuse a trainer.

| Component | Host file and registry | YAML selector |
| --- | --- | --- |
| State transition/model | `nca/core/models/model_factory.py`: `MODEL_REGISTRY` | `MODEL.ARCHITECTURE` |
| Rollout strategy | `nca/training/evolve_factory.py`: `EVOLVER_REGISTRY` | `TRAINING.EVOLVE_MODE` |
| Training objective/loop | `nca/training/trainer_factory.py`: `TRAINER_REGISTRY` | `TRAINING.TRAINER_TYPE` |
| Dataset, if needed | `nca/data/dataset_factory.py`: `DATASET_REGISTRY` | `DATASET.NAME` |
| Configuration fields/constraints | `nca/utils/config.py` | Corresponding config section |

`MODEL.NAME` selects the update network, such as `MLP`; `MODEL.ARCHITECTURE`
selects the complete state-transition architecture. For features that only add
one neighborhood operator or update network, see the narrower
[perception](custom_perception_guide.md) and
[update module](custom_update_module_guide.md) guides.

## Directory structure

```text
nca/extensions/
    __init__.py
    my_extension/
        __init__.py       # Lightweight package description.
        README.md         # Wiring, supported combinations, and run command.
        config.py         # Extension fields and validation.
        registration.py   # Explicit hooks for host registries.
        model.py          # State transition, if needed.
        factory.py        # Model construction, if needed.
        evolver.py        # Rollout implementation, if needed.
        trainer.py        # Training objective, if needed.
        configs/
            my_extension_config.yaml  # Runnable example configurations.
tests/test_my_extension.py
```

Keep example YAML configs in the extension's `configs/` directory alongside its
implementation. `config.py` defines the schema; `configs/` holds runnable settings.
Keep feature-specific helpers here too. Reuse host perception/update factories,
losses, and training infrastructure where their interfaces fit. Keep package
initializers free of eager trainer imports and registration side effects.

## Registration

Define registration functions in the extension's `registration.py`. Each function
receives the corresponding host registry and adds the extension's components.

```python
# nca/extensions/my_extension/registration.py
def register_models(registry):
    from .factory import create_my_model
    registry["my_extension"] = create_my_model


def register_evolvers(registry):
    from .evolver import MyEvolver
    registry["my_extension"] = lambda config: MyEvolver(config)


def register_trainers(registry):
    from .trainer import MyTrainer
    registry["my_extension"] = MyTrainer
```

Adapt `MyEvolver(config)` to your constructor. Omit hooks for components you
reuse. Choose unused keys: assigning an existing key replaces its implementation.

In each relevant host file, call the hook **after** defining its registry,
retaining existing entries and registration calls:

```python
# nca/core/models/model_factory.py, after MODEL_REGISTRY is defined
from nca.extensions.my_extension.registration import register_models
register_models(MODEL_REGISTRY)

# nca/training/evolve_factory.py, after EVOLVER_REGISTRY is defined
from nca.extensions.my_extension.registration import register_evolvers
register_evolvers(EVOLVER_REGISTRY)

# nca/training/trainer_factory.py, after TRAINER_REGISTRY is defined
from nca.extensions.my_extension.registration import register_trainers
register_trainers(TRAINER_REGISTRY)
```

These are edits in three separate files. Registration happens during factory
initialization because config validators consult the registries. Registering
only in a training script misses other consumers, including inference tools.

## Configuration schema

Declare a strict Pydantic model in the extension and add an optional section
under `EXTENSIONS` in the host config. Keep feature-specific fields together:

```python
# nca/extensions/my_extension/config.py
from pydantic import BaseModel, ConfigDict, Field


class MyExtensionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    LOSS_WEIGHT: float = Field(default=0.0, ge=0.0)


def validate_my_extension_config(config):
    # Example constraint: this trainer requires its own model and evolver.
    if config.TRAINING.TRAINER_TYPE == "my_extension":
        if config.MODEL.ARCHITECTURE != "my_extension":
            raise ValueError("my_extension trainer requires its model architecture")
        if config.TRAINING.EVOLVE_MODE != "my_extension":
            raise ValueError("my_extension trainer requires its evolver")
```

Import the schema and validator in `nca/utils/config.py`, and add a field to the
existing `ExtensionsConfig` class:

```python
from nca.extensions.my_extension.config import (
    MyExtensionConfig,
    validate_my_extension_config,
)


class ExtensionsConfig(StrictModel):
    MY_EXTENSION: MyExtensionConfig | None = None
    # Retain all existing host fields and validators here.
    ...
```

This is an editing sketch, not a replacement class. Implementation code can read
`config.EXTENSIONS.MY_EXTENSION`; if the section is optional, use an extension-owned
helper that returns `config.EXTENSIONS.MY_EXTENSION or MyExtensionConfig()`.
The host `Config` already provides `EXTENSIONS` with an empty default instance.

Add `validate_my_extension_config(self)` to the existing `Config.model_post_init`,
preserving its other validation and any superclass behavior. Validate only the
combinations your feature actually requires. Preserve the host defaults:
`MODEL.ARCHITECTURE: residual`, `TRAINING.EVOLVE_MODE: base`, and
`TRAINING.TRAINER_TYPE: null`.

Keep extension config modules independent of model/trainer modules and the host
`Config` class to avoid circular imports. For implementation modules, use deferred
annotations and a `TYPE_CHECKING` import when `Config` is needed only as a type.

Dataset-derived dimensions may be unavailable during YAML validation. The training
entry point passes `cond_dim`, `img_height`, and `img_width` to the model factory,
then records them in `Config`. Validate those values during construction or
trainer initialization as appropriate.

## Runtime interfaces

### Model

The model registry expects a callable with this signature:

```python
def create_my_model(config, cond_dim, img_height, img_width):
    # Construct and return a torch.nn.Module on config.DEVICE.
    ...
```

If reusing `BaseEvolver`, support `model(state, conds, freeze_channels=...)` and
return `(next_state, delta)`. A custom evolver may define a different step interface
with its model. Document which residual features apply: custom architectures do
not automatically use the residual firing, noise, living-mask, or clamp pipeline.

### Evolver

Subclass [`Evolver`](../nca/training/evolve.py) and preserve its `forward` arguments:
`ca_model`, `state_in`, `conds`, `iter_n`, `logger`, `freeze_channels`, `logging`,
`step_observers`, and `return_rollout`.

Return a state tensor normally, or `(final_state, RolloutOutput)` when
`return_rollout=True`. Implement the requested observer/logging behavior or
explicitly reject unsupported combinations. The host's `BaseEvolver` provides a
reference implementation of the standard rollout interface.

### Trainer

Subclass [`BaseTrainer`](../nca/training/trainers/base_trainer.py) and implement:

- `_initialize_additional_components()` for feature-specific setup.
- `_compute_losses(initial_state, cond, target, logging=False)`, returning
  `(prediction_image, final_state, loss_dict)`, with a differentiable
  `loss_dict["total_loss"]` for training.

Use `self._to_device(...)`, `self.forward(...)`, or `self._evolve(...)` as needed.
Preserve the constructor contract used by the trainer factory:
`MyTrainer(model, dataloader, config, config_path, loss_fn=..., use_latent=...)`.
Inheriting the base constructor satisfies this contract.

The base trainer provides the optimizer, scheduler, accumulation, optional AMP,
sample pool, logging, and model saving. Extra optimizers or trainable modules need
explicit handling, including any additional checkpoint state. See the
[custom trainer guide](custom_trainer_guide.md) for hook examples.

Set `USES_FRAMEWORK_LOSS = False` on trainers that construct their own loss in
`_initialize_additional_components()`. The factory then skips `create_loss_fn`;
other trainers inherit `True` from `BaseTrainer`.

Select a new trainer explicitly in YAML, or register its datasets in
`_DATASET_REQUIRED_TRAINER` in the trainer factory. That mapping controls both
automatic selection and rejection of incompatible explicit trainers. A registration
hook can receive both mappings and add entries to each when needed.

### Dataset, if needed

Reuse `(seed, condition, target)` batches. In pixel space, seeds normally have
shape `[B, MODEL.CHANNEL_N, H, W]` and targets share the spatial grid. Access dataset
metadata through `dataloader.get_dataset()` and preserve image conversion for
logging.

A new `DATASET_REGISTRY` constructor accepts `(config, train)` and returns
`(dataset, cond_dim, im_height, im_width)`. Register it directly or through another
extension hook called after that registry is defined. The host `create_dataset`
builds the loader and wrapper. Pass new config fields through the constructor;
adding fields to YAML alone does not change dataset behavior. See the
[custom dataset guide](custom_dataset_guide.md).

Document any additional dataset metadata the trainer requires, including channel
counts, layout conventions, and optional attributes. Validate those requirements
when initializing the trainer.

## Example configuration and usage

Create a complete `nca/extensions/my_extension/configs/my_extension_config.yaml`
from a compatible task configuration. When providing all three components, select:

```yaml
MODEL:
  ARCHITECTURE: my_extension
TRAINING:
  TRAINER_TYPE: my_extension
  EVOLVE_MODE: my_extension
EXTENSIONS:
  MY_EXTENSION:
    LOSS_WEIGHT: 0.1
```

This fragment only shows extension selection. Include the task's dataset, model,
optimizer, and logging settings in the full file. A trainer-only extension should
retain a compatible existing architecture and evolver.

After implementing and wiring the extension, run from the repository root:

```bash
uv run python scripts/train_ca.py \
  --config nca/extensions/my_extension/configs/my_extension_config.yaml \
  --device cpu
```

Use `--device cuda:0` when appropriate. Training registration does not itself add
custom UI controls or inference behavior; check those entry points if they are
part of the feature.

CLI overrides use the full section path, for example
`-o EXTENSIONS.MY_EXTENSION.LOSS_WEIGHT=0.25`.

## Verification

- Load the example YAML in a fresh process. Check that new fields retain their
  values and selectors resolve to the intended implementations.
- Check invalid combinations produce useful errors and an existing residual
  configuration still loads and runs.
- Run a small synthetic batch through the actual factories, rollout, loss, and
  optimizer step. Check finite losses and gradients.
- Save and reload the model, comparing outputs for fixed inputs and settings.
- Add focused checks for the extension's behavior, such as rollout scheduling,
  custom loss calculations, and gradient propagation.

Start with the simplest supported configuration. Test optional conditioning,
latent training, pools, AMP, and checkpointing before claiming support. Document
required dependencies and unsupported combinations in the extension's README.

## Extension documentation

Each extension's README should describe its purpose, registered component keys,
configuration fields, required dependencies, supported combinations, and training
command. Keep runnable YAML examples in its `configs/` directory and document any
additional dataset or checkpoint requirements.
