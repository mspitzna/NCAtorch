<div align="center">
  <h1>Custom Trainer Guide</h1>
  <p><em>Extend NCAtorch with custom training loops that plug straight into the pipeline.</em></p>
</div>

---

## Overview

A trainer owns the training loop for one experiment. It receives the CA model, dataloader, and config, and is responsible for the forward pass, loss computation, backward pass, logging, and checkpointing.

There are only two steps: implement the trainer and add one entry to the registry. The config validator updates itself automatically.

## Understand `BaseTrainer`

All trainers inherit from [`BaseTrainer`](../nca/training/trainers/base_trainer.py). It handles everything that is shared across training setups:

| Provided by `BaseTrainer` | You implement |
|---------------------------|---------------|
| Optimizer and LR scheduler | `_initialize_additional_components()` |
| Gradient accumulation | `_compute_losses()` |
| Mixed-precision scaler (`self.scaler`) | *(optional)* `_on_step_end()` |
| NaN detection and backward pass | *(optional)* `_on_train_end()` |
| Sample pool | |
| Latent encode/decode via `self.latent_wrapper` | |
| Logging and checkpointing | |
| `forward()` — runs `_evolve()` then optional decode | |

The two **abstract methods** you must implement:

### `_initialize_additional_components()`

Called once at the end of `__init__`. Use it to set up anything specific to your trainer — extra models, optimizers, loss functions, etc.

### `_compute_losses(initial_state, cond, target, logging=False)`

Called every step by `_run_train_step`. Responsible for:
1. Moving tensors to device (use `self._to_device(...)`)
2. Forward pass (call `self.forward(initial_state, cond, target)`)
3. Loss computation
4. Returning `(prediction_image, final_state, loss_dict)`

`BaseTrainer` takes care of the rest: gradient accumulation scaling, AMP backward pass, NaN check, metric logging, and returning detached tensors for the pool.

```python
def _compute_losses(self, initial_state, cond, target, logging=False):
    initial_state, cond, target = self._to_device(initial_state, cond, target)
    prediction_image, final_state = self.forward(
        initial_state, cond, target, logging=logging
    )
    loss_dict = self.loss_fn(prediction_image, target)
    return prediction_image, final_state, loss_dict
```

`loss_dict` must contain a `"total_loss"` key. Any additional keys are logged automatically.

### Optional rollout metadata and step observers

Most trainers only need the final CA state, so the normal `forward()` call returns two values:

```python
prediction_image, final_state = self.forward(
    initial_state, cond, target, logging=logging
)
```

If a trainer needs information from every CA step, pass `return_rollout=True`. This returns a third value, `rollout`, without changing the default API for existing trainers:

```python
prediction_image, final_state, rollout = self.forward(
    initial_state,
    cond,
    target,
    logging=logging,
    return_rollout=True,
)
```

`rollout` is a `RolloutOutput` object. It contains the final state plus any losses, metrics, or states collected by optional step observers.

A step observer is a callable that receives a `StepContext` once per rollout step. It can return `None`, or a `StepObserverOutput` with named losses and metrics:

```python
from nca.training.evolve import StepObserverOutput


def smoothness_observer(ctx):
    step_delta = ctx.next_state - ctx.previous_state
    return StepObserverOutput(
        losses={"step_smoothness": step_delta.square().mean()},
        metrics={"mean_state": ctx.next_state.mean()},
    )


def _compute_losses(self, initial_state, cond, target, logging=False):
    initial_state, cond, target = self._to_device(initial_state, cond, target)

    prediction_image, final_state, rollout = self.forward(
        initial_state,
        cond,
        target,
        logging=logging,
        return_rollout=True,
        step_observers=[smoothness_observer],
    )

    loss_dict = self.loss_fn(prediction_image, target)
    loss_dict["smoothness_loss"] = rollout.losses["step_smoothness"]
    loss_dict["total_loss"] = (
        loss_dict["total_loss"] + 0.1 * loss_dict["smoothness_loss"]
    )
    return prediction_image, final_state, loss_dict
```

Observer losses and metrics are aggregated across rollout steps by mean and exposed as `rollout.losses` and `rollout.metrics`. They are not logged automatically; add anything you want to track to your trainer's `loss_dict`.

Step observers currently require `TRAINING.GRADIENT_CHECKPOINTING: false`. If observers are used with gradient checkpointing enabled, training raises a clear error.

### Optional hooks

Override `_run_train_step` only if you need multi-optimizer logic (e.g. a GAN with a separate discriminator update). See [`AdversarialTrainer`](../nca/training/trainers/adversarial_trainer.py) for a reference.

Override these if your trainer needs per-step or end-of-training logic:

```python
def _on_step_end(self, step: int):
    """Called at the end of every training step."""
    pass

def _on_train_end(self):
    """Called once after training completes."""
    pass
```


---

## Add your custom Trainer

### Step 1 – Implement your trainer

Create a new file in [`nca/training/trainers/`](../nca/training/trainers/):

```python
# nca/training/trainers/my_custom_trainer.py
import torch
from nca.training.trainers.base_trainer import BaseTrainer
from nca.core.losses.loss_factory import create_metric


class MyCustomTrainer(BaseTrainer):

    def _initialize_additional_components(self):
        # Set up your loss function, extra models, or any trainer-specific state here.

    def _compute_losses(self, initial_state, cond, target, logging=False):
        initial_state, cond, target = self._to_device(initial_state, cond, target)

        # Some *fancy_stuff* that needs a new trainer class

        prediction_image, final_state = self.forward(
            initial_state, cond, target, logging=logging
        )
        loss_dict = self.loss_fn(prediction_image, target)
        return prediction_image, final_state, loss_dict
```

### Step 2 – Add one entry to the registry ([nca/training/trainer_factory.py](../nca/training/trainer_factory.py))

```python
from nca.training.trainers.my_custom_trainer import MyCustomTrainer

TRAINER_REGISTRY = {
    # ... existing entries ...
    "my_custom": MyCustomTrainer,
}
```

The `TRAINING.TRAINER_TYPE` config validator imports `TRAINER_REGISTRY` at runtime, so `"my_custom"` becomes a valid value immediately.

If your trainer only makes sense for specific datasets, declare that constraint too:

```python
_DATASET_REQUIRED_TRAINER = {
    "mnist":     "classification",
    "cifar10":   "classification",
    "my_dataset": "my_custom",   # add this
}
```

## Step 3 – Use it in a config YAML

```yaml
TRAINING:
  TRAINER_TYPE: "my_custom"
  LOSS_FN: "mse"
  STEPS: 10000
```

Leave `TRAINER_TYPE` unset (or `null`) for automatic selection based on dataset and `ADVERSARIAL.ENABLED`.
