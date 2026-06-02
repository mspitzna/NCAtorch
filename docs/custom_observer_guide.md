<div align="center">
  <h1>Custom Logging Observer Guide</h1>
  <p><em>Extend NCAtorch with diagnostic observers that hook into the CA rollout and log themselves.</em></p>
</div>

---

## Overview

A **logging observer** is a small, self-contained diagnostic. On logging steps it hooks into the CA rollout, collects whatever it needs (state, update delta `dx`, …), and logs it itself — to W&B and/or the console. Observers are declared in `LOGGING.OBSERVERS` and built through a registry, so a new diagnostic is three touchpoints: implement, register, reference in the config. The trainer and logger stay untouched.

Observers run **only on logging steps** (`LOGGING.LOG_INTERVAL`), so there is no overhead on normal steps. They are incompatible with `TRAINING.GRADIENT_CHECKPOINTING` (which hides individual rollout steps) and are disabled with a warning in that case.

### Lifecycle

The trainer drives each observer, on logging steps only:

1. `reset()` — clear per-rollout buffers
2. `observe(context)` — called once per CA iteration during the rollout (collection)
3. `log(logger, step)` — emit to W&B / console, then clear (the logging phase, inside `commit_logs`)

Because the base class is callable and matches the rollout's `StepObserver` protocol, an observer plugs straight into the evolver. It is a pure side channel: nothing it collects affects the training loss.

## Step 1 – Implement the observer ([nca/training/observers/](../nca/training/observers/))

Subclass [`LoggingObserver`](../nca/training/observers/base.py) and fill in three methods:

```python
from nca.training.observers.base import LoggingObserver


class MyObserver(LoggingObserver):
    def __init__(self):
        ...           # create empty buffers for what you'll collect

    def reset(self):
        ...           # empty those buffers (called before each logging rollout)

    def observe(self, context):
        ...           # one rollout step: read from `context`, append to buffers

    def log(self, logger, step):
        ...           # turn the buffers into a W&B payload, emit it, then reset()
```

- **`observe(context)`** is called once per CA iteration during the rollout. The [`StepContext`](../nca/training/evolve.py) gives you `previous_state` (the input to that iteration), `next_state`, `dx` (the update delta), `condition`, `step_index`, and `freeze_channels`. Pull what you need (detach it) and accumulate into your buffers — keep this cheap, it runs every step.
- **`log(logger, step)`** is called once at the end of the rollout, during the logging phase. Turn your buffers into a payload and send it with `logger.wandb_log(payload, step)` (a no-op when W&B is disabled) and/or `print(...)`. Guard on `logger.use_wandb` if the work is W&B-only, and call `reset()` at the end so buffers never carry over.
- **`reset()`** clears the buffers; the trainer also calls it before each logging rollout.

If your observer needs constructor arguments, add them to `__init__`; they are supplied from the config via `PARAMS` (see Step 3).

For a complete, working example see [`IterationStatsObserver`](../nca/training/observers/iteration_stats.py) (per-channel state stats logged as one interactive W&B panel per statistic).

**Picking a W&B representation.** Logging an interactive `wandb.Plotly` (or `wandb.Image`) under the same key every logging phase gives each panel a **media step-slider**, so you can inspect the values at step 500, 1000, … Native custom charts (`wandb.plot.line_series`) are interactive but show only the latest step — no slider — so prefer media objects when you want to scrub across training.

## Step 2 – Add one entry to the registry ([nca/training/observers/registry.py](../nca/training/observers/registry.py))

Import your class and add it to `LOGGING_OBSERVER_REGISTRY`, keyed by the `TYPE` you'll use in the config:

```python
from nca.training.observers.iteration_stats import IterationStatsObserver

LOGGING_OBSERVER_REGISTRY = {
    # ... existing entries ...
    "iteration_stats": IterationStatsObserver,
}
```

Also export it from [`nca/training/observers/__init__.py`](../nca/training/observers/__init__.py).

**That's it.** Two things happen automatically:

- The `ObserverConfig.TYPE` validator imports `LOGGING_OBSERVER_REGISTRY` at runtime, so `"iteration_stats"` becomes a valid value immediately — no manual list to maintain.
- `create_logging_observers(config)` instantiates each entry as `cls(**PARAMS)`, so a wrong `PARAMS` key surfaces as a clear constructor error.

## Step 3 – Reference it in a config YAML

```yaml
LOGGING:
  OBSERVERS:
    - TYPE: iteration_stats        # registry key
    # - TYPE: my_other_observer    # list more to run several at once
    #   PARAMS:                    # forwarded as keyword args to __init__
    #     some_arg: 0.5
```

`OBSERVERS` is a list; to disable one, remove or comment out its entry. Each entry collects independently during the same rollout and logs itself during the logging phase. Observers without constructor arguments need only the `TYPE` line.

---

With those touchpoints wired up, rerun training: on every `LOG_INTERVAL` step the observer collects across the CA iterations and emits its diagnostic, while normal steps stay untouched.
