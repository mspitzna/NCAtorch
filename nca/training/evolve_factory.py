from nca.training.evolve import BaseEvolver


# Registry of all available rollout strategies.
#
# Maps TRAINING.EVOLVE_MODE to a factory function with signature:
#   (config: Config) -> Evolver
#
# Adding a new evolver:
#   1. Implement the evolver in nca/training/evolve.py or a sibling module.
#   2. Add one entry here.
#   3. Set TRAINING.EVOLVE_MODE in the YAML config.
EVOLVER_REGISTRY = {
    "base": lambda config: BaseEvolver(
        gradient_checkpointing=config.TRAINING.GRADIENT_CHECKPOINTING,
        checkpoint_segments=config.TRAINING.GRADIENT_CHECKPOINT_SEGMENTS,
        intermediate_logging_steps=config.TRAINING.INTERMEDIATE_LOGGING_STEPS,
    ),
}


def create_evolver(config):
    mode = config.TRAINING.EVOLVE_MODE
    if mode not in EVOLVER_REGISTRY:
        raise ValueError(
            f"Invalid TRAINING.EVOLVE_MODE '{mode}'. "
            f"Valid options: {sorted(EVOLVER_REGISTRY)}"
        )
    print(f"Evolver: {mode}")
    return EVOLVER_REGISTRY[mode](config)
