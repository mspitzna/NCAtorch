# NCA-Torch Framework - Claude Context

## Project Overview
This is a modular PyTorch framework for Neural Cellular Automata (NCA) research. The codebase emphasizes flexibility, allowing researchers to experiment with different perception mechanisms, update rules, and training strategies.

## Core Architecture

### Cellular Automata Model (`nca/models/cellular_automata/CAModel.py`)
The main CA model follows this flow:
1. **Input Preparation**: Concatenate state with optional conditioning vectors/images
2. **Perception**: Extract local features (gradients, learned filters, attention)
3. **Update Model**: Compute state delta (dx) from perception features
4. **State Update**: Apply dx with fire rate masking, living mask, and noise injection

Key parameters:
- `channel_n`: Number of state channels (typically 16)
- `channel_out`: Output channels after perception
- `fire_rate`: Probability of cell update each step (0.0-1.0)
- `living_mask`: Zero out "dead" cells based on alpha channel threshold
- `use_positional_embeddings`: Add learnable (x,y) coordinates to perception input

### Perception Modules (`nca/models/cellular_automata/perceptions.py`)
All perception classes inherit from `Perception` base class and must implement:
- `forward(x)`: Process input state
- `get_out_channel()`: Return output channel count

**Available Perceptions:**
- `SobelPerception`: Fixed filters (identity, sobel_x, sobel_y, laplacian) with circular padding
- `ConvPerception`: Learnable 3x3 conv with LeakyReLU, circular padding
- `DeformableConvPerception`: Learnable spatial offsets for adaptive receptive fields
- `ResidualConvPerception`: Two conv layers with skip connection
- `AttentionPerception`: Local attention over kernel_size × kernel_size neighborhood
- `MultiHeadAttentionPerception`: Full transformer-style with multi-head attention, LayerNorm, optional FFN
- `MultiPerception`: Concatenate outputs from multiple perception modules

**Important**: Perceptions use `padding_mode="circular"` for toroidal topology (wraparound boundaries).

### Update Models (`nca/models/cellular_automata/update_models.py`)
All update models inherit from `UpdateModelBase(in_channels, out_channels)`.

**Available Update Models:**
- `SimpleMLPUpdate`: Stack of 1×1 convs (channel mixing only, no spatial mixing)
- `ResNetUpdate`: Initial conv → residual blocks → final conv (more stable for deep networks)
- `PathInvertibleUpdate`: Reversible updates using coupling layers (requires even state channels)
  - Only updates first half of state channels
  - Requires `current_state` parameter in forward pass

**Weight Initialization**: CAModel automatically zeros the last conv layer in update models for stable training start.

## Training System

### Trainers (`nca/utils/trainers/`)
- `BaseTrainer`: Core training loop with pattern pool, gradient checkpointing, learning rate schedules
- `ImageGenTrainer`: Image generation tasks (growing patterns, texture synthesis)
- `ClassificationTrainer`: Self-classifying NCAs (e.g., MNIST digit recognition)
- `AdversarialTrainer`: GAN-style training with critic network

### Configuration System (`nca/utils/config_handler.py`)
YAML-based configuration with Pydantic validation. Key sections:
- `MODEL`: Architecture params (perception type, channels, living mask, etc.)
- `TRAINING`: Optimizer, learning rate, batch size, iteration ranges
- `DATASET`: Dataset name, preprocessing, augmentation
- `PATTERN_POOL`: Enable/disable pool, size, damage/mutation ratios

### Pattern Pool
A replay buffer that stores intermediate CA states during training. Benefits:
- **Robustness**: Train on partially-grown/damaged patterns
- **Diversity**: Prevents overfitting to seed patterns
- **Damage**: Randomly zero out regions to learn self-repair
- **Mutation**: Randomly perturb states to explore variations

Typical settings:
```yaml
PATTERN_POOL:
  ENABLED: true
  POOL_SIZE: 1024
  POOL_DMG_RATIO: 0.5    # 50% of samples get damage
  POOL_MUTATION_RATIO: 0.5
```

## Dataset Handling (`nca/utils/dataset_handling/`)
All datasets return `(image, label, condition_vector)` tuples.

**Key Datasets:**
- `emoji_dataset`: Render emoji as PIL images
- `growing_mnist_dataset`: MNIST with temporal growing targets
- `moving_mnist_dataset`: Video sequences for temporal prediction
- `organizing_textures_dataset`: Texture synthesis benchmarks
- `cfg_dataset_wrapper`: Classifier-free guidance wrapper (randomly drops conditioning)

**Dataloader Factory** (`dataloader_handler.py`): Maps dataset names to instantiation logic.

## Model Factory (`nca/models/model_factory.py`)
Central factory for creating CA models from config. Handles:
- Perception instantiation based on `PERCEPTIONS` list in config
- Update model selection (`SimpleMLPUpdate`, `ResNetUpdate`, etc.)
- Channel computation (perception out → update in)
- Conditional models (VAE, AE wrappers)

## Web Interface (`fastapi_backend.py`, `app/`)
FastAPI server for real-time CA interaction:
- WebSocket for bidirectional state updates
- Load trained models from `train_log/` checkpoints
- Interactive painting/erasing tools
- Real-time inference and visualization

**Key Files:**
- `connection_manager.py`: WebSocket connection handling
- `ca_handler.py`: CA step execution and state management
- `state_handler.py`: Coordinate transformation between canvas and CA grid
- `tools/brushes.py`: Paint/delete operations on CA state

## Code Conventions

### File Organization
```
nca/
├── models/           # Neural network architectures
├── losses/           # Loss functions (MSE, LPIPS, overflow penalty)
├── utils/
│   ├── trainers/     # Training loops
│   ├── dataset_handling/  # Datasets and dataloaders
│   ├── config_handler.py  # YAML config loading
│   ├── train_utils.py     # Helper functions (seed, optimizer creation)
│   └── vutils.py          # Visualization utilities
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `CAModel`, `SobelPerception`)
- **Functions/Methods**: snake_case (e.g., `get_living_mask`, `_prepare_input`)
- **Config Keys**: UPPER_SNAKE_CASE (e.g., `CHANNEL_N`, `FIRE_RATE`)
- **Private Methods**: Prefix with `_` (e.g., `_build_perception`)

### Tensor Shapes
Standard shape notation: `[B, C, H, W]` where:
- `B` = batch size
- `C` = channels
- `H` = height
- `W` = width

## Common Workflows

### Adding a New Perception
1. Create class in `perceptions.py` inheriting from `Perception`
2. Implement `forward(x)` and `get_out_channel()`
3. Add instantiation logic to `model_factory.py`
4. Update config schema if new parameters needed

### Adding a New Dataset
1. Create dataset class in `nca/utils/dataset_handling/`
2. Implement `__getitem__` returning `(image, label, condition)`
3. Add to `create_dataset()` in `util_factorys.py`
4. Add config option in YAML schema

### Training a New Model
1. Create YAML config in `config/`
2. Run: `python scripts/train_ca.py --config config/your_config.yaml`
3. Monitor: W&B dashboard (if enabled) or console logs
4. Checkpoints saved to `train_log/<project>/<run>/`

## Important Implementation Details

### Circular Padding
Most operations use `padding_mode="circular"` to create toroidal topology. This prevents edge artifacts and allows patterns to wrap around borders.

### Living Mask
When enabled, cells are considered "dead" if their alpha channel < 0.1. Dead cells are zeroed after each update. This enables:
- Growing patterns from seeds
- Self-organizing boundaries
- Stable pattern persistence

### Fire Rate
Stochastic update masking. Lower fire rates (0.5) encourage:
- Asynchronous updates (more biologically plausible)
- Emergent collective behavior
- Robustness to partial updates

### Gradient Checkpointing
When enabled, splits temporal rollout into segments and recomputes activations during backward pass. Essential for:
- Long rollouts (64+ steps)
- Memory-constrained GPUs
- Large batch sizes

### Iteration Range Training
Training uses random rollout lengths between `ITER_N_MIN` and `ITER_N_MAX`. This prevents:
- Overfitting to specific timesteps
- Mode collapse to static solutions
- Instability at different rollout lengths

## Debugging Tips

### Common Issues
1. **NaN/Inf Loss**: Enable overflow loss penalty, reduce learning rate, check gradient clipping
2. **Static Patterns**: Increase fire rate, reduce update magnitude, check perception gradients
3. **Unstable Growth**: Enable living mask, reduce learning rate, increase pattern pool damage
4. **Memory Issues**: Enable gradient checkpointing, reduce batch size, reduce `ITER_N_MAX`

### Useful Visualization
- `vutils.py`: `to_rgb()`, `to_rgba()`, `make_seed()` for visualizing CA states
- W&B logs: Intermediate steps, final outputs, gradient norms, loss curves

### Testing
Run pytest suite: `pytest nca/` (if tests exist)

## Research Extensions

### Current Capabilities
- Multi-modal conditioning (class labels, images, text embeddings)
- Adversarial training with critic networks
- Path-invertible architectures for reversible dynamics
- Attention-based perception for non-local interactions
- Autoencoder wrappers for latent-space NCAs

### Potential Extensions
- 3D cellular automata (volumetric data)
- Video prediction (temporal conditioning)
- Multi-agent systems (heterogeneous cell types)
- Physics-informed losses (conservation laws)
- Hierarchical NCAs (multi-scale grids)

## Key Papers & References
- Growing Neural Cellular Automata (Mordvintsev et al., 2020)
- Self-Organising Textures (Niklasson et al., 2021)
- Self-classifying MNIST Digits (Randazzo et al., 2020)

## When Working on This Codebase

### Before Making Changes
1. Check existing perception/update implementations for similar functionality
2. Verify config schema supports new parameters
3. Consider backward compatibility with existing checkpoints

### Testing Changes
1. Test with small toy config (8×8 grid, 100 steps)
2. Verify gradient flow (print grad norms)
3. Check memory usage (use gradient checkpointing if needed)
4. Test with pattern pool enabled/disabled

### Code Quality
- Use type hints where possible
- Add docstrings for new modules/classes
- Follow existing naming conventions
- Keep backward compatibility with existing configs

This codebase prioritizes **flexibility** and **modularity** over performance optimization. Prefer clear, extensible code over micro-optimizations.
