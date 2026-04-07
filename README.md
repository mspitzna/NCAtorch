<div align="center">
  <img src="figures/nca_torch_logo.png" alt="NCA-torch Logo" width="400"/>
  <p>
    <em>A comprehensive PyTorch-based framework for Neural Cellular Automata research and applications</em>
  </p>


  <!-- <p>
    📄 Paper (Coming Soon) &nbsp&nbsp | &nbsp&nbsp 📚 <a href="#-documentation">Documentation</a> &nbsp&nbsp | &nbsp&nbsp 🚀 <a href="#-quick-start">Quick Start</a> &nbsp&nbsp | &nbsp&nbsp 🎨 <a href="#-interactive-visualization">Demo</a>
  </p> -->

  ---
</div>

## 🌟 Highlights

**NCAtorch** is an open-source, modular research framework that combines classical Cellular Automata concepts with learnable neural networks. This implementation provides a unified codebase for training, evaluating, and visualizing Neural Cellular Automata across diverse tasks.

Key features:

- 🎯 **Modular Architecture**: Composable perception and update modules for flexible experimentation
- 🎨 **Diverse Tasks**: Image generation (emoji, handbags), texture synthesis, self-classifying NCAs, video prediction
- 🖼️ **Latent Space NCAs**: High-resolution generation (512x512) via pre-trained autoencoders
- 🎮 **Interactive Visualization**: Real-time FastAPI-based web interface with painting tools
- 📊 **Experiment Tracking**: Integrated [Weights & Biases](https://wandb.ai/site/) logging
- ⚙️ **YAML Configuration**: Pydantic-validated configuration system

## 📑 What's New

- **[TBA, 2025]** 🎉 Initial release of NCA-torch framework

## Community Works

If your work has improved **NCAtorch** and you would like more people to see it, please inform us!


## 🚀 Quick Start

### Installation

#### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-capable GPU (recommended)

#### Setup

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone <repo-url>
cd NCAtorch
```

3. Install all dependencies (creates `.venv` automatically, pulls the correct PyTorch CUDA build):
```bash
uv sync --dev
```

### Training Your First Model

Train an NCA model using a configuration file:

```bash
ncatorch-train --config config/emoji_config.yaml
```

For latent space NCA (requires training an autoencoder first):
```bash
# Step 1 — train the autoencoder (checkpoint saved to train_log/<run_folder>/ae_checkpoints/)
ncatorch-train-ae --config config/your_config.yaml

# Step 2 — train the CA, pointing --folder to the AE training log
ncatorch-train --folder train_log/<run_folder>
```

💡 **Tip**: Start with the emoji generation task for quick results and visual feedback!

## 🎨 Interactive Visualization

Launch the web interface to interact with trained models:

```bash
ncatorch-ui
```

Then open your browser to `http://localhost:8000`

### Options

```bash
# Force a specific device
ncatorch-ui --device cuda:0
ncatorch-ui --device cpu

# Custom host/port
ncatorch-ui --host 0.0.0.0 --port 8080

# Auto-reload on code changes (development)
ncatorch-ui --reload
```

### 📹 Demo Video

<div align="center">
  <a href="https://youtu.be/TWF4HYgWQwY">
    <img src="https://img.youtube.com/vi/TWF4HYgWQwY/maxresdefault.jpg" alt="NCAtorch Interactive Demo" width="700">
  </a>
  <p><em>Click to watch the toolkit demo video</em></p>
</div>

## ⚙️ Configuration

Models and training are configured via YAML files. Each perception entry declares its own `OUT_CHANNEL` to set the number of filters emitted from that branch. Here's a basic example:

```yaml
PROJECT_NAME: "your_project"        # High-level grouping for experiment tracking
TRAIN_NAME: "your_training_0"       # Unique name for this training run

MODEL:
  NAME: "MLP"                      # Select the MLP architecture for the update module
  HIDDEN_CHANNELS: [64, 128]       # Hidden units per layer in the update module
  CHANNEL_N: 16                    # Number of state channels per cell
  PERCEPTIONS:
    - MODE: "conv"                 # Standard convolutional neighborhood perception
      KERNEL_SIZE: 5
      OUT_CHANNEL: 48              # Filters for this perception branch
    - MODE: "residual_conv"        # Residual convolutional neighborhood perception
      OUT_CHANNEL: 32              # Filters for this perception branch

TRAINING:
  BATCH_SIZE: 18                   # Cell grids processed per optimization step
  LEARNING_RATE: 0.0005            # learning rate (see config/training)
  STEPS: 50000                     # Total optimization steps (in batches)
  LOSS_FN: "mse"                   # Target reconstruction loss
  ITER_N_MIN: 20                   # Minimum rollout iterations per batch
  ITER_N_MAX: 26                   # Maximum rollout iterations per batch
  INTERMEDIATE_LOGGING_STEPS: [5, 10, 15]  # Intermediate logging states for visualization

DATASET:
  NAME: "emoji"                    # Use built-in emoji dataset
  TARGET_SIZE: 64                  # Resolution for targets and predictions
  TARGET_PADDING: 16               # Padding arround target emoji
  EMOJIS:
    - "😭"
    - "🔥"
```

💡 **See [config] directory for complete examples of all supported tasks.**

## 🎯 Supported Tasks

### 🖼️ Image Generation
- **Emoji Generation**: Generate emoji from Unicode characters
- **Edge-to-Handbag (E2H)**: Conditional generation from edge maps

### 🎨 Texture Synthesis
- **Organizing Textures**: DTD texture synthesis with style loss

### 🔢 Classification
- **MNIST**: Self-classifying digit recognition
- **CIFAR-10**: Multi-class image classification

### 🎬 Video Prediction
- **Moving MNIST**: Temporal dynamics and video prediction

### 🖼️ High-Resolution Generation
- **Latent Space NCAs**: 512x512 generation via pre-trained autoencoders

## 📂 Project Structure

```
nca-torch/
├── nca/                      # Core library
│   ├── core/
│   │   ├── models/          # NCA models, autoencoders, critics
│   │   └── losses/          # Loss functions
│   ├── data/
│   │   └── datasets/        # Dataset implementations
│   ├── training/
│   │   └── trainers/        # Training logic
│   └── utils/               # Utilities and visualization
├── app/                      # FastAPI web application
│   ├── fastapi_backend.py   # Server entry point
│   ├── templates/           # HTML templates
│   └── scripts/             # Frontend JavaScript
├── train_scripts/            # Training entry points
├── config/                   # YAML configuration files
└── datasets/                 # Dataset storage
└── train_log/                # Training logs
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Custom Perception](docs/custom_perception_guide.md) | Add a new neighborhood operator |
| [Custom Update Module](docs/custom_update_module_guide.md) | Add a new update architecture |
| [Custom Dataset](docs/custom_dataset_guide.md) | Add a new dataset and wire it into the training pipeline |
| [Custom Trainer](docs/custom_trainer_guide.md) | Add a new training loop by implementing two methods and registering one entry |

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@software{ncatorch2025,
  title={TBA},
  author={TBA},
  year={TBA},
  url={TBA}
}
```

## 👥 Authors

TBA

## 📄 License

TBA

## Contact Us

TBA
