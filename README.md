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

- Python 3.12 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

#### Setup

1. Clone the repository:
```bash
git clone https://github.com/mspitzna/nca-torch.git
cd nca-torch
```
2. (Recommended) Create and activate a virtual environment so the dependencies stay isolated:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install PyTorch (if not already installed) using the command generated for your system:
Visit https://pytorch.org/get-started/locally/ for platform-specific instructions.

4. Install the remaining project requirements with the environment active:
```bash
pip install -r requirements.txt
```

### Training Your First Model

Train an NCA model using a configuration file:

```bash
python train_scripts/train_ca.py --config config/emoji_config.yaml --device cuda
```

💡 **Tip**: Start with the emoji generation task for quick results and visual feedback!

## 🎨 Interactive Visualization

Launch the web interface to interact with trained models:

```bash
uvicorn app.fastapi_backend:app
```

Then open your browser to `http://localhost:8000`

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

For detailed documentation on:
- Implementing custom perception modules
- Loss function implementations
- Dataset preparation
- Advanced training techniques

Please refer to the [docs] directory (coming soon).

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
