<div align="center">
  <img src="figures/nca_torch_logo.png" alt="NCA-torch Logo" width="400"/>

  <h1>NCAtorch</h1>

  <p>
    <em>A comprehensive PyTorch-based framework for Neural Cellular Automata research and applications</em>
  </p>

  [📚 Documentation](#documentation) | [🚀 Quick Start](#quick-start) | [💡 Examples](#examples) | [🎨 Demo](#interactive-visualization)

  ---
</div>

## 🌟 Highlights

**NCA-torch** is an open-source, modular research framework that combines classical Cellular Automata concepts with learnable neural networks. This implementation provides a unified codebase for training, evaluating, and visualizing Neural Cellular Automata across diverse tasks.

Key features:

- 🎯 **Modular Architecture**: Composable perception and update modules for flexible experimentation
- 🎨 **Diverse Tasks**: Image generation (emoji, handbags), texture synthesis, self-classifying NCAs, video prediction
- 🖼️ **Latent Space NCAs**: High-resolution generation (512x512) via pre-trained autoencoders
- 🎮 **Interactive Visualization**: Real-time FastAPI-based web interface with painting tools
- 📊 **Experiment Tracking**: Integrated [Weights & Biases](https://wandb.ai/site/) logging
- ⚙️ **YAML Configuration**: Pydantic-validated configuration system

## 📑 What's New

- **[2025-11]** 🎉 Initial release of NCA-torch framework

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

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (if not already installed):
Visit https://pytorch.org/get-started/locally/ for platform-specific instructions.

### Training Your First Model

Train an NCA model using a configuration file:

```bash
python train_scripts/train_ca.py --config config/emoji_config.yaml --device cuda
```

💡 **Tip**: Start with the emoji generation task for quick results and visual feedback!

### Interactive Visualization

Launch the web interface to interact with trained models:

```bash
uvicorn app.fastapi_backend:app --reload
```

Then open your browser to `http://localhost:8000`

## ⚙️ Configuration

Models and training are configured via YAML files. Here's a basic example:

```yaml
PROJECT_NAME: "your_project"
TRAIN_NAME: "your_training_0"

MODEL:
  NAME: "ConvCA"
  CHANNEL_N: 16
  HIDDEN_CHANNELS: [80]
  PERCEPTIONS:
    - MODE: "conv"
      KERNEL_SIZE: 3
    - MODE: "residual_conv"
      KERNEL_SIZE: 3

TRAINING:
  BATCH_SIZE: 18
  LEARNING_RATE: 0.0005
  STEPS: 50000
  LOSS_FN: "mse"
  ITER_N_MIN: 20
  ITER_N_MAX: 26
  INTERMEDIATE_LOGGING_STEPS: [5, 10, 15]

DATASET:
  NAME: "emoji"
  TARGET_SIZE: 64
  EMOJIS:
    - "😭"
    - "🔥"
```

💡 **See** [config/](config/) **directory for complete examples of all supported tasks.**

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

Please refer to the [docs/](docs/) directory (coming soon).

## 🔗 Related Projects

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Original work by Mordvintsev et al.
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) - Texture synthesis with NCAs
- [Self-classifying MNIST Digits](https://distill.pub/2020/selforg/mnist/) - Classification with NCAs
- TODO

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

TODO

## 📄 License

TODO

## Contact Us

TODO
