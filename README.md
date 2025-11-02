# NCAtorch

A comprehensive PyTorch-based framework for Neural Cellular Automata (NCA) research and applications.

## Overview

NCAtorch is a modular research framework that combines classical Cellular Automata concepts with learnable neural networks. This implementation provides a unified codebase for training, evaluating, and visualizing Neural Cellular Automata across diverse tasks including image generation, texture synthesis, classification, and video prediction.

## Features

- **Modular Architecture**: Composable perception and update modules for flexible experimentation
- **Multiple Perception Strategies**: Sobel filters, standard convolutions, deformable convolutions, residual connections, and attention mechanisms
- **Diverse Tasks**: Image generation (emoji, handbags), texture synthesis, self-classifying NCAs, video prediction
- **Latent Space NCAs**: High-resolution generation via pre-trained autoencoders
- **Interactive Visualization**: Real-time FastAPI-based web interface with painting tools
- **Robust Training**: Sample pooling with damage and mutation mechanisms
- **Experiment Tracking**: Integrated Weights & Biases logging
- **YAML Configuration**: Pydantic-validated configuration system

## Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nca-torch.git
cd nca-torch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (if not already installed):
https://pytorch.org/get-started/locally/

## Quick Start

### Training

Train an NCA model using a configuration file:

```bash
python train_scripts/train_ca.py --config config/emoji_config.yaml --device cuda
```

### Interactive Visualization

Launch the web interface to interact with trained models:

```bash
uvicorn app.fastapi_backend:app --reload
```

Then open your browser to `http://localhost:8000`

## Project Structure

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
```

## Configuration

Models and training are configured via YAML files. Example:

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

See [config/](config/) directory for complete examples.

## Supported Tasks

### Image Generation
- **Emoji Generation**: Generate emoji from Unicode characters
- **Edge-to-Handbag (E2H)**: Conditional generation from edge maps

### Texture Synthesis
- **Organizing Textures**: DTD texture synthesis with style loss

### Classification
- **MNIST**: Self-classifying digit recognition
- **CIFAR-10**: Multi-class image classification

### Video Prediction
- **Moving MNIST**: Temporal dynamics and video prediction

### High-Resolution Generation
- **Latent Space NCAs**: 512x512 generation via pre-trained autoencoders


## Related Projects

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) - Original work by Mordvintsev et al.
- [Self-Organising Textures](https://distill.pub/selforg/2021/textures/) - Texture synthesis with NCAs
- [Self-classifying MNIST Digits](https://distill.pub/2020/selforg/mnist/) - Classification with NCAs

## Authors

TODO
