# Spatio-Temporal Remote Sensing Image Synthesis

## Introduction
This repository contains a PyTorch-based pipeline for training, testing, and evaluating image generation models, specifically using GANs (Generative Adversarial Networks) and UNets. The models are designed to generate and process high-resolution satellite images. The repository includes various utilities, configurations, and metrics necessary for model evaluation and checkpoint management.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Utilities](#utilities)
- [Configuration](#configuration)
- [Models](#models)
- [Metrics](#metrics)
- [License](#license)

## Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/ShahmirAltamash/Spatio-Temporal-Remote_Sensing_Image_Synthesis.git
   cd Spatio-Temporal-Remote_Sensing_Image_Synthesis
   \`\`\`

2. Install the required dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

### Training

To train the models, use the \`train.py\` script:

   \`\`\`bash
   python train.py
   \`\`\`

This script initializes the models, loads the dataset, and starts the training process. It also saves the model checkpoints and logs the training progress.

### Testing

To test the models, use the \`test.py\` script:

   \`\`\`bash
   python test.py
   \`\`\`

This script loads the trained models and evaluates them on the test dataset, computing various metrics and optionally saving generated images.

## Utilities
Utility functions are provided in \`utils.py\`, which include functions for saving/loading checkpoints, saving model outputs, and more.

## Configuration
All the configurations for the training and testing process are located in the \`config.py\` file. Here are some key configurations:

- \`DEVICE\`: Device to run the computations (\`cuda\` or \`cpu\`).
- \`LEARNING_RATE\`: Learning rate for the optimizer.
- \`BATCH_SIZE\`: Batch size for data loaders.
- \`NUM_EPOCHS\`: Number of training epochs.
- \`LOAD_MODEL\`: Flag to load a pre-trained model.
- \`SAVE_MODEL\`: Flag to save the model during training.

## Models
### Discriminator
The Discriminator model is defined in \`models.py\` and consists of convolutional layers to classify real and generated images.

### Generator
The Generator model is also defined in \`models.py\` and is responsible for generating images from input noise.

### UNet
The UNet model is used for segmentation tasks and is defined in \`models.py\`.

### Attention UNet
An enhanced version of the UNet with attention mechanisms to focus on important regions of the image.

## Metrics
Metrics for model evaluation are implemented in \`metrics.py\`. These include:

- LPIPS: Learned Perceptual Image Patch Similarity
- SSIM: Structural Similarity Index
- PSNR: Peak Signal-to-Noise Ratio
- F1 Score

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
"""
