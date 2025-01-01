# Denoising Diffusion Probabilistic Model (DDPM) Implementation

An unofficial PyTorch implementation of the paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) (Ho et al., 2020). This implementation is based on the [official TensorFlow implementation](https://github.com/hojonathanho/diffusion)

## Overview

This implementation includes:
- A complete DDPM training pipeline
- Support for CelebA-HQ and CIFAR-10 datasets
- Multi-GPU training support via PyTorch Lightning
- Configurable model architecture and training parameters
- TensorBoard logging for training metrics and image generation

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Dependencies listed in `environment.yml` or `requirements.txt`

## Cost

* Training on CelebA-HQ dataset (256x256 images) costs around $435 for 0.5M steps
* Training on CIFAR-10 dataset (32x32 images) costs around $70 for 0.8M steps

## Installation

1. Clone the repository:
```bash
$ git clone https://github.com/AhmedEssam19/ddpm-pytorch.git
$ cd ddpm-pytorch
```

2. Create and activate conda environment:
```bash
$ conda env create -f environment.yml
$ conda activate image-generation-finetuning
```

Alternatively, you can use pip:
```bash
$ pip install --no-cache-dir -r requirements.txt
```

## Project Structure

- `model.py`: Contains the U-Net architecture with attention mechanisms
- `diffusion_utils.py`: Implementation of the diffusion process utilities
- `train.py`: Training script with PyTorch Lightning
- `pl_utils.py`: Lightning model and callbacks
- `dataset.py`: Dataset implementations for CelebA-HQ and CIFAR-10
- `config.py`: Configuration management
- `sample.py`: Image generation script
- `configs/`: Configuration files for different datasets

## Training

1. Choose or modify a configuration file in the `configs/` directory. Two default configurations are provided:
   - `celeba.yml`: For CelebA-HQ dataset (256x256 images)
   - `cifar10.yml`: For CIFAR-10 dataset (32x32 images)

2. Start training:
```bash
$ python train.py configs/celeba.yml
```

To resume training from a checkpoint:
```bash
$ python train.py configs/celeba.yml --continue-training --checkpoint-path path/to/checkpoint.ckpt
```

## Generating Images

To generate images using a trained model:

```bash
$ python3 sample.py --help          
                                                                                                                                                                 
Usage: sample.py [OPTIONS] CHECKPOINT_PATH                                                                                                                      
                                                                                                                                                                 
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    checkpoint_path      TEXT  [default: None] [required]                                                                                                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --device            TEXT     [default: cuda]                                                                                                                  │
│ --num-images        INTEGER  [default: 8]                                                                                                                     │
│ --image-size        INTEGER  [default: 256]                                                                                                                   │
│ --timesteps         INTEGER  [default: 1000]                                                                                                                  │
│ --batch-size        INTEGER  [default: 8]                                                                                                                     │
│ --output-dir        TEXT     [default: samples]                                                                                                               │
│ --help                       Show this message and exit.                                                                                                      │

```

Generated images will be saved in the `samples/` directory.

## Model Architecture

The implementation uses a U-Net architecture with:
- Residual blocks with group normalization
- Self-attention layers at specified resolutions
- Time embedding through sinusoidal positional encoding
- Skip connections between encoder and decoder

## Training Process

The training follows the DDPM paper's approach to the tiniest details:

1. Forward diffusion process adds Gaussian noise gradually
2. Model learns to reverse the diffusion process
3. Uses linear noise schedule
4. Implements linear warmup for learning rate

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir lightning_logs
```

This will show:
- Training loss
- Generated samples during training
- Validation metrics

## Citation

If you use this implementation in your research, please cite the original DDPM paper:

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2006.11239},
  year={2020}
}
```

