import typer

from torchvision.transforms import ToTensor, Compose, Lambda, ToPILImage, RandomHorizontalFlip
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import CIFAR10Dataset
from diffusion_utils import DiffusionUtils

from pl_utils import PLModel, ImageGenerationCallback
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import lightning as L


def main(
    image_size: int = 32,
    timesteps: int = 1000,
    batch_size: int = 128,
    view_sample_size: int = 10,
    num_workers: int = 8,
    max_steps: int = 800000,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5000,
    dropout: float = 0.1,
    first_layer_channels: int = 128,
    channels_multiplier: list[int] = [1, 2, 2, 2],
    num_res_blocks: int = 2,
    attn_resolutions: list[int] = [16],
    log_interval: int = 100,
    results_folder: str = "./results",
    accelerator: str = "gpu",
    num_gpus: int = 1,
    continue_training: bool = False,
    checkpoint_path: str = None
):
    results_folder = Path(results_folder)
    results_folder.mkdir(exist_ok=True)
    
    transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda x: (x * 2) - 1)
    ])

    train_dataset = CIFAR10Dataset(transform=transform, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = CIFAR10Dataset(transform=transform, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    diffusion_utils = DiffusionUtils(timesteps)
    model = PLModel(image_size, first_layer_channels, channels_multiplier, num_res_blocks, attn_resolutions, dropout, learning_rate, warmup_steps, diffusion_utils)
    logger = TensorBoardLogger(save_dir=str(results_folder))
    image_generation_callback = ImageGenerationCallback(view_sample_size, image_size, log_interval, diffusion_utils)
    trainer = L.Trainer(
        max_steps=max_steps, 
        accelerator=accelerator, 
        devices=num_gpus, 
        default_root_dir=str(results_folder), 
        log_every_n_steps=log_interval, 
        logger=logger, 
        callbacks=[image_generation_callback],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=checkpoint_path if continue_training else None)


if __name__ == "__main__":
    typer.run(main)
