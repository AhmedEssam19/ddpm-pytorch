import typer
from typing import Optional
from torchvision.transforms import ToTensor, Compose, Lambda, RandomHorizontalFlip
from torch.utils.data import DataLoader
from dataset import CelebADataset, CIFAR10Dataset
from diffusion_utils import DiffusionUtils
from pl_utils import PLModel, ImageGenerationCallback
from lightning.pytorch.loggers import TensorBoardLogger
from config import Config
import lightning as L


def main(
    config_path: str,
    continue_training: bool = False,
    checkpoint_path: Optional[str] = None
):
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Setup paths
    config.training.results_folder.mkdir(exist_ok=True)
    
    # Setup transforms
    transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda x: (x * 2) - 1)
    ])

    # Setup dataset based on config
    if config.dataset.name.lower() == 'celeba':
        train_dataset = CelebADataset(timesteps=config.model.timesteps, transform=transform, train=True)
        val_dataset = CelebADataset(timesteps=config.model.timesteps, transform=transform, train=False)
    elif config.dataset.name.lower() == 'cifar10':
        train_dataset = CIFAR10Dataset(transform=transform, train=True)
        val_dataset = CIFAR10Dataset(transform=transform, train=False)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")

    # Setup data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        num_workers=config.training.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.training.num_workers
    ) if val_dataset else None

    # Setup model and training
    diffusion_utils = DiffusionUtils(config.model.timesteps)
    model = PLModel(
        image_size=config.model.image_size,
        first_layer_channels=config.model.first_layer_channels,
        channels_multiplier=config.model.channels_multiplier,
        num_res_blocks=config.model.num_res_blocks,
        attn_resolutions=config.model.attn_resolutions,
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        diffusion_utils=diffusion_utils
    )

    # Setup logging and callbacks
    logger = TensorBoardLogger(save_dir=str(config.training.results_folder))
    image_generation_callback = ImageGenerationCallback(
        config.training.view_sample_size,
        config.model.image_size,
        config.training.log_interval,
        diffusion_utils
    )

    # Setup trainer
    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator=config.training.accelerator,
        devices=config.training.num_gpus,
        default_root_dir=str(config.training.results_folder),
        log_every_n_steps=config.training.log_interval,
        logger=logger,
        callbacks=[image_generation_callback],
        gradient_clip_val=config.training.gradient_clip_val,
    )

    # Train
    trainer.fit(
        model, 
        train_dataloader, 
        val_dataloader, 
        ckpt_path=checkpoint_path if continue_training else None
    )


if __name__ == "__main__":
    typer.run(main) 