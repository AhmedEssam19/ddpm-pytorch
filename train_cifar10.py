import torch

from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Lambda, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from pathlib import Path
from dataset import CIFAR10Dataset
from diffusion_utils import DiffusionUtils
from model import UNet
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
import lightning as L


IMAGE_SIZE = 32
TIMESTEPS = 1000
BATCH_SIZE = 128
VIEW_SAMPLE_SIZE = 10
NUM_WORKERS = 8
MAX_STEPS = 800000
LEARNING_RATE = 2e-4
FIRST_LAYER_CHANNELS = 128
CHANNELS_MULTIPLIER = [1, 2, 2, 2]
NUM_RES_BLOCKS = 2
ATTN_RESOLUTIONS = [16]
LOG_INTERVAL = 100
RESULTS_FOLDER = Path("./results")
RESULTS_FOLDER.mkdir(exist_ok=True)
ACCELERATOR = "gpu"
NUM_GPUS = 1


def main():
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x * 2) - 1)
    ])


    reverse_transform = Compose([
        Lambda(lambda x: (x + 1) / 2),
        Lambda(lambda x: x.permute(1, 2, 0)),
        Lambda(lambda x: x * 255.),
        Lambda(lambda x: x.numpy().astype(np.uint8)),
        ToPILImage()
    ])

    dataset = CIFAR10Dataset(timesteps=TIMESTEPS, transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    diffusion_utils = DiffusionUtils(TIMESTEPS)
    model = PLModel(diffusion_utils)
    logger = TensorBoardLogger(save_dir=str(RESULTS_FOLDER))
    image_generation_callback = ImageGenerationCallback(VIEW_SAMPLE_SIZE, IMAGE_SIZE, LOG_INTERVAL, diffusion_utils)
    trainer = L.Trainer(max_steps=MAX_STEPS, accelerator=ACCELERATOR, devices=NUM_GPUS, default_root_dir=str(RESULTS_FOLDER), log_every_n_steps=LOG_INTERVAL, logger=logger, callbacks=[image_generation_callback])
    trainer.fit(model, dataloader)


class PLModel(L.LightningModule):
    def __init__(self, diffusion_utils: DiffusionUtils):
        super().__init__()
        self.model = UNet(IMAGE_SIZE, FIRST_LAYER_CHANNELS, CHANNELS_MULTIPLIER, NUM_RES_BLOCKS, ATTN_RESOLUTIONS)
        self.loss_fn = MSELoss()
        self.diffusion_utils = diffusion_utils

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x_0, t = batch
        t = t.squeeze(1)
        noise = torch.randn_like(x_0).to(x_0.device)
        x_t = self.diffusion_utils.q_sample(x_0, t, noise)
        pred = self(x_t, t)
        loss = self.loss_fn(pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=LEARNING_RATE)
    

class ImageGenerationCallback(Callback):
    def __init__(self, num_samples, image_size, every_n_steps, diffusion_utils: DiffusionUtils):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.every_n_steps = every_n_steps
        self.diffusion_utils = diffusion_utils

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: PLModel, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0:
            milestone = trainer.global_step // self.every_n_steps
            all_images = self.diffusion_utils.p_sample_loop(pl_module, (self.num_samples, 3, self.image_size, self.image_size))
            all_images = (all_images + 1) * 0.5
            img = make_grid(all_images, nrow = 1)
            logger: TensorBoardLogger = trainer.logger
            logger.experiment.add_image(f"generated_images_{trainer.global_step}", img, milestone)


if __name__ == "__main__":
    main()
