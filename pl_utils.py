import torch

from torchvision.utils import make_grid
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

from model import UNet
from diffusion_utils import DiffusionUtils

import lightning as L


class PLModel(L.LightningModule):
    def __init__(self, image_size: int, first_layer_channels: int, channels_multiplier: list[int], num_res_blocks: int, attn_resolutions: list[int], dropout: float, learning_rate: float, warmup_steps: int, diffusion_utils: DiffusionUtils):
        super().__init__()
        self.model = UNet(image_size, first_layer_channels, channels_multiplier, num_res_blocks, attn_resolutions, dropout)
        self.loss_fn = MSELoss()
        self.diffusion_utils = diffusion_utils
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x_0 = batch
        t = torch.randint(0, self.diffusion_utils.timesteps, (x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0).to(x_0.device)
        x_t = self.diffusion_utils.q_sample(x_0, t, noise)
        pred = self(x_t, t)
        loss = self.loss_fn(pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_0 = batch
        for i in range(0, self.diffusion_utils.timesteps, self.diffusion_utils.timesteps // 10):
            t = torch.full((x_0.shape[0],), i, device=x_0.device)
            noise = torch.randn_like(x_0).to(x_0.device)
            x_t = self.diffusion_utils.q_sample(x_0, t, noise)
            pred = self(x_t, t)
            loss = self.loss_fn(pred, noise)
            self.log(f"val_loss@{i}", loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        def warmup_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    

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
            pl_module.eval()
            all_images = self.diffusion_utils.p_sample_loop(pl_module, (self.num_samples, 3, self.image_size, self.image_size))
            all_images = (all_images + 1) * 0.5
            img = make_grid(all_images, nrow = 1)
            logger: TensorBoardLogger = trainer.logger
            logger.experiment.add_image(f"generated_images_{trainer.global_step}", img, milestone)
            pl_module.train()
