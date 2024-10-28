import torch

from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Lambda, ToPILImage
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from pathlib import Path

from dataset import CIFAR10Dataset
from diffusion_utils import DiffusionUtils
from model import UNet

import numpy as np


IMAGE_SIZE = 128
TIMESTEPS = 300
BATCH_SIZE = 16
VIEW_SAMPLE_SIZE = 10
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
FIRST_LAYER_CHANNELS = 128
CHANNELS_MULTIPLIER = [1, 2, 2, 2]
NUM_RES_BLOCKS = 2
ATTN_RESOLUTIONS = [16]
LOG_INTERVAL = 100
RESULTS_FOLDER = Path("./results")
RESULTS_FOLDER.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    transform = Compose([
        Resize(IMAGE_SIZE),
        CenterCrop(IMAGE_SIZE),
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
    model = UNet(IMAGE_SIZE, FIRST_LAYER_CHANNELS, CHANNELS_MULTIPLIER, NUM_RES_BLOCKS, ATTN_RESOLUTIONS)
    model.to(DEVICE)
    diffusion = DiffusionUtils(TIMESTEPS, DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = MSELoss()

    for epoch in range(EPOCHS):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataset) // BATCH_SIZE):
            optimizer.zero_grad()

            x_0, t = batch
            x_0 = x_0.to(DEVICE)
            t = t.to(DEVICE)
            t = t.squeeze(1)
            
            noise = torch.randn_like(x_0).to(DEVICE)
            x_t = diffusion.q_sample(x_0, t, noise)
            
            pred = model(x_t, t)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()

            if step % LOG_INTERVAL == 0:
                print(f"\nEpoch {epoch} Step {step} Loss: {loss.item()}")

            # save generated images
            if step != 0 and step % LOG_INTERVAL == 0:
                model.eval()
                milestone = step // LOG_INTERVAL
                all_images = diffusion.p_sample_loop(model, (VIEW_SAMPLE_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
                print(all_images.shape)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(RESULTS_FOLDER / f'sample-{milestone}.png'), nrow = 1)
                model.train()


        torch.save(model.state_dict(), str(RESULTS_FOLDER / f'model-{epoch}.pth'))


if __name__ == "__main__":
    main()
