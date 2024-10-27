import torch

from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose, Lambda, ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss

from dataset import CIFAR10Dataset
from diffusion_utils import DiffusionUtils
from model import UNet

import numpy as np


IMAGE_SIZE = 128
TIMESTEPS = 2000
BATCH_SIZE = 2
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
FIRST_LAYER_CHANNELS = 128
CHANNELS_MULTIPLIER = [1, 2, 2, 2]
NUM_RES_BLOCKS = 2
ATTN_RESOLUTIONS = [16]
LOG_INTERVAL = 100
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
        for step, batch in tqdm(enumerate(dataloader)):
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
                print("Loss:", loss.item())


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


if __name__ == "__main__":
    main()
