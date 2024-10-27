import torch

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, timesteps, transform, train):
        super().__init__()
        self.timesteps = timesteps
        self.transform = transform
        self.dataset = CIFAR10(root='./data', train=train, download=True)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return self.transform(image), torch.randint(0, self.timesteps, (1,))