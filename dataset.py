import torch
import kagglehub

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, transform, train):
        super().__init__()
        self.train = train
        self.transform = transform
        self.dataset = CIFAR10(root='./data', train=True, download=True)
        
    def __len__(self):
        if self.train:
            return len(self.dataset)
        else:
            return 1000
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return self.transform(image)

class CelebADataset(Dataset):
    def __init__(self, timesteps, transform):
        super().__init__()
        self.timesteps = timesteps
        self.transform = transform
        self.path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")

    def __len__(self):
        return 30000
    
    def __getitem__(self, idx):
        image = Image.open(f"{self.path}/celeba_hq_256/{idx:05d}.jpg")
        return self.transform(image)
